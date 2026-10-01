import json
import logging
import re
import time
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from groq import Groq, RateLimitError
from langgraph.graph import StateGraph, END
import numpy as np

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K, MAX_RETRIES
from store import faiss_index, embed_model, all_chunks, meta, chunks_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_pipeline")

_groq_client = Groq(api_key=GROQ_API_KEY)
_token_counts = {"total": 0}

def _llm(**kwargs):
    kwargs.setdefault("max_tokens", 512)
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.2
    kwargs["model"] = GROQ_MODEL
    try:
        response = _groq_client.chat.completions.create(**kwargs)
        if hasattr(response, 'usage') and response.usage:
            _token_counts["total"] += response.usage.total_tokens
        return response
    except RateLimitError:
        time.sleep(2)
        try:
            response = _groq_client.chat.completions.create(**kwargs)
            if hasattr(response, 'usage') and response.usage:
                _token_counts["total"] += response.usage.total_tokens
            return response
        except RateLimitError:
            raise RuntimeError("Demo is rate-limited, try again shortly.")

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list
    draft_answer: str
    grounded: bool
    retry_count: int
    final_answer: str
    escalated: bool
    node_path: list
    unsupported_claims: list
    critic_reason: str

class CriticResult(BaseModel):
    grounded: bool           = Field(..., description="True if every claim is grounded.")
    unsupported_claims: list = Field(default_factory=list)
    reason: str              = Field(..., description="One-sentence explanation.")

ROUTER_SYSTEM_PROMPT = (
    'You are an intent classifier for a customer-support system.\n'
    'Return a SINGLE intent label from this EXACT list (lowercase_underscored only, no other words):\n'
    '  account_access, order_management, shipping_delivery, returns_refunds,\n'
    '  billing_payment, product_information, technical_support, general_inquiry,\n'
    '  contact_human_agent, contact_customer_service\n'
    '\n'
    'RULES (follow exactly):\n'
    '- If the user wants to SPEAK TO, REACH, CONNECT TO, TRANSFER TO, or GET a human, agent, '
    'live person, representative, or manager -> return: contact_human_agent\n'
    '- If the user mentions customer service, support team, or helpdesk -> return: contact_customer_service\n'
    '- Respond with ONLY the label. No punctuation. No explanation. No extra words.\n'
    '\n'
    'FEW-SHOT EXAMPLES:\n'
    'Q: I need to speak to a real human agent right now -> contact_human_agent\n'
    'Q: Connect me to a customer service representative -> contact_customer_service\n'
    'Q: Let me talk to your manager -> contact_human_agent\n'
    'Q: Transfer me to a live agent -> contact_human_agent\n'
    'Q: I want to talk to a real person -> contact_human_agent\n'
    'Q: Can I get help from someone live? -> contact_human_agent\n'
    'Q: How do I reset my password? -> account_access\n'
    'Q: Where is my order? -> order_management\n'
    'Q: I want to return my item -> returns_refunds'
)

def build_router_messages(query: str) -> list:
    return [
        {'role': 'system', 'content': ROUTER_SYSTEM_PROMPT},
        {'role': 'user',   'content': f'Query: {query}'},
    ]

GENERATOR_SYSTEM_PROMPT = (
    'You are a helpful customer-support assistant.\n'
    'Answer ONLY using the provided context excerpts. Do NOT fabricate information.\n'
    'If context is insufficient, say what you cannot confirm.\n'
    'Be concise (2-5 sentences unless the topic genuinely requires more).'
)

def build_generator_messages(query: str, chunks: list) -> list:
    ctx = '\n\n'.join(f'[Excerpt {i+1}]\n{c}' for i, c in enumerate(chunks))
    user_msg = f'--- CONTEXT EXCERPTS ---\n{ctx}\n--- END ---\n\nCustomer question: {query}\n\nGrounded answer:'
    return [
        {'role': 'system', 'content': GENERATOR_SYSTEM_PROMPT},
        {'role': 'user',   'content': user_msg},
    ]

CRITIC_SYSTEM_PROMPT = (
    'You are a factual-grounding critic for a customer-support AI.\n'
    'Verify whether every claim in the answer is supported by the context excerpts.\n'
    'IMPORTANT: if the answer declines to answer, says it cannot confirm something, '
    'or states the information is not available/not covered by context — treat this '
    'as grounded=false, since it means the context did not actually answer the '
    'customer\'s question and the query should be escalated, not treated as resolved.\n'
    'Return ONLY this JSON object (no markdown fences, no extra keys):\n'
    '{"grounded": true/false, "unsupported_claims": ["..."], "reason": "..."}\n'
    'grounded=true ONLY if the answer both (a) makes claims fully supported by the '
    'context, AND (b) actually addresses the customer\'s question.\n'
    'unsupported_claims: verbatim phrases not backed by any excerpt (empty list if grounded=true).'
)

def build_critic_messages(answer: str, chunks: list) -> list:
    ctx = '\n\n'.join(f'[Excerpt {i+1}]\n{c}' for i, c in enumerate(chunks))
    user_msg = f'--- CONTEXT ---\n{ctx}\n--- END ---\n\n--- ANSWER ---\n{answer}\n--- END ---\n\nReturn JSON verdict.'
    return [
        {'role': 'system', 'content': CRITIC_SYSTEM_PROMPT},
        {'role': 'user',   'content': user_msg},
    ]

def router_node(state: SupportState) -> SupportState:
    logger.info('[Router] classifying: %r', state['query'][:80])
    r = _llm(messages=build_router_messages(state['query']),
             temperature=0.0, max_tokens=32)
    intent = r.choices[0].message.content.strip().lower().split('\n')[0].strip(' .,;"\'')
    logger.info('[Router] intent=%r', intent)
    path = list(state.get('node_path') or []) + ['router']
    return {**state, 'intent': intent, 'node_path': path}

def retriever_node(state: SupportState) -> SupportState:
    logger.info('[Retriever] top-%d search for: %r', TOP_K, state['query'][:80])
    qv = embed_model.encode([state['query']], normalize_embeddings=meta.get("normalize_embeddings", True)).astype(np.float32)
    dists, idxs = faiss_index.search(qv, TOP_K)
    chunks = [all_chunks[i] for i, _ in zip(idxs[0], dists[0]) if 0 <= i < len(all_chunks)]
    logger.info('[Retriever] %d chunks retrieved', len(chunks))
    path = list(state.get('node_path') or []) + ['retriever']
    return {**state, 'retrieved_chunks': chunks, 'node_path': path}

def generator_node(state: SupportState) -> SupportState:
    retry = (state.get('retry_count') or 0) + 1
    logger.info('[Generator] attempt #%d', retry)
    msgs = build_generator_messages(state['query'], state.get('retrieved_chunks') or [])
    if retry > 1:   # stricter system prompt on retries
        msgs[0]['content'] += '\n\nPrevious answer was flagged ungrounded. Be extra conservative.'
    r = _llm(messages=msgs,
             temperature=0.2 if retry == 1 else 0.0, max_tokens=512)
    draft = r.choices[0].message.content.strip()
    logger.info('[Generator] draft: %d chars', len(draft))
    path = list(state.get('node_path') or []) + ['generator']
    return {**state, 'draft_answer': draft, 'retry_count': retry, 'node_path': path}

def _extract_json(text: str) -> dict:
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    raise ValueError(f'No JSON in critic output: {text[:200]}')

def critic_node(state: SupportState) -> SupportState:
    logger.info('[Critic] evaluating draft...')
    r = _llm(messages=build_critic_messages(
                 state.get('draft_answer', ''), state.get('retrieved_chunks') or []),
             temperature=0.0, max_tokens=512)
    raw = r.choices[0].message.content.strip()
    try:
        verdict = CriticResult(**_extract_json(raw))
    except Exception as exc:
        logger.warning('[Critic] parse failed (%s) → ungrounded', exc)
        verdict = CriticResult(grounded=False, unsupported_claims=[], reason='Parse error — treating as ungrounded.')
    logger.info('[Critic] grounded=%s | unsupported=%d | %r',
                verdict.grounded, len(verdict.unsupported_claims), verdict.reason[:60])
    path = list(state.get('node_path') or []) + ['critic']
    out = {**state, 'grounded': verdict.grounded, 'unsupported_claims': verdict.unsupported_claims,
           'critic_reason': verdict.reason, 'node_path': path}
    if verdict.grounded:
        out['final_answer'] = state.get('draft_answer', '')
    return out

ESCALATION_MSG = (
    "I'm sorry, I wasn't able to find a fully verified answer in our knowledge base. "
    "Your query has been escalated to our support team who will follow up shortly. "
    "For urgent issues please contact us via phone or live chat."
)

def escalation_node(state: SupportState) -> SupportState:
    logger.warning('[Escalation] escalating after %d attempts — query: %r',
                   state.get('retry_count', 0), state.get('query', '')[:80])
    path = list(state.get('node_path') or []) + ['escalation']
    return {**state, 'escalated': True, 'final_answer': ESCALATION_MSG, 'node_path': path}

CONTACT_INTENTS = {'contact_human_agent', 'contact_customer_service'}

def _after_router(state: SupportState) -> str:
    intent = state.get('intent', '')
    # Robust check in case the LLM returned slight variations
    if intent in CONTACT_INTENTS or 'human' in intent or 'agent' in intent or 'representative' in intent:
        logger.info('[Graph] contact-intent (%r) -> bypass Generator/Critic -> Escalation', state.get('intent'))
        return 'escalation'
    return 'retriever'

def _after_critic(state: SupportState) -> str:
    if state.get('grounded'):
        logger.info('[Graph] grounded after %d attempt(s) → END', state.get('retry_count', 0))
        return 'end'
    if (state.get('retry_count') or 0) < MAX_RETRIES:
        logger.info('[Graph] not grounded (attempt %d/%d) → retry Generator',
                    state.get('retry_count', 0), MAX_RETRIES)
        return 'generator'
    logger.warning('[Graph] retry budget exhausted → Escalation')
    return 'escalation'

builder = StateGraph(SupportState)
builder.add_node('router',     router_node)
builder.add_node('retriever',  retriever_node)
builder.add_node('generator',  generator_node)
builder.add_node('critic',     critic_node)
builder.add_node('escalation', escalation_node)

builder.set_entry_point('router')
builder.add_conditional_edges(
    'router', _after_router,
    {'retriever': 'retriever', 'escalation': 'escalation'},
)
builder.add_edge('retriever', 'generator')
builder.add_edge('generator', 'critic')
builder.add_conditional_edges(
    'critic', _after_critic,
    {'generator': 'generator', 'escalation': 'escalation', 'end': END},
)
builder.add_edge('escalation', END)
graph = builder.compile()

def run_query(query: str):
    _token_counts["total"] = 0
    start_time = time.time()
    
    initial_state = {
        'query': query, 'intent': '', 'retrieved_chunks': [],
        'draft_answer': '', 'grounded': False, 'retry_count': 0,
        'final_answer': '', 'escalated': False, 'node_path': [],
        'unsupported_claims': [], 'critic_reason': '',
    }
    
    result = graph.invoke(initial_state)
    
    retrieved_chunks_with_id = []
    for chunk_str in result.get("retrieved_chunks", []):
        matching_rows = chunks_df[chunks_df["text"] == chunk_str]
        chunk_id = matching_rows.iloc[0]["chunk_id"] if not matching_rows.empty else "unknown"
        retrieved_chunks_with_id.append({"chunk_id": chunk_id, "text": chunk_str})
        
    critic_verdict = {
        "grounded": result.get("grounded", False),
        "unsupported_claims": result.get("unsupported_claims", []),
        "reason": result.get("critic_reason", "")
    }
    
    latency = time.time() - start_time
    return {
        "answer": result.get("final_answer"),
        "intent": result.get("intent"),
        "retrieved_chunks": retrieved_chunks_with_id,
        "critic_verdict": critic_verdict,
        "retry_count": result.get("retry_count", 0),
        "escalated": result.get("escalated", False),
        "model": GROQ_MODEL,
        "latency": latency,
        "tokens": _token_counts["total"]
    }
