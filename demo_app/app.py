import streamlit as st
import json
import time
import os

st.set_page_config(
    page_title="Grounded Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #070b16 0%, #0c1120 60%, #080c18 100%); }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero { text-align:center; padding: 2.2rem 1rem 0.6rem; }
.hero-title {
    font-size: 2.8rem; font-weight: 900; letter-spacing:-0.02em;
    background: linear-gradient(130deg,#818cf8 0%,#a78bfa 40%,#38bdf8 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { color:#475569; font-size:0.9rem; margin-top:0.3rem; letter-spacing:0.01em; }

/* ── Status strip ── */
.status-strip {
    display:flex; gap:0.6rem; justify-content:center; flex-wrap:wrap;
    margin: 0.8rem 0 0;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sp {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07);
    border-radius:30px; padding:0.3rem 1rem; display:flex; align-items:center; gap:0.5rem;
}
.sp-label { color:#334155; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; }
.sp-value { color:#e2e8f0; font-size:0.8rem; font-weight:700; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:transparent; border-bottom: 2px solid rgba(255,255,255,0.06);
    gap:0; padding: 0 2rem;
}
.stTabs [data-baseweb="tab"] {
    color:#475569; font-weight:600; font-size:0.9rem;
    border-radius:0; padding:0.9rem 2rem;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
    background:transparent !important; border-top:none !important;
    border-left:none !important; border-right:none !important;
    transition: color .2s, border-color .2s;
}
.stTabs [aria-selected="true"] {
    color:#818cf8 !important;
    border-bottom: 3px solid #818cf8 !important;
    background:transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#94a3b8 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:2rem; }

/* ── Centre query card ── */
.query-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2.4rem 1.8rem;
    max-width: 680px;
    margin: 0 auto 1.8rem;
}
.qcard-label {
    color:#64748b; font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;
}

/* ── Input override ── */
.stTextInput input {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:#f1f5f9 !important;
    font-size:0.92rem !important; padding:0.7rem 1rem !important;
}
.stTextInput input:focus {
    border-color:rgba(129,140,248,0.6) !important;
    box-shadow:0 0 0 3px rgba(129,140,248,0.1) !important;
}
.stTextInput input::placeholder { color:#334155 !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:#e2e8f0 !important;
}

/* ── Run button ── */
div[data-testid="stFormSubmitButton"] > button,
.run-btn button {
    background: linear-gradient(130deg, #6366f1, #8b5cf6) !important;
    border: none !important; color:#fff !important;
    font-weight:700 !important; font-size:0.9rem !important;
    border-radius:10px !important; padding:0.65rem 2.4rem !important;
    width:100% !important; cursor:pointer !important;
    transition: opacity .2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
div[data-testid="stFormSubmitButton"] > button:hover { opacity:0.88 !important; }

/* ── Node badges ── */
.node-path { display:flex; align-items:center; gap:5px; flex-wrap:wrap; justify-content:center; margin:1rem 0 0.5rem; }
.nb { padding:3px 13px; border-radius:20px; font-size:0.72rem; font-weight:700; letter-spacing:.03em; }
.nb-router    { background:rgba(99,102,241,.18); color:#a5b4fc; border:1px solid rgba(99,102,241,.35); }
.nb-retriever { background:rgba(6,182,212,.14);  color:#67e8f9; border:1px solid rgba(6,182,212,.3);  }
.nb-generator { background:rgba(245,158,11,.14); color:#fcd34d; border:1px solid rgba(245,158,11,.3); }
.nb-critic    { background:rgba(168,85,247,.14);  color:#d8b4fe; border:1px solid rgba(168,85,247,.3); }
.nb-escalation{ background:rgba(239,68,68,.14);   color:#fca5a5; border:1px solid rgba(239,68,68,.3);  }

/* ── Answer boxes ── */
.answer-box {
    background:linear-gradient(135deg,rgba(99,102,241,.07),rgba(139,92,246,.05));
    border:1px solid rgba(99,102,241,.22); border-radius:14px;
    padding:1.4rem 1.8rem; color:#e2e8f0; line-height:1.78; font-size:0.93rem;
    max-width:680px; margin:0 auto;
}
.escalated-box {
    background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.22);
    border-radius:14px; padding:1.4rem 1.8rem;
    color:#fca5a5; font-size:.93rem; line-height:1.78;
    max-width:680px; margin:0 auto;
}

/* ── Stat chips ── */
.stat-row { display:flex; gap:.7rem; justify-content:center; flex-wrap:wrap; margin:.9rem 0; }
.sc {
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:.5rem 1rem; text-align:center; min-width:100px;
}
.sc-label { color:#334155; font-size:.65rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.sc-val   { color:#e2e8f0; font-size:.86rem; font-weight:700; margin-top:3px; }

/* ── Badges ── */
.b-ok  { background:rgba(16,185,129,.16); color:#34d399; border:1px solid rgba(16,185,129,.3); padding:2px 12px; border-radius:20px; font-size:.72rem; font-weight:700; }
.b-bad { background:rgba(239,68,68,.16);  color:#f87171; border:1px solid rgba(239,68,68,.3);  padding:2px 12px; border-radius:20px; font-size:.72rem; font-weight:700; }

/* ── Pipeline diagram ── */
.pipe-wrap {
    display:flex; align-items:center; justify-content:center; gap:7px; flex-wrap:wrap;
    padding:1rem 2rem;
    background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.05);
    border-radius:14px; margin:1.6rem auto 0; max-width:680px;
}
.pn { padding:5px 15px; border-radius:25px; font-size:.76rem; font-weight:700; }
.pa { color:#1e293b; }

/* ── Chunk card ── */
.chunk-card {
    background:rgba(255,255,255,.03); border-left:3px solid #6366f1;
    border-radius:8px; padding:.65rem 1rem; margin-bottom:.45rem;
    font-size:.79rem; color:#94a3b8; line-height:1.5;
}

/* ── Eval cards ── */
.card {
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
    border-radius:14px; padding:1.4rem; margin-bottom:1rem;
}
.mc { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07); border-radius:12px; padding:1.1rem; text-align:center; }
.mc-label { color:#475569; font-size:.65rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.mc-val   { font-size:1.8rem; font-weight:800; color:#f1f5f9; line-height:1.15; }
.mc-base  { color:#475569; font-size:.7rem; margin-top:1px; }
.mc-pos   { color:#10b981; font-size:.76rem; font-weight:700; }
.mc-neg   { color:#f87171; font-size:.76rem; font-weight:700; }

/* ── Regression cards ── */
.reg-card {
    border-radius:14px; padding:1.3rem 1.6rem;
}
.reg-bad { background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.22); }
.reg-ok  { background:rgba(16,185,129,.05); border:1px solid rgba(16,185,129,.18); }

/* ── Table ── */
.ctable { width:100%; border-collapse:collapse; font-size:.87rem; }
.ctable th { background:rgba(129,140,248,.13); color:#a5b4fc; padding:.7rem 1rem; text-align:left; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.ctable td { padding:.62rem 1rem; border-bottom:1px solid rgba(255,255,255,.05); color:#cbd5e1; }
.ctable tr:hover td { background:rgba(255,255,255,.03); }
.tpos { color:#34d399; font-weight:700; }
.tneg { color:#f87171; font-weight:700; }
.tneut { color:#475569; }

/* ── Sec title ── */
.sec { font-size:1.05rem; font-weight:700; color:#f1f5f9; margin:1.5rem 0 .7rem; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background:rgba(255,255,255,.03) !important; border:1px solid rgba(255,255,255,.06) !important;
    border-radius:10px !important; color:#64748b !important;
}
.stSpinner > div { border-top-color:#818cf8 !important; }
hr { border-color:rgba(255,255,255,.06) !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:rgba(129,140,248,.4); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from pipeline import run_query, GROQ_MODEL
    return run_query, GROQ_MODEL

@st.cache_resource(show_spinner=False)
def get_store_info():
    from store import faiss_index, meta
    return faiss_index.ntotal, faiss_index.d, meta.get("embedding_model", "all-MiniLM-L6-v2")

# ─── HERO (full width) ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🤖 Grounded Support Agent</div>
    <div class="hero-sub">LangGraph RAG Pipeline &nbsp;·&nbsp; Router → Retriever → Generator → Critic &nbsp;·&nbsp; Auto-Escalation with Retry</div>
</div>
""", unsafe_allow_html=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
lp = st.empty()
with lp.container():
    with st.spinner("🔄 Loading embedding model & FAISS index…"):
        try:
            run_query, GROQ_MODEL = load_pipeline()
            n_chunks, embed_dim, embed_model_name = get_store_info()
            pipeline_ready = True
        except Exception as e:
            pipeline_ready = False
            load_err = str(e)
lp.empty()

if not pipeline_ready:
    st.error(f"❌ Failed to load pipeline: `{load_err}`")
    st.info("Ensure `data/` folder contains `faiss.index`, `chunks.parquet`, `meta.json`, `set_a_ids.json` and `GROQ_API_KEY` is set in `.env`.")
    st.stop()

# ─── STATUS BAR (full width) ──────────────────────────────────────────────────
st.markdown(f"""
<div class="status-strip">
    <div class="sp"><span class="sp-label">Status</span><span class="sp-value" style="color:#34d399">● Live</span></div>
    <div class="sp"><span class="sp-label">Model</span><span class="sp-value" style="color:#a5b4fc">{GROQ_MODEL}</span></div>
    <div class="sp"><span class="sp-label">KB Chunks</span><span class="sp-value" style="color:#67e8f9">{n_chunks:,}</span></div>
    <div class="sp"><span class="sp-label">Embed Dim</span><span class="sp-value" style="color:#fcd34d">{embed_dim}d</span></div>
    <div class="sp"><span class="sp-label">Retriever</span><span class="sp-value" style="color:#d8b4fe">FAISS · Inner Product</span></div>
    <div class="sp"><span class="sp-label">Max Retries</span><span class="sp-value" style="color:#fca5a5">2</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS (only 2) ────────────────────────────────────────────────────────────
tab_demo, tab_eval = st.tabs(["🚀  Live Demo", "📊  Evaluation Results"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:

    TEST_QUERIES = {
        "── Select a test query ──": "",
        # Account Access
        "🔑 [account_access]  How do I reset my password?": "How do I reset my password?",
        "🔑 [account_access]  Why was my account locked after 3 failed logins?": "Why was my account locked after 3 failed login attempts?",
        "🔑 [account_access]  How do I change my email address?": "How do I change my email address on my account?",
        "🔑 [account_access]  Can I have two accounts with the same email? ❌ not in KB": "Can I have two accounts with the same email address?",
        # Order Management
        "📦 [order_management]  How do I cancel my order?": "How do I cancel my order?",
        "📦 [order_management]  Can I change delivery address after placing an order?": "Can I change the delivery address after placing an order?",
        "📦 [order_management]  I placed two orders by mistake — can you merge them? ❌ not in KB": "I placed two orders by mistake, can you merge them into one?",
        # Shipping
        "🚚 [shipping_delivery]  Where is my package?": "Where is my package? It hasn't arrived yet.",
        "🚚 [shipping_delivery]  My package shows delivered but I never got it.": "My package shows delivered but I never received it.",
        "🚚 [shipping_delivery]  How long does standard shipping take?": "How long does standard shipping take?",
        "🚚 [shipping_delivery]  Can I schedule a 2-hour delivery window? ❌ not in KB": "Can I schedule a specific 2-hour delivery time window?",
        # Returns
        "↩️ [returns_refunds]  What is your return policy?": "What is your return policy?",
        "↩️ [returns_refunds]  How long does a refund take to process?": "How long does a refund take to process?",
        "↩️ [returns_refunds]  Can I return a digital download? ❌ not in KB": "Can I return a digital download that I already accessed?",
        # Billing
        "💳 [billing_payment]  Why was I charged twice for the same order?": "Why was I charged twice for the same order?",
        "💳 [billing_payment]  My payment failed but money left my account.": "My payment failed but the money left my bank account.",
        "💳 [billing_payment]  Do you accept cryptocurrency? ❌ not in KB": "Do you accept cryptocurrency as payment?",
        # Technical
        "🔧 [technical_support]  The app keeps crashing when I open it.": "The app keeps crashing every time I open it.",
        "🔧 [technical_support]  How do I clear the app cache?": "How do I clear the app cache?",
        "🔧 [technical_support]  I'm getting Server Error 503. ❌ not in KB": "I'm getting a Server Error 503 message, what does it mean?",
        # Contact Intent (bypass)
        "👤 [contact_human_agent]  I need to speak to a real human agent right now.  ⚡ bypass": "I need to speak to a real human agent right now.",
        "👤 [contact_human_agent]  Transfer me to a live representative.  ⚡ bypass": "Transfer me to a live representative please.",
        "👤 [contact_human_agent]  Let me talk to your manager.  ⚡ bypass": "Let me talk to your manager immediately.",
        # Out of scope
        "🌀 [out_of_scope]  What is quantum entanglement? ❌ hard OOS": "What is the quantum entanglement of dark matter?",
        "🌀 [out_of_scope]  Who won the 2022 FIFA World Cup? ❌ hard OOS": "Who won the 2022 FIFA World Cup?",
        "🌀 [out_of_scope]  Are you hiring software engineers? ❌ hard OOS": "Are you hiring software engineers right now?",
    }

    # ── Centre card ────────────────────────────────────────────────────────────
    _, cc, _ = st.columns([1, 3, 1])
    with cc:
        with st.form("query_form"):
            st.markdown('<div class="query-card">', unsafe_allow_html=True)

            # Dropdown
            st.markdown('<div class="qcard-label">Choose a query from dropdown</div>', unsafe_allow_html=True)
            selected_key = st.selectbox("", options=list(TEST_QUERIES.keys()), label_visibility="collapsed")
            dropdown_val = TEST_QUERIES[selected_key]

            st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

            # Manual input
            st.markdown('<div class="qcard-label">Or write a manual query</div>', unsafe_allow_html=True)
            manual_val = st.text_input("", placeholder="e.g.  How do I return an item I bought last week?", label_visibility="collapsed")

            st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("▶  Run Pipeline", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Resolve which query to use ─────────────────────────────────────────────
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

    if submitted:
        query = manual_val.strip() or dropdown_val
        if not query:
            _, wc, _ = st.columns([1, 3, 1])
            with wc:
                st.warning("Please select a query from the dropdown or type one manually.")
        elif st.session_state.query_count >= 20:
            _, wc, _ = st.columns([1, 3, 1])
            with wc:
                st.warning("⚠️ Session limit of 20 queries reached. Refresh to reset.")
        else:
            _, rc, _ = st.columns([1, 3, 1])
            with rc:
                with st.spinner("🔄 Routing → Retrieving → Generating → Critiquing…"):
                    try:
                        res = run_query(query)
                        st.session_state.query_count += 1
                    except Exception as e:
                        st.error(f"❌ Pipeline error: `{e}`")
                        st.stop()

                is_escalated = res.get("escalated", False)
                intent = res.get("intent", "")
                retries = res.get("retry_count", 0)
                verdict = res.get("critic_verdict", {})
                grounded = verdict.get("grounded", False)

                # Answer
                if is_escalated:
                    st.markdown(f'<div class="escalated-box">🔴 <strong>Escalated —</strong> {res["answer"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="answer-box">{res["answer"]}</div>', unsafe_allow_html=True)

                # Stat chips
                gb = '<span class="b-ok">✓ Grounded</span>' if grounded else '<span class="b-bad">✗ Ungrounded</span>'
                eb = '<span class="b-bad">Yes</span>' if is_escalated else '<span class="b-ok">No</span>'
                st.markdown(f"""
                <div class="stat-row">
                    <div class="sc"><div class="sc-label">Critic</div><div class="sc-val">{gb}</div></div>
                    <div class="sc"><div class="sc-label">Intent</div><div class="sc-val" style="color:#a5b4fc;font-size:.74rem">{intent}</div></div>
                    <div class="sc"><div class="sc-label">Escalated</div><div class="sc-val">{eb}</div></div>
                    <div class="sc"><div class="sc-label">Retries</div><div class="sc-val">{retries}</div></div>
                    <div class="sc"><div class="sc-label">Latency</div><div class="sc-val">{res.get('latency',0):.2f}s</div></div>
                    <div class="sc"><div class="sc-label">Tokens</div><div class="sc-val">{res.get('tokens',0)}</div></div>
                </div>
                """, unsafe_allow_html=True)

                # Internals
                with st.expander("🔍 Pipeline Internals — Critic reasoning & retrieved chunks"):
                    if verdict.get("reason"):
                        st.markdown(f"**Critic Reason:** {verdict['reason']}")
                    if verdict.get("unsupported_claims"):
                        st.markdown("**Unsupported Claims:**")
                        for cl in verdict["unsupported_claims"]:
                            st.markdown(f"- `{cl}`")
                    chunks = res.get("retrieved_chunks", [])
                    if chunks:
                        st.markdown(f"**{len(chunks)} Retrieved Chunks (FAISS Top-4):**")
                        for i, ch in enumerate(chunks):
                            txt = ch.get("text", ch) if isinstance(ch, dict) else ch
                            cid = (ch.get("chunk_id", "")[:8] + "…") if isinstance(ch, dict) else ""
                            st.markdown(f'<div class="chunk-card"><strong>[{i+1}] {cid}</strong><br>{txt[:380]}{"…" if len(txt)>380 else ""}</div>', unsafe_allow_html=True)

                # Node path
                if intent in ("contact_human_agent", "contact_customer_service") or "human" in intent or "agent" in intent:
                    path_nodes = ["router", "escalation"]
                elif is_escalated:
                    path_nodes = ["router", "retriever"] + ["generator", "critic"] * min(retries, 2) + ["escalation"]
                else:
                    path_nodes = ["router", "retriever"] + ["generator", "critic"] * max(1, retries)

                path_html = ""
                for i, nd in enumerate(path_nodes):
                    path_html += f'<span class="nb nb-{nd}">{nd}</span>'
                    if i < len(path_nodes) - 1:
                        path_html += '<span style="color:#1e293b;padding:0 1px">›</span>'

                st.markdown(f"""
                <div style="color:#1e293b;text-align:center;font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-top:1.2rem;margin-bottom:.3rem">
                    Execution Path
                </div>
                <div class="node-path">{path_html}</div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div style="color:#1e2940;font-size:.68rem;text-align:right;margin-top:.5rem">{st.session_state.query_count}/20 queries this session</div>', unsafe_allow_html=True)

    # ── Pipeline diagram (always visible) ─────────────────────────────────────
    _, dc, _ = st.columns([1, 3, 1])
    with dc:
        st.markdown("""
        <div class="pipe-wrap">
            <span class="pn nb-router">Router</span><span class="pa">→</span>
            <span class="pn nb-retriever">Retriever</span><span class="pa">→</span>
            <span class="pn nb-generator">Generator</span><span class="pa">→</span>
            <span class="pn nb-critic">Critic</span><span class="pa">→</span>
            <span class="pn" style="background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)">✓ Answer</span>
            <span class="pa" style="color:#1e293b;font-size:.75rem;padding:0 6px">or retry / exhaust →</span>
            <span class="pn nb-escalation">Escalation</span>
        </div>
        <div style="text-align:center;color:#1e3a4a;font-size:.7rem;margin-top:.5rem;padding-bottom:.5rem">
            Contact intents (speak to human / agent) bypass Retriever → Generator → Critic entirely
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    try:
        with open("results/results.json") as f:
            results = json.load(f)
        p2 = results["baseline"]
        p3 = results["qlora"]

        st.markdown('<div class="sec">📊 Set A Evaluation — Phase 2 Baseline vs Phase 3 QLoRA Fine-Tune</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="background:rgba(129,140,248,.05);border-color:rgba(129,140,248,.18)">
            <div style="color:#a5b4fc;font-weight:700;margin-bottom:.5rem">📋 Methodology</div>
            <div style="color:#94a3b8;font-size:.875rem;line-height:1.75">
            Both phases were evaluated on the <strong style="color:#e2e8f0">exact same 250 Set-A held-out queries</strong>
            — never used for training or indexing.<br>
            · <strong style="color:#e2e8f0">Phase 2 Baseline</strong>: Qwen2.5-7B-Instruct (4-bit NF4, no fine-tuning) + FAISS RAG pipeline.<br>
            · <strong style="color:#e2e8f0">Phase 3 QLoRA</strong>: Same pipeline, model fine-tuned via QLoRA on
            <strong style="color:#e2e8f0">1,265 stratified training examples</strong> (5% held-out from each intent category).<br>
            · Scored by an LLM-as-a-Judge on faithfulness, relevancy, correctness, completeness, and tone (RAGAS + custom rubric).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards (top 7)
        top_keys = [
            ("Faithfulness", "faithfulness", "{:.3f}", False),
            ("Answer Relevancy", "answer_relevancy", "{:.3f}", False),
            ("Correctness /5", "correctness_1to5", "{:.2f}", False),
            ("Completeness /5", "completeness_1to5", "{:.2f}", False),
            ("Tone /5", "tone_1to5", "{:.2f}", False),
            ("Grounded Rate", "grounded_rate", "{:.1%}", False),
            ("Escalation Rate", "escalation_rate", "{:.1%}", True),
        ]
        cols = st.columns(len(top_keys))
        for col, (label, key, fmt, inv) in zip(cols, top_keys):
            v2, v3 = p2.get(key, 0), p3.get(key, 0)
            delta = v3 - v2
            good = (delta >= 0) ^ inv
            dc_cls = "mc-pos" if good else "mc-neg"
            arrow = ("▲ " if delta > 0 else "▼ ") + fmt.format(abs(delta))
            col.markdown(f"""
            <div class="mc">
                <div class="mc-label">{label}</div>
                <div class="mc-val">{fmt.format(v3)}</div>
                <div class="mc-base">Base: {fmt.format(v2)}</div>
                <div class="{dc_cls}">{arrow}</div>
            </div>
            """, unsafe_allow_html=True)

        # Full table
        st.markdown('<div class="sec" style="margin-top:1.8rem">Full Metric Comparison Table</div>', unsafe_allow_html=True)
        metric_labels = {
            "faithfulness": "Faithfulness (RAGAS)",
            "answer_relevancy": "Answer Relevancy (RAGAS)",
            "context_precision": "Context Precision (RAGAS)",
            "context_recall": "Context Recall (RAGAS)",
            "correctness_1to5": "Correctness (Judge 1–5)",
            "completeness_1to5": "Completeness (Judge 1–5)",
            "tone_1to5": "Tone (Judge 1–5)",
            "grounded_rate": "Grounded Rate",
            "escalation_rate": "Escalation Rate ⚠️",
        }
        rows = ""
        for key, label in metric_labels.items():
            v2, v3 = p2.get(key, 0), p3.get(key, 0)
            delta = v3 - v2
            inv = key == "escalation_rate"
            good = (delta >= 0) ^ inv
            if abs(delta) < 0.0005:
                ds = '<span class="tneut">≈ 0.000</span>'
            elif good:
                ds = f'<span class="tpos">▲ {abs(delta):.3f}</span>'
            else:
                ds = f'<span class="tneg">▼ {abs(delta):.3f}</span>'
            rows += f"<tr><td>{label}</td><td>{v2:.3f}</td><td>{v3:.3f}</td><td>{ds}</td></tr>"

        st.markdown(f"""
        <div class="card" style="padding:0;overflow:hidden">
        <table class="ctable">
            <thead><tr>
                <th>Metric</th>
                <th>Phase 2 — Baseline (N=250)</th>
                <th>Phase 3 — QLoRA (N=250)</th>
                <th>Δ Change</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table></div>
        """, unsafe_allow_html=True)

        # ── Regression boxes BELOW table ──────────────────────────────────────
        st.markdown('<div class="sec" style="margin-top:1.8rem">Regression Breakdown</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("""
            <div class="reg-card reg-bad">
                <div style="color:#f87171;font-weight:700;font-size:.95rem;margin-bottom:.5rem">
                    📉 Grounded Rate: 88.4% → 64.8%
                </div>
                <div style="color:#94a3b8;font-size:.85rem;line-height:1.7">
                    <strong style="color:#fca5a5">−23.6 percentage points.</strong>
                    Nearly 1 in 4 queries that the base model answered correctly were now escalated after QLoRA fine-tuning.
                    The retrieval layer was identical — this is purely a Generator regression.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.markdown("""
            <div class="reg-card reg-bad">
                <div style="color:#f87171;font-weight:700;font-size:.95rem;margin-bottom:.5rem">
                    📈 Escalation Rate: 11.6% → 35.2%
                </div>
                <div style="color:#94a3b8;font-size:.85rem;line-height:1.7">
                    <strong style="color:#fca5a5">+23.6 percentage points — escalations tripled.</strong>
                    The fine-tuned Generator produced more hedging / refusal language which the Critic
                    correctly flagged as ungrounded, exhausting the retry budget and escalating.
                </div>
            </div>
            """, unsafe_allow_html=True)

        rc3, rc4 = st.columns(2)
        with rc3:
            st.markdown("""
            <div class="reg-card reg-ok" style="margin-top:.7rem">
                <div style="color:#34d399;font-weight:700;font-size:.95rem;margin-bottom:.5rem">
                    ✅ Context Retrieval: Unchanged
                </div>
                <div style="color:#94a3b8;font-size:.85rem;line-height:1.7">
                    Context precision (+0.5pp) and recall (+1.1pp) were essentially unchanged.
                    FAISS retrieval is model-agnostic — confirming the regression is isolated to the Generator.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with rc4:
            st.markdown("""
            <div class="reg-card reg-ok" style="margin-top:.7rem">
                <div style="color:#34d399;font-weight:700;font-size:.95rem;margin-bottom:.5rem">
                    ✅ Critic is Working Correctly
                </div>
                <div style="color:#94a3b8;font-size:.85rem;line-height:1.7">
                    The verify-or-abstain Critic layer correctly caught every hedging answer the fine-tuned model
                    produced. The pipeline escalated as designed — not silently returning bad answers.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Key Finding ───────────────────────────────────────────────────────
        st.markdown('<div class="sec" style="margin-top:1.8rem">🔎 Key Finding</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="background:rgba(239,68,68,.04);border-color:rgba(239,68,68,.2);line-height:1.85;color:#94a3b8;font-size:.9rem">
        Fine-tuning the Generator with QLoRA made the system worse at the metric that matters.
        On a paired, held-out 250-query eval, grounded-answer rate dropped from
        <strong style="color:#f87171">88.4% to 64.8%</strong> and escalations tripled
        (<strong style="color:#f87171">11.6% → 35.2%</strong>), with retrieval unchanged.
        I isolated the regression to the Generator by fixing a bug where the adapter was also corrupting
        the Critic and Judge, and the drop persisted. My leading hypothesis is that SFT on templated gold
        answers taught generic phrasing the Critic can't ground. The verify-or-abstain layer caught it,
        so I shipped the base model.
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("results/results.json not found in results/ folder.")
