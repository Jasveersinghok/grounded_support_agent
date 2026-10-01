import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import ROUTER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT, graph, run_query
from store import faiss_index, chunks_df, meta, all_chunks, embed_model

def check_parity():
    print("--- PARITY CHECK ---")
    print("1. Store Assertions")
    print(f"  Index ntotal: {faiss_index.ntotal} | Chunks len: {len(chunks_df)}")
    assert faiss_index.ntotal == len(chunks_df), "ntotal mismatch!"
    print(f"  Index dim: {faiss_index.d} | Meta dim: {meta['embedding_dim']}")
    assert faiss_index.d == meta['embedding_dim'], "dim mismatch!"
    print("  Held-out overlap check passed during load_store().")

    print("\n2. Prompt Check")
    print(f"  Router Prompt Length: {len(ROUTER_SYSTEM_PROMPT)}")
    print(f"  Generator Prompt Length: {len(GENERATOR_SYSTEM_PROMPT)}")
    print(f"  Critic Prompt Length: {len(CRITIC_SYSTEM_PROMPT)}")
    
    print("\n3. Graph Edges")
    for edge in graph.builder.edges:
        print(f"  {edge}")

    queries = [
        "how to reset password", "where is my order", "i want to speak to a human",
        "cancel subscription", "refund status", "technical support needed",
        "what are the specs of product x", "general inquiry", "login issue",
        "wrong item delivered", "change shipping address", "payment failed",
        "update billing info", "delete account", "connect to agent",
        "return policy", "lost package", "exchange item", "create account",
        "quantum entanglement of dark matter"
    ]
    print("\n4. Retrieval and Routing for 20 Queries:")
    for i, q in enumerate(queries):
        qv = embed_model.encode([q], normalize_embeddings=meta.get("normalize_embeddings", True)).astype(np.float32)
        dists, idxs = faiss_index.search(qv, 4)
        chunk_ids = [chunks_df.iloc[j]["chunk_id"] for j in idxs[0] if 0 <= j < len(chunks_df)]
        
        # Determine expected intent bypass without hitting LLM
        if q in ["i want to speak to a human", "connect to agent"]:
            route = "contact_intent_bypass -> escalation"
        else:
            route = "retriever -> generator -> critic"
            
        print(f"Q{i+1}: {q}")
        print(f"  Expected Route: {route}")
        print(f"  Retrieved Chunk IDs: {chunk_ids}")
        print(f"  Scores: {dists[0]}")

if __name__ == "__main__":
    check_parity()
