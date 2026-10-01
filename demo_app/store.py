import os
import json
import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_store():
    data_dir = "data"
    repo_id = os.environ.get("DATA_REPO")
    
    if repo_id:
        from huggingface_hub import hf_hub_download
        def get_path(filename):
            return hf_hub_download(repo_id=repo_id, filename=f"data/{filename}", repo_type="dataset")
    else:
        def get_path(filename):
            return os.path.join(data_dir, filename)
            
    meta_path = get_path("meta.json")
    index_path = get_path("faiss.index")
    chunks_path = get_path("chunks.parquet")
    set_a_ids_path = get_path("set_a_ids.json")
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    faiss_index = faiss.read_index(index_path)
    chunks_df = pd.read_parquet(chunks_path)
    
    with open(set_a_ids_path, "r") as f:
        set_a_ids = set(json.load(f))
        
    if faiss_index.ntotal != len(chunks_df):
        raise ValueError(f"Index size ({faiss_index.ntotal}) does not match chunks size ({len(chunks_df)})")
        
    if faiss_index.d != meta["embedding_dim"]:
        raise ValueError(f"Index dimension ({faiss_index.d}) does not match meta dimension ({meta['embedding_dim']})")
        
    chunk_ids = set(chunks_df["chunk_id"])
    overlap = set_a_ids.intersection(chunk_ids)
    if overlap:
        raise ValueError(f"Leakage detected: {len(overlap)} Set A IDs appear in chunks!")
        
    embed_model = SentenceTransformer(meta["embedding_model"])
    
    return faiss_index, chunks_df, embed_model, meta

faiss_index, chunks_df, embed_model, meta = load_store()
all_chunks = chunks_df["text"].tolist()
