import faiss
import pandas as pd
import json
from datetime import datetime
import hashlib
import os

# Run this inside the Colab notebook after the index is built!

os.makedirs("demo_app/data", exist_ok=True)
faiss.write_index(faiss_index, "demo_app/data/faiss.index")

def get_id(row):
    return hashlib.md5(json.dumps(row, sort_keys=True).encode('utf-8')).hexdigest()

set_a_ids = [get_id(row) for row in set_a_rows]
with open("demo_app/data/set_a_ids.json", "w") as f:
    json.dump(set_a_ids, f)

chunks_data = []
for i, row in enumerate(index_rows):
    chunks_data.append({
        "chunk_id": get_id(row),
        "text": all_chunks[i],
        "intent": row.get("intent", ""),
        "category": row.get("category", "")
    })

df = pd.DataFrame(chunks_data)
df.to_parquet("demo_app/data/chunks.parquet")

meta = {
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dim": faiss_index.d,
    "normalize_embeddings": True,
    "faiss_index_type": "IndexFlatIP",
    "n_chunks": faiss_index.ntotal,
    "dataset_name": DATASET_NAME,
    "export_date": datetime.now().isoformat()
}
with open("demo_app/data/meta.json", "w") as f:
    json.dump(meta, f)
print("Saved FAISS index, chunks.parquet, set_a_ids.json, and meta.json to demo_app/data/")
