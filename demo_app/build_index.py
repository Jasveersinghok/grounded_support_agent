import numpy as np, faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print('Using held-out split from previous cell (eval rows excluded)...')
ds = index_rows
print(f'Indexing {len(ds)} rows (eval rows held out, never indexed).')

all_chunks = []
for row in ds:
    chunk = (
        f"[Category: {row.get('category','')}] [Intent: {row.get('intent','')}]\n"
        f"Q: {row.get('instruction','')}\nA: {row.get('response','')}"
    )
    all_chunks.append(chunk)

print(f'  {len(ds)} records → {len(all_chunks)} chunks (1:1, no fragmentation)')

print(f'Embedding with {EMBEDDING_MODEL}...')
embed_model = SentenceTransformer(EMBEDDING_MODEL)

all_vecs = []
for i in range(0, len(all_chunks), 256):
    vecs = embed_model.encode(
        all_chunks[i:i+256], normalize_embeddings=True, show_progress_bar=False)
    all_vecs.append(vecs)
    if (i // 256) % 5 == 0:
        print(f'  ... {min(i+256, len(all_chunks))}/{len(all_chunks)} embedded')

embeddings = np.vstack(all_vecs).astype(np.float32)

faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

print(f'\nFAISS index ready: {faiss_index.ntotal} vectors, dim={embeddings.shape[1]}')
