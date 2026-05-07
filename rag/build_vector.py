import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_PATH = 'db/evidence.db'
INDEX_PATH = 'data/faiss_index.bin'
META_PATH = 'data/faiss_metadata.json'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 256

def main():
    model = SentenceTransformer(MODEL_NAME)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, text FROM evidence")
    rows = cursor.fetchall()

    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]

    print(f"Loaded {len(texts)} data, start to vectorize...")

    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump({'ids': ids}, f)

    print("Vectorize complete!")
    conn.close()

if __name__ == "__main__":
    main()