import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class FaissBuilder:
    def __init__(self, 
                 db_path='data/evidence.db', 
                 index_path='data/faiss_index.bin', 
                 meta_path='data/faiss_metadata.json', 
                 model_name='BAAI/bge-large-en-v1.5', 
                 batch_size=64):
        self.db_path = db_path
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.batch_size = batch_size

    def build(self):
        print(f"Loading Bi-Encoder model: {self.model_name}...")
        model = SentenceTransformer(self.model_name)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, text FROM evidence")
        rows = cursor.fetchall()

        ids = [row[0] for row in rows]
        texts = [row[1] for row in rows]

        print(f"Loaded {len(texts)} documents, starting vectorization...")

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i+self.batch_size]
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) 
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump({'ids': ids}, f)

        print("Vectorization complete!")
        conn.close()

if __name__ == "__main__":
    builder = FaissBuilder()
    builder.build()