import sqlite3
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = 'db/evidence.db'
INDEX_PATH = 'data/faiss_index.bin'
META_PATH = 'data/faiss_metadata.json'
BI_ENCODER_NAME = 'all-MiniLM-L6-v2'
CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.bi_encoder = SentenceTransformer(BI_ENCODER_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.ids = self.metadata['ids']

        self.conn = sqlite3.connect(DB_PATH)

    def get_text_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT text FROM evidence WHERE id = ?", (id,))
        result = cursor.fetchone()
        return result[0] if result else ""

    def process_claim(self, claim_text, top_k_retrieve=30, threshold=2.0, max_results=5):
        claim_embedding = self.bi_encoder.encode([claim_text], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(claim_embedding, top_k_retrieve)

        retrieved_docs = []
        for idx in indices[0]:
            doc_id = self.ids[idx]
            doc_text = self.get_text_by_id(doc_id)
            retrieved_docs.append({
                "id": doc_id, 
                "text": doc_text
            })

        cross_inp = [[claim_text, doc["text"]] for doc in retrieved_docs]
        cross_scores = self.cross_encoder.predict(cross_inp)

        for i in range(len(retrieved_docs)):
            retrieved_docs[i]["rerank_score"] = float(cross_scores[i])

        reranked_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

        final_docs = [doc for doc in reranked_docs if doc["rerank_score"] >= threshold]

        if not final_docs and reranked_docs:
            final_docs = [reranked_docs[0]]

        return final_docs[:max_results]