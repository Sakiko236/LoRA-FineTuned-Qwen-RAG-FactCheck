import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = 'data/evidence.db'
INDEX_PATH = 'data/faiss_index.bin'
META_PATH = 'data/faiss_metadata.json'

BI_ENCODER_NAME = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_NAME = 'ms-marco-MiniLM-L-6-v2'

class RAGPipeline:
    def __init__(self):
        print("Initializing English RAG Pipeline with Dense Search (FAISS)...")
        
        self.bi_encoder = SentenceTransformer(BI_ENCODER_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.ids = self.metadata['ids']

        self.conn = sqlite3.connect(DB_PATH)

        print("Bi-encoder device:", self.bi_encoder.device)
        print("Cross-encoder device:", self.cross_encoder.device)

    def get_text_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT text FROM evidence WHERE id = ?", (id,))
        result = cursor.fetchone()
        return result[0] if result else ""

    def process_claim(self, claim_text, top_k_retrieve=20, threshold=0.05, max_results=5):
        retrieved_candidates = {} 

        query_prompt = f"Represent this sentence for searching relevant passages: {claim_text}"
        claim_embedding = self.bi_encoder.encode([query_prompt], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(claim_embedding, top_k_retrieve)
        
        for idx in indices[0]:
            if idx != -1: 
                doc_id = self.ids[idx]
                retrieved_candidates[doc_id] = self.get_text_by_id(doc_id)

        if not retrieved_candidates:
            return []

        retrieved_docs = [{"id": k, "text": v} for k, v in retrieved_candidates.items()]
        
        cross_inp = [[claim_text, doc["text"]] for doc in retrieved_docs]
        cross_scores = self.cross_encoder.predict(cross_inp)

        for i in range(len(retrieved_docs)):
            retrieved_docs[i]["rerank_score"] = float(cross_scores[i])

        reranked_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

        final_docs = [doc for doc in reranked_docs if doc["rerank_score"] >= threshold]

        if not final_docs and reranked_docs:
            final_docs = [reranked_docs[0]]

        return final_docs[:max_results]