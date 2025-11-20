# vector_store.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedder import Embedder
from chunker import chunk_document

"""
Vector Store Module
-------------------
- 문서 청크를 받아 embedding 생성
- numpy 기반 vector store 구성
- cosine similarity로 topK 검색
"""

class VectorStore:
    def __init__(self, embedder: Embedder):
        """
        embedder: Embedder 객체
        """
        self.embedder = embedder
        self.chunk_texts = []
        self.chunk_embeds = None
        self.chunk_doc_ids = []

    def build(self, documents, chunk_fn=chunk_document):
        """
        documents: [{"docid": "...", "content": "..."}]
        chunk_fn: chunking 함수 (기본: Section-aware chunk_document)
        """

        chunks = []
        doc_ids = []

        print("[VectorStore] Building chunks...")
        for doc in documents:
            text = doc["content"]
            docid = doc["docid"]

            cks = chunk_fn(text)
            for ck in cks:
                chunks.append(ck)
                doc_ids.append(docid)

        print(f"[VectorStore] Total chunks created: {len(chunks)}")

        print("[VectorStore] Embedding chunks...")
        embeds = self.embedder.embed_chunks(chunks)

        self.chunk_texts = chunks
        self.chunk_doc_ids = doc_ids
        self.chunk_embeds = np.array(embeds)

        print("[VectorStore] Build completed.")
        return self

    def retrieve_topk(self, query: str, k=20):
        """
        단일 query 기준 top-K chunk index 반환
        """
        q_emb = self.embedder.embed_query(query)
        sims = cosine_similarity([q_emb], self.chunk_embeds)[0]
        topk_idx = sims.argsort()[::-1][:k]
        return topk_idx.tolist()

    def get_chunk(self, idx: int):
        return self.chunk_texts[idx]

    def get_doc_id(self, idx: int):
        return self.chunk_doc_ids[idx]
