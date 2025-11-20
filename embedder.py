# embedder.py
import os
from sentence_transformers import SentenceTransformer

"""
Embedder Module
---------------
- 임베딩 모델 로딩
- 텍스트 리스트 임베딩
- 문서/쿼리 통일된 embedding 인터페이스 제공
"""

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"   # 기본 모델: 한국어+과학 QA 최적


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        """
        임베딩 모델 로드
        """
        print(f"[Embedder] Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        문자열 리스트를 batch로 임베딩하여 numpy array 반환
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings

    def embed_query(self, query: str):
        """
        쿼리 한 문장을 embedding
        """
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks):
        """
        chunk 리스트 embedding
        """
        return self.embed_texts(chunks)

