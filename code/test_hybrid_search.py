#!/usr/bin/env python3
"""
Hybrid Search 테스트 스크립트
Sparse와 Dense 검색 결과를 비교하고 Hybrid Search의 효과를 확인합니다.
"""

import os
import json
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# .env 파일 로드
load_dotenv()

# Elasticsearch 연결
es_username = "elastic"
es_password = os.getenv("ELASTICSEARCH_PASSWORD")

es = Elasticsearch(
    ['http://localhost:9200'],
    basic_auth=(es_username, es_password),
)

# 임베딩 모델 로드
encoder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def get_embedding(text_list):
    """텍스트를 벡터로 변환"""
    return encoder.encode(text_list, show_progress_bar=False)

def sparse_retrieve(query_str, size):
    """BM25 기반 키워드 검색"""
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size):
    """벡터 유사도 검색"""
    query_embedding = get_embedding([query_str])[0]

    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    return es.search(index="test", knn=knn)

def hybrid_retrieve(query_str, size=3, alpha=0.4):
    """Hybrid Search 구현"""
    candidate_size = size * 3

    sparse_results = sparse_retrieve(query_str, candidate_size)
    sparse_hits = sparse_results['hits']['hits']

    dense_results = dense_retrieve(query_str, candidate_size)
    dense_hits = dense_results['hits']['hits']

    doc_scores = {}

    # Sparse 점수 처리
    max_sparse_score = sparse_hits[0]['_score'] if sparse_hits else 1.0
    for hit in sparse_hits:
        docid = hit['_source']['docid']
        normalized_score = hit['_score'] / max_sparse_score if max_sparse_score > 0 else 0

        doc_scores[docid] = {
            'sparse_score': alpha * normalized_score,
            'dense_score': 0,
            'content': hit['_source']['content'][:200],  # 첫 200자만
            'original_sparse_score': hit['_score']
        }

    # Dense 점수 처리
    for i, hit in enumerate(dense_hits):
        docid = hit['_source']['docid']
        rank_score = 1.0 - (i / len(dense_hits))

        if docid in doc_scores:
            doc_scores[docid]['dense_score'] = (1 - alpha) * rank_score
        else:
            doc_scores[docid] = {
                'sparse_score': 0,
                'dense_score': (1 - alpha) * rank_score,
                'content': hit['_source']['content'][:200],
                'original_sparse_score': 0
            }

    # 최종 점수 계산
    for docid in doc_scores:
        doc_scores[docid]['final_score'] = (
            doc_scores[docid]['sparse_score'] +
            doc_scores[docid]['dense_score']
        )

    # 정렬
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1]['final_score'],
        reverse=True
    )[:size]

    return sorted_docs

def test_queries():
    """테스트 쿼리 실행"""
    test_queries = [
        "DNA의 구조는 어떻게 되어 있나요?",
        "광합성 과정에서 엽록소의 역할",
        "바이러스와 세균의 차이점",
        "화성 탐사의 역사",
        "인공지능과 머신러닝의 차이"
    ]

    for query in test_queries:
        print("\n" + "="*80)
        print(f"쿼리: {query}")
        print("="*80)

        # Sparse 검색
        sparse_result = sparse_retrieve(query, 3)
        print("\n[Sparse Search (BM25)]")
        for i, hit in enumerate(sparse_result['hits']['hits'], 1):
            print(f"{i}. DocID: {hit['_source']['docid'][:8]}... | Score: {hit['_score']:.3f}")
            print(f"   내용: {hit['_source']['content'][:100]}...")

        # Dense 검색
        dense_result = dense_retrieve(query, 3)
        print("\n[Dense Search (Vector)]")
        for i, hit in enumerate(dense_result['hits']['hits'], 1):
            print(f"{i}. DocID: {hit['_source']['docid'][:8]}... | Score: {hit['_score']:.3f}")
            print(f"   내용: {hit['_source']['content'][:100]}...")

        # Hybrid 검색
        print("\n[Hybrid Search (α=0.4)]")
        hybrid_result = hybrid_retrieve(query, 3, alpha=0.4)
        for i, (docid, scores) in enumerate(hybrid_result, 1):
            print(f"{i}. DocID: {docid[:8]}... | Final: {scores['final_score']:.3f}")
            print(f"   Sparse: {scores['sparse_score']:.3f} | Dense: {scores['dense_score']:.3f}")
            print(f"   내용: {scores['content'][:100]}...")

        # 다른 alpha 값으로 테스트
        for alpha in [0.3, 0.5, 0.6]:
            print(f"\n[Hybrid Search (α={alpha})]")
            hybrid_result = hybrid_retrieve(query, 3, alpha=alpha)
            top_doc = hybrid_result[0] if hybrid_result else None
            if top_doc:
                docid, scores = top_doc
                print(f"Top-1: {docid[:8]}... | Score: {scores['final_score']:.3f}")

def compare_alpha_values():
    """다양한 alpha 값 비교"""
    query = "DNA 복제 과정에서 헬리케이스의 역할"

    print(f"\n쿼리: {query}")
    print("\nAlpha 값에 따른 상위 3개 문서:")

    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\nα={alpha} (Sparse: {alpha*100:.0f}%, Dense: {(1-alpha)*100:.0f}%)")
        hybrid_result = hybrid_retrieve(query, 3, alpha=alpha)
        for i, (docid, scores) in enumerate(hybrid_result[:3], 1):
            print(f"  {i}. {docid[:8]}... | Score: {scores['final_score']:.3f}")

if __name__ == "__main__":
    print("Hybrid Search 테스트 시작\n")

    # Elasticsearch 연결 확인
    if es.ping():
        print("✓ Elasticsearch 연결 성공")

        # 인덱스 확인
        if es.indices.exists(index="test"):
            print("✓ 'test' 인덱스 존재")

            # 문서 수 확인
            count = es.count(index="test")['count']
            print(f"✓ 총 {count}개 문서 인덱싱됨\n")

            # 테스트 실행
            test_queries()

            print("\n" + "="*80)
            print("Alpha 값 비교 테스트")
            print("="*80)
            compare_alpha_values()

        else:
            print("✗ 'test' 인덱스가 없습니다. 먼저 데이터를 인덱싱하세요.")
    else:
        print("✗ Elasticsearch 연결 실패. Elasticsearch가 실행 중인지 확인하세요.")