"""
Dense Retrieval + BM25 Hybrid
의미 기반 검색으로 BM25의 한계 극복

목표: 0.7+ 달성
"""

import json
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Dense 모델 로드
print("Loading Dense Retrieval model...")
dense_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
print("✅ Model loaded")

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def load_documents_with_embeddings():
    """
    문서 임베딩 생성 (한 번만 실행)
    """
    print("\n문서 임베딩 생성 중...")

    with open('../data/documents.jsonl', 'r') as f:
        documents = [json.loads(line) for line in f]

    # 문서 내용 추출
    doc_texts = [doc['content'] for doc in documents]
    doc_ids = [doc['docid'] for doc in documents]

    # 배치로 임베딩 생성 (메모리 효율)
    batch_size = 32
    embeddings = []

    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding documents"):
        batch = doc_texts[i:i+batch_size]
        batch_embeddings = dense_model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)

    # 저장
    np.save('document_embeddings.npy', embeddings)
    with open('document_ids.json', 'w') as f:
        json.dump(doc_ids, f)

    print(f"✅ {len(embeddings)}개 문서 임베딩 완료")
    return embeddings, doc_ids

def load_or_create_embeddings():
    """임베딩 로드 또는 생성"""
    try:
        embeddings = np.load('document_embeddings.npy')
        with open('document_ids.json', 'r') as f:
            doc_ids = json.load(f)
        print(f"✅ 기존 임베딩 로드: {len(embeddings)}개 문서")
        return embeddings, doc_ids
    except FileNotFoundError:
        return load_documents_with_embeddings()

def search_bm25(query, top_k=10):
    """BM25 검색"""
    response = es.search(
        index='test',
        body={
            'query': {
                'match': {
                    'content': {
                        'query': query,
                        'analyzer': 'nori'
                    }
                }
            },
            'size': top_k
        }
    )

    if not response['hits']['hits']:
        return {}, 0.0

    results = {hit['_source']['docid']: hit['_score']
               for hit in response['hits']['hits']}
    max_score = response['hits']['hits'][0]['_score']

    return results, max_score

def search_dense(query, embeddings, doc_ids, top_k=10):
    """Dense 검색 (cosine similarity)"""
    query_embedding = dense_model.encode([query], convert_to_numpy=True)[0]

    # Cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = {doc_ids[idx]: float(similarities[idx])
               for idx in top_indices}

    return results

def hybrid_search(query, embeddings, doc_ids, alpha=0.7, top_k=10):
    """
    Hybrid Search: BM25 + Dense

    alpha: BM25 가중치 (0~1)
    1-alpha: Dense 가중치
    """
    # BM25 검색
    bm25_results, max_score = search_bm25(query, top_k=top_k)

    if not bm25_results:
        return [], 0.0

    # Dense 검색
    dense_results = search_dense(query, embeddings, doc_ids, top_k=top_k)

    # Normalize scores
    if bm25_results:
        max_bm25 = max(bm25_results.values())
        bm25_norm = {doc: score/max_bm25 for doc, score in bm25_results.items()}
    else:
        bm25_norm = {}

    max_dense = max(dense_results.values())
    dense_norm = {doc: score/max_dense for doc, score in dense_results.items()}

    # Combine
    all_docs = set(bm25_norm.keys()) | set(dense_norm.keys())
    hybrid_scores = {}

    for doc in all_docs:
        bm25_s = bm25_norm.get(doc, 0.0)
        dense_s = dense_norm.get(doc, 0.0)
        hybrid_scores[doc] = alpha * bm25_s + (1 - alpha) * dense_s

    # Sort
    sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_docs[:top_k], max_score

def adaptive_topk_selection(max_score):
    """동적 TopK 선택"""
    if max_score < 2.0:
        return 0
    elif max_score < 5.0:
        return 1
    elif max_score < 8.0:
        return 2
    else:
        return 3

def process_eval(eval_path, output_path, embeddings, doc_ids):
    """평가 데이터 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"Hybrid Search (BM25 + Dense) 실행")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc="Processing queries"):
        eval_id = item['eval_id']
        msg = item['msg']

        # 일반 대화
        if eval_id in SMALLTALK_IDS:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # 쿼리 추출
        if isinstance(msg, list):
            query = msg[-1]['content']
        else:
            query = msg

        # Hybrid 검색
        hybrid_results, max_score = hybrid_search(
            query, embeddings, doc_ids,
            alpha=0.7,  # BM25 70%, Dense 30%
            top_k=10
        )

        # TopK 선택
        topk_count = adaptive_topk_selection(max_score)
        topk_docs = [doc_id for doc_id, _ in hybrid_results[:topk_count]]

        results.append({
            'eval_id': eval_id,
            'topk': topk_docs
        })

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 통계
    topk_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for result in results:
        topk_counts[len(result['topk'])] += 1

    print(f"\n{'='*80}")
    print(f"✅ 완료: {output_path}")
    print(f"{'='*80}")
    print(f"\nTopK 분포:")
    print(f"  TopK=0: {topk_counts[0]:3d}개 ({topk_counts[0]/len(results)*100:5.1f}%)")
    print(f"  TopK=1: {topk_counts[1]:3d}개 ({topk_counts[1]/len(results)*100:5.1f}%)")
    print(f"  TopK=2: {topk_counts[2]:3d}개 ({topk_counts[2]/len(results)*100:5.1f}%)")
    print(f"  TopK=3: {topk_counts[3]:3d}개 ({topk_counts[3]/len(results)*100:5.1f}%)")
    print(f"{'='*80}")

def main():
    print("=" * 80)
    print("Dense Retrieval + BM25 Hybrid System")
    print("의미 기반 검색으로 성능 향상")
    print("=" * 80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # 임베딩 로드/생성
    embeddings, doc_ids = load_or_create_embeddings()

    # 처리
    process_eval(
        eval_path='../data/eval.jsonl',
        output_path='hybrid_dense_submission.csv',
        embeddings=embeddings,
        doc_ids=doc_ids
    )

if __name__ == "__main__":
    main()
