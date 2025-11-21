"""
BM25 중심 Hybrid 최적화

Dense가 BM25보다 성능이 낮으므로 BM25 비중을 높임
Alpha: 0.8, 0.85, 0.9, 0.95, 1.0 (순수 BM25)
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

def load_embeddings():
    """저장된 임베딩 로드"""
    embeddings = np.load('document_embeddings.npy')
    with open('document_ids.json', 'r') as f:
        doc_ids = json.load(f)
    print(f"✅ 임베딩 로드: {len(embeddings)}개 문서")
    return embeddings, doc_ids

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
    """Dense 검색"""
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

def hybrid_search(query, embeddings, doc_ids, alpha=0.9, top_k=10):
    """
    Hybrid Search: BM25 중심

    alpha: BM25 가중치 (0.8~1.0)
    """
    # BM25 검색
    bm25_results, max_score = search_bm25(query, top_k=top_k)

    if not bm25_results:
        return [], 0.0

    if alpha == 1.0:
        # 순수 BM25
        sorted_docs = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k], max_score

    # Dense 검색
    dense_results = search_dense(query, embeddings, doc_ids, top_k=top_k)

    # Normalize scores
    max_bm25 = max(bm25_results.values())
    bm25_norm = {doc: score/max_bm25 for doc, score in bm25_results.items()}

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

def process_with_alpha(eval_path, output_path, embeddings, doc_ids, alpha):
    """특정 alpha 값으로 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"Alpha = {alpha:.2f} (BM25: {alpha*100:.0f}%, Dense: {(1-alpha)*100:.0f}%)")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc=f"Alpha {alpha:.2f}"):
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
            alpha=alpha,
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

    print(f"\n✅ 완료: {output_path}")
    print(f"{'='*80}\n")

def main():
    print("=" * 80)
    print("BM25 중심 Hybrid 최적화")
    print("=" * 80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # 임베딩 로드
    embeddings, doc_ids = load_embeddings()

    # BM25 비중 높은 alpha 값 실험
    alpha_values = [0.8, 0.85, 0.9, 0.95, 1.0]

    for alpha in alpha_values:
        if alpha == 1.0:
            output_path = 'pure_bm25_submission.csv'
        else:
            output_path = f'hybrid_alpha_{int(alpha*100):02d}_submission.csv'

        process_with_alpha(
            eval_path='../data/eval.jsonl',
            output_path=output_path,
            embeddings=embeddings,
            doc_ids=doc_ids,
            alpha=alpha
        )

    print(f"\n{'='*80}")
    print(f"✅ 모든 Alpha 실험 완료")
    print(f"{'='*80}")
    print(f"\n생성된 파일:")
    for alpha in alpha_values:
        if alpha == 1.0:
            print(f"  - pure_bm25_submission.csv (순수 BM25 100%)")
        else:
            print(f"  - hybrid_alpha_{int(alpha*100):02d}_submission.csv (BM25: {alpha*100:.0f}%, Dense: {(1-alpha)*100:.0f}%)")
    print(f"\n기대 효과:")
    print(f"  - Dense 비중을 낮춰서 BM25 중심으로 회귀")
    print(f"  - Alpha 0.8~0.95 중 하나가 0.63보다 높을 가능성")
    print(f"  - 순수 BM25는 super_simple과 다른 TopK 전략 적용")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
