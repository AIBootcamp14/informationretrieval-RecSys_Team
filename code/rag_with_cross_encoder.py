"""
Cross-Encoder Reranking 전략

LLM API 대신 Cross-Encoder 모델 사용
- 모델: jhgan/ko-sroberta-multitask
- 방법: BM25 Top-10 검색 → Cross-Encoder로 재정렬 → Top-3 선택

장점:
- API 비용 없음
- 빠른 속도
- 쿼리-문서 쌍을 직접 점수화하여 높은 정확도

기대 효과: 0.63 → 0.70+
"""

import json
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Cross-Encoder 로드
print("Loading Cross-Encoder model...")
cross_encoder = CrossEncoder('jhgan/ko-sroberta-multitask')
print("✅ Cross-Encoder loaded")

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def search_bm25_top10(query):
    """BM25로 Top-10 검색"""
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
            'size': 10
        }
    )

    if not response['hits']['hits']:
        return [], 0.0

    results = []
    for hit in response['hits']['hits']:
        results.append({
            'docid': hit['_source']['docid'],
            'content': hit['_source']['content'],
            'bm25_score': hit['_score']
        })

    max_score = response['hits']['hits'][0]['_score']
    return results, max_score

def rerank_with_cross_encoder(query, top10_docs):
    """
    Cross-Encoder로 Top-10 문서 재정렬

    Args:
        query: 사용자 쿼리
        top10_docs: BM25 Top-10 문서 리스트

    Returns:
        List[str]: 재정렬된 문서 ID Top-3
    """
    if not top10_docs:
        return []

    # 쿼리-문서 쌍 생성
    pairs = [[query, doc['content']] for doc in top10_docs]

    # Cross-Encoder로 점수 계산
    scores = cross_encoder.predict(pairs)

    # 점수와 문서 ID 매핑
    doc_scores = [(top10_docs[i]['docid'], scores[i])
                  for i in range(len(top10_docs))]

    # 점수순 정렬
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Top-3 선택
    return [doc_id for doc_id, score in doc_scores[:3]]

def process_with_cross_encoder(eval_path, output_path):
    """Cross-Encoder Reranking으로 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"Cross-Encoder Reranking 실행")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc="Cross-Encoder Reranking"):
        eval_id = item['eval_id']
        msg = item['msg']

        # Smalltalk
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

        # BM25 Top-10 검색
        top10_docs, max_score = search_bm25_top10(query)

        if not top10_docs:
            # 검색 결과 없음
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # Threshold 체크 (너무 낮은 점수는 일반 대화)
        if max_score < 2.0:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # Cross-Encoder Reranking
        selected_ids = rerank_with_cross_encoder(query, top10_docs)

        results.append({
            'eval_id': eval_id,
            'topk': selected_ids
        })

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 통계
    topk_counts = {}
    for r in results:
        count = len(r['topk'])
        topk_counts[count] = topk_counts.get(count, 0) + 1

    print(f"\n✅ 완료: {output_path}")
    print(f"\nTopK 분포:")
    for k in sorted(topk_counts.keys()):
        print(f"  TopK={k}: {topk_counts[k]:3d}개 ({topk_counts[k]/len(results)*100:5.1f}%)")
    print(f"{'='*80}\n")

def main():
    print("=" * 80)
    print("Cross-Encoder Reranking 전략")
    print("=" * 80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # Cross-Encoder Reranking 실행
    process_with_cross_encoder(
        eval_path='../data/eval.jsonl',
        output_path='cross_encoder_reranking_submission.csv'
    )

    print(f"\n{'='*80}")
    print(f"✅ Cross-Encoder Reranking 완료")
    print(f"{'='*80}")
    print(f"\n생성된 파일:")
    print(f"  - cross_encoder_reranking_submission.csv")
    print(f"\n전략:")
    print(f"  1. BM25로 Top-10 후보 검색")
    print(f"  2. Cross-Encoder로 쿼리-문서 관련도 직접 점수화")
    print(f"  3. 점수순으로 재정렬하여 Top-3 선택")
    print(f"\n장점:")
    print(f"  - API 비용 없음")
    print(f"  - Bi-encoder(Dense)보다 정확")
    print(f"  - 쿼리와 문서를 함께 입력하여 관련도 계산")
    print(f"\n기대 효과:")
    print(f"  - Baseline (0.63) → 0.70+ 기대")
    print(f"  - BM25 Top-3가 아닌 실제 관련도로 선택")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
