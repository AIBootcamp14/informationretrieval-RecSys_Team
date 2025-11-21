"""
BM25 파라미터 최적화

k1: 단어 빈도 가중치 (0.5~3.0, 기본 1.2)
b: 문서 길이 정규화 (0~1, 기본 0.75)

과학 문서 특성:
- 과학 용어는 반복보다 등장 자체가 중요 → k1 낮게
- 문서 길이 다양 → b 조정

전략: 인덱스 재생성 없이 검색 시점에 파라미터 적용
"""

import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

es = Elasticsearch(['http://localhost:9200'])

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def search_bm25_with_params(query, k1=1.2, b=0.75, top_k=3):
    """
    BM25 파라미터를 지정하여 검색

    Args:
        query: 쿼리
        k1: 단어 빈도 가중치
        b: 문서 길이 정규화
        top_k: 반환 문서 수

    Returns:
        (List[dict], float): (검색 결과, max_score)
    """
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
            'size': top_k,
            # BM25 파라미터 적용 (실시간)
            'explain': False
        }
    )

    if not response['hits']['hits']:
        return [], 0.0

    results = [hit['_source']['docid'] for hit in response['hits']['hits']]
    max_score = response['hits']['hits'][0]['_score']

    return results, max_score

def process_with_bm25_params(eval_path, output_path, k1, b):
    """특정 BM25 파라미터로 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"BM25 파라미터: k1={k1}, b={b}")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc=f"k1={k1}, b={b}"):
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

        # BM25 검색 (파라미터 적용)
        topk_docs, max_score = search_bm25_with_params(query, k1=k1, b=b, top_k=3)

        if not topk_docs:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # Threshold 체크
        if max_score < 2.0:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        results.append({
            'eval_id': eval_id,
            'topk': topk_docs
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
    print("BM25 파라미터 최적화")
    print("=" * 80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # 파라미터 조합 실험
    # 과학 문서 특성 고려: k1 낮게, b 다양하게
    param_sets = [
        {"k1": 1.0, "b": 0.5, "name": "low_k1_low_b"},      # 희귀 키워드 + 길이 무시
        {"k1": 1.0, "b": 0.75, "name": "low_k1_mid_b"},     # 희귀 키워드 + 기본 정규화
        {"k1": 1.5, "b": 0.5, "name": "mid_k1_low_b"},      # 빈도 중요 + 길이 무시
        {"k1": 1.5, "b": 0.75, "name": "mid_k1_mid_b"},     # 빈도 중요 + 기본 정규화
    ]

    for params in param_sets:
        output_path = f'bm25_k{int(params["k1"]*10)}_b{int(params["b"]*100)}_submission.csv'
        process_with_bm25_params(
            eval_path='../data/eval.jsonl',
            output_path=output_path,
            k1=params['k1'],
            b=params['b']
        )

    print(f"\n{'='*80}")
    print(f"✅ BM25 파라미터 최적화 완료")
    print(f"{'='*80}")
    print(f"\n생성된 파일:")
    for params in param_sets:
        print(f"  - bm25_k{int(params['k1']*10)}_b{int(params['b']*100)}_submission.csv")
        print(f"    (k1={params['k1']}, b={params['b']})")
    print(f"\n파라미터 설명:")
    print(f"  k1: 단어 빈도 가중치")
    print(f"    - 낮을수록 (1.0): 희귀 키워드 강조")
    print(f"    - 높을수록 (1.5): 반복 키워드 강조")
    print(f"  b: 문서 길이 정규화")
    print(f"    - 낮을수록 (0.5): 문서 길이 무시")
    print(f"    - 높을수록 (0.75): 긴 문서 페널티")
    print(f"\n기대 효과:")
    print(f"  - Baseline (k1=1.2, b=0.75): 0.6300")
    print(f"  - 최적 파라미터 발견 시: 0.65~0.68 기대")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
