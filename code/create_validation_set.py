"""
Validation Set 생성 도구

전략 1: 수동 Annotation
- eval.jsonl에서 샘플링
- 각 쿼리에 대해 정답 문서를 수동으로 레이블링
- validation.jsonl 생성
"""

import json
import random
from elasticsearch import Elasticsearch

def sample_queries(eval_path, n_samples=20):
    """eval.jsonl에서 랜덤 샘플링"""
    with open(eval_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # 다양한 타입 샘플링
    science_queries = [item for item in data if item['eval_id'] not in
                      {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}]
    smalltalk_queries = [item for item in data if item['eval_id'] in
                        {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}]

    # 과학 질문 15개, 일반 대화 5개
    samples = random.sample(science_queries, min(15, len(science_queries))) + \
              random.sample(smalltalk_queries, min(5, len(smalltalk_queries)))

    return samples

def search_and_display(es, query, top_k=10):
    """검색 결과를 보여주고 정답 선택"""
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

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")

    results = []
    for i, hit in enumerate(response['hits']['hits'], 1):
        docid = hit['_source']['docid']
        score = hit['_score']
        content = hit['_source']['content'][:200]

        print(f"\n[{i}] Score: {score:.2f}")
        print(f"DocID: {docid}")
        print(f"Content: {content}...")

        results.append({
            'docid': docid,
            'score': score,
            'content': hit['_source']['content']
        })

    return results

def annotate_samples(samples, es):
    """샘플에 정답 레이블 추가"""
    annotated = []

    for item in samples:
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        print(f"\n\n{'#'*80}")
        print(f"Eval ID: {eval_id}")
        print(f"{'#'*80}")

        # 검색 결과 표시
        results = search_and_display(es, query, top_k=10)

        # 정답 입력
        print(f"\n정답 문서 번호를 입력하세요 (1-10, 여러 개는 콤마로 구분, 일반 대화는 0):")
        answer = input("> ").strip()

        if answer == "0":
            ground_truth = []
        else:
            indices = [int(x.strip())-1 for x in answer.split(',')]
            ground_truth = [results[i]['docid'] for i in indices if 0 <= i < len(results)]

        annotated.append({
            'eval_id': eval_id,
            'query': query,
            'msg': item['msg'],
            'ground_truth': ground_truth
        })

        print(f"✅ 저장: {len(ground_truth)}개 문서")

    return annotated

def save_validation_set(annotated, output_path):
    """validation set 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"✅ Validation set 저장 완료: {output_path}")
    print(f"총 {len(annotated)}개 샘플")
    print(f"{'='*80}")

def main():
    print("=" * 80)
    print("Validation Set 생성 도구")
    print("=" * 80)

    # Elasticsearch 연결
    es = Elasticsearch(['http://localhost:9200'])
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공\n")

    # 샘플 선택
    samples = sample_queries('../data/eval.jsonl', n_samples=20)
    print(f"📋 {len(samples)}개 샘플 선택됨")

    # Annotation
    annotated = annotate_samples(samples, es)

    # 저장
    save_validation_set(annotated, 'validation.jsonl')

if __name__ == "__main__":
    main()
