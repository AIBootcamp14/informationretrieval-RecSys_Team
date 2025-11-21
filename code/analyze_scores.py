"""BM25 점수 분포 분석"""

import json
from elasticsearch import Elasticsearch
import numpy as np

# Elasticsearch 연결
es = Elasticsearch(['http://localhost:9200'])

# eval.jsonl 로드
dataset = []
with open('../data/eval.jsonl', 'r') as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

# 점수 수집
all_scores = []
smalltalk_scores = []
science_scores = []

SMALLTALK_IDS = {276, 261, 233, 90, 222, 37, 70, 235, 91, 265, 26, 260, 51, 30, 60}

print("점수 분포 분석 중...")
for item in dataset[:50]:  # 처음 50개만 샘플링
    eval_id = item['eval_id']

    if 'msg' in item and isinstance(item['msg'], list) and item['msg']:
        query = item['msg'][-1].get('content', '')
    else:
        query = item.get('msg', item.get('query', ''))

    if not query:
        continue

    # BM25 검색
    try:
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

        if response['hits']['hits']:
            top_score = response['hits']['hits'][0]['_score']
            all_scores.append(top_score)

            if eval_id in SMALLTALK_IDS:
                smalltalk_scores.append(top_score)
            else:
                science_scores.append(top_score)

            # 상위 3개 점수 출력
            scores = [hit['_score'] for hit in response['hits']['hits'][:3]]
            if eval_id in SMALLTALK_IDS:
                print(f"[일반] eval_id {eval_id}: {scores} - '{query[:30]}...'")
            else:
                print(f"[과학] eval_id {eval_id}: {scores}")

    except Exception as e:
        continue

print("\n" + "=" * 60)
print("📊 점수 분포 요약")
print("=" * 60)

if all_scores:
    print(f"\n전체 점수 (n={len(all_scores)}):")
    print(f"  최소: {min(all_scores):.2f}")
    print(f"  25%: {np.percentile(all_scores, 25):.2f}")
    print(f"  중앙값: {np.median(all_scores):.2f}")
    print(f"  75%: {np.percentile(all_scores, 75):.2f}")
    print(f"  최대: {max(all_scores):.2f}")

if smalltalk_scores:
    print(f"\n일반 대화 점수 (n={len(smalltalk_scores)}):")
    print(f"  최소: {min(smalltalk_scores):.2f}")
    print(f"  중앙값: {np.median(smalltalk_scores):.2f}")
    print(f"  최대: {max(smalltalk_scores):.2f}")

if science_scores:
    print(f"\n과학 쿼리 점수 (n={len(science_scores)}):")
    print(f"  최소: {min(science_scores):.2f}")
    print(f"  중앙값: {np.median(science_scores):.2f}")
    print(f"  최대: {max(science_scores):.2f}")

print("\n💡 권장 Threshold:")
print(f"  매우 낮음: < {np.percentile(all_scores, 10):.1f}")
print(f"  낮음: < {np.percentile(all_scores, 30):.1f}")
print(f"  중간: < {np.percentile(all_scores, 60):.1f}")
print(f"  높음: >= {np.percentile(all_scores, 60):.1f}")