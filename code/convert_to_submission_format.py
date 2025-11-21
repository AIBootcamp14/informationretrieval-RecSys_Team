"""
제출 형식 변환 스크립트
optimized_submission.csv를 표준 제출 형식으로 변환
"""

import json
import pandas as pd
from elasticsearch import Elasticsearch

# Elasticsearch 연결
es = Elasticsearch(['http://localhost:9200'])

# CSV 파일 읽기
df = pd.read_csv('optimized_submission.csv')

# eval.jsonl에서 쿼리 정보 가져오기
eval_data = {}
with open('../data/eval.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        eval_data[item['eval_id']] = item

# 결과 저장할 리스트
results = []

for _, row in df.iterrows():
    eval_id = row['eval_id']

    # topk_docs 파싱
    if isinstance(row['topk_docs'], str):
        try:
            topk = eval(row['topk_docs'])
        except:
            topk = []
    else:
        topk = [] if pd.isna(row['topk_docs']) else row['topk_docs']

    # 쿼리 정보 가져오기
    item = eval_data.get(eval_id, {})

    # standalone_query 생성
    if 'msg' in item and isinstance(item['msg'], list) and item['msg']:
        query = item['msg'][-1].get('content', '')
    else:
        query = item.get('msg', item.get('query', ''))

    # references 생성 (topk에서 문서 정보 가져오기)
    references = []
    for docid in topk if isinstance(topk, list) else []:
        try:
            # Elasticsearch에서 문서 검색
            response = es.search(
                index='test',
                body={
                    'query': {
                        'match': {
                            'docid': docid
                        }
                    },
                    'size': 1
                }
            )

            if response['hits']['hits']:
                hit = response['hits']['hits'][0]
                references.append({
                    'docid': docid,
                    'score': hit['_score'],
                    'content': hit['_source']['content'][:500]  # 처음 500자만
                })
        except:
            # 에러 발생 시 기본값
            references.append({
                'docid': docid,
                'score': 0.0,
                'content': ''
            })

    # 간단한 answer 생성
    if not topk:
        answer = "관련 문서를 찾을 수 없습니다."
    else:
        answer = f"검색 결과 {len(topk)}개 문서를 찾았습니다."

    # 결과 저장
    results.append({
        'eval_id': eval_id,
        'standalone_query': query,
        'topk': topk if isinstance(topk, list) else [],
        'answer': answer,
        'references': references
    })

# JSON lines 형식으로 저장
output_file = 'final_submission.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"✅ 변환 완료: {output_file}")
print(f"- 총 {len(results)}개 항목")
print(f"- 문서 0개: {sum(1 for r in results if not r['topk'])}개")
print(f"- 문서 1개: {sum(1 for r in results if len(r['topk']) == 1)}개")
print(f"- 문서 2개: {sum(1 for r in results if len(r['topk']) == 2)}개")
print(f"- 문서 3개: {sum(1 for r in results if len(r['topk']) == 3)}개")