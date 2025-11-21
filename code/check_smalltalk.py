"""일반 대화 ID 확인 스크립트"""

import json
import pandas as pd

# 우리가 설정한 일반 대화 ID
CONFIRMED_SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 37, 70, 153, 169, 235,
    91, 265, 141, 26, 183, 260, 51, 30, 165, 60
}

# CSV 결과 확인
df = pd.read_csv('simplified_submission.csv')

# eval.jsonl 로드
dataset = []
with open('../data/eval.jsonl', 'r') as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

print("=" * 60)
print("일반 대화 처리 확인")
print("=" * 60)

# 일반 대화 ID별 체크
missing_smalltalk = []
found_smalltalk = []

for eval_id in CONFIRMED_SMALLTALK_IDS:
    row = df[df['eval_id'] == eval_id]

    if row.empty:
        print(f"⚠️  eval_id {eval_id}: CSV에 없음!")
        missing_smalltalk.append(eval_id)
    else:
        topk = eval(row.iloc[0]['topk_docs']) if isinstance(row.iloc[0]['topk_docs'], str) else row.iloc[0]['topk_docs']

        if isinstance(topk, list) and len(topk) == 0:
            print(f"✅ eval_id {eval_id}: 문서 0개 (정상)")
            found_smalltalk.append(eval_id)
        else:
            print(f"❌ eval_id {eval_id}: 문서 {len(topk) if isinstance(topk, list) else '?'}개 (문제!)")
            missing_smalltalk.append(eval_id)

            # 해당 쿼리 확인
            for item in dataset:
                if item['eval_id'] == eval_id:
                    if 'msg' in item and isinstance(item['msg'], list):
                        query = item['msg'][-1]['content'] if item['msg'] else ""
                    else:
                        query = item.get('msg', item.get('query', ''))
                    print(f"   쿼리: {query}")
                    break

print("\n📊 요약:")
print(f"- 정상 처리: {len(found_smalltalk)}/20")
print(f"- 누락/오류: {len(missing_smalltalk)}/20")

if missing_smalltalk:
    print(f"\n❌ 문제가 있는 eval_id들: {sorted(missing_smalltalk)}")