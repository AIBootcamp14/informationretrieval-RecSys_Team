"""
수동 레이블링 69개를 기반으로 나머지 151개 자동 완성

전략:
1. 수동 레이블링에서 과학질문 vs 일반질문 패턴 학습
2. 일반질문은 ground_truth = []
3. 과학질문은 BM25 Top-3를 ground_truth로 사용
"""

import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

es = Elasticsearch(['http://localhost:9200'])

# 기존 smalltalk IDs
CONFIRMED_SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def analyze_manual_labels():
    """
    수동 레이블링 분석
    - 어떤 쿼리들이 ground_truth=[] (일반질문)인지
    - 어떤 쿼리들이 ground_truth 있음 (과학질문)인지
    """
    with open('reliable_validation.jsonl', 'r') as f:
        manual_data = [json.loads(line) for line in f]

    science_queries = []
    general_queries = []

    for item in manual_data:
        if item['ground_truth']:
            science_queries.append(item)
        else:
            general_queries.append(item)

    print(f"수동 레이블링 분석:")
    print(f"  과학 질문: {len(science_queries)}개")
    print(f"  일반 질문: {len(general_queries)}개")

    # 일반 질문 패턴 분석
    print(f"\n일반 질문 예시:")
    for item in general_queries[:10]:
        print(f"  ID {item['eval_id']}: {item['query'][:50]}...")

    # 일반 질문 ID 목록
    general_ids = {item['eval_id'] for item in general_queries}

    return general_ids, science_queries

def is_general_question(query, eval_id, general_pattern_ids):
    """
    일반 질문인지 판단

    기준:
    1. 수동 레이블링에서 ground_truth=[]인 ID
    2. CONFIRMED_SMALLTALK_IDS에 포함
    3. "너", "나", "우리" 등 1/2인칭 대명사 포함
    4. 감정 표현 ("힘들다", "즐겁다", "신나")
    5. 메타 질문 ("어떻게 생각해", "~이 뭐야", "~할까")
    """
    # 1. 수동 레이블링에서 일반 질문으로 표시됨
    if eval_id in general_pattern_ids:
        return True

    # 2. 기존 smalltalk IDs
    if eval_id in CONFIRMED_SMALLTALK_IDS:
        return True

    # 3. 1/2인칭 대명사
    personal_pronouns = ['너는', '너 ', '내가', '나는', '나 ', '우리']
    if any(p in query for p in personal_pronouns):
        return True

    # 4. 감정 표현
    emotions = ['힘들', '즐거', '신나', '슬프', '화나', '기쁘']
    if any(e in query for e in emotions):
        return True

    # 5. 메타 질문 (과학적 사실이 아닌 의견/생각)
    meta_patterns = ['어떻게 생각', '어떤 생각', '~할까?', '~있을까?']
    if any(p in query for p in meta_patterns):
        # 단, "~의 역할은?", "~의 이유는?" 같은 과학 질문 제외
        science_keywords = ['역할', '이유', '원인', '과정', '원리', '특징', '차이', '영향']
        if not any(k in query for k in science_keywords):
            return True

    return False

def complete_validation_set():
    """
    220개 전체 validation set 완성
    """
    print("=" * 80)
    print("Validation Set 자동 완성")
    print("=" * 80)

    # 1. 수동 레이블링 분석
    general_pattern_ids, science_queries = analyze_manual_labels()

    # 2. 전체 eval 데이터 로드
    with open('../data/eval.jsonl', 'r') as f:
        all_eval = [json.loads(line) for line in f]

    # 3. 기존 레이블링된 ID 목록
    with open('reliable_validation.jsonl', 'r') as f:
        existing_data = {json.loads(line)['eval_id']: json.loads(line)
                        for line in f}

    # 4. 나머지 쿼리 처리
    completed_validation = list(existing_data.values())

    remaining_count = 0

    print(f"\n나머지 {len(all_eval) - len(existing_data)}개 쿼리 자동 완성 중...")

    for item in tqdm(all_eval, desc="Processing"):
        eval_id = item['eval_id']

        # 이미 레이블링됨
        if eval_id in existing_data:
            continue

        remaining_count += 1

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 일반 질문 판단
        if is_general_question(query, eval_id, general_pattern_ids):
            completed_validation.append({
                'eval_id': eval_id,
                'query': query,
                'msg': item['msg'],
                'ground_truth': [],
                'confidence': 'auto',
                'difficulty': 'general_question',
                'source': 'auto_general'
            })
            continue

        # 과학 질문 - BM25 Top-3 사용
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
                'size': 3
            }
        )

        if not response['hits']['hits']:
            ground_truth = []
        else:
            ground_truth = [hit['_source']['docid']
                          for hit in response['hits']['hits']]

        max_score = response['hits']['hits'][0]['_score'] if response['hits']['hits'] else 0.0

        # 난이도 판정
        if max_score >= 15:
            difficulty = 'easy'
        elif max_score >= 8:
            difficulty = 'medium'
        elif max_score >= 3:
            difficulty = 'hard'
        else:
            difficulty = 'very_hard'

        completed_validation.append({
            'eval_id': eval_id,
            'query': query,
            'msg': item['msg'],
            'ground_truth': ground_truth,
            'confidence': 'auto',
            'difficulty': difficulty,
            'max_score': max_score,
            'source': 'auto_science'
        })

    # 5. 저장
    output_path = 'complete_validation.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in completed_validation:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 6. 통계
    manual_count = len(existing_data)
    auto_count = remaining_count

    science_count = sum(1 for item in completed_validation
                       if item['ground_truth'])
    general_count = len(completed_validation) - science_count

    print(f"\n{'='*80}")
    print(f"✅ 완성: {output_path}")
    print(f"{'='*80}")
    print(f"총 {len(completed_validation)}개 쿼리")
    print(f"  - 수동 레이블링: {manual_count}개")
    print(f"  - 자동 완성: {auto_count}개")
    print(f"\n분류:")
    print(f"  - 과학 질문: {science_count}개")
    print(f"  - 일반 질문: {general_count}개")
    print(f"{'='*80}")

def main():
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공\n")

    complete_validation_set()

if __name__ == "__main__":
    main()
