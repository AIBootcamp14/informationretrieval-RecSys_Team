"""
Solar Pro 기반 Validation Set 생성

전략:
1. Solar Pro로 각 질문 분석 (난이도, 주제, 유형)
2. 계층적 샘플링으로 대표성 있는 val set 생성
3. Train/Val 분할 후 저장
"""

import json
import os
import random
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Solar API 클라이언트
upstage_api_key = os.getenv("UPSTAGE_API_KEY")
client = OpenAI(
    base_url="https://api.upstage.ai/v1/solar",
    api_key=upstage_api_key
)

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def analyze_question_with_solar(query, eval_id):
    """
    Solar Pro로 질문 특성 분석

    Returns:
        dict: {difficulty, topic, question_type}
    """
    # Smalltalk은 별도 처리
    if eval_id in SMALLTALK_IDS:
        return {
            'difficulty': 'easy',
            'topic': 'smalltalk',
            'question_type': 'greeting'
        }

    prompt = f"""다음 생물학 질문을 분석하세요.

질문: {query}

다음을 JSON 형식으로 출력하세요:
{{
  "difficulty": "easy/medium/hard",
  "topic": "DNA/세포/광합성/효소/유전/진화/생태/기타",
  "question_type": "정의/과정/비교/설명/원리"
}}

출력 예시:
{{"difficulty": "medium", "topic": "DNA", "question_type": "과정"}}"""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        result = response.choices[0].message.content.strip()

        # JSON 파싱
        # 코드 블록이 있으면 제거
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]

        analysis = json.loads(result.strip())

        # 검증
        if 'difficulty' not in analysis or 'topic' not in analysis or 'question_type' not in analysis:
            raise ValueError("Missing required fields")

        return analysis

    except Exception as e:
        print(f"⚠️ 분석 실패 (eval_id={eval_id}): {e}")
        # 기본값 반환
        return {
            'difficulty': 'medium',
            'topic': '기타',
            'question_type': '설명'
        }

def create_stratified_validation_set(eval_path, val_ratio=0.2, seed=42):
    """
    계층적 샘플링으로 validation set 생성

    Args:
        eval_path: eval.jsonl 경로
        val_ratio: validation set 비율 (기본 20%)
        seed: 랜덤 시드

    Returns:
        train_data, val_data
    """
    random.seed(seed)

    # 데이터 로드
    print(f"\n📂 로딩: {eval_path}")
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    print(f"✅ 총 {len(eval_data)}개 질문 로드")

    # 각 질문 분석
    print(f"\n🔍 Solar Pro로 질문 분석 중...")
    analyzed_data = []

    for item in tqdm(eval_data, desc="Analyzing"):
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 분석
        analysis = analyze_question_with_solar(query, eval_id)

        analyzed_data.append({
            **item,
            'difficulty': analysis['difficulty'],
            'topic': analysis['topic'],
            'question_type': analysis['question_type']
        })

    # 계층별 그룹화 (difficulty x topic)
    print(f"\n📊 계층별 그룹화 중...")
    groups = defaultdict(list)

    for item in analyzed_data:
        key = (item['difficulty'], item['topic'])
        groups[key].append(item)

    # 각 그룹 통계
    print(f"\n그룹 분포:")
    for key, items in sorted(groups.items()):
        difficulty, topic = key
        print(f"  [{difficulty:6s}] {topic:8s}: {len(items):3d}개")

    # 각 그룹에서 계층적 샘플링
    print(f"\n✂️  계층적 샘플링 (val_ratio={val_ratio})...")
    val_data = []
    train_data = []

    for key, items in groups.items():
        random.shuffle(items)

        # 최소 1개는 val에 포함 (그룹이 크면 비율대로)
        val_size = max(1, int(len(items) * val_ratio))

        val_data.extend(items[:val_size])
        train_data.extend(items[val_size:])

    # 섞기
    random.shuffle(val_data)
    random.shuffle(train_data)

    # 통계 출력
    print(f"\n{'='*80}")
    print(f"✅ Validation Set 생성 완료")
    print(f"{'='*80}")
    print(f"\n통계:")
    print(f"  Train: {len(train_data):3d}개 ({len(train_data)/len(eval_data)*100:5.1f}%)")
    print(f"  Val:   {len(val_data):3d}개 ({len(val_data)/len(eval_data)*100:5.1f}%)")

    # Val set 난이도 분포
    val_difficulty = defaultdict(int)
    val_topic = defaultdict(int)
    for item in val_data:
        val_difficulty[item['difficulty']] += 1
        val_topic[item['topic']] += 1

    print(f"\nValidation Set 난이도 분포:")
    for diff in ['easy', 'medium', 'hard']:
        count = val_difficulty[diff]
        print(f"  {diff:6s}: {count:3d}개 ({count/len(val_data)*100:5.1f}%)")

    print(f"\nValidation Set 주제 분포:")
    for topic, count in sorted(val_topic.items(), key=lambda x: -x[1]):
        print(f"  {topic:10s}: {count:3d}개 ({count/len(val_data)*100:5.1f}%)")

    return train_data, val_data

def save_splits(train_data, val_data, output_dir='../data'):
    """Train/Val 분할 저장"""
    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')

    # Train 저장
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            # 분석 정보 제거 (eval.jsonl 형식 유지)
            clean_item = {
                'eval_id': item['eval_id'],
                'msg': item['msg']
            }
            f.write(json.dumps(clean_item, ensure_ascii=False) + '\n')

    # Val 저장
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            clean_item = {
                'eval_id': item['eval_id'],
                'msg': item['msg']
            }
            f.write(json.dumps(clean_item, ensure_ascii=False) + '\n')

    print(f"\n💾 저장 완료:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")

def main():
    print("=" * 80)
    print("Solar Pro 기반 Validation Set 생성")
    print("=" * 80)

    if not upstage_api_key:
        print("❌ UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    print("✅ Upstage Solar API Key 확인")

    # Validation set 생성
    train_data, val_data = create_stratified_validation_set(
        eval_path='../data/eval.jsonl',
        val_ratio=0.2,  # 20% validation
        seed=42
    )

    # 저장
    save_splits(train_data, val_data)

    print(f"\n{'='*80}")
    print(f"✅ 전체 프로세스 완료")
    print(f"{'='*80}")
    print(f"\n💡 다음 단계:")
    print(f"  1. val.jsonl로 빠른 실험 (40개, 5분 이내)")
    print(f"  2. 최적 전략 찾기")
    print(f"  3. 전체 eval.jsonl로 최종 제출")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
