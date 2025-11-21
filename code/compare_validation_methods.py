"""
2번(BM25 자동 완성)과 3번(AI 자동 생성) 비교 분석
"""

import json
from collections import defaultdict

def load_jsonl(path):
    """JSONL 파일 로드"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def analyze_differences():
    """두 validation 방법의 차이점 분석"""

    # 데이터 로드
    complete_val = load_jsonl('complete_validation.jsonl')
    ai_val = load_jsonl('ai_validation.jsonl')

    # eval_id로 딕셔너리 변환
    complete_dict = {item['eval_id']: item for item in complete_val}
    ai_dict = {item['eval_id']: item for item in ai_val}

    print("=" * 80)
    print("2번(BM25 자동 완성) vs 3번(AI 자동 생성) 비교")
    print("=" * 80)

    # 1. 전체 통계
    complete_science = sum(1 for item in complete_val if item['ground_truth'])
    complete_general = len(complete_val) - complete_science

    ai_science = sum(1 for item in ai_val if item['query_type'] == 'science')
    ai_general = sum(1 for item in ai_val if item['query_type'] == 'general')

    print(f"\n📊 전체 통계")
    print(f"{'Method':<20} {'Science':>10} {'General':>10} {'Total':>10}")
    print(f"{'-'*60}")
    print(f"{'Complete (2번)':<20} {complete_science:>10} {complete_general:>10} {len(complete_val):>10}")
    print(f"{'AI (3번)':<20} {ai_science:>10} {ai_general:>10} {len(ai_val):>10}")

    # 2. Classification 차이 분석
    classification_diff = []

    for eval_id in complete_dict:
        complete_item = complete_dict[eval_id]
        ai_item = ai_dict[eval_id]

        complete_is_science = bool(complete_item['ground_truth'])
        ai_is_science = (ai_item['query_type'] == 'science')

        if complete_is_science != ai_is_science:
            classification_diff.append({
                'eval_id': eval_id,
                'query': complete_item['query'],
                'complete': 'science' if complete_is_science else 'general',
                'ai': ai_item['query_type']
            })

    print(f"\n🔍 Classification 차이: {len(classification_diff)}개")
    print(f"\n상위 10개 예시:")
    for i, item in enumerate(classification_diff[:10], 1):
        print(f"\n[{i}] ID {item['eval_id']}: {item['query'][:60]}...")
        print(f"    Complete(2번): {item['complete']} | AI(3번): {item['ai']}")

    # 3. Ground truth 차이 분석 (science 쿼리만)
    gt_differences = {
        'identical': 0,
        'different': 0,
        'partial_overlap': 0,
        'examples': []
    }

    for eval_id in complete_dict:
        complete_item = complete_dict[eval_id]
        ai_item = ai_dict[eval_id]

        # 둘 다 science인 경우만
        if complete_item['ground_truth'] and ai_item['query_type'] == 'science':
            complete_gt = set(complete_item['ground_truth'])
            ai_gt = set(ai_item['ground_truth'])

            if complete_gt == ai_gt:
                gt_differences['identical'] += 1
            else:
                overlap = len(complete_gt & ai_gt)
                if overlap > 0:
                    gt_differences['partial_overlap'] += 1
                else:
                    gt_differences['different'] += 1

                if len(gt_differences['examples']) < 5:
                    gt_differences['examples'].append({
                        'eval_id': eval_id,
                        'query': complete_item['query'],
                        'complete_gt': list(complete_gt)[:3],
                        'ai_gt': list(ai_gt)[:3],
                        'overlap': overlap,
                        'complete_count': len(complete_gt),
                        'ai_count': len(ai_gt)
                    })

    print(f"\n📋 Ground Truth 비교 (science 쿼리)")
    total = gt_differences['identical'] + gt_differences['partial_overlap'] + gt_differences['different']
    print(f"  - 완전 일치: {gt_differences['identical']}개 ({gt_differences['identical']/total*100:.1f}%)")
    print(f"  - 부분 일치: {gt_differences['partial_overlap']}개 ({gt_differences['partial_overlap']/total*100:.1f}%)")
    print(f"  - 완전 불일치: {gt_differences['different']}개 ({gt_differences['different']/total*100:.1f}%)")

    print(f"\n예시 (Ground Truth 차이):")
    for i, ex in enumerate(gt_differences['examples'], 1):
        print(f"\n[{i}] ID {ex['eval_id']}: {ex['query'][:60]}...")
        print(f"    Complete(2번): {ex['complete_count']}개 문서")
        print(f"    AI(3번): {ex['ai_count']}개 문서")
        print(f"    Overlap: {ex['overlap']}개")
        print(f"    Complete GT: {ex['complete_gt']}")
        print(f"    AI GT: {ex['ai_gt']}")

    # 4. 일반 질문으로 분류된 쿼리 중 차이 분석
    complete_general_queries = [item for item in complete_val if not item['ground_truth']]
    ai_general_queries = [item for item in ai_val if item['query_type'] == 'general']

    complete_general_ids = {item['eval_id'] for item in complete_general_queries}
    ai_general_ids = {item['eval_id'] for item in ai_general_queries}

    only_complete_general = complete_general_ids - ai_general_ids
    only_ai_general = ai_general_ids - complete_general_ids
    both_general = complete_general_ids & ai_general_ids

    print(f"\n🗣️  일반 대화 분류 차이")
    print(f"  - 둘 다 일반: {len(both_general)}개")
    print(f"  - Complete만 일반: {len(only_complete_general)}개")
    print(f"  - AI만 일반: {len(only_ai_general)}개")

    if only_complete_general:
        print(f"\nComplete만 일반으로 분류 (AI는 science):")
        for eval_id in list(only_complete_general)[:5]:
            print(f"  ID {eval_id}: {complete_dict[eval_id]['query'][:60]}...")

    if only_ai_general:
        print(f"\nAI만 일반으로 분류 (Complete는 science):")
        for eval_id in list(only_ai_general)[:5]:
            print(f"  ID {eval_id}: {ai_dict[eval_id]['query'][:60]}...")

    # 5. 난이도 분포 비교
    print(f"\n📊 난이도 분포")

    complete_diff_dist = defaultdict(int)
    ai_diff_dist = defaultdict(int)

    for item in complete_val:
        if 'difficulty' in item:
            complete_diff_dist[item['difficulty']] += 1

    for item in ai_val:
        if 'difficulty' in item:
            ai_diff_dist[item['difficulty']] += 1

    print(f"{'Difficulty':<15} {'Complete(2번)':>15} {'AI(3번)':>15}")
    print(f"{'-'*50}")
    all_difficulties = set(complete_diff_dist.keys()) | set(ai_diff_dist.keys())
    for diff in sorted(all_difficulties):
        print(f"{diff:<15} {complete_diff_dist[diff]:>15} {ai_diff_dist[diff]:>15}")

    print(f"\n{'='*80}\n")

def main():
    analyze_differences()

if __name__ == "__main__":
    main()
