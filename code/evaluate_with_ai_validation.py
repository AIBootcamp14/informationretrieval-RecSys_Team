"""
AI Validation Set으로 Submissions 평가

LLM이 자동 생성한 validation set으로 평가
"""

import json
import numpy as np
from collections import defaultdict

def load_validation(path):
    """Validation set 로드"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def load_submission(path):
    """Submission 로드"""
    with open(path, 'r') as f:
        return {json.loads(line)['eval_id']: json.loads(line) for line in f}

def calculate_average_precision(ground_truth, predicted):
    """
    Average Precision 계산

    Args:
        ground_truth: 정답 문서 리스트
        predicted: 예측 문서 리스트

    Returns:
        AP score (0~1)
    """
    if not ground_truth:
        # Smalltalk (정답이 없는 경우)
        return 1.0 if len(predicted) == 0 else 0.0

    if not predicted:
        return 0.0

    ap = 0.0
    hits = 0

    for i, pred_doc in enumerate(predicted, 1):
        if pred_doc in ground_truth:
            hits += 1
            precision_at_i = hits / i
            ap += precision_at_i

    if hits == 0:
        return 0.0

    # Normalize by number of relevant documents
    ap /= len(ground_truth)

    return ap

def evaluate_submission(submission_path, validation_path):
    """
    Submission을 AI Validation set으로 평가
    """
    print(f"\n{'='*80}")
    print(f"평가: {submission_path}")
    print(f"{'='*80}\n")

    # Load data
    val_data = load_validation(validation_path)
    sub_data = load_submission(submission_path)

    # 통계
    stats = {
        'overall': {'total': 0, 'ap_sum': 0, 'perfect': 0},
        'by_query_type': defaultdict(lambda: {'total': 0, 'ap_sum': 0}),
        'by_difficulty': defaultdict(lambda: {'total': 0, 'ap_sum': 0}),
    }

    ap_scores = []
    failures = []

    for val_item in val_data:
        eval_id = val_item['eval_id']

        if eval_id not in sub_data:
            continue

        ground_truth = val_item['ground_truth']
        predicted = sub_data[eval_id]['topk']
        query_type = val_item.get('query_type', 'unknown')
        difficulty = val_item.get('difficulty', 'unknown')

        # AP 계산
        ap = calculate_average_precision(ground_truth, predicted)
        ap_scores.append(ap)

        # Overall
        stats['overall']['total'] += 1
        stats['overall']['ap_sum'] += ap

        if ap == 1.0:
            stats['overall']['perfect'] += 1

        # By query type
        stats['by_query_type'][query_type]['total'] += 1
        stats['by_query_type'][query_type]['ap_sum'] += ap

        # By difficulty
        stats['by_difficulty'][difficulty]['total'] += 1
        stats['by_difficulty'][difficulty]['ap_sum'] += ap

        # Track failures (AP < 0.5)
        if ap < 0.5:
            failures.append({
                'eval_id': eval_id,
                'query': val_item['query'],
                'ap': ap,
                'ground_truth': ground_truth[:3],
                'predicted': predicted[:3],
                'query_type': query_type,
                'difficulty': difficulty
            })

    # 결과 출력
    if stats['overall']['total'] > 0:
        overall_map = stats['overall']['ap_sum'] / stats['overall']['total']
        perfect_rate = stats['overall']['perfect'] / stats['overall']['total'] * 100

        print(f"📊 Overall Results")
        print(f"  Total queries: {stats['overall']['total']}")
        print(f"  MAP: {overall_map:.4f}")
        print(f"  Perfect (AP=1.0): {stats['overall']['perfect']}개 ({perfect_rate:.1f}%)")
        print(f"  Median AP: {np.median(ap_scores):.4f}")
        print(f"  Min AP: {min(ap_scores):.4f}")
        print(f"  Max AP: {max(ap_scores):.4f}")

    # Query Type별
    print(f"\n📊 By Query Type")
    for qtype in ['science', 'general']:
        if stats['by_query_type'][qtype]['total'] > 0:
            qtype_map = stats['by_query_type'][qtype]['ap_sum'] / stats['by_query_type'][qtype]['total']
            print(f"  {qtype.upper()}: MAP {qtype_map:.4f} ({stats['by_query_type'][qtype]['total']}개)")

    # Difficulty별
    print(f"\n📊 By Difficulty")
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        if stats['by_difficulty'][diff]['total'] > 0:
            diff_map = stats['by_difficulty'][diff]['ap_sum'] / stats['by_difficulty'][diff]['total']
            print(f"  {diff.upper()}: MAP {diff_map:.4f} ({stats['by_difficulty'][diff]['total']}개)")

    # Failures
    if failures:
        print(f"\n❌ Failures (AP < 0.5): {len(failures)}개")
        for i, fail in enumerate(failures[:5], 1):
            print(f"\n  [{i}] ID {fail['eval_id']}: {fail['query'][:60]}...")
            print(f"      AP: {fail['ap']:.3f} | Type: {fail['query_type']} | Difficulty: {fail['difficulty']}")
            print(f"      Ground truth: {fail['ground_truth']}")
            print(f"      Predicted: {fail['predicted']}")

    print(f"\n{'='*80}\n")

    return {
        'map': overall_map if stats['overall']['total'] > 0 else 0.0,
        'stats': stats,
        'failures': failures
    }

def compare_submissions(validation_path):
    """
    여러 submissions 비교
    """
    submissions = [
        ('super_simple', 'super_simple_submission.csv', 0.6300),
        ('context_aware', 'context_aware_submission.csv', 0.6220),
        ('selective_context', 'selective_context_submission.csv', 0.6038),
    ]

    print(f"\n{'='*80}")
    print(f"AI Validation Set으로 Submissions 비교")
    print(f"{'='*80}")

    results = []

    for name, path, leaderboard_map in submissions:
        try:
            result = evaluate_submission(path, validation_path)
            results.append({
                'name': name,
                'validation_map': result['map'],
                'leaderboard_map': leaderboard_map,
                'gap': leaderboard_map - result['map']
            })
        except FileNotFoundError:
            print(f"⚠️ {path} 파일 없음\n")
        except Exception as e:
            print(f"⚠️ {name} 평가 실패: {e}\n")

    # 비교 테이블
    print(f"\n{'='*80}")
    print(f"📊 종합 비교")
    print(f"{'='*80}\n")

    print(f"{'Submission':<25} {'Validation MAP':<15} {'Leaderboard MAP':<18} {'Gap':<10}")
    print(f"{'-'*80}")

    for r in results:
        gap_str = f"{r['gap']:+.4f}"
        print(f"{r['name']:<25} {r['validation_map']:<15.4f} {r['leaderboard_map']:<18.4f} {gap_str:<10}")

    # 상관관계 분석
    if len(results) >= 2:
        val_maps = [r['validation_map'] for r in results]
        lead_maps = [r['leaderboard_map'] for r in results]

        correlation = np.corrcoef(val_maps, lead_maps)[0, 1]

        print(f"\n{'='*80}")
        print(f"📈 상관관계 분석")
        print(f"{'='*80}")
        print(f"Validation MAP ↔ Leaderboard MAP: {correlation:.4f}")

        if correlation > 0.9:
            print(f"✅ 매우 높은 상관관계 - AI Validation set 신뢰 가능!")
        elif correlation > 0.7:
            print(f"✅ 높은 상관관계 - AI Validation set 유용")
        elif correlation > 0.5:
            print(f"⚠️ 중간 상관관계 - 주의 필요")
        else:
            print(f"❌ 낮은 상관관계 - AI Validation set 재검토 필요")

def main():
    print("=" * 80)
    print("AI Validation Set 기반 평가")
    print("=" * 80)

    validation_path = 'ai_validation.jsonl'

    # 비교 평가
    compare_submissions(validation_path)

if __name__ == "__main__":
    main()
