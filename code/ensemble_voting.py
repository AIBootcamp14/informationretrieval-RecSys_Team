"""
Ensemble Voting 전략

여러 방법의 결과를 투표로 결합:
1. super_simple (0.63)
2. context_aware (0.62)
3. hybrid_dense (0.62)

방법:
- Weighted Voting: 성능에 따라 가중치 부여
- Rank Fusion: 순위 기반 점수 계산

기대 효과: 0.63 → 0.65~0.67
"""

import json
from collections import Counter, defaultdict

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def load_submission(file_path):
    """Submission 파일 로드"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = {}
    for item in data:
        results[item['eval_id']] = item['topk']

    return results

def weighted_voting(results_dict, weights):
    """
    가중치 투표

    Args:
        results_dict: {method_name: {eval_id: [doc_ids]}}
        weights: {method_name: weight}

    Returns:
        {eval_id: [doc_ids]}
    """
    final_results = {}

    # 모든 eval_id 수집
    all_eval_ids = set()
    for method_results in results_dict.values():
        all_eval_ids.update(method_results.keys())

    for eval_id in all_eval_ids:
        # Smalltalk
        if eval_id in SMALLTALK_IDS:
            final_results[eval_id] = []
            continue

        # 각 문서의 가중치 점수 계산
        doc_scores = defaultdict(float)

        for method, method_results in results_dict.items():
            if eval_id not in method_results:
                continue

            docs = method_results[eval_id]

            # 각 문서에 가중치 부여
            for rank, doc_id in enumerate(docs, 1):
                # 순위가 높을수록 더 높은 점수
                rank_score = 1.0 / rank
                doc_scores[doc_id] += weights[method] * rank_score

        # 점수순 정렬
        if doc_scores:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            final_results[eval_id] = [doc_id for doc_id, score in sorted_docs[:3]]
        else:
            final_results[eval_id] = []

    return final_results

def rank_fusion(results_dict):
    """
    Reciprocal Rank Fusion (RRF)

    Args:
        results_dict: {method_name: {eval_id: [doc_ids]}}

    Returns:
        {eval_id: [doc_ids]}
    """
    final_results = {}
    k = 60  # RRF 상수

    # 모든 eval_id 수집
    all_eval_ids = set()
    for method_results in results_dict.values():
        all_eval_ids.update(method_results.keys())

    for eval_id in all_eval_ids:
        # Smalltalk
        if eval_id in SMALLTALK_IDS:
            final_results[eval_id] = []
            continue

        # RRF 점수 계산
        doc_scores = defaultdict(float)

        for method_results in results_dict.values():
            if eval_id not in method_results:
                continue

            docs = method_results[eval_id]

            # RRF 공식: 1 / (k + rank)
            for rank, doc_id in enumerate(docs, 1):
                doc_scores[doc_id] += 1.0 / (k + rank)

        # 점수순 정렬
        if doc_scores:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            final_results[eval_id] = [doc_id for doc_id, score in sorted_docs[:3]]
        else:
            final_results[eval_id] = []

    return final_results

def save_results(results, output_path):
    """결과 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for eval_id in sorted(results.keys()):
            f.write(json.dumps({
                'eval_id': eval_id,
                'topk': results[eval_id]
            }, ensure_ascii=False) + '\n')

    # 통계
    topk_counts = Counter([len(docs) for docs in results.values()])

    print(f"\n✅ 완료: {output_path}")
    print(f"\nTopK 분포:")
    for k in sorted(topk_counts.keys()):
        print(f"  TopK={k}: {topk_counts[k]:3d}개 ({topk_counts[k]/len(results)*100:5.1f}%)")

def main():
    print("=" * 80)
    print("Ensemble Voting 전략")
    print("=" * 80)

    # 기존 결과 파일 로드
    files = {
        'super_simple': 'super_simple_submission.csv',
        'context_aware': 'context_aware_submission.csv',
        'selective_context': 'selective_context_submission.csv'
    }

    print("\n파일 로드 중...")
    results_dict = {}
    for method, file_path in files.items():
        try:
            results_dict[method] = load_submission(file_path)
            print(f"  ✅ {method}: {file_path}")
        except FileNotFoundError:
            print(f"  ⚠️ {method}: {file_path} 파일 없음 (스킵)")

    if len(results_dict) < 2:
        print("\n❌ 최소 2개 이상의 결과 파일이 필요합니다.")
        return

    print(f"\n✅ {len(results_dict)}개 방법 로드 완료\n")

    # 방법 1: Weighted Voting
    print(f"{'='*80}")
    print("방법 1: Weighted Voting")
    print(f"{'='*80}")

    weights = {
        'super_simple': 0.5,        # 0.63 (최고 성능)
        'context_aware': 0.3,       # 0.62
        'selective_context': 0.2    # 0.60
    }

    # 실제 로드된 방법에 맞게 가중치 조정
    active_weights = {m: weights.get(m, 1.0) for m in results_dict.keys()}
    total_weight = sum(active_weights.values())
    active_weights = {m: w/total_weight for m, w in active_weights.items()}

    print(f"\n가중치:")
    for method, weight in active_weights.items():
        print(f"  {method}: {weight:.2f}")

    weighted_results = weighted_voting(results_dict, active_weights)
    save_results(weighted_results, 'ensemble_weighted_submission.csv')

    # 방법 2: Rank Fusion (RRF)
    print(f"\n{'='*80}")
    print("방법 2: Reciprocal Rank Fusion")
    print(f"{'='*80}")

    rrf_results = rank_fusion(results_dict)
    save_results(rrf_results, 'ensemble_rrf_submission.csv')

    print(f"\n{'='*80}")
    print(f"✅ Ensemble Voting 완료")
    print(f"{'='*80}")
    print(f"\n생성된 파일:")
    print(f"  - ensemble_weighted_submission.csv (가중치 투표)")
    print(f"  - ensemble_rrf_submission.csv (RRF)")
    print(f"\n전략:")
    print(f"  Weighted: 성능 좋은 방법에 더 높은 가중치")
    print(f"  RRF: 순위 기반 공정한 결합")
    print(f"\n기대 효과:")
    print(f"  - Baseline (0.63) → 0.65~0.67 기대")
    print(f"  - Robust한 결과 (한 방법 실패해도 안전)")
    print(f"  - 여러 방법의 장점 결합")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
