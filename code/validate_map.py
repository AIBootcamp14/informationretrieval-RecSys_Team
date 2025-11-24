"""
로컬 MAP@3 검증 스크립트
Ground Truth를 사용하여 제출 파일의 MAP 점수를 계산
"""

import json
import sys

def calculate_map_at_k(ground_truth_path, submission_path, k=3):
    """
    MAP@K 계산

    Args:
        ground_truth_path: Ground truth JSONL 파일 경로
        submission_path: 제출 파일 경로 (CSV 형식)
        k: Top-K (기본값: 3)

    Returns:
        MAP 점수
    """
    # Ground truth 로드
    ground_truth = {}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            ground_truth[item['eval_id']] = set(item['ground_truth'])

    # 제출 파일 로드
    submissions = {}
    with open(submission_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            submissions[item['eval_id']] = item['topk']

    # MAP 계산
    total_ap = 0.0
    num_queries = 0

    detailed_results = []

    for eval_id in sorted(ground_truth.keys()):
        gt_docs = ground_truth[eval_id]

        # Ground truth가 비어있으면 건너뛰기 (일반 대화 또는 정답 없음)
        if not gt_docs:
            continue

        # 제출 결과가 없으면 AP=0
        if eval_id not in submissions:
            detailed_results.append({
                'eval_id': eval_id,
                'ap': 0.0,
                'retrieved': [],
                'ground_truth': list(gt_docs),
                'hits': 0
            })
            num_queries += 1
            continue

        retrieved_docs = submissions[eval_id][:k]

        # Average Precision 계산
        num_hits = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in gt_docs:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                sum_precisions += precision_at_i

        # AP = (관련 문서에서의 precision들의 합) / (관련 문서 총 개수)
        ap = sum_precisions / len(gt_docs) if gt_docs else 0.0

        detailed_results.append({
            'eval_id': eval_id,
            'ap': ap,
            'retrieved': retrieved_docs,
            'ground_truth': list(gt_docs),
            'hits': num_hits
        })

        total_ap += ap
        num_queries += 1

    # MAP = 모든 쿼리의 AP 평균
    map_score = total_ap / num_queries if num_queries > 0 else 0.0

    return map_score, detailed_results, num_queries

def print_results(map_score, detailed_results, num_queries, show_details=False):
    """결과 출력"""
    print("="*80)
    print("MAP@3 검증 결과")
    print("="*80)
    print(f"총 쿼리 수:        {num_queries}개")
    print(f"MAP@3 점수:        {map_score:.4f}")
    print("="*80)

    if show_details:
        print("\n상세 결과 (AP 낮은 순):")
        print("-"*80)

        # AP 낮은 순으로 정렬
        sorted_results = sorted(detailed_results, key=lambda x: x['ap'])

        for i, result in enumerate(sorted_results[:20], 1):
            print(f"\n{i}. eval_id={result['eval_id']}, AP={result['ap']:.3f}, Hits={result['hits']}/{len(result['ground_truth'])}")
            print(f"   Retrieved: {result['retrieved']}")
            print(f"   GT:        {result['ground_truth']}")

    # AP 분포 통계
    ap_bins = {
        'AP=0.0': 0,
        '0.0<AP≤0.3': 0,
        '0.3<AP≤0.6': 0,
        '0.6<AP≤0.9': 0,
        'AP>0.9': 0
    }

    for result in detailed_results:
        ap = result['ap']
        if ap == 0.0:
            ap_bins['AP=0.0'] += 1
        elif ap <= 0.3:
            ap_bins['0.0<AP≤0.3'] += 1
        elif ap <= 0.6:
            ap_bins['0.3<AP≤0.6'] += 1
        elif ap <= 0.9:
            ap_bins['0.6<AP≤0.9'] += 1
        else:
            ap_bins['AP>0.9'] += 1

    print("\n📊 AP 분포:")
    print("-"*80)
    for bin_name, count in ap_bins.items():
        pct = count / len(detailed_results) * 100 if detailed_results else 0
        print(f"   {bin_name:15s}: {count:3d}개 ({pct:5.1f}%)")
    print("="*80)

def main():
    if len(sys.argv) < 2:
        print("사용법: python validate_map.py <submission_file> [--details]")
        print("예시: python validate_map.py strategy_1_bm25_only_submission.csv --details")
        sys.exit(1)

    submission_path = sys.argv[1]
    show_details = '--details' in sys.argv

    ground_truth_path = 'ground_truth_solar_auto.jsonl'

    print(f"Ground Truth: {ground_truth_path}")
    print(f"Submission:   {submission_path}")
    print()

    try:
        map_score, detailed_results, num_queries = calculate_map_at_k(
            ground_truth_path,
            submission_path,
            k=3
        )

        print_results(map_score, detailed_results, num_queries, show_details)

        # 상세 결과를 JSON으로 저장
        if show_details:
            output_path = submission_path.replace('.csv', '_validation_details.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'map_score': map_score,
                    'num_queries': num_queries,
                    'detailed_results': detailed_results
                }, f, ensure_ascii=False, indent=2)
            print(f"\n💾 상세 결과 저장: {output_path}")

    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
