"""
여러 제출 파일을 한번에 검증하고 순위 매기기
"""

import json
import glob
import sys
from validate_map import calculate_map_at_k

def batch_validate():
    """모든 submission 파일 검증"""
    ground_truth_path = 'ground_truth_solar_auto.jsonl'

    # 모든 submission 파일 찾기
    submission_files = glob.glob('*submission*.csv')

    if not submission_files:
        print("제출 파일을 찾을 수 없습니다.")
        return

    print("="*80)
    print(f"총 {len(submission_files)}개 제출 파일 검증")
    print("="*80)

    results = []

    for submission_file in submission_files:
        try:
            map_score, detailed_results, num_queries = calculate_map_at_k(
                ground_truth_path,
                submission_file,
                k=3
            )

            # AP 분포 계산
            ap_bins = {
                'zero': 0,
                'low': 0,
                'mid': 0,
                'high': 0,
                'perfect': 0
            }

            for r in detailed_results:
                ap = r['ap']
                if ap == 0.0:
                    ap_bins['zero'] += 1
                elif ap <= 0.3:
                    ap_bins['low'] += 1
                elif ap <= 0.6:
                    ap_bins['mid'] += 1
                elif ap <= 0.9:
                    ap_bins['high'] += 1
                else:
                    ap_bins['perfect'] += 1

            results.append({
                'file': submission_file,
                'map_score': map_score,
                'num_queries': num_queries,
                'ap_bins': ap_bins
            })

        except Exception as e:
            print(f"⚠️  {submission_file}: 검증 실패 - {e}")
            continue

    # MAP 점수 순으로 정렬
    results.sort(key=lambda x: x['map_score'], reverse=True)

    # 결과 출력
    print("\n📊 전체 결과 (MAP@3 높은 순):")
    print("="*80)
    print(f"{'순위':<4} {'MAP@3':<8} {'파일명':<50} {'AP>0.9':<8}")
    print("-"*80)

    for i, result in enumerate(results, 1):
        perfect_pct = result['ap_bins']['perfect'] / result['num_queries'] * 100 if result['num_queries'] > 0 else 0
        filename = result['file'][:47] + "..." if len(result['file']) > 50 else result['file']
        print(f"{i:<4} {result['map_score']:<8.4f} {filename:<50} {perfect_pct:>5.1f}%")

    print("="*80)

    # Top 5 상세 정보
    print("\n🏆 Top 5 전략 상세:")
    print("="*80)

    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result['file']}")
        print(f"   MAP@3:        {result['map_score']:.4f}")
        print(f"   검증 쿼리:    {result['num_queries']}개")
        print(f"   AP 분포:")
        bins = result['ap_bins']
        total = result['num_queries']
        print(f"      AP=0.0:      {bins['zero']:3d}개 ({bins['zero']/total*100:5.1f}%)")
        print(f"      0.0<AP≤0.3:  {bins['low']:3d}개 ({bins['low']/total*100:5.1f}%)")
        print(f"      0.3<AP≤0.6:  {bins['mid']:3d}개 ({bins['mid']/total*100:5.1f}%)")
        print(f"      0.6<AP≤0.9:  {bins['high']:3d}개 ({bins['high']/total*100:5.1f}%)")
        print(f"      AP>0.9:      {bins['perfect']:3d}개 ({bins['perfect']/total*100:5.1f}%)")

    print("\n" + "="*80)

    # JSON으로 저장
    output_path = 'validation_summary.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"💾 전체 결과 저장: {output_path}")
    print("="*80)

if __name__ == "__main__":
    batch_validate()
