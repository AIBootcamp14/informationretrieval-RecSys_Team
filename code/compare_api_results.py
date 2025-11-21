"""
Solar API vs OpenAI API 성능 비교 분석 스크립트
"""

import json
from collections import Counter

def load_results(filepath):
    """결과 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def analyze_results(results, api_name):
    """결과 분석"""
    print(f"\n{'='*80}")
    print(f"{api_name} 분석 결과")
    print(f"{'='*80}")

    # TopK 분포
    topk_counts = Counter(len(r['topk']) for r in results)
    print(f"\n📊 TopK 분포:")
    for k in sorted(topk_counts.keys()):
        count = topk_counts[k]
        percentage = count / len(results) * 100
        print(f"  TopK={k}: {count:3d}개 ({percentage:5.1f}%)")

    # 통계
    total = len(results)
    with_results = sum(1 for r in results if len(r['topk']) > 0)
    without_results = total - with_results

    print(f"\n📈 기본 통계:")
    print(f"  전체 쿼리: {total}개")
    print(f"  결과 있음: {with_results}개 ({with_results/total*100:.1f}%)")
    print(f"  결과 없음: {without_results}개 ({without_results/total*100:.1f}%)")

    # 문서 개수 통계
    doc_counts = [len(r['topk']) for r in results]
    avg_docs = sum(doc_counts) / len(doc_counts) if doc_counts else 0
    print(f"  평균 문서 수: {avg_docs:.2f}개")

    return {
        'topk_distribution': dict(topk_counts),
        'total': total,
        'with_results': with_results,
        'without_results': without_results,
        'avg_docs': avg_docs
    }

def compare_results(solar_results, openai_results):
    """두 결과 비교"""
    print(f"\n{'='*80}")
    print(f"🔍 상세 비교 분석")
    print(f"{'='*80}")

    # eval_id 순서 맞추기
    solar_dict = {r['eval_id']: r['topk'] for r in solar_results}
    openai_dict = {r['eval_id']: r['topk'] for r in openai_results}

    # 동일 쿼리 비교
    same_count = 0
    different_count = 0
    solar_better = 0  # Solar만 결과 있음
    openai_better = 0  # OpenAI만 결과 있음

    differences = []

    for eval_id in solar_dict.keys():
        if eval_id not in openai_dict:
            continue

        solar_topk = solar_dict[eval_id]
        openai_topk = openai_dict[eval_id]

        # 완전히 동일한지 확인
        if solar_topk == openai_topk:
            same_count += 1
        else:
            different_count += 1
            differences.append({
                'eval_id': eval_id,
                'solar': solar_topk,
                'openai': openai_topk
            })

        # 어느 쪽이 더 많은 결과를 제공했는지
        if len(solar_topk) > 0 and len(openai_topk) == 0:
            solar_better += 1
        elif len(solar_topk) == 0 and len(openai_topk) > 0:
            openai_better += 1

    print(f"\n🔄 결과 일치도:")
    total_compared = same_count + different_count
    print(f"  비교된 쿼리: {total_compared}개")
    print(f"  동일한 결과: {same_count}개 ({same_count/total_compared*100:.1f}%)")
    print(f"  다른 결과: {different_count}개 ({different_count/total_compared*100:.1f}%)")

    print(f"\n⚖️  결과 제공 능력:")
    print(f"  Solar만 결과 있음: {solar_better}개")
    print(f"  OpenAI만 결과 있음: {openai_better}개")

    # 차이나는 케이스 샘플
    if differences:
        print(f"\n📋 차이나는 케이스 샘플 (최대 5개):")
        for i, diff in enumerate(differences[:5], 1):
            print(f"\n  [{i}] eval_id: {diff['eval_id']}")
            print(f"      Solar:  {len(diff['solar'])}개 문서")
            print(f"      OpenAI: {len(diff['openai'])}개 문서")

            # 문서 ID 비교
            solar_set = set(diff['solar'])
            openai_set = set(diff['openai'])
            common = solar_set & openai_set
            solar_only = solar_set - openai_set
            openai_only = openai_set - solar_set

            if common:
                print(f"      공통: {len(common)}개")
            if solar_only:
                print(f"      Solar만: {len(solar_only)}개")
            if openai_only:
                print(f"      OpenAI만: {len(openai_only)}개")

def main():
    print("="*80)
    print("Solar API vs OpenAI API 성능 비교 분석")
    print("="*80)

    # 파일 로드
    print("\n📂 결과 파일 로드 중...")
    try:
        solar_results = load_results('llm_optimized_submission.csv')
        print(f"✅ Solar API 결과: {len(solar_results)}개 쿼리")
    except Exception as e:
        print(f"❌ Solar API 결과 로드 실패: {e}")
        return

    try:
        openai_results = load_results('llm_optimized_openai_submission.csv')
        print(f"✅ OpenAI API 결과: {len(openai_results)}개 쿼리")
    except Exception as e:
        print(f"❌ OpenAI API 결과 로드 실패: {e}")
        return

    # 개별 분석
    solar_stats = analyze_results(solar_results, "Solar API (solar-mini + solar-pro)")
    openai_stats = analyze_results(openai_results, "OpenAI API (gpt-4o-mini)")

    # 비교 분석
    compare_results(solar_results, openai_results)

    # 종합 요약
    print(f"\n{'='*80}")
    print(f"📊 종합 요약")
    print(f"{'='*80}")

    print(f"\n🎯 결과 제공률:")
    print(f"  Solar API:  {solar_stats['with_results']}/{solar_stats['total']} ({solar_stats['with_results']/solar_stats['total']*100:.1f}%)")
    print(f"  OpenAI API: {openai_stats['with_results']}/{openai_stats['total']} ({openai_stats['with_results']/openai_stats['total']*100:.1f}%)")

    print(f"\n📝 평균 문서 수:")
    print(f"  Solar API:  {solar_stats['avg_docs']:.2f}개")
    print(f"  OpenAI API: {openai_stats['avg_docs']:.2f}개")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
