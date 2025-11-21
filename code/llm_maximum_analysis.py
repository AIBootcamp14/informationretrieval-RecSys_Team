"""
LLM Maximum vs LLM Optimized 성능 비교 분석
"""

import json
from collections import Counter

def load_results(filepath):
    """결과 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def analyze_submission(filepath, name):
    """결과 분석"""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]

    topk_counts = {}
    for r in results:
        k = len(r['topk'])
        topk_counts[k] = topk_counts.get(k, 0) + 1

    print(f'\n{"="*80}')
    print(f'{name}')
    print(f'{"="*80}')
    print(f'Total queries: {len(results)}')
    print(f'\n📊 TopK Distribution:')
    for k in sorted(topk_counts.keys()):
        count = topk_counts[k]
        pct = count / len(results) * 100
        print(f'  TopK={k}: {count:3d} ({pct:5.1f}%)')

    with_results = sum(1 for r in results if len(r['topk']) > 0)
    print(f'\n✅ Results provided: {with_results}/{len(results)} ({with_results/len(results)*100:.1f}%)')

    avg_docs = sum(len(r['topk']) for r in results) / len(results)
    print(f'📈 Average documents: {avg_docs:.2f}')

    return {
        'results': results,
        'topk_counts': topk_counts,
        'total': len(results),
        'with_results': with_results,
        'avg_docs': avg_docs
    }

def compare_results(baseline_data, maximum_data):
    """두 결과 비교"""
    print(f'\n{"="*80}')
    print(f'🔍 상세 비교 분석')
    print(f'{"="*80}')

    baseline_results = baseline_data['results']
    maximum_results = maximum_data['results']

    # eval_id 순서 맞추기
    baseline_dict = {r['eval_id']: r['topk'] for r in baseline_results}
    maximum_dict = {r['eval_id']: r['topk'] for r in maximum_results}

    # 동일 쿼리 비교
    same_count = 0
    different_count = 0
    baseline_better = 0  # Baseline만 결과 있음
    maximum_better = 0  # Maximum만 결과 있음

    differences = []

    for eval_id in baseline_dict.keys():
        if eval_id not in maximum_dict:
            continue

        baseline_topk = baseline_dict[eval_id]
        maximum_topk = maximum_dict[eval_id]

        # 완전히 동일한지 확인
        if baseline_topk == maximum_topk:
            same_count += 1
        else:
            different_count += 1
            differences.append({
                'eval_id': eval_id,
                'baseline': baseline_topk,
                'maximum': maximum_topk
            })

        # 어느 쪽이 더 많은 결과를 제공했는지
        if len(baseline_topk) > 0 and len(maximum_topk) == 0:
            baseline_better += 1
        elif len(baseline_topk) == 0 and len(maximum_topk) > 0:
            maximum_better += 1

    print(f'\n🔄 결과 일치도:')
    total_compared = same_count + different_count
    print(f'  비교된 쿼리: {total_compared}개')
    print(f'  동일한 결과: {same_count}개 ({same_count/total_compared*100:.1f}%)')
    print(f'  다른 결과: {different_count}개 ({different_count/total_compared*100:.1f}%)')

    print(f'\n⚖️  결과 제공 능력:')
    print(f'  Baseline만 결과 있음: {baseline_better}개')
    print(f'  Maximum만 결과 있음: {maximum_better}개')

    # TopK 분포 비교
    print(f'\n📊 TopK 분포 변화:')
    print(f'  {"TopK":<10} {"Baseline":<15} {"Maximum":<15} {"차이"}')
    print(f'  {"-"*60}')
    for k in sorted(set(list(baseline_data['topk_counts'].keys()) + list(maximum_data['topk_counts'].keys()))):
        baseline_count = baseline_data['topk_counts'].get(k, 0)
        maximum_count = maximum_data['topk_counts'].get(k, 0)
        diff = maximum_count - baseline_count
        diff_str = f'{diff:+d}' if diff != 0 else '0'
        print(f'  TopK={k:<5} {baseline_count:<15} {maximum_count:<15} {diff_str}')

def main():
    print("="*80)
    print("LLM Maximum vs LLM Optimized 성능 비교 분석")
    print("="*80)

    # 파일 로드 및 분석
    print("\n📂 결과 파일 로드 및 분석 중...")

    try:
        baseline_data = analyze_submission(
            'Solar_optimized_submission.csv',
            'Baseline: LLM Optimized (Score: 0.6856) - 2 LLM Stages'
        )
    except Exception as e:
        print(f"❌ Baseline 결과 로드 실패: {e}")
        return

    try:
        maximum_data = analyze_submission(
            'llm_maximum_submission.csv',
            'LLM Maximum - 6 LLM Stages'
        )
    except Exception as e:
        print(f"❌ LLM Maximum 결과 로드 실패: {e}")
        return

    # 비교 분석
    compare_results(baseline_data, maximum_data)

    # 종합 요약
    print(f'\n{"="*80}')
    print(f'📊 종합 요약')
    print(f'{"="*80}')

    print(f'\n🎯 결과 제공률 비교:')
    print(f'  Baseline: {baseline_data["with_results"]}/{baseline_data["total"]} ({baseline_data["with_results"]/baseline_data["total"]*100:.1f}%)')
    print(f'  Maximum:  {maximum_data["with_results"]}/{maximum_data["total"]} ({maximum_data["with_results"]/maximum_data["total"]*100:.1f}%)')

    print(f'\n📝 평균 문서 수:')
    print(f'  Baseline: {baseline_data["avg_docs"]:.2f}개')
    print(f'  Maximum:  {maximum_data["avg_docs"]:.2f}개')

    print(f'\n🚀 LLM Maximum 주요 개선사항:')
    print(f'  ✅ LLM Intent Classification (하드코딩 → LLM)')
    print(f'  ✅ Context-Aware Rewriting (다중 턴 대화 처리)')
    print(f'  ✅ LLM Query Enhancement (키워드 확장)')
    print(f'  ✅ LLM Document Relevance Scoring (0-100점 평가)')
    print(f'  ✅ LLM Final Reranking (70점 이상만 선택)')

    print(f'\n{"="*80}')

if __name__ == "__main__":
    main()
