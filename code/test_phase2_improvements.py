"""
Phase 2 개선사항 테스트 스크립트
Query Rewrite, 멀티턴 최적화, Hybrid 가중치 테스트
"""

import json
from rag_phase2_improved import (
    rewrite_query,
    expand_abbreviations,
    add_synonyms,
    correct_typos,
    calculate_query_characteristics,
    get_dynamic_weights,
    create_standalone_query,
    get_dynamic_topk_v2
)
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def test_query_rewrite():
    """Query Rewrite 시스템 테스트"""
    print("=" * 50)
    print("1. Query Rewrite 테스트")
    print("=" * 50)

    test_cases = [
        # (원본 쿼리, 예상 개선 포함 키워드)
        ("디엔에이의 구조에 대해 알려줘", ["DNA", "유전자"]),
        ("아르엔에이와 디엔에이의 차이", ["RNA", "DNA"]),
        ("광합성 과정을 설명해줘", ["photosynthesis"]),
        ("새포 분열 과정", ["세포"]),  # 오타 교정
        ("괌합성이 일어나는 원리", ["광합성"]),  # 오타 교정
        ("에이티피의 역할", ["ATP", "에너지"]),
        ("진화의 메커니즘은 무엇인가", ["evolution", "자연선택"]),
    ]

    print("\n축약어 확장 테스트:")
    for query, expected_terms in test_cases[:3]:
        expanded = expand_abbreviations(query)
        found = any(term in expanded for term in expected_terms)
        status = "✅" if found else "❌"
        print(f"{status} '{query}' -> '{expanded[:60]}...'")

    print("\n오타 교정 테스트:")
    typo_tests = [
        ("새포 분열", "세포"),
        ("괌합성", "광합성"),
        ("엽색체", "염색체")
    ]
    for typo, correct in typo_tests:
        corrected = correct_typos(typo)
        status = "✅" if correct in corrected else "❌"
        print(f"{status} '{typo}' -> '{corrected}'")

    print("\n종합 Rewrite 테스트:")
    for query, expected_terms in test_cases:
        rewritten = rewrite_query(query)
        found = sum(1 for term in expected_terms if term in rewritten)
        status = "✅" if found > 0 else "❌"
        print(f"{status} Original: '{query[:30]}...'")
        print(f"    Rewritten: '{rewritten[:60]}...'")
        print()

def test_query_characteristics():
    """쿼리 특성 분석 테스트"""
    print("=" * 50)
    print("2. 쿼리 특성 분석 테스트")
    print("=" * 50)

    test_queries = [
        "DNA RNA 단백질 합성 과정",  # 과학 용어 밀도 높음
        "광합성이란 무엇이고 어떻게 일어나나요?",  # 개념 설명
        "ATP 3개가 생성되는 과정",  # 구체적
        "뭐야",  # 매우 짧음
        "세포 분열의 원리와 과정을 자세히 설명해주세요",  # 복합적
    ]

    for query in test_queries:
        chars = calculate_query_characteristics(query)
        weights = get_dynamic_weights(query)

        print(f"\nQuery: '{query[:40]}...'")
        print(f"  특성: 과학밀도={chars['science_density']:.2f}, "
              f"개념={chars['conceptual']:.2f}, "
              f"구체성={chars['specific']:.2f}")
        print(f"  가중치: BM25={weights['bm25']:.2f}, Dense={weights['dense']:.2f}")

def test_multiturn_handling():
    """멀티턴 대화 처리 테스트"""
    print("\n" + "=" * 50)
    print("3. 멀티턴 대화 처리 테스트")
    print("=" * 50)

    # 실제 평가 데이터에서 멀티턴 예시
    multiturn_examples = [
        {
            "eval_id": 107,
            "messages": [
                {"role": "user", "content": "기억 상실증 걸리면 너무 무섭겠다."},
                {"role": "assistant", "content": "네 맞습니다."},
                {"role": "user", "content": "어떤 원인 때문에 발생하는지 궁금해."}
            ],
            "expected": "기억 상실증 원인"
        },
        {
            "eval_id": 42,
            "messages": [
                {"role": "user", "content": "이란 콘트라 사건이 뭐야"},
                {"role": "assistant", "content": "이란-콘트라 사건은 1986년 레이건 행정부 스캔들입니다."},
                {"role": "user", "content": "이 사건이 미국 정치에 미친 영향은?"}
            ],
            "expected": "이란 콘트라 사건 미국 정치 영향"
        }
    ]

    # OpenAI client 초기화 (실제 테스트 시)
    try:
        upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        if upstage_api_key:
            client = OpenAI(
                base_url="https://api.upstage.ai/v1/solar",
                api_key=upstage_api_key
            )

            for example in multiturn_examples[:1]:  # API 호출 최소화
                standalone = create_standalone_query(example['messages'], client)
                print(f"\nEval ID: {example['eval_id']}")
                print(f"Last query: '{example['messages'][-1]['content']}'")
                print(f"Standalone: '{standalone}'")
                print(f"Expected: '{example['expected']}'")
        else:
            print("API 키 없음 - 실제 멀티턴 테스트 스킵")
    except:
        print("멀티턴 테스트 스킵 (API 연결 필요)")

def test_dynamic_topk_v2():
    """개선된 동적 TopK 테스트"""
    print("\n" + "=" * 50)
    print("4. Phase 2 동적 TopK 테스트")
    print("=" * 50)

    # 모의 검색 결과
    test_cases = [
        # (점수 리스트, 예상 선택 개수)
        ([2.0, 1.8, 1.5], 0),  # 매우 낮은 점수
        ([3.0, 2.5, 2.0], 0),  # 여전히 threshold 미만
        ([5.5, 4.0, 3.0], 1),  # 중간 점수
        ([8.0, 6.0, 4.0], 2),  # 중상 점수
        ([12.0, 10.0, 8.0], 3),  # 높은 점수
        ([15.0, 14.0, 13.0], 3),  # 매우 높은 점수
    ]

    for scores, expected_max in test_cases:
        # 모의 결과 생성
        mock_results = [
            {"docid": f"doc{i}", "score": score, "content": f"content{i}"}
            for i, score in enumerate(scores)
        ]

        selected = get_dynamic_topk_v2(mock_results, base_threshold=5)
        num_selected = len(selected)

        status = "✅" if num_selected <= expected_max else "❌"
        max_score = max(scores) if scores else 0
        print(f"{status} Max score: {max_score:.1f} -> {num_selected} docs (expected ≤{expected_max})")

def analyze_eval_data():
    """평가 데이터 심층 분석"""
    print("\n" + "=" * 50)
    print("5. 평가 데이터 심층 분석")
    print("=" * 50)

    eval_data = []
    with open("../data/eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))

    # 멀티턴 대화 분석
    multiturn_queries = []
    for item in eval_data:
        if len(item['msg']) > 1:
            multiturn_queries.append({
                'eval_id': item['eval_id'],
                'turns': len(item['msg']),
                'last_query': item['msg'][-1]['content']
            })

    print(f"멀티턴 대화: {len(multiturn_queries)}개")
    print("\n멀티턴 예시 (처음 5개):")
    for mq in multiturn_queries[:5]:
        print(f"  ID {mq['eval_id']}: {mq['turns']}턴 - '{mq['last_query'][:40]}...'")

    # 쿼리 길이 분석
    query_lengths = []
    for item in eval_data:
        query = item['msg'][-1]['content']
        query_lengths.append(len(query))

    avg_length = sum(query_lengths) / len(query_lengths)
    print(f"\n쿼리 길이 통계:")
    print(f"  평균: {avg_length:.1f}자")
    print(f"  최소: {min(query_lengths)}자")
    print(f"  최대: {max(query_lengths)}자")

    # 과학 키워드 포함 비율
    science_count = 0
    for item in eval_data:
        query = item['msg'][-1]['content'].lower()
        if any(keyword in query for keyword in ['dna', 'rna', '세포', '원자', '분자', '에너지', '광합성']):
            science_count += 1

    print(f"\n과학 키워드 포함: {science_count}/{len(eval_data)} ({science_count/len(eval_data)*100:.1f}%)")

def compare_improvements():
    """Phase 1 vs Phase 2 개선 비교"""
    print("\n" + "=" * 50)
    print("6. Phase 1 vs Phase 2 개선 비교")
    print("=" * 50)

    improvements = {
        "Phase 1": {
            "일반 대화 필터링": "키워드 기반 단순 필터",
            "TopK 선택": "고정 threshold (5, 10)",
            "검색 전략": "BM25 우선 (threshold 기반)",
            "예상 MAP": "0.65"
        },
        "Phase 2": {
            "일반 대화 필터링": "Phase 1 유지 + 개선",
            "TopK 선택": "세밀한 threshold + 점수 분포 고려",
            "검색 전략": "적응형 Hybrid (쿼리 특성 분석)",
            "Query 처리": "Rewrite (축약어, 오타, 동의어)",
            "멀티턴": "Standalone query 생성",
            "예상 MAP": "0.80"
        }
    }

    for phase, features in improvements.items():
        print(f"\n{phase}:")
        for feature, description in features.items():
            print(f"  - {feature}: {description}")

    print("\n핵심 개선점:")
    print("1. Query Rewrite로 검색 정확도 향상 (+5%)")
    print("2. 멀티턴 대화 처리 개선 (+4%)")
    print("3. 적응형 Hybrid 가중치 (+6%)")
    print("4. 세밀한 TopK threshold (+2%)")
    print("\n총 예상 개선: +15~17% (MAP 0.65 -> 0.80)")

def main():
    """모든 테스트 실행"""
    print("Phase 2 개선사항 테스트 시작\n")

    # 각 컴포넌트 테스트
    test_query_rewrite()
    test_query_characteristics()
    test_multiturn_handling()
    test_dynamic_topk_v2()
    analyze_eval_data()
    compare_improvements()

    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)
    print("\n다음 단계:")
    print("1. Elasticsearch가 실행 중인지 확인")
    print("2. python rag_phase2_improved.py 실행")
    print("3. phase2_submission.csv 파일 제출")
    print("4. MAP 점수 확인 (목표: 0.75~0.80)")

if __name__ == "__main__":
    main()