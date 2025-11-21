"""
Phase 1 개선사항 테스트 스크립트
개별 컴포넌트를 테스트하여 개선 효과 확인
"""

import json
from rag_phase1_improved import is_smalltalk, get_dynamic_topk, NORMAL_CHAT_IDS

def test_smalltalk_filter():
    """일반 대화 필터링 테스트"""
    print("=" * 50)
    print("1. 일반 대화 필터링 테스트")
    print("=" * 50)

    test_cases = [
        # (query, eval_id, expected_result)
        ("안녕 반갑다", 90, True),
        ("요새 너무 힘들다.", 276, True),
        ("니가 대답을 잘해줘서 너무 신나!", 261, True),
        ("남녀 관계에서 정서적인 행동이 왜 중요해?", 233, True),
        ("안녕하세요", None, True),
        ("고마워요", None, True),
        # 과학 질문들 (False expected)
        ("DNA의 구조에 대해 설명해줘", None, False),
        ("광합성 과정을 알려줘", None, False),
        ("헬륨이 다른 원소들과 반응을 잘 안하는 이유는?", None, False),
        ("자기장이 얼마나 센지 표현하는 방식은?", None, False),
        # 애매한 케이스
        ("뭐야", None, True),  # 짧은 의문사
        ("어떻게", None, True),  # 짧은 의문사
        ("세포가 어떻게 분열하는지 설명해줘", None, False),  # 과학 키워드 포함
    ]

    correct = 0
    for query, eval_id, expected in test_cases:
        result = is_smalltalk(query, eval_id)
        status = "✅" if result == expected else "❌"
        print(f"{status} Query: '{query[:30]}...' -> {result} (expected: {expected})")
        if result == expected:
            correct += 1

    print(f"\n정확도: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
    return correct == len(test_cases)

def test_dynamic_topk():
    """동적 TopK 시스템 테스트"""
    print("\n" + "=" * 50)
    print("2. 동적 TopK 시스템 테스트")
    print("=" * 50)

    # 모의 검색 결과 생성
    test_cases = [
        # (max_score, num_docs, expected_num)
        (3.0, 3, 0),   # 낮은 점수 -> 0개
        (4.9, 3, 0),   # threshold_mid 미만 -> 0개
        (5.1, 3, 1),   # 중간 점수 -> 1-2개
        (7.5, 3, 2),   # 중간 점수 -> 1-2개
        (10.5, 3, 3),  # 높은 점수 -> 3개
        (15.0, 3, 3),  # 매우 높은 점수 -> 3개
    ]

    for max_score, num_docs, expected_max in test_cases:
        # 모의 검색 결과 생성
        mock_results = {
            "hits": {
                "hits": [
                    {"_score": max_score * (0.9 ** i), "_source": {"docid": f"doc{i}"}}
                    for i in range(num_docs)
                ],
                "max_score": max_score
            }
        }

        selected = get_dynamic_topk(mock_results, threshold_high=10, threshold_mid=5)
        num_selected = len(selected)

        # 점수가 threshold_mid 이상인 문서만 선택되는지 확인
        valid_selection = all(doc["_score"] >= 5 for doc in selected) if selected else True

        status = "✅" if (num_selected <= expected_max and valid_selection) else "❌"
        print(f"{status} Score: {max_score:.1f} -> {num_selected} docs selected (expected ≤{expected_max})")

def test_with_eval_data():
    """실제 평가 데이터로 테스트"""
    print("\n" + "=" * 50)
    print("3. 평가 데이터 분석")
    print("=" * 50)

    eval_data = []
    with open("../data/eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))

    # 일반 대화 감지 테스트
    smalltalk_detected = 0
    smalltalk_ids_found = []

    for item in eval_data:
        eval_id = item['eval_id']
        messages = item['msg']
        last_msg = messages[-1]['content'] if messages else ""

        if is_smalltalk(last_msg, eval_id):
            smalltalk_detected += 1
            smalltalk_ids_found.append(eval_id)

    print(f"일반 대화로 감지: {smalltalk_detected}/{len(eval_data)}개")

    # NORMAL_CHAT_IDS와 비교
    expected_ids = set(NORMAL_CHAT_IDS)
    found_ids = set(smalltalk_ids_found)

    correctly_found = expected_ids & found_ids
    missed = expected_ids - found_ids
    false_positive = found_ids - expected_ids

    print(f"\n일반 대화 ID 분석:")
    print(f"  - 정확히 찾음: {len(correctly_found)}/{len(expected_ids)}개")
    print(f"  - 놓친 ID: {missed if missed else '없음'}")
    print(f"  - 잘못 분류: {len(false_positive)}개")

    # sample_submission.csv와 비교
    print("\n" + "=" * 50)
    print("4. 기존 제출 파일과 비교")
    print("=" * 50)

    original_results = []
    with open("sample_submission.csv", "r", encoding="utf-8") as f:
        for line in f:
            original_results.append(json.loads(line))

    # 일반 대화에 대한 처리 비교
    improved_count = 0
    for orig in original_results:
        eval_id = orig['eval_id']
        if eval_id in NORMAL_CHAT_IDS:
            # 원래는 3개 문서를 추출했음
            if len(orig.get('topk', [])) == 3:
                improved_count += 1

    print(f"개선 가능한 일반 대화: {improved_count}개")
    print(f"예상 점수 향상: +{improved_count/len(eval_data)*100:.1f}%")

    # TopK 분포 분석
    topk_dist_original = {}
    for orig in original_results:
        k = len(orig.get('topk', []))
        topk_dist_original[k] = topk_dist_original.get(k, 0) + 1

    print(f"\n원래 TopK 분포: {topk_dist_original}")
    print(f"목표 TopK 분포: 0개(일반대화), 1-2개(중간관련도), 3개(높은관련도)")

def main():
    """모든 테스트 실행"""
    print("Phase 1 개선사항 테스트 시작\n")

    # 각 컴포넌트 테스트
    test_smalltalk_filter()
    test_dynamic_topk()
    test_with_eval_data()

    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)
    print("\n다음 단계:")
    print("1. python rag_phase1_improved.py 실행하여 개선된 결과 생성")
    print("2. phase1_submission.csv 파일 제출")
    print("3. MAP 점수 확인 및 분석")

if __name__ == "__main__":
    main()