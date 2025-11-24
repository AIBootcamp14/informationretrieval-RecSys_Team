# Validation Set 생성 방법 비교 보고서

## 개요

3가지 validation set 생성 방법을 시도했으나, 모두 negative correlation을 보이며 실패했습니다.

## 방법 비교

### 1번: 수동 레이블링 (69개)
- **방법**: 사람이 직접 BM25 Top-10 결과를 보고 판단
- **결과**:
  - MAP correlation: **-0.7353**
  - 65/69개를 일반 질문으로 잘못 분류
  - 문제: 비전문가가 과학 질문을 일반 질문으로 착각

### 2번: BM25 자동 완성 (220개)
- **방법**:
  - 수동 레이블링 69개 패턴 학습
  - 나머지 151개는 BM25 Top-3를 ground_truth로 자동 설정
- **결과**:
  - Science: 143개 (65%)
  - General: 77개 (35%)
  - MAP correlation: **-0.7871**
  - 문제: BM25로 ground_truth 생성 → circular reasoning

### 3번: AI 자동 생성 (220개)
- **방법**:
  - LLM이 각 쿼리를 science/general 분류
  - Science 쿼리는 BM25 Top-10 중 LLM이 관련 문서 선택
- **결과**:
  - Science: 179개 (81%)
  - General: 41개 (19%)
  - MAP correlation: **-0.7730**
  - 문제: 여전히 BM25 기반이라 circular reasoning 존재

## 2번 vs 3번 상세 비교

### Classification 차이

**38개 쿼리에서 분류 차이** (17.3%)

주요 패턴:
- **Complete만 일반으로 분류한 경우**: 37개
  - 예: "이렇게 집단으로 이주하게 되는 계기는?" → 실제로는 과학 질문
  - 예: "피임을 하기 위한 방법중 약으로 처리하는 방법은?" → 명확한 과학 질문
  - **원인**: 수동 레이블링 패턴이 너무 보수적 (대명사 "이렇게" 등에 과민 반응)

- **AI만 일반으로 분류한 경우**: 1개만
  - "예외처리가 필요한 경우를 알려줘" → 프로그래밍 질문 (애매함)

**결론**: AI가 더 정확하게 science 질문 식별

### Ground Truth 차이 (Science 쿼리만)

142개 science 쿼리 중:
- **완전 일치**: 19개 (13.4%)
- **부분 일치**: 100개 (70.4%)
- **완전 불일치**: 23개 (16.2%)

예시:
```
Query: "DNA 조각들이 서로 결합되도록 돕는 것은?"
- Complete(2번): 3개 문서 [A, B, C]
- AI(3번): 2개 문서 [B, C]
- Overlap: 2개

Query: "바람이 부는 이유는?"
- Complete(2번): 2개 문서 [A, B]
- AI(3번): 1개 문서 [A]
- Overlap: 1개
```

**차이 발생 이유**:
- Complete: BM25 Top-3를 무조건 선택
- AI: BM25 Top-10 중에서 실제 관련성 판단하여 0~3개 선택
- AI가 더 selective (평균 2.2개 vs Complete 2.8개)

### 난이도 분포 차이

```
              Complete(2번)    AI(3번)
Easy               147           143
Medium              51            34
Hard                 4             2
General/Smalltalk   18            41
```

**해석**:
- AI가 더 많은 쿼리를 general로 분류 (41 vs 18)
- 그러나 여전히 81%는 science로 분류 (더 정확)

## 왜 모두 실패했는가?

### 공통 문제: Circular Reasoning

1. **BM25 기반 Ground Truth**
   ```
   BM25로 검색 → BM25 Top-K를 ground_truth로 설정
   → BM25 변형 알고리즘 평가 → BM25와 비슷할수록 높은 점수
   ```

2. **실제 Leaderboard는 다른 기준 사용**
   - Leaderboard ground_truth는 사람이 직접 레이블링했을 가능성
   - 또는 완전히 다른 검색 방법으로 생성
   - BM25와 무관한 기준

3. **결과**:
   ```
   Validation: BM25와 유사한 방법이 높은 점수
   → selective_context (0.49) > context_aware (0.49) > super_simple (0.39)

   Leaderboard: 다른 기준
   → super_simple (0.63) > context_aware (0.62) > selective_context (0.60)

   → Negative correlation: -0.77
   ```

## 어떤 방법이 더 나은가?

### AI 자동 생성 (3번)이 더 우수

**Classification 정확도**:
- 37개 과학 질문을 더 정확하게 식별
- 예: "피임을 하기 위한 방법중 약으로 처리하는 방법은?"
  - Complete: 일반 질문 (❌)
  - AI: 과학 질문 (✓)

**Ground Truth 품질**:
- Selective하게 선택 (관련 없는 문서 제외)
- 평균 2.2개 vs Complete 2.8개
- 더 정확한 relevance 판단

**신뢰성**:
- Correlation: -0.77 (Complete: -0.79보다 약간 나음)
- 여전히 negative지만 조금 더 나은 결과

## 최종 결론

### Validation Set 기반 개발은 불가능

**3가지 시도 모두 실패**:
1. 수동 레이블링: -0.74
2. BM25 자동 완성: -0.79
3. AI 자동 생성: -0.77

**근본 원인**:
- Leaderboard ground_truth를 알 수 없음
- BM25 기반 pseudo-labeling은 circular reasoning 유발
- 비전문가 레이블링은 부정확

### 다음 단계 권장사항

#### ✅ 추천: Leaderboard 직접 테스트

1. **Dense Retrieval 시도** (준비 완료)
   ```bash
   python3 rag_with_dense.py
   ```
   - BM25와 완전히 다른 방법
   - 의미 기반 검색으로 0.7+ 기대

2. **Hybrid Search 최적화**
   - BM25 + Dense 가중치 조정
   - Alpha 파라미터 실험 (0.3, 0.5, 0.7, 0.9)

3. **Reranking 추가**
   - Top-10 검색 후 LLM으로 재정렬
   - 더 정확한 Top-3 선택

#### ❌ 비추천: Validation 기반 개발

- 더 이상 validation set 생성 시도하지 말 것
- Correlation이 negative면 개선 방향 판단 불가
- 시간 낭비

## 부록: 파일 정보

### 생성된 Validation Sets

1. **reliable_validation.jsonl** (69개, 수동)
   - 사람이 직접 레이블링
   - 품질 낮음 (과학 질문을 일반으로 오분류)

2. **complete_validation.jsonl** (220개, 2번)
   - 수동 69개 + BM25 자동 151개
   - Science: 143개 (65%)

3. **ai_validation.jsonl** (220개, 3번)
   - LLM 자동 생성
   - Science: 179개 (81%)
   - **가장 품질 높음**

### 평가 스크립트

- `evaluate_with_reliable_validation.py` - 수동 레이블링 평가
- `evaluate_with_ai_validation.py` - AI 자동 생성 평가
- `compare_validation_methods.py` - 2번 vs 3번 비교

### 다음 실험 파일

- `rag_with_dense.py` - Dense Retrieval 준비 완료
- `test_hybrid_search.py` - Hybrid 파라미터 실험용

## 교훈

1. **Pseudo-labeling의 함정**
   - 같은 방법으로 label과 prediction 생성하면 circular reasoning
   - Ground truth는 독립적인 출처에서 와야 함

2. **Negative correlation의 의미**
   - Validation이 leaderboard와 반대 방향
   - 이런 경우 validation 무용지물

3. **Domain expert 필요성**
   - 과학 질문 판단에는 전문가 필요
   - 비전문가는 많은 과학 질문을 일반 대화로 착각

4. **다음 프로젝트에서**
   - Ground truth는 반드시 독립적으로 구축
   - 또는 competition organizer가 제공하는 public validation 사용
   - Pseudo-labeling 사용 시 다른 방법론 적용 (teacher-student, consistency 등)
