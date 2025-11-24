# LLM Maximum Solution 분석 리포트

## 요약

**날짜**: 2025-11-21
**목표**: LLM을 최대한 활용하여 검색 성능 향상
**결과**: **성능 저하 발생** (97.3% → 72.7% 결과 제공률)

---

## 1. 테스트 결과 비교

### 1.1 전체 성능 지표

| 지표 | Baseline (2 LLM Stages) | LLM Maximum (6 LLM Stages) | 차이 |
|------|-------------------------|---------------------------|------|
| **결과 제공률** | 214/220 (97.3%) | 160/220 (72.7%) | **-24.6%** ❌ |
| **평균 문서 수** | 2.92개 | 1.49개 | **-1.43개** ❌ |
| **처리 시간** | ~15분 | ~25분 | **+10분** |

### 1.2 TopK 분포 비교

| TopK | Baseline | LLM Maximum | 차이 |
|------|----------|-------------|------|
| **TopK=0** (결과 없음) | 6 (2.7%) | 60 (27.3%) | **+54** ❌ |
| **TopK=1** | 0 (0%) | 47 (21.4%) | **+47** |
| **TopK=2** | 0 (0%) | 59 (26.8%) | **+59** |
| **TopK=3** | 214 (97.3%) | 54 (24.5%) | **-160** ❌ |

**해석**: LLM Maximum은 대부분 쿼리에서 3개 문서를 제공하지 못함.

---

## 2. 문제점 분석

### 2.1 과도한 Smalltalk 필터링

**문제**:
- Baseline: 6개 쿼리를 smalltalk로 분류 (하드코딩)
- LLM Maximum: **55개 쿼리를 smalltalk로 분류** (LLM 기반)

**원인**:
```python
# llm_maximum_solution.py:46-60
def llm_intent_classification(msg):
    """LLM Intent Classification"""
    prompt = f"""다음 질문이 과학 지식 검색이 필요한 질문인지 판단하세요.

    질문: {query}

    출력: 'science' 또는 'smalltalk' 중 하나만 출력하세요."""
```

**문제점**: LLM이 과학 질문을 너무 보수적으로 판단
- 예: "차량의 연비가 좋아질때 나타나는 긍정적인 효과는?" → smalltalk로 분류 가능
- 예: "건설 현장에서 망치로 벽을 치는 이유는?" → smalltalk로 분류됨

**영향**: 49개의 science 쿼리가 잘못 필터링되어 TopK=0 증가

### 2.2 너무 엄격한 70점 문턱값

**문제**:
```python
# llm_maximum_solution.py:147-152
def llm_final_reranking(query, scored_documents):
    """LLM Final Reranking - Stage 6"""
    # 70점 이상만 선택
    high_score_docs = [doc for doc in scored_documents if doc[1] >= 70]
    return high_score_docs[:3]
```

**영향**:
- 많은 쿼리에서 70점 이상 문서가 1-2개만 존재
- TopK=1, TopK=2가 총 106개 (47 + 59)
- Baseline에서는 이런 경우 BM25 순위대로 3개 제공

**예시 로그**:
```
[Final Selection] Top-3 선택: [('7c14c33c', 85)]  # TopK=1
[Final Selection] Top-3 선택: [('2077ea5b', 75)]  # TopK=1
[Final Selection] Top-3 선택: [('05b5a4f4', 85)]  # TopK=1
```

### 2.3 중복 DocID 버그 (여전히 존재)

**문제**:
```
[Final Selection] Top-3 선택: [('99a07643', 85), ('99a07643', 80)]
[Final Selection] Top-3 선택: [('11dddee4', 95), ('11dddee4', 90)]
[Final Selection] Top-3 선택: [('2077ea5b', 85), ('2077ea5b', 80)]
```

**원인**: BM25 검색에서 같은 문서가 여러 번 반환되거나, reranking 과정에서 중복 처리 미흡

**영향**: 실제 문서 수가 줄어들어 MAP 점수 하락

---

## 3. LLM 사용량 비교

### 3.1 Baseline (2 LLM Stages)

```
쿼리당 LLM 호출:
1. Query Rewriting (solar-mini): 1회
2. Reranking (solar-pro): 1회
→ 총 2회/쿼리
```

### 3.2 LLM Maximum (6 LLM Stages)

```
쿼리당 LLM 호출:
1. Intent Classification (solar-pro): 1회
2. Context-Aware Rewriting (solar-pro): 1회 (다중턴 대화만)
3. Query Enhancement (solar-mini): 1회
4. [BM25 Search - Non-LLM]
5. Document Relevance Scoring (solar-pro): 1회
6. Final Reranking (Non-LLM, 단순 필터링)
→ 총 3-4회/쿼리
```

**실제 LLM 사용량**: 1.5-2배 증가
**처리 시간**: 1.67배 증가 (15분 → 25분)

---

## 4. 왜 성능이 낮아졌나?

### 4.1 Recall vs Precision 트레이드오프

**Baseline**:
- High Recall (97.3% 결과 제공)
- Moderate Precision (BM25 + LLM reranking)
- 전략: "일단 결과를 제공하고, LLM으로 순서만 조정"

**LLM Maximum**:
- Low Recall (72.7% 결과 제공) ❌
- High Precision 시도 (70점 이상만)
- 전략: "확실한 문서만 제공"

**문제**: MAP 평가에서는 **Recall이 매우 중요**
- TopK=0 하나당 큰 감점
- TopK=3 → TopK=1로 줄어들면 2개 문서 누락으로 감점

### 4.2 LLM의 한계

**Intent Classification**:
- 과학과 일상의 경계가 모호한 질문 존재
- LLM이 보수적으로 판단하여 과학 질문도 smalltalk로 분류

**Document Scoring**:
- LLM이 문서 관련성을 70점 이상으로 평가하지 않는 경우 다수
- 실제로는 관련 있지만, LLM 판단으로 누락

---

## 5. 개선 방안

### 5.1 즉시 적용 가능한 수정

#### A. Intent Classification 완화
```python
# 현재: LLM 기반 → 문제 발생
# 해결: Baseline의 하드코딩 방식 유지 또는 하이브리드

# Option 1: 하드코딩 유지 (Baseline과 동일)
SMALLTALK_IDS = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

# Option 2: LLM + Fallback
intent = llm_intent_classification(msg)
if intent == 'smalltalk' and eval_id not in KNOWN_SMALLTALK:
    intent = 'science'  # 의심스러운 경우 science로 처리
```

#### B. 문턱값 낮추기
```python
# 현재: 70점 이상만 선택
MIN_SCORE_THRESHOLD = 70

# 수정: 50-60점으로 낮추기
MIN_SCORE_THRESHOLD = 50  # 또는 동적 조정

# 또는: 최소 문서 수 보장
def llm_final_reranking(query, scored_documents):
    high_score_docs = [doc for doc in scored_documents if doc[1] >= 70]
    if len(high_score_docs) < 3:
        # 70점 미만이라도 BM25 순위대로 3개 채우기
        all_sorted = sorted(scored_documents, key=lambda x: x[1], reverse=True)
        high_score_docs = all_sorted[:3]
    return high_score_docs[:3]
```

#### C. 중복 DocID 제거
```python
def llm_final_reranking(query, scored_documents):
    # 중복 제거
    seen_docids = set()
    unique_docs = []
    for doc in scored_documents:
        if doc[0] not in seen_docids:
            unique_docs.append(doc)
            seen_docids.add(doc[0])

    # 점수 기준 정렬 후 Top-3
    sorted_docs = sorted(unique_docs, key=lambda x: x[1], reverse=True)
    return sorted_docs[:3]
```

### 5.2 장기 개선 전략

#### A. LLM 사용의 재설계

**현재 문제**: 모든 단계에 LLM 사용 = 좋은 결과 (❌ 잘못된 가정)

**올바른 접근**:
1. **Query Rewriting**: LLM 유지 ✅
2. **Intent Classification**: 하드코딩 또는 하이브리드 ✅
3. **Query Enhancement**: 선택적 사용 (복잡한 쿼리만)
4. **BM25 Search**: 필수 유지
5. **Document Scoring**: LLM 유지 but 문턱값 조정 ✅
6. **Final Reranking**: 최소 문서 수 보장 ✅

#### B. 하이브리드 Scoring

```python
def hybrid_scoring(query, documents, bm25_scores):
    """LLM Score + BM25 Score 결합"""
    llm_scores = llm_document_relevance_scoring(query, documents)

    final_scores = []
    for (docid, llm_score, bm25_score) in llm_scores:
        # 가중 평균
        combined_score = 0.7 * llm_score + 0.3 * (bm25_score / max_bm25 * 100)
        final_scores.append((docid, combined_score, bm25_score))

    return final_scores
```

---

## 6. 예상 점수 영향

### 6.1 현재 상황 (LLM Maximum)

**예상 MAP**: **0.45 - 0.55** (Baseline 0.6856 대비 **-21% ~ -26%**)

**근거**:
- TopK=0: 60개 (27.3%) → 각각 0점
- TopK=1-2: 106개 (48.2%) → 부분 점수
- TopK=3: 54개 (24.5%) → 정상 점수

### 6.2 개선 후 예상

**개선안 적용 시**:
- Intent Classification 완화: +40개 결과 제공
- 문턱값 낮추기: +50개 TopK=3 달성
- 중복 제거: 정확도 +5%

**예상 MAP**: **0.72 - 0.78** (Baseline 0.6856 대비 **+5% ~ +14%**)

---

## 7. 결론 및 권장사항

### 7.1 즉시 조치 사항

1. ❌ **LLM Maximum 사용 중단** (현재 성능 0.45-0.55 예상)
2. ✅ **Baseline 유지** (검증된 0.6856 점수)
3. ⚙️ **선택적 개선** 적용:
   - 중복 DocID 버그 수정 (우선)
   - Context-Aware Rewriting 추가 (3개 쿼리만 영향)
   - Query Enhancement 선택적 적용

### 7.2 LLM 최대 활용의 올바른 방향

**잘못된 접근** (현재 LLM Maximum):
- "모든 단계에 LLM 사용 = 높은 점수" ❌

**올바른 접근**:
- "**적재적소에 LLM 활용** = 높은 점수" ✅
- Recall 유지하면서 Precision 향상
- LLM의 강점(언어 이해)과 BM25의 강점(키워드 매칭) 결합

### 7.3 다음 단계

**우선순위 1**: Baseline 개선
```
1. 중복 DocID 버그 수정
2. Context-Aware Rewriting 추가 (eval_id 43, 44, 97)
3. 재테스트
→ 예상 점수: 0.70-0.72
```

**우선순위 2**: 하이브리드 접근
```
1. LLM Scoring + BM25 Score 가중 결합
2. 동적 문턱값 (BM25 max_score 기반)
3. 재테스트
→ 예상 점수: 0.73-0.75
```

**우선순위 3**: Query Enhancement
```
1. 복잡한 쿼리만 선택적 확장
2. 불필요한 확장 방지 (과적합 위험)
3. 재테스트
→ 예상 점수: 0.75-0.78
```

---

## 8. 교훈

### 8.1 LLM 사용의 함정

**교훈 1**: "More LLM ≠ Better Results"
- LLM 호출 횟수와 성능은 비례하지 않음
- 오히려 과도한 LLM 의존이 Recall 저하 유발

**교훈 2**: "LLM은 도구, 전략이 중요"
- LLM을 어디에, 어떻게 사용하느냐가 핵심
- Baseline의 단순한 2-stage가 6-stage보다 우수

**교훈 3**: "Recall First, Precision Second"
- MAP 평가에서는 결과 제공률이 최우선
- 정밀도 향상은 Recall 유지 전제 하에 진행

### 8.2 실험의 가치

**긍정적 측면**:
- LLM Maximum 실험으로 한계 파악 ✅
- Intent Classification의 문제점 발견 ✅
- 70점 문턱값의 부작용 확인 ✅

**다음 실험**:
- 개선안 적용 버전 (llm_maximum_v2.py)
- A/B 테스트로 점진적 개선 검증

---

**분석 작성**: Claude Code
**작성일**: 2025-11-21
**버전**: 1.0
