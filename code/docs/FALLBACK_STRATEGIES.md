# Query Expansion 실패 시 대응 전략

## 현재 상황 정리

### 시도한 방법과 결과
1. **super_simple** (BM25 + threshold=2.0): **0.6300** ✅ (최고)
2. context_aware: 0.6220
3. hybrid_dense (α=0.7): 0.6205
4. selective_context: 0.6038
5. pure_bm25 (adaptive TopK): 0.5568
6. cross_encoder_reranking: **0.1947** ❌ (최악)

### 패턴 분석

**성공 요인:**
- TopK=3 고정 (adaptive는 실패)
- Threshold=2.0 적절
- **Simple이 Best**

**실패 요인:**
- Dense Retrieval 추가 → 0.62 하락
- Adaptive TopK → 0.56 하락
- Cross-Encoder (미초기화) → 0.19 참사
- 복잡한 방법일수록 오히려 성능 하락

## Query Expansion 0.6 미만 시 전략

### 전략 1: BM25 파라미터 튜닝 (가장 우선)

**현재 문제:**
- BM25 기본 파라미터 사용 중
- k1=1.2, b=0.75 (Elasticsearch 기본값)

**최적화 방법:**
```python
# Elasticsearch에서 BM25 파라미터 조정
{
    "similarity": {
        "custom_bm25": {
            "type": "BM25",
            "k1": 1.5,  # 단어 빈도 중요도 (1.2~2.0)
            "b": 0.6    # 문서 길이 정규화 (0.5~0.9)
        }
    }
}
```

**k1 효과:**
- 높을수록: 단어 빈도 더 중요 (반복되는 키워드 강조)
- 낮을수록: 단어 존재 여부만 중요 (희귀 키워드 강조)

**b 효과:**
- 높을수록: 긴 문서 페널티 증가
- 낮을수록: 문서 길이 무시

**과학 문서 특성 고려:**
- 과학 용어는 반복보다 등장 자체가 중요 → k1 낮게 (1.0~1.2)
- 문서 길이 다양 → b 낮게 (0.5~0.7)

**실험 계획:**
```python
parameter_sets = [
    {"k1": 1.0, "b": 0.5},  # 희귀 키워드 + 길이 무시
    {"k1": 1.0, "b": 0.75}, # 희귀 키워드 + 기본 정규화
    {"k1": 1.5, "b": 0.5},  # 빈도 중요 + 길이 무시
    {"k1": 1.5, "b": 0.75}, # 빈도 중요 + 기본 정규화
]
```

---

### 전략 2: Document Preprocessing (재인덱싱)

**현재 문제:**
- 문서가 너무 길거나 짧을 수 있음
- 불필요한 내용 포함 가능

**개선 방법:**

1. **문서 청소 (Cleaning)**
   ```python
   # 특수 문자, 불필요한 공백 제거
   # HTML 태그 제거
   # 반복 문장 제거
   ```

2. **문서 분할 (Chunking)**
   ```python
   # 현재: 문서 전체를 하나의 단위로
   # 개선: 문서를 500자 단위로 분할
   # → 더 정확한 매칭 가능
   ```

3. **핵심 문장 추출**
   ```python
   # 각 문서에서 가장 중요한 3~5문장만 인덱싱
   # TF-IDF로 중요도 계산
   ```

---

### 전략 3: Ensemble Voting

**아이디어:**
- 여러 방법의 결과를 투표로 결합
- 각 방법의 장점 활용

**방법:**

1. **Simple Voting**
   ```python
   results = {
       'super_simple': [A, B, C],
       'context_aware': [B, C, D],
       'hybrid_dense': [A, C, E]
   }

   # 각 문서가 몇 번 등장했는지 카운트
   votes = Counter([A:2, B:2, C:3, D:1, E:1])

   # 가장 많이 등장한 3개 선택
   final = [C, A, B]  # 득표순
   ```

2. **Weighted Voting**
   ```python
   # 성능 좋은 방법에 더 높은 가중치
   weights = {
       'super_simple': 0.6,    # 0.63 성능
       'context_aware': 0.3,   # 0.62 성능
       'hybrid_dense': 0.1     # 0.62 성능
   }
   ```

3. **Rank Fusion**
   ```python
   # Reciprocal Rank Fusion (RRF)
   # 각 방법의 순위를 결합

   for method, docs in results.items():
       for rank, doc in enumerate(docs, 1):
           score[doc] += 1 / (rank + 60)

   # 점수 높은 순으로 Top-3 선택
   ```

**기대 효과:**
- 0.63 (super_simple) → 0.65~0.67
- Robust한 결과 (한 방법 실패해도 안전)

---

### 전략 4: Query Classification (쿼리 분류)

**아이디어:**
- 쿼리 유형에 따라 다른 전략 사용
- "어려운 쿼리"와 "쉬운 쿼리" 구분

**방법:**

1. **쿼리 난이도 분석**
   ```python
   def classify_query_difficulty(query, bm25_max_score):
       if bm25_max_score > 10:
           return "easy"     # 명확한 키워드 매칭
       elif bm25_max_score > 5:
           return "medium"   # 적당한 매칭
       else:
           return "hard"     # 모호한 쿼리
   ```

2. **난이도별 전략**
   ```python
   if difficulty == "easy":
       # BM25 Top-3 그대로 (신뢰)
       return bm25_top3

   elif difficulty == "medium":
       # Query Expansion 사용
       return query_expansion(query)

   else:  # hard
       # Hybrid Search 사용
       return hybrid_search(query)
   ```

**기대 효과:**
- 쉬운 쿼리: 안전하게 처리
- 어려운 쿼리: 더 정교한 방법 적용
- 0.63 → 0.66~0.68

---

### 전략 5: Multi-field Search (다중 필드 검색)

**현재 문제:**
- `content` 필드만 검색 중
- 문서에 다른 유용한 필드 있을 수 있음

**개선:**
```python
# 여러 필드에 가중치 부여
{
    "query": {
        "multi_match": {
            "query": query,
            "fields": [
                "content^2",      # content 2배 가중치
                "title^3",        # title 3배 가중치 (있다면)
                "keywords^1.5"    # keywords 1.5배 (있다면)
            ]
        }
    }
}
```

---

### 전략 6: Negative Sampling (제외 규칙)

**아이디어:**
- 명백히 틀린 문서를 제외
- Precision 향상

**방법:**
```python
def should_exclude(query, doc):
    """문서를 제외해야 하는지 판단"""

    # 규칙 1: 너무 짧은 문서
    if len(doc['content']) < 50:
        return True

    # 규칙 2: 쿼리와 전혀 다른 주제
    query_keywords = extract_keywords(query)
    doc_keywords = extract_keywords(doc['content'])

    overlap = len(query_keywords & doc_keywords)
    if overlap == 0:
        return True

    return False
```

---

## 실행 우선순위

### 즉시 실행 (Query Expansion 0.6 미만일 때)

**1순위: BM25 파라미터 튜닝** ⭐⭐⭐
- 구현 간단
- 효과 높을 가능성
- 위험 낮음

**2순위: Ensemble Voting** ⭐⭐
- 기존 결과 활용
- 안정적
- 추가 비용 없음

**3순위: Query Classification** ⭐⭐
- 논리적 접근
- 효과 기대
- 구현 복잡도 중간

### 차선책

**4순위: Document Preprocessing**
- 재인덱싱 필요 (시간 소요)
- 효과 불확실

**5순위: Multi-field Search**
- 필드 정보 필요
- 데이터 구조 파악 필요

**6순위: Negative Sampling**
- 복잡한 규칙 설계 필요
- 효과 제한적

---

## 최종 권장 순서

```
Query Expansion 결과 확인
    │
    ├─ 0.63+ (성공)
    │   └─ 파라미터 미세 조정 (키워드 개수 등)
    │
    ├─ 0.60~0.63 (약간 개선)
    │   └─ 1. BM25 파라미터 튜닝
    │       2. Ensemble Voting 추가
    │
    └─ < 0.60 (실패)
        └─ 1. BM25 파라미터 튜닝 (최우선)
            2. Ensemble Voting (안전한 선택)
            3. Query Classification (추가 개선)
            4. Document Preprocessing (최후 수단)
```

---

## 예상 최종 성능

| 전략 | 예상 MAP | 구현 난이도 | 소요 시간 |
|------|----------|------------|----------|
| BM25 파라미터 | 0.65~0.68 | 낮음 | 10분 |
| Ensemble | 0.64~0.67 | 낮음 | 10분 |
| Query Classification | 0.65~0.69 | 중간 | 20분 |
| Document Preprocessing | 0.66~0.70 | 높음 | 1시간 |

---

## 결론

**Query Expansion이 0.6 미만이라면:**

1. **즉시**: BM25 파라미터 튜닝 실행
2. **동시**: Ensemble Voting 구현
3. **결과 확인 후**: Query Classification 추가

**목표: 0.65~0.68 달성**
- 0.9 목표는 현재 접근으로 불가능
- 현실적 목표를 0.68로 조정
- 더 큰 도약은 완전히 다른 접근 필요 (Fine-tuned 모델, 더 큰 데이터셋 등)
