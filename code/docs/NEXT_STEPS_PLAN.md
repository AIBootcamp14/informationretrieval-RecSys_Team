# 다음 단계 실행 계획

## 현재 상황
- Validation set 기반 개발 포기 (3가지 방법 모두 negative correlation)
- Direct Leaderboard Testing으로 전환
- Dense Retrieval 실행 중 (문서 임베딩 생성)

## 단계별 실행 계획

### ✅ 단계 1: Dense Retrieval (진행 중)
**목표**: 의미 기반 검색으로 BM25 한계 극복

**실행 중**:
```bash
python3 rag_with_dense.py
```

**예상 결과**:
- 파일: `hybrid_dense_submission.csv`
- 기대 MAP: 0.65~0.70 (BM25 0.63 대비 개선)
- 소요 시간: 약 10~15분 (임베딩 생성 + 검색)

**완료 후 작업**:
1. Leaderboard 제출
2. 점수 확인 후 다음 단계 결정

---

### 📋 단계 2: Hybrid 파라미터 최적화
**조건**: 단계 1에서 0.65+ 달성 시

**전략**:
```python
# Alpha 파라미터 실험
alpha_values = [0.3, 0.5, 0.7, 0.9]  # BM25 가중치
# 0.3: Dense 70%, BM25 30%
# 0.9: Dense 10%, BM25 90%
```

**실행 파일 생성 필요**:
```bash
python3 optimize_hybrid_alpha.py
```

**예상 결과**:
- 최적 alpha 발견
- MAP 0.70+ 가능

**소요 시간**: alpha당 5분 × 4 = 20분

---

### 📋 단계 3: Query Expansion
**조건**: 단계 2에서 개선 확인 시

**방법**:
1. **LLM 기반 Query Rewriting** (신중하게)
   - Context-aware는 실패했지만 다른 방식 시도
   - 예: 쿼리 확장 (동의어, 관련 키워드 추가)

2. **Pseudo Relevance Feedback**
   - Top-3 문서에서 핵심 단어 추출
   - 원래 쿼리에 추가하여 재검색

**구현 파일**:
```python
# query_expansion.py
def expand_query_with_prf(query, top_docs):
    # Top-3 문서에서 TF-IDF 상위 단어 추출
    # 원래 쿼리에 추가
    pass
```

**예상 결과**:
- 어려운 쿼리 (BM25 score < 10) 개선
- MAP 0.72+ 가능

---

### 📋 단계 4: Reranking with LLM
**조건**: 단계 3에서 0.70+ 달성 시

**전략**:
1. Hybrid Search로 Top-10 검색
2. LLM으로 Top-10을 Top-3으로 재정렬
3. 관련도 판단에 LLM의 언어 이해력 활용

**구현**:
```python
def rerank_with_llm(query, top_10_docs):
    prompt = f"""
    Query: {query}

    다음 문서들을 관련도 순으로 정렬하세요:
    {format_docs(top_10_docs)}

    가장 관련도 높은 3개의 문서 번호를 출력하세요.
    """
    # LLM 호출하여 재정렬
```

**예상 결과**:
- Precision 대폭 향상
- MAP 0.75~0.80 가능

**주의사항**:
- LLM 비용 증가 (220 쿼리 × 1회 = 약 $0.5~1.0)
- 속도 느림 (약 30분 소요)

---

### 📋 단계 5: Ensemble Methods
**조건**: 여러 방법이 0.70+ 달성 시

**전략**:
```python
# 여러 검색 방법 결과 결합
results_bm25 = search_bm25(query)
results_dense = search_dense(query)
results_hybrid = search_hybrid(query)

# Voting or Score Fusion
final_results = ensemble([results_bm25, results_dense, results_hybrid])
```

**예상 결과**:
- 안정적인 성능
- MAP 0.80+ 가능

---

### 📋 단계 6: Advanced Techniques (0.9 목표)
**조건**: 단계 5에서 0.80+ 달성 시

**방법들**:

1. **Cross-Encoder Reranking**
   - Bi-encoder (현재 Dense) → Cross-encoder
   - 쿼리-문서 쌍을 직접 점수화
   - 훨씬 정확하지만 느림

2. **ColBERT (Late Interaction)**
   - Token-level interaction
   - Dense보다 정확, Cross-encoder보다 빠름

3. **Document Augmentation**
   - 문서에 LLM으로 요약 추가
   - 검색 정확도 향상

4. **Multi-vector Retrieval**
   - 문서를 여러 벡터로 표현
   - 더 풍부한 의미 표현

---

## 의사결정 트리

```
Dense Retrieval 실행
    │
    ├─ MAP < 0.65
    │   └─ Query Expansion 시도 (단계 3)
    │
    ├─ 0.65 ≤ MAP < 0.70
    │   └─ Hybrid 최적화 (단계 2) → 단계 3
    │
    ├─ 0.70 ≤ MAP < 0.80
    │   └─ Reranking (단계 4) → Ensemble (단계 5)
    │
    └─ MAP ≥ 0.80
        └─ Advanced Techniques (단계 6)로 0.9 도전
```

---

## 실행 체크리스트

### 현재 진행 중
- [x] Dense Retrieval 실행 (`rag_with_dense.py`)
- [ ] 임베딩 생성 완료 대기
- [ ] Submission 파일 생성 확인
- [ ] Leaderboard 제출
- [ ] 결과 확인

### 다음 작업 (결과에 따라)
- [ ] Hybrid alpha 최적화 스크립트 작성
- [ ] Query expansion 구현
- [ ] Reranking 구현
- [ ] Ensemble 방법 구현

---

## 예상 타임라인

| 단계 | 소요 시간 | 누적 시간 |
|------|----------|----------|
| Dense Retrieval | 15분 | 15분 |
| Leaderboard 제출 | 5분 | 20분 |
| Hybrid 최적화 | 30분 | 50분 |
| Query Expansion | 20분 | 70분 |
| Reranking | 40분 | 110분 |
| Ensemble | 30분 | 140분 |
| **총 소요 시간** | | **약 2.5시간** |

---

## 핵심 원칙

1. **Validation 사용 금지**
   - 모든 평가는 Leaderboard에서만
   - Correlation이 negative면 의미 없음

2. **점진적 개선**
   - 한 번에 하나씩 변경
   - 각 변경사항의 효과 명확히 파악

3. **비용 고려**
   - LLM 호출 최소화
   - 필요한 경우에만 사용

4. **재현 가능성**
   - 모든 실험 스크립트 저장
   - Random seed 고정

---

## 최종 목표

**Target: MAP 0.9**

현실적 경로:
1. Dense Retrieval: 0.65~0.70
2. Hybrid 최적화: 0.70~0.75
3. Reranking: 0.75~0.80
4. Ensemble: 0.80~0.85
5. Advanced: 0.85~0.90

각 단계에서 **0.05~0.10** 개선 필요
