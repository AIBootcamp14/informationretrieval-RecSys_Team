# Task 4: Cascaded Stages 증가 실험

## 실험 목표
Task 3 failure analysis에서 발견한 문제 해결:
- **문제점**: 30→10 급격한 필터링이 3점 문서를 누락시킴
- **가설**: 더 세밀한 단계적 필터링으로 3점 문서 보존 가능
- **목표**: MAP@3 0.75 → 0.80+ 향상

## Cascaded Reranking v1 vs v2 비교

### v1 (현재 최고 성능: MAP@3 0.7939)
```
검색: Top 30
   ↓
Stage 1: Top 30 → Top 10 (빠른 필터링)
   ↓
Stage 2: Top 10 → Top 3 (정밀 Reranking)
```

**문제점**:
- 30→10에서 20개 문서를 한 번에 제거
- 너무 급격한 필터링으로 3점 문서가 소실됨
- Validation 결과: 4/8 partial matches (0 < AP@3 < 1.0)

### v2 (실험 중)
```
검색: Top 50 (30 → 50 확장)
   ↓
Stage 1: Top 50 → Top 30 (1차 필터링)
   ↓
Stage 2: Top 30 → Top 20 (2차 필터링)
   ↓
Stage 3: Top 20 → Top 10 (3차 필터링)
   ↓
Stage 4: Top 10 → Top 3 (최종 정밀 Reranking)
```

**개선점**:
- 각 단계별로 제거되는 문서 수 감소 (20개 → 10개씩)
- 더 많은 단계로 점진적 필터링
- 초기 검색 범위 확대 (30 → 50)

## 주요 변경사항

### 1. 초기 검색 확대
```python
# v1
hybrid_results = hybrid_search_rrf(
    rewritten_query,
    embeddings_dict,
    top_k=30,  # Top 30
    k=60,
    query_max_length=128
)

# v2
hybrid_results = hybrid_search_rrf(
    rewritten_query,
    embeddings_dict,
    top_k=50,  # ✅ Top 50으로 확장
    k=60,
    query_max_length=128
)
```

### 2. Cascaded Stage 증가
```python
# v1: 2-Stage
def llm_cascaded_rerank(query, docs, top_k=3):
    # Stage 1: Top 30 → Top 10
    stage1_indices = llm_stage1_filter(query, docs, top_k=10)
    stage1_docs = [docs[i] for i in stage1_indices if i < len(docs)]

    # Stage 2: Top 10 → Top 3
    final_docids = llm_stage2_rerank(query, stage1_docs, top_k=top_k)
    return final_docids

# v2: 4-Stage (세밀한 필터링)
def llm_cascaded_rerank_v2(query, docs, top_k=3):
    # Stage 1: Top 50 → Top 30
    stage1_indices = llm_stage_filter(query, docs[:50], top_k=30, stage_name="1차")
    stage1_docs = [docs[i] for i in stage1_indices if i < len(docs)]

    # Stage 2: Top 30 → Top 20
    stage2_indices = llm_stage_filter(query, stage1_docs, top_k=20, stage_name="2차")
    stage2_docs = [stage1_docs[i] for i in stage2_indices if i < len(stage1_docs)]

    # Stage 3: Top 20 → Top 10
    stage3_indices = llm_stage_filter(query, stage2_docs, top_k=10, stage_name="3차")
    stage3_docs = [stage2_docs[i] for i in stage3_indices if i < len(stage2_docs)]

    # Stage 4: Top 10 → Top 3 (최종)
    final_docids = llm_final_rerank(query, stage3_docs, top_k=top_k)
    return final_docids
```

### 3. 범용 Stage Filter 함수
```python
def llm_stage_filter(query, docs, top_k, stage_name):
    """
    범용 LLM Stage 필터링
    각 단계별로 동일한 로직 사용
    """
    # Solar Pro로 관련성 판단하여 top_k 선택
    # ...
```

## 예상 결과

### 시나리오 1: 성공 (MAP@3 향상)
- **가설 검증**: 더 세밀한 필터링이 3점 문서 보존에 효과적
- **예상 성능**: 0.78~0.80 MAP@3
- **다음 단계**: v2를 baseline으로 하여 Task 5 진행

### 시나리오 2: 성능 유지 또는 하락
- **원인 분석**:
  - 더 많은 LLM 호출로 인한 노이즈 증가
  - Stage 증가가 오히려 혼란을 야기
  - 실제 문제는 Retrieval Recall (Task 3 분석 결과)
- **다음 단계**:
  - Retrieval Recall 개선으로 방향 전환
  - Query Decomposition (Task 5) 또는 Document Context Expansion (Task 6)

## Validation 방법

```bash
cd /Users/dongjunekim/dev_team/ai14/ir/code
export UPSTAGE_API_KEY=up_sv4ka64IAQVM0kw07iclUbvB5ZRZe
python3 auto_validate.py cascaded_reranking_v2 cascaded_reranking_strategy
```

**Ultra Validation Set** (8 samples):
- eval_id: 205, 18, 43, 226, 24, 200, 41, 47
- Solar Pro 5-phase validation으로 생성된 ground truth
- 각 문서별 detailed scores (1-5점) 제공

## Task 3 Failure Analysis 요약

| eval_id | v1 AP@3 | Hits | 문제 유형 |
|---------|---------|------|-----------|
| 18      | 0.6667  | 2/3  | Retrieval 실패 (1개 missed) |
| 24      | 0.3333  | 1/3  | Retrieval 실패 (2개 missed) |
| 200     | 0.3333  | 1/3  | Retrieval 실패 (2개 missed) |
| 41      | 0.6667  | 2/3  | Reranking 실패 (3점 문서 swap) |

**핵심 발견**:
- 7개 잘못된 문서 중 6개가 Solar Pro 스코어링 세트에 없음
- Retrieval Recall 문제가 Reranking 문제보다 심각
- 3점 문서들이 초기 검색 단계에서 누락됨

## 실험 타임라인

1. ✅ cascaded_reranking_v2.py 작성 완료
2. 🔄 auto_validate.py로 ultra validation set 테스트 중
3. ⏳ 결과 분석 및 비교
4. ⏳ 다음 Task 결정

---

**작성일**: 2025-11-24
**실험자**: Claude Code
**관련 파일**:
- [cascaded_reranking_v2.py](cascaded_reranking_v2.py)
- [auto_validate.py](auto_validate.py)
- [analyze_failures.py](analyze_failures.py)
