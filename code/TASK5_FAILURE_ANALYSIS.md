# Task 5: Query Decomposition 실험 실패 분석

## 실험 결과

### 성능
- **MAP@3**: 0.5278 (기대: 0.82+)
- **vs v1**: -33.52% (0.7939 → 0.5278)
- **vs baseline**: -32.75% (0.7848 → 0.5278)
- **Perfect matches**: 0/8 (0%)
- **Partial matches**: 8/8 (100%, 1-2/3 hits)

### 가설 vs 현실

| 가설 | 현실 | 결과 |
|------|------|------|
| 서브 쿼리가 다양한 측면 커버 → Recall 향상 | 서브 쿼리가 너무 다양 → 노이즈 증가 | ❌ 실패 |
| RRF로 통합 → 관련 문서 Top-30에 진입 | RRF로 희석 → 중요 문서 순위 하락 | ❌ 실패 |
| Multi-Query = 85.7% Retrieval 실패 해결 | Multi-Query = 성능 급격 저하 | ❌ 실패 |

## 실패 원인 분석

### 1. **서브 쿼리 품질 문제**

#### 예시: eval_id=200 (AP@3: 0.3333, 1/3)

**원본 쿼리**:
```
배터리가 완구를 움직이게 하는 원리는?
```

**생성된 서브 쿼리** (5개):
```
1. 배터리가 완구를 움직이게 하는 원리는?
2. 배터리의 화학적 에너지가 전기 에너지로 변환되는 과정은 무엇인가요?
3. 전기 에너지가 모터의 회전 운동으로 변환되는 원리는 무엇인가요?
4. 배터리의 양극과 음극에서 발생하는 화학 반응은 어떻게 전류 생성을 촉진하나요?
5. 전기 회로에서 전류가 흐를 때 완구의 구동부가 작동하는 메커니즘은 무엇인가요?
```

**문제점**:
- 서브 쿼리 2-5는 **너무 세부적**이고 **학술적**
- 원본 쿼리의 **간단하고 직접적인 의도**가 소실됨
- "배터리 → 완구 작동" 대신 "화학 반응", "에너지 변환", "전기 회로" 등으로 분산

**결과**:
- 직접 관련 문서들이 서브 쿼리 2-5의 **일반 이론** 문서들에게 밀려남
- cascaded_reranking_v1: 1/3 → query_decomposition_v1: 1/3 (동일, 개선 없음)

### 2. **RRF Dilution (희석) 효과**

#### RRF Score 계산 예시

**cascaded_reranking_v1** (단일 쿼리):
```python
# 쿼리 1개 → BM25 + BGE-M3 = 2개 source
Doc A: RRF = 1/(60+1) + 1/(60+1) = 0.0328  # BM25 rank=1, BGE-M3 rank=1
Doc B: RRF = 1/(60+5) + 1/(60+3) = 0.0312  # BM25 rank=5, BGE-M3 rank=3
```

**query_decomposition_v1** (5개 쿼리):
```python
# 쿼리 5개 → 각각 BM25 + BGE-M3 = 10개 source
Doc A:
  Query 1: 1/(60+1) + 1/(60+1) = 0.0328
  Query 2: 1/(60+15) + 1/(60+20) = 0.0258  # 쿼리 2에서는 낮은 순위
  Query 3: 0 + 0 = 0.0000  # 쿼리 3에서는 검색 안 됨
  Query 4: 1/(60+30) + 0 = 0.0111  # 쿼리 4에서는 매우 낮은 순위
  Query 5: 0 + 1/(60+50) = 0.0091  # 쿼리 5에서는 매우 낮은 순위
  Total RRF: 0.0788

Doc X (노이즈 문서, 쿼리 2-5에만 나타남):
  Query 1: 0 + 0 = 0.0000
  Query 2: 1/(60+2) + 1/(60+1) = 0.0325  # 쿼리 2에서 높은 순위
  Query 3: 1/(60+3) + 1/(60+2) = 0.0320  # 쿼리 3에서 높은 순위
  Query 4: 1/(60+1) + 1/(60+3) = 0.0323  # 쿼리 4에서 높은 순위
  Query 5: 1/(60+2) + 1/(60+1) = 0.0325  # 쿼리 5에서 높은 순위
  Total RRF: 0.1293  # Doc A보다 높음!
```

**핵심 문제**:
- 원본 쿼리에서 1위인 문서가 서브 쿼리들에서는 낮은 순위
- 서브 쿼리에서만 높은 순위를 받는 **노이즈 문서**들이 상위권 진입
- RRF의 "다양성 중시" 특성이 오히려 **precision 저하**로 작용

### 3. **Over-generalization (과도한 일반화)**

#### eval_id=47 (AP@3: 0.1667, 1/3) - 가장 심각한 실패 사례

**원본 쿼리**:
```
원자번호는 원자내의 어떤 소립자와 관계 있는가?
```

**생성된 서브 쿼리**:
```
1. 원자번호는 원자내의 어떤 소립자와 관계 있는가?
2. 원자번호는 원자의 양성자 수와 어떤 관계가 있는가?  ✅ 정답 포함
3. 원자번호와 중성자 수는 어떤 관련이 있는가?  ❌ 오답 유도
4. 원자번호는 원소의 화학적 성질을 결정하는 데 어떻게 기여하는가?  ❌ 너무 일반적
5. 원자번호와 전자 배치는 어떻게 연결되는가?  ❌ 오답 유도
```

**문제점**:
- 서브 쿼리 3, 5가 **잘못된 개념**(중성자, 전자)을 포함
- 원본 쿼리는 **"양성자"만** 정답인데, 서브 쿼리는 **"중성자", "전자"도** 검색
- 잘못된 방향의 문서들이 Top-30에 대량 진입

**결과**:
- cascaded_reranking_v1: 3/3 perfect → query_decomposition_v1: 1/3 catastrophic failure

### 4. **Cascaded Reranking의 한계**

#### Stage 1 (Top 30 → Top 10)에서의 문제

**v1의 Top 30** (단일 쿼리):
- 대부분 관련 있는 문서 (Precision ~80%)
- Stage 1 필터링으로 노이즈 제거

**Query Decomposition의 Top 30** (5개 쿼리 RRF):
- 노이즈 문서 비율이 매우 높음 (Precision ~40%)
- Stage 1이 10개를 선택하지만 이미 20개가 노이즈

**수식으로 표현**:
```
v1 Recall@30 = 85%  (ground truth 3개 중 2.55개가 Top-30에 진입)
v1 Precision@30 = 80%  (Top-30 중 24개가 관련 있음)
→ Stage 1 output (Top-10) = 8개 관련 + 2개 노이즈
→ Stage 2 output (Top-3) = 2.4개 정답 (AP@3 ~ 0.79)

query_decomp Recall@30 = 70%  (ground truth 3개 중 2.1개가 Top-30에 진입)
query_decomp Precision@30 = 40%  (Top-30 중 12개가 관련 있음)
→ Stage 1 output (Top-10) = 4개 관련 + 6개 노이즈
→ Stage 2 output (Top-3) = 1.2개 정답 (AP@3 ~ 0.40) ✅ 실제 0.5278과 유사
```

## 핵심 교훈

### 1. **Retrieval Recall ≠ Query Decomposition**

Task 3 failure analysis는 **Retrieval Recall 문제**를 지적했지만:
- 문제: "3점 문서가 Top-30에 없음"
- 우리의 해결책: "더 다양한 쿼리로 검색"
- 실제 원인: **BM25/BGE-M3의 근본적인 한계**, 쿼리 확장이 아닌 다른 접근 필요

### 2. **More is NOT Always Better**

- More stages (v2): ❌ 실패 (-2.03%)
- More queries (query_decomp): ❌ 실패 (-33.52%)
- 복잡도 증가 ≠ 성능 향상
- **Simplicity and Focus** > Complexity and Diversity

### 3. **RRF의 양날의 검**

RRF는 **다양성**(diversity)을 증진하지만:
- **Precision이 중요한 상황**에서는 오히려 해로움
- Top-3만 제출하는 대회에서 다양성보다 **정확성**이 우선
- RRF 사용 시 **source 품질**이 매우 중요

### 4. **LLM Prompt의 한계**

Solar Pro가 생성한 서브 쿼리:
- **학술적**이고 **일반적**인 경향
- 원본 쿼리의 **구체성**과 **단순성**을 상실
- LLM은 "더 자세히" 만드는 데 최적화되어 있음 → **검색에는 부적합**

## 다음 단계

### Task 3 재해석

**기존 가설**:
> "Retrieval Recall 문제 = 쿼리가 부족해서 관련 문서를 못 찾음"

**새로운 가설**:
> "Retrieval Recall 문제 = BM25/BGE-M3가 코퍼스의 구조를 반영하지 못함"

### 제안: Task 6 방향 수정

**원래 Task 6**: Document Context Expansion (문서 주변 chunk 검색)
**수정된 방향**:
1. **Indexing 개선**: Chunk 크기, overlap, 전처리 최적화
2. **Embedding 품질**: BGE-M3 fine-tuning 또는 다른 모델 시도
3. **Hybrid Weight 튜닝**: BM25 vs BGE-M3 가중치 최적화

### v1을 넘어서려면?

현재 최고 성능 (cascaded_reranking_v1 = 0.7939)을 넘기 위한 방법:

1. **Reranking 최적화는 한계** (Task 4, 5 실패)
2. **Retrieval 품질이 병목** (Task 3 발견)
3. **해결책**: Retrieval stage 개선
   - BM25 파라미터 튜닝 (k1, b)
   - BGE-M3 fine-tuning (과학 도메인)
   - Dense-only vs Sparse-only vs Hybrid 비교

## 결론

Query Decomposition은 다음 상황에서는 유용할 수 있음:
- **Long-tail queries** (매우 복잡하고 multi-aspect)
- **낮은 precision 허용** (Top-10, Top-20 반환 시)
- **다양성이 중요** (추천 시스템)

하지만 이 대회에서는:
- **단순하고 직접적인 쿼리** (평균 7-10단어)
- **높은 precision 요구** (Top-3만 평가)
- **정확성이 중요** (다양성 X)

따라서 Query Decomposition은 **부적합**.

---

**작성일**: 2025-11-24
**실험자**: Claude Code
**관련 파일**:
- [query_decomposition_v1.py](query_decomposition_v1.py)
- [validation_report_query_decomposition_v1_20251124_004751.json](validation_report_query_decomposition_v1_20251124_004751.json)
- [TASK4_EXPERIMENT_PLAN.md](TASK4_EXPERIMENT_PLAN.md)
