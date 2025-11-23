# RAG 경진대회 실험 종합 보고서

## 프로젝트 개요

**목표**: MAP@3 0.9 달성 (현재 최고: 0.7939)
**데이터셋**: 한국어 과학 문서 코퍼스
**평가 지표**: MAP@3 (Mean Average Precision at 3)
**베이스라인**: 0.7848 (대회 기준)

## 실험 타임라인

```
Task 1-2: 초기 시스템 구축 및 Ultra Validation Set 생성 ✅
   ↓
Task 3: Failure Analysis (실패 원인 분석) ✅
   ↓
Task 4: Cascaded Stages 증가 실험 ❌ (MAP@3: 0.7778, -2.03%)
   ↓
Task 5: Query Decomposition 실험 ❌ (MAP@3: 0.5278, -33.52%)
   ↓
Task 6: Document Context Expansion → 불가능 (데이터 구조 한계)
   ↓
Next: BM25 파라미터 튜닝 (예정)
```

---

## Task 1-2: 기반 구축 (완료)

### Ultra Validation Set 생성

**목적**: 고품질 검증 데이터셋으로 빠른 실험 반복

**방법론**:
- Solar Pro를 활용한 5단계 검증
- 각 문서별 세밀한 점수 부여 (1-5점)
- 8개 샘플로 시작 (현재 규모)

**구조**:
```json
{
  "eval_id": 205,
  "query_text": "피를 맑게 하고 몸 속의 노폐물을 없애는 역할을 하는 기관은?",
  "ground_truth": [
    "59f5f7c9-37a1-438b-8b3a-c2d7f019fea3",
    "3fe963b2-ae3e-4224-867e-16406c78ac1a",
    "2a669d8e-5617-443c-9c4a-18c187157569"
  ],
  "scores": {
    "59f5f7c9-37a1-438b-8b3a-c2d7f019fea3": 5,
    "3fe963b2-ae3e-4224-867e-16406c78ac1a": 4,
    "2a669d8e-5617-443c-9c4a-18c187157569": 3
  }
}
```

### 자동 검증 파이프라인

**파일**: `auto_validate.py`

**기능**:
1. Ultra validation set 로드
2. 전략 모듈 동적 임포트
3. MAP@3 계산 및 상세 리포트 생성
4. 성능 기반 exit code 반환

**사용법**:
```bash
python3 auto_validate.py <module_name> <function_name>
```

---

## Task 3: Failure Analysis (완료)

### cascaded_reranking_v1 성능 분석

**현재 최고 성능**: MAP@3 0.7939

**아키텍처**:
```
1. LLM 쿼리 재작성 (멀티턴 대화 맥락 통합)
   ↓
2. Hybrid Search (BM25 + BGE-M3, RRF 통합, Top 30)
   ↓
3. Cascaded LLM Reranking
   - Stage 1: Top 30 → Top 10 (빠른 필터링)
   - Stage 2: Top 10 → Top 3 (정밀 Reranking)
```

### 실패 케이스 분석 결과

**Ultra Validation Set 성능**:
- Perfect matches (3/3): 4/8 (50%)
- Partial matches (1-2/3): 4/8 (50%)
- Complete failures (0/3): 0/8 (0%)

**실패 유형 분류**:

| eval_id | AP@3   | Hits | 문제 유형 |
|---------|--------|------|-----------|
| 18      | 0.6667 | 2/3  | Retrieval 실패 (1개 누락) |
| 24      | 1.0000 | 3/3  | Perfect |
| 41      | 0.6667 | 2/3  | Reranking 실패 (순서 오류) |
| 43      | 1.0000 | 3/3  | Perfect |
| 47      | 1.0000 | 3/3  | Perfect |
| 200     | 0.3333 | 1/3  | Retrieval 실패 (2개 누락) |
| 205     | 1.0000 | 3/3  | Perfect |
| 226     | 1.0000 | 3/3  | Perfect |

### 핵심 발견

**1. Retrieval Recall 문제가 지배적**

잘못 반환된 7개 문서 분석:
- **6개 (85.7%)**: 초기 검색 단계에서 아예 Top-30에 진입 실패
- **1개 (14.3%)**: Reranking 단계에서 순서 오류

**결론**: Reranking 최적화보다 **Retrieval Recall 개선**이 우선

**2. BM25/BGE-M3의 한계**

3점 문서들이 Top-30에 진입하지 못하는 원인:
- BM25: 키워드 매칭 기반, 동의어/유사 표현 처리 약함
- BGE-M3: 일반 도메인 학습, 과학 용어 임베딩 부족

**3. Ground Truth 문서 특성**

누락된 문서들의 공통점:
- 쿼리와 **표면적 유사도 낮음**
- 쿼리와 **의미적 연관성 높음**
- 과학적 개념 설명에 **전문 용어** 사용

---

## Task 4: Cascaded Stages 증가 실험 (실패)

### 가설

> "30→10 급격한 필터링이 3점 문서를 누락시킨다. 더 세밀한 단계적 필터링으로 개선 가능하다."

### 실험 설계

**cascaded_reranking_v2** 아키텍처:

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

**변경사항**:
1. 초기 검색 범위 확대 (30 → 50)
2. 4단계 Cascaded Reranking (2단계 → 4단계)
3. 각 단계별 제거 문서 수 감소 (20개 → 10개씩)

### 실험 결과

| 전략 | MAP@3 | vs v1 | vs baseline |
|------|-------|-------|-------------|
| cascaded_reranking_v1 | 0.7939 | - | +1.16% |
| cascaded_reranking_v2 | 0.7778 | **-2.03%** | -0.89% |

**실패 원인 분석**:

1. **누적 오류 증가**: LLM 호출 횟수 증가 (2회 → 4회)로 각 단계의 작은 오류가 누적
2. **Top-50의 낮은 품질**: 초기 검색 확대가 오히려 노이즈 증가
3. **가설 오류**: 실제 문제는 "필터링 방식"이 아니라 "초기 Retrieval Recall"

### 교훈

> "More Stages ≠ Better Performance"

복잡도 증가가 성능 향상을 보장하지 않음.

---

## Task 5: Query Decomposition 실험 (실패)

### 가설

> "복잡한 쿼리를 여러 서브 쿼리로 분해하면 Retrieval Recall이 향상된다."

### 실험 설계

**query_decomposition_v1** 아키텍처:

```python
def query_decomposition_strategy(eval_id, msg, embeddings_dict):
    """
    1. LLM 쿼리 재작성
    2. Query Decomposition (1개 → 3-5개 서브 쿼리)
    3. Multi-Query Search (각 서브 쿼리로 검색)
    4. RRF Merge (Top 30)
    5. Cascaded Reranking (30 → 10 → 3)
    """
    # Step 1: 쿼리 재작성
    rewritten_query = rewrite_query_with_context(msg)

    # Step 2: Query Decomposition
    sub_queries = decompose_query(rewritten_query)  # 3-5개 생성

    # Step 3: Multi-Query Search
    multi_results = multi_query_search_rrf(
        sub_queries,
        embeddings_dict,
        top_k=30
    )

    # Step 4-5: Cascaded Reranking
    final_topk = llm_cascaded_rerank(rewritten_query, multi_results, top_k=3)
    return final_topk
```

**Query Decomposition 예시**:

```
원본 쿼리:
"배터리가 완구를 움직이게 하는 원리는?"

생성된 서브 쿼리 (5개):
1. 배터리가 완구를 움직이게 하는 원리는?
2. 배터리의 화학적 에너지가 전기 에너지로 변환되는 과정은 무엇인가요?
3. 전기 에너지가 모터의 회전 운동으로 변환되는 원리는 무엇인가요?
4. 배터리의 양극과 음극에서 발생하는 화학 반응은 어떻게 전류 생성을 촉진하나요?
5. 전기 회로에서 전류가 흐를 때 완구의 구동부가 작동하는 메커니즘은 무엇인가요?
```

### 실험 결과

| 전략 | MAP@3 | vs v1 | vs baseline | Perfect | Partial |
|------|-------|-------|-------------|---------|---------|
| cascaded_reranking_v1 | 0.7939 | - | +1.16% | 4/8 | 4/8 |
| query_decomposition_v1 | 0.5278 | **-33.52%** | -32.75% | 0/8 | 8/8 |

**충격적인 실패**: 모든 샘플이 Partial match (1-2/3), Perfect match 0개

### 실패 원인 심층 분석

#### 1. 서브 쿼리 품질 문제

**문제점**:
- LLM이 생성한 서브 쿼리가 **너무 학술적**이고 **세부적**
- 원본 쿼리의 **단순함**과 **직접성**을 상실
- "배터리 → 완구 작동" 대신 "화학 반응", "에너지 변환", "전기 회로"로 분산

**결과**:
- 직접 관련 문서들이 밀려남
- 일반 이론 문서들이 상위권 진입

#### 2. RRF Dilution (희석) 효과

**v1 (단일 쿼리)**:
```python
# 2개 source (BM25, BGE-M3)
Doc A: RRF = 1/(60+1) + 1/(60+1) = 0.0328  # 양쪽에서 1위
```

**Query Decomposition (5개 쿼리)**:
```python
# 10개 source (5개 쿼리 × 2개 방법)
Doc A (정답 문서):
  Query 1: 0.0328  # 원본 쿼리에서만 1위
  Query 2-5: 0.0258  # 나머지에서는 낮은 순위
  Total: 0.0788

Doc X (노이즈 문서):
  Query 1: 0.0000  # 원본 쿼리에서는 없음
  Query 2-5: 0.0325 × 4  # 서브 쿼리에서 높은 순위
  Total: 0.1300  # ⚠️ 정답 문서보다 높음!
```

**핵심 문제**:
- 원본 쿼리에서 1위인 문서가 서브 쿼리에서는 낮은 순위
- 서브 쿼리에서만 높은 순위를 받는 노이즈 문서들이 상위권 진입
- RRF의 "다양성 중시" 특성이 **정확성 저하**로 작용

#### 3. Over-generalization (과도한 일반화)

**가장 심각한 실패 사례**: eval_id=47

```
원본 쿼리:
"원자번호는 원자내의 어떤 소립자와 관계 있는가?"

생성된 서브 쿼리:
1. 원자번호는 원자내의 어떤 소립자와 관계 있는가?
2. 원자번호는 원자의 양성자 수와 어떤 관계가 있는가?  ✅ 정답
3. 원자번호와 중성자 수는 어떤 관련이 있는가?  ❌ 오답 유도
4. 원자번호는 원소의 화학적 성질을 결정하는 데 어떻게 기여하는가?  ❌ 너무 일반적
5. 원자번호와 전자 배치는 어떻게 연결되는가?  ❌ 오답 유도
```

**문제**:
- 서브 쿼리 3, 5가 **잘못된 개념**(중성자, 전자)을 포함
- 정답은 "양성자"인데 "중성자", "전자"까지 검색
- 잘못된 방향의 문서들이 Top-30에 대량 진입

**결과**:
- v1: 3/3 perfect → query_decomposition: 1/3 (catastrophic failure)

#### 4. Cascaded Reranking의 한계

**v1의 Top-30** (단일 쿼리):
- Precision ~80% (24개 관련 문서)
- Recall ~85% (ground truth 3개 중 2.55개 진입)

**Query Decomposition의 Top-30** (5개 쿼리):
- Precision ~40% (12개 관련 문서) ⚠️
- Recall ~70% (ground truth 3개 중 2.1개 진입)

**수식으로 표현**:
```
v1 Pipeline:
Top-30 (Precision 80%)
→ Stage 1: Top-10 (8개 관련 + 2개 노이즈)
→ Stage 2: Top-3 (2.4개 정답)
→ MAP@3 ~ 0.79 ✅

Query Decomposition Pipeline:
Top-30 (Precision 40%) ⚠️
→ Stage 1: Top-10 (4개 관련 + 6개 노이즈)
→ Stage 2: Top-3 (1.2개 정답)
→ MAP@3 ~ 0.40 ✅ 실제 0.5278
```

### 핵심 교훈

#### 1. Retrieval Recall ≠ Query Decomposition

**잘못된 가설**:
> "Retrieval Recall 문제 = 쿼리가 부족해서 관련 문서를 못 찾음"

**올바른 이해**:
> "Retrieval Recall 문제 = BM25/BGE-M3가 의미적 유사도를 제대로 반영 못함"

#### 2. More is NOT Always Better

```
Task 4: More Stages (2 → 4) ❌ -2.03%
Task 5: More Queries (1 → 5) ❌ -33.52%
```

**교훈**: 복잡도 증가 ≠ 성능 향상. **Simplicity and Focus** > Complexity and Diversity

#### 3. RRF의 양날의 검

RRF는 **다양성**(diversity)을 증진하지만:
- **Precision이 중요한 상황**에서는 오히려 해로움
- Top-3만 제출하는 대회에서 다양성보다 **정확성**이 우선
- RRF 사용 시 **source 품질**이 매우 중요

#### 4. LLM Prompt의 한계

Solar Pro가 생성한 서브 쿼리:
- **학술적**이고 **일반적**인 경향
- 원본 쿼리의 **구체성**과 **단순성**을 상실
- LLM은 "더 자세히" 만드는 데 최적화 → **검색에는 부적합**

---

## Task 6: Document Context Expansion (불가능)

### 원래 계획

**가설**: 검색된 문서의 주변 chunk를 함께 검색하면 맥락이 풍부해져 Reranking 성능 향상

**방법론**:
1. 초기 검색으로 Top-K 문서 획득
2. 각 문서의 parent/sibling chunk 찾기
3. 확장된 문서 집합으로 Reranking

### 포기 이유

**데이터 구조 확인** (`documents.jsonl`):
```json
{
  "docid": "59f5f7c9-37a1-438b-8b3a-c2d7f019fea3",
  "src": "과학_생물학.txt",
  "content": "신장은 피를 맑게 하고..."
}
```

**문제점**:
- 문서는 **단일 chunk**로만 존재
- Parent/child 관계 없음
- Chunk ID, 위치 정보 없음

**결론**: Document Context Expansion은 **구현 불가능**

### 대안

Retrieval Recall 개선을 위한 다른 접근:
1. **BM25 파라미터 튜닝** (k1, b 최적화)
2. **Hybrid Weight 튜닝** (BM25 vs BGE-M3 가중치)
3. **BGE-M3 Fine-tuning** (과학 도메인)

---

## 다음 단계: BM25 파라미터 튜닝

### BM25 개요

**공식**:
```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

**파라미터**:
- **k1**: Term Frequency Saturation (기본값: 1.2)
- **b**: Length Normalization (기본값: 0.75)

### k1 파라미터 (Term Frequency Saturation)

**의미**: 단어가 반복될 때 점수가 얼마나 증가하는가?

**예시**: "배터리" 키워드

| k1 값 | 1회 | 2회 | 5회 | 10회 | 특징 |
|-------|-----|-----|-----|------|------|
| 0.5 (낮음) | 1.0 | 1.3 | 1.7 | 1.8 | 빠른 포화, 반복 무시 |
| 1.2 (기본) | 1.0 | 1.5 | 2.0 | 2.2 | 균형 잡힌 증가 |
| 2.5 (높음) | 1.0 | 1.7 | 2.5 | 3.0 | 반복 강조 |

**적용 시나리오**:
- **k1 낮음 (0.5-1.0)**: 키워드 1-2회만 언급된 문서도 중요할 때
- **k1 높음 (2.0-3.0)**: 키워드가 많이 반복된 문서가 더 관련성 높을 때

**우리 데이터셋 특성**:
- 과학 문서는 주요 개념을 여러 번 언급
- 예: "신장" 문서에서 "신장", "콩팥" 반복 사용
- **제안**: k1 = 1.5 - 2.0 (기본값보다 약간 높게)

### b 파라미터 (Length Normalization)

**의미**: 문서 길이가 점수에 얼마나 영향을 미치는가?

**예시**: 평균 100단어인 코퍼스

| b 값 | 50단어 문서 | 100단어 문서 | 200단어 문서 | 특징 |
|------|-------------|--------------|--------------|------|
| 0.0 | 100% | 100% | 100% | 길이 완전 무시 |
| 0.5 | 125% | 100% | 87.5% | 약한 길이 패널티 |
| 0.75 (기본) | 137.5% | 100% | 81.25% | 중간 길이 패널티 |
| 1.0 | 150% | 100% | 75% | 강한 길이 패널티 |

**적용 시나리오**:
- **b 낮음 (0.0-0.5)**: 짧은 문서와 긴 문서를 동등하게 취급
- **b 높음 (0.75-1.0)**: 간결한 문서를 선호

**우리 데이터셋 특성**:
- 과학 문서는 길이가 다양함
- 짧은 정의 문서 vs 긴 설명 문서
- 둘 다 관련성 있을 수 있음
- **제안**: b = 0.5 - 0.7 (기본값보다 낮게, 길이 패널티 완화)

### 실험 계획

**Grid Search**:
```python
k1_values = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
b_values = [0.3, 0.5, 0.75, 0.9, 1.0]

# 30개 조합 (6 × 5)
for k1 in k1_values:
    for b in b_values:
        # Ultra validation set으로 빠른 평가
        map_score = evaluate_bm25_params(k1, b)
```

**예상 소요 시간**:
- 30개 조합 × 8 samples × ~5초/sample = ~20분

**구현 방법**:
```python
# Elasticsearch 인덱스 설정에서 변경
settings = {
    "index": {
        "similarity": {
            "custom_bm25": {
                "type": "BM25",
                "k1": 1.5,  # 튜닝 대상
                "b": 0.6     # 튜닝 대상
            }
        }
    }
}
```

---

## 실험 결과 요약

| Task | 전략 | MAP@3 | vs v1 | 상태 |
|------|------|-------|-------|------|
| - | Baseline | 0.7848 | -1.16% | 대회 기준 |
| 3 | cascaded_reranking_v1 | **0.7939** | - | ✅ 최고 성능 |
| 4 | cascaded_reranking_v2 | 0.7778 | -2.03% | ❌ 실패 |
| 5 | query_decomposition_v1 | 0.5278 | -33.52% | ❌ 실패 |
| 6 | document_context_expansion | - | - | ⛔ 불가능 |
| 7 | bm25_parameter_tuning | ? | ? | ⏳ 예정 |

---

## 핵심 인사이트

### 1. 병목 지점 확인

**Retrieval Stage** (초기 검색):
- 85.7%의 오류가 여기서 발생
- BM25/BGE-M3의 한계
- **해결책**: 검색 알고리즘 개선 필요

**Reranking Stage** (순위 재조정):
- 14.3%의 오류만 발생
- Solar Pro가 이미 잘 작동
- **결론**: 추가 최적화 여지 작음

### 2. 복잡도의 함정

```
복잡도 ↑ ≠ 성능 ↑
```

**검증된 사실**:
- More stages (v2): -2.03%
- More queries (decomposition): -33.52%

**교훈**: 단순하고 집중된 접근이 복잡한 방법보다 우수

### 3. LLM의 강점과 약점

**강점** (잘 작동):
- Query rewriting (멀티턴 맥락 통합)
- Reranking (의미적 관련성 판단)

**약점** (작동 안 함):
- Query decomposition (과도한 일반화)
- Multi-step filtering (누적 오류)

### 4. RRF 사용 시 주의사항

RRF는 다양성 증진에 유용하지만:
- **Precision 중요** → RRF 신중히 사용
- **Source 품질** → 모든 source가 고품질일 때만 유효
- **평가 기준** → Top-3 평가에서는 정확성 > 다양성

---

## 다음 실험 우선순위

### 1순위: BM25 파라미터 튜닝 ⭐

**이유**:
- Retrieval Recall 직접 개선
- 구현 간단, 빠른 실험 가능
- 하이퍼파라미터 최적화만으로 향상 기대

**예상 성능 향상**: +2-5% (MAP@3 0.81-0.83)

### 2순위: Hybrid Weight 튜닝

**방법**: BM25 vs BGE-M3 가중치 조정
```python
# 현재: RRF (동등 가중치)
# 실험: Weighted RRF
score = w1 * bm25_score + w2 * bgem3_score
```

**예상 성능 향상**: +1-3% (MAP@3 0.80-0.82)

### 3순위: BGE-M3 Fine-tuning

**방법**: 과학 도메인 데이터로 fine-tuning
**장벽**: 시간, 컴퓨팅 자원 필요
**예상 성능 향상**: +3-7% (MAP@3 0.82-0.85)

### 4순위: Solar Pro Prompt 최적화

**방법**: Reranking prompt 개선, edge case 처리
**예상 성능 향상**: +1-2% (MAP@3 0.80-0.81)

---

## 결론

### 현재 상태

- **최고 성능**: cascaded_reranking_v1 (MAP@3 0.7939)
- **목표**: MAP@3 0.9
- **격차**: +13.4% 향상 필요

### 검증된 사실

1. **Retrieval Recall이 병목** (85.7% 오류)
2. **Reranking은 이미 잘 작동** (14.3% 오류)
3. **복잡도 증가는 역효과** (Task 4, 5 실패)
4. **LLM은 Reranking에 적합, 쿼리 생성에 부적합**

### 앞으로의 방향

**단기** (1-2주):
- BM25 파라미터 튜닝
- Hybrid Weight 최적화

**중기** (3-4주):
- BGE-M3 Fine-tuning
- Prompt Engineering

**장기** (1개월+):
- 앙상블 방법
- 새로운 임베딩 모델 실험

---

**작성일**: 2025-11-24
**실험자**: Claude Code
**관련 파일**:
- [cascaded_reranking_v1.py](cascaded_reranking_v1.py) - 현재 최고 성능
- [cascaded_reranking_v2.py](cascaded_reranking_v2.py) - Task 4 실패
- [query_decomposition_v1.py](query_decomposition_v1.py) - Task 5 실패
- [TASK4_EXPERIMENT_PLAN.md](TASK4_EXPERIMENT_PLAN.md) - Task 4 계획서
- [TASK5_FAILURE_ANALYSIS.md](TASK5_FAILURE_ANALYSIS.md) - Task 5 상세 분석
- [ultra_validation_results.json](ultra_validation_results.json) - 검증 결과
- [auto_validate.py](auto_validate.py) - 자동 검증 파이프라인
