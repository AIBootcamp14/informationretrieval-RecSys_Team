# Information Retrieval System - Pipeline Summary

## 📊 시스템 개요

질문 응답 시스템(QA System)을 위한 고급 정보 검색(IR) 파이프라인으로, 하이브리드 검색과 다단계 리랭킹을 통해 높은 정확도의 문서 검색을 수행합니다.

**최종 성능**: 79.09% Top-3 Accuracy

---

## 🏗️ 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                          사용자 질문 입력                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: 쿼리 분류 및 재작성 (LLM Query Rewriting)                    │
│  ├─ LLM 기반 검색 가능성 분류 (SEARCHABLE/NOT_SEARCHABLE)              │
│  ├─ 대화 맥락 반영 standalone 쿼리 생성                                │
│  └─ 1~3개의 쿼리 변형 생성                                            │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: 쿼리 확장 (Query Expansion)                                 │
│  ├─ 동의어 기반 확장 (Additive Expansion)                             │
│  ├─ 복합명사 분해 (Compound Noun Splitting)                          │
│  └─ 다중 쿼리 생성 및 중복 제거                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: 하이브리드 검색 (Hybrid Retrieval)                           │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │  BM25 검색       │    │  Dense 검색      │                       │
│  │  (키워드 매칭)   │    │  (의미 유사도)   │                       │
│  │  Top-50 청크     │    │  Top-50 청크     │                       │
│  └────────┬─────────┘    └─────────┬────────┘                       │
│           └──────────┬──────────────┘                                │
│                      ▼                                               │
│            청크 풀 생성 (Pooling)                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Evidence Gating (증거 기반 필터링)                           │
│  ├─ 토큰 오버랩 검사 (최소 1~2개)                                      │
│  ├─ BM25 점수 기준 (최고 점수 대비 40% 이상)                           │
│  ├─ Dense 점수 기준 (최고 점수 대비 0.03 이내)                         │
│  └─ 도메인 특화 중의성 해소 (예: 버스 - 통학/하드웨어)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: 청크 수준 리랭킹 (Cross-Encoder Reranking)                   │
│  ├─ BGE-reranker-v2-m3 모델 사용                                     │
│  ├─ 쿼리-청크 쌍의 정확한 관련성 점수 계산                              │
│  └─ [0, 1] 범위의 신뢰도 점수 생성                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: 문서 수준 집계 (Document Aggregation)                       │
│  ├─ MAX Aggregation (최고 청크 점수 → 문서 점수)                      │
│  ├─ Secondary Score (상위 2개 청크 평균)                             │
│  └─ BM25/Dense 최대값 유지                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: Margin 기반 필터링 (Margin-based Filtering)                 │
│  ├─ 1위: 항상 유지                                                   │
│  ├─ 2위: doc_score >= 0.12                                          │
│  └─ 3위: doc_score >= 0.15                                          │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 8: [선택적] LLM 리랭킹 (2-Stage Cascaded Reranking)            │
│  ├─ Stage 1: 30개 → 10개 (Coarse Filtering)                         │
│  └─ Stage 2: 10개 → 3개 (Fine-grained Ranking)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    최종 Top-K 문서 반환                               │
│                    (docid + 상세 메타데이터)                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 기술 스택

### 1. 언어 모델 (LLMs)

| 구성 요소 | 모델 | 용도 |
|----------|------|------|
| Query Rewriting | GPT-4o-mini | 쿼리 분류 및 다중 변형 생성 |
| Answer Generation | GPT-4o-mini | 검색된 문서 기반 답변 생성 |
| LLM Reranking | GPT-4o-mini | 2단계 계층적 문서 재순위 결정 |

**특징**:
- Temperature 0.0으로 결정론적 출력
- JSON 모드로 구조화된 응답 보장
- 대화 맥락(최대 3턴) 반영

### 2. 임베딩 및 검색 모델

| 구성 요소 | 모델/기술 | 용도 |
|----------|----------|------|
| Dense Retrieval | BAAI/bge-m3 | 의미 기반 청크 검색 |
| Sparse Retrieval | Whoosh BM25 | 키워드 기반 청크 검색 |
| Tokenization | BGE-M3 Tokenizer | BM25와 Dense 검색의 일관된 토크나이징 |
| Cross-Encoder Reranking | BAAI/bge-reranker-v2-m3 | 청크-쿼리 쌍 정밀 점수화 |

**특징**:
- Multilingual 지원 (한국어 최적화)
- BM25 Field Boosting: title(1.3x), section(1.15x), content(1.0x)
- Character N-gram 필터로 띄어쓰기 변형 처리

### 3. 프레임워크 및 라이브러리

```python
# 핵심 라이브러리
├── transformers      # HuggingFace 모델 로딩
├── torch            # 딥러닝 추론
├── whoosh           # BM25 인덱싱/검색
├── numpy            # 벡터 연산
├── openai           # GPT API 호출
├── diskcache        # 검색 결과 캐싱
└── loguru           # 구조화된 로깅
```

### 4. 인프라

| 항목 | 설정 |
|-----|------|
| 임베딩 저장 | In-Memory NumPy Array |
| BM25 인덱스 | RamStorage (Whoosh) |
| 캐싱 | DiskCache (TTL: 3600s) |
| GPU 지원 | CUDA (FP16 최적화) |
| 배치 처리 | Cross-Encoder 32개/배치 |

---

## 🔬 핵심 기법 상세

### 1. LLM Query Rewriting

**목적**: 대화형 질문을 검색에 최적화된 standalone 쿼리로 변환

**프로세스**:
```
사용자: "그거 키우는 방법은?"
맥락: 이전 질문 "토마토에 대해 알려줘"

→ LLM 분류: SEARCHABLE
→ LLM 재작성:
   1. "토마토 재배 방법"
   2. "토마토 키우기"
   3. "토마토 재배 노하우"
```

**구현**:
- System prompt로 검색 목적 명시
- Few-shot examples로 품질 향상
- JSON 스키마 강제로 파싱 안정성 확보

### 2. Additive Query Expansion

**목적**: 원본 쿼리를 훼손하지 않고 검색 범위 확대

**전략**:
```python
# 동의어 맵
{
    "키우": ["재배", "재배법", "기르기"],
    "재배": ["재배법", "재배 방법"],
    "노하우": ["방법", "요령", "팁"],
    "병해충": ["병충해", "해충", "질병 관리"]
}

# 복합명사 분해 (도메인 어휘 기반)
"병해충관리" → "병해충 관리"
"평형분극비율" → "평형 분극 비율"
```

**결과**:
- 원본: "토마토 재배"
- 확장: "토마토 재배", "토마토 재배 재배법", "토마토 재배 방법"

### 3. Hybrid Retrieval

**BM25 (Sparse)**:
- 정확한 키워드 매칭 (전문 용어, 고유명사)
- Okapi BM25 알고리즘
- 필드별 가중치 차등 적용

**BGE-M3 (Dense)**:
- 의미적 유사성 측정
- 768차원 임베딩 벡터
- 코사인 유사도 계산

**병합 전략**:
- Sparse 50개 + Dense 50개 → Pool
- 중복 제거 후 메타데이터 병합
- 각 청크에 BM25/Dense/Overlap 점수 보존

### 4. Evidence Gating

**목적**: 신뢰도 낮은 후보 조기 제거

**다중 신호 검증**:
```python
def passes_evidence_gate(candidate):
    # 1. 토큰 오버랩
    if searchable:
        min_overlap = 1
    else:
        min_overlap = 2

    # 2. BM25 상대 점수
    if candidate.bm25 < 0.4 * max_bm25:
        return False

    # 3. Dense 상대 점수
    if candidate.dense < max_dense - 0.03:
        return False

    # 4. 도메인 중의성 해소
    if is_bus_ambiguous(query, candidate):
        return False

    return True
```

**버스 중의성 해소 예시**:
```python
# 쿼리: "학생 통학 버스"
school_cues = {"통학", "등교", "학교", "학생"}
hw_cues = {"pci", "데이터", "전송", "주소", "메모리"}

# 문서에 "PCI 버스" 포함 → 필터링
# 문서에 "통학버스" 포함 → 통과
```

### 5. Cross-Encoder Reranking

**Bi-Encoder vs Cross-Encoder**:

| 구분 | Bi-Encoder (BGE-M3) | Cross-Encoder (BGE-reranker) |
|-----|---------------------|------------------------------|
| 방식 | 쿼리/문서 별도 인코딩 | 쿼리+문서 동시 인코딩 |
| 속도 | 빠름 (사전 계산 가능) | 느림 (실시간 계산) |
| 정확도 | 중간 | 높음 |
| 용도 | 1차 검색 (많은 후보) | 2차 리랭킹 (적은 후보) |

**Cross-Encoder 프로세스**:
```python
# 입력
query = "RAM의 역할은?"
chunk = "RAM은 컴퓨터의 주기억장치로..."

# 토크나이징
tokens = tokenizer(
    query, chunk,
    max_length=512,
    truncation=True
)

# 모델 추론
logits = model(tokens)
score = sigmoid(logits)  # [0, 1] 범위
# → 0.92 (높은 관련성)
```

### 6. MAX Aggregation

**문제**: 청크 점수를 어떻게 문서 점수로?

**해결책**: 최고 청크가 문서를 대표

```python
# 예시: doc1의 청크 3개
chunks = [
    {"chunk_id": "doc1_0", "rerank_score": 0.92},
    {"chunk_id": "doc1_1", "rerank_score": 0.35},
    {"chunk_id": "doc1_2", "rerank_score": 0.28}
]

# 집계
doc_score = max([0.92, 0.35, 0.28]) = 0.92  # Primary
doc_score2 = mean([0.92, 0.35]) = 0.635     # Secondary (타이브레이커)
```

**정렬 우선순위**:
1. doc_score (MAX rerank) ↓
2. doc_score2 (TOP-2 MEAN) ↓
3. bm25_max ↓
4. dense_max ↓

### 7. Margin-based Filtering

**기존 방식의 문제**:
```python
# 절대 점수 기준
if doc_score >= 0.8:
    keep_document()

# 문제: 모든 문서가 0.81, 0.80, 0.79일 때?
# → 1개 문서만 반환 (과소 검색)
```

**개선된 방식**:
```python
# 상대 점수 기준
results = []
results.append(docs[0])  # 1위 항상 유지

if docs[1].score >= 0.12:  # 절대값
    results.append(docs[1])

if docs[2].score >= 0.15:  # 절대값
    results.append(docs[2])
```

**장점**:
- 점수 분포와 무관하게 작동
- 낮은 임계값으로 재현율 향상
- 명확한 1위가 있으면 1개만 반환 (정밀도 유지)

### 8. 2-Stage Cascaded Reranking

**Stage 1 - Coarse Filtering** (30 → 10):
```python
system_prompt = """
다음 문서 중 질문과 관련된 문서를 선택하세요.
관대하게 선택하되, 명백히 무관한 것은 제외하세요.

출력: {"indices": [0, 2, 5, 7, 9]}
"""

# 문서는 200자 미리보기만 사용 (속도 향상)
```

**Stage 2 - Fine-grained Ranking** (10 → 3):
```python
system_prompt = """
문서를 관련성 순으로 정밀하게 순위를 매기세요.

기준:
1. 질문에 대한 직접적 답변
2. 정보의 완전성
3. 과학적 정확성
4. 명확성과 상세함

출력: {
  "rankings": [
    {"index": 2, "score": 0.95, "reason": "..."},
    {"index": 0, "score": 0.80, "reason": "..."}
  ]
}
"""

# 문서는 500자 사용 (정밀도 향상)
```

---

## 📈 성능 최적화 전략

### 1. 캐싱 전략

```python
# 캐시 키 생성
cache_key = md5(query + messages + eval_id)

# 히트율 향상
- 동일 질문 즉시 응답
- TTL 3600초 (1시간)
- DiskCache로 영구 보존
```

### 2. 배치 처리

```python
# Cross-Encoder
- Batch size: 32
- GPU 메모리 효율화
- FP16 사용 (속도 2배, 메모리 50%)
```

### 3. 인덱스 최적화

```python
# BM25
- RamStorage (메모리 인덱스)
- limitmb=512MB (대용량 처리)

# Dense
- NumPy array (빠른 코사인 유사도)
- 사전 정규화로 dot product 사용
```

---

## 🎯 주요 설정값

### 검색 파라미터

```python
# Hybrid Retrieval
CHUNK_N_SPARSE = 50      # BM25 top-k
CHUNK_N_DENSE = 50       # Dense top-k
CHUNK_N_POOL = 100       # 병합 후 풀 크기

# Evidence Gating
BM25_THRESHOLD = 0.4     # 최고 대비 40%
DENSE_THRESHOLD = 0.03   # 최고 대비 -0.03

# Margin Filtering
CHUNK_MIN_SCORE_2ND = 0.12
CHUNK_MIN_SCORE_3RD = 0.15

# Final Output
FINAL_TOP_K = 3          # 최종 반환 문서 수
```

### 모델 파라미터

```python
# Cross-Encoder
CHUNK_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CHUNK_RERANKER_MAX_LENGTH = 512
CHUNK_RERANKER_BATCH_SIZE = 32
CHUNK_RERANKER_USE_FP16 = True

# Embedder
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"  # or "cpu"
```

---

## 📊 데이터 흐름 예시

### 입력 질문
```
"토마토 키우는 방법 알려줘"
```

### STEP 1: Query Rewriting
```json
{
  "original": "토마토 키우는 방법",
  "rewrites": [
    "토마토 재배 방법",
    "토마토 기르기",
    "토마토 재배 노하우"
  ],
  "searchable": true
}
```

### STEP 2: Query Expansion
```python
[
  "토마토 키우는 방법",
  "토마토 키우는 방법 재배",
  "토마토 키우는 방법 재배법",
  "토마토 재배 방법",
  "토마토 재배 방법 재배법",
  "토마토 기르기",
  "토마토 재배 노하우",
  "토마토 재배 노하우 방법"
]
```

### STEP 3: Hybrid Retrieval
```python
# BM25 결과 (Top-5)
[
  {"chunk_id": "doc42_2", "bm25": 8.5, "text": "토마토 재배 시 물주기..."},
  {"chunk_id": "doc15_0", "bm25": 7.2, "text": "방울토마토 키우기..."},
  {"chunk_id": "doc88_1", "bm25": 6.8, "text": "토마토 병해충 관리..."},
  ...
]

# Dense 결과 (Top-5)
[
  {"chunk_id": "doc42_2", "dense": 0.89, "text": "토마토 재배 시 물주기..."},
  {"chunk_id": "doc23_3", "dense": 0.85, "text": "가지과 식물 재배법..."},
  {"chunk_id": "doc15_0", "dense": 0.82, "text": "방울토마토 키우기..."},
  ...
]
```

### STEP 4: Evidence Gating
```python
# 필터링 전: 75개 청크
# 필터링 후: 42개 청크
[
  {"chunk_id": "doc42_2", "bm25": 8.5, "dense": 0.89, "overlap": 3},
  {"chunk_id": "doc15_0", "bm25": 7.2, "dense": 0.82, "overlap": 2},
  ...
]
```

### STEP 5: Cross-Encoder Reranking
```python
[
  {"chunk_id": "doc42_2", "rerank_score": 0.94},
  {"chunk_id": "doc15_0", "rerank_score": 0.88},
  {"chunk_id": "doc42_3", "rerank_score": 0.76},
  {"chunk_id": "doc88_1", "rerank_score": 0.65},
  ...
]
```

### STEP 6: Document Aggregation
```python
[
  {
    "docid": "doc42",
    "doc_score": 0.94,      # MAX(0.94, 0.76) from chunks 2,3
    "doc_score2": 0.85,     # MEAN(0.94, 0.76)
    "chunk_count": 2,
    "best_chunk_id": "doc42_2"
  },
  {
    "docid": "doc15",
    "doc_score": 0.88,
    "doc_score2": 0.88,
    "chunk_count": 1,
    "best_chunk_id": "doc15_0"
  },
  {
    "docid": "doc88",
    "doc_score": 0.65,
    "doc_score2": 0.65,
    "chunk_count": 1,
    "best_chunk_id": "doc88_1"
  }
]
```

### STEP 7: Margin Filtering
```python
# doc42: 0.94 → 유지 (1위)
# doc15: 0.88 → 유지 (0.88 >= 0.12)
# doc88: 0.65 → 유지 (0.65 >= 0.15)

filtered = ["doc42", "doc15", "doc88"]
```

### 최종 출력
```json
{
  "standalone_query": "토마토 키우는 방법",
  "topk": ["doc42", "doc15", "doc88"],
  "references": [
    {
      "docid": "doc42",
      "content": "토마토 재배 시 물주기는...",
      "score": 0.94,
      "bm25": 8.5,
      "dense": 0.89
    },
    {
      "docid": "doc15",
      "content": "방울토마토 키우기는...",
      "score": 0.88,
      "bm25": 7.2,
      "dense": 0.82
    },
    {
      "docid": "doc88",
      "content": "토마토 병해충 관리는...",
      "score": 0.65,
      "bm25": 6.8,
      "dense": 0.71
    }
  ],
  "answer": "토마토를 키우는 방법은 다음과 같습니다..."
}
```

---

## 🔍 주요 디자인 결정

### 1. 왜 청크 수준 검색인가?

**문제**: 긴 문서는 여러 주제를 포함
```
문서 "과학백과사전_식물편"
├─ 청크 0: 토마토 재배 ✓
├─ 청크 1: 감자 병해충
├─ 청크 2: 고추 수확
└─ 청크 3: 오이 저장
```

**해결**: 청크 단위로 검색 후 MAX 집계
- 관련 청크만 높은 점수
- 문서 전체가 부정확하게 높은 점수 받지 않음

### 2. 왜 MAX Aggregation인가?

**대안**:
- MEAN: 모든 청크 평균 → 무관한 청크가 점수 하락
- SUM: 청크 많은 문서 유리 → 길이 편향
- MAX: 최고 청크만 → 질 높은 청크 1개면 충분

**선택 이유**:
- "하나라도 관련성 높은 청크가 있으면 좋은 문서"
- 문서 길이에 불변

### 3. 왜 Margin Filtering인가?

**절대 점수의 문제**:
```python
# 점수 분포가 높을 때
docs = [0.95, 0.94, 0.93]
threshold = 0.8
→ 3개 모두 반환 (과다 검색)

# 점수 분포가 낮을 때
docs = [0.50, 0.20, 0.10]
threshold = 0.8
→ 0개 반환 (과소 검색)
```

**상대 점수의 해결**:
```python
# 명확한 1위
docs = [0.95, 0.20, 0.10]
→ 1개 반환 (0.20 < 0.12 필터링)

# 근소한 차이
docs = [0.50, 0.45, 0.40]
→ 3개 모두 반환
```

### 4. 왜 2-Stage LLM Reranking인가?

**비용 최적화**:
```
1-Stage: 30개 문서 × 500토큰 = 15,000 토큰
2-Stage:
  - Stage1: 30개 × 200토큰 = 6,000 토큰
  - Stage2: 10개 × 500토큰 = 5,000 토큰
  - 총: 11,000 토큰 (27% 절감)
```

**정확도 유지**:
- Stage 1이 명백히 무관한 것만 제거
- Stage 2가 정밀 순위 결정

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정 (.env)
OPENAI_API_KEY=sk-...

# 3. GPU 확인 (선택)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 데이터 준비

```bash
# data/ 폴더 구조
data/
├── documents.jsonl    # 문서 컬렉션
└── eval.jsonl        # 평가 질문
```

### 3. 실행

```bash
# 전체 파이프라인 실행
python main.py

# 출력: submission/submission.csv
```

---

## 📝 로그 예시

```
2025-11-27 10:15:23 | INFO     | Initializing RAG pipeline...
2025-11-27 10:15:25 | INFO     | Building BM25 index for 15234 chunks...
2025-11-27 10:15:30 | INFO     | Indexing complete: 15234 chunks indexed
2025-11-27 10:15:31 | INFO     | ============================================================
2025-11-27 10:15:31 | INFO     | Eval ID: 1
2025-11-27 10:15:31 | INFO     | Messages: [{'role': 'user', 'content': '토마토 키우는 방법'}]
2025-11-27 10:15:32 | INFO     | Query classified as: SEARCHABLE (factual question)
2025-11-27 10:15:32 | INFO     | Final queries to search: ['토마토 키우는 방법', '토마토 재배 방법', ...]
2025-11-27 10:15:33 | INFO     | [CHUNK-RERANK] Search queries: ['토마토 키우는 방법', ...]
2025-11-27 10:15:33 | INFO     | [CHUNK-RERANK] Retrieved 42 chunks
2025-11-27 10:15:35 | INFO     | [CHUNK-RERANK] Aggregated to 15 documents
2025-11-27 10:15:35 | INFO     | [CHUNK-RERANK] After margin filtering: 3 documents
2025-11-27 10:15:35 | INFO     | Top-K: ['doc42', 'doc15', 'doc88']
2025-11-27 10:15:37 | INFO     | Answer generated: 토마토를 키우는 방법은...
```

---

## 🎓 학습 포인트

### 1. Hybrid Search의 중요성
- BM25: 전문 용어, 고유명사 정확 매칭
- Dense: 의미 유사성, 동의어 처리
- 상호 보완으로 재현율 향상

### 2. Multi-Stage Reranking
- 초기: 넓은 후보군 (BM25 + Dense)
- 중간: 증거 기반 필터링
- 최종: 정밀 순위 결정 (Cross-Encoder)

### 3. Query Optimization
- LLM으로 대화 맥락 반영
- Additive expansion으로 원본 보존
- 복합명사 처리로 한국어 특화

### 4. Chunk-Level Processing
- 긴 문서의 관련 부분만 검색
- MAX aggregation으로 문서 점수화
- 길이 편향 제거

### 5. Dynamic Filtering
- 절대 점수보다 상대 점수
- 점수 분포에 적응적 반응
- 과소/과다 검색 방지

---

## 🔮 향후 개선 방향

### 1. Query Expansion 고도화
- 도메인 특화 동의어 사전 확장
- Named Entity Recognition 통합
- 문법 기반 변형 생성

### 2. Reranking 최적화
- Larger cross-encoder 모델 테스트
- Ensemble reranking (여러 모델 조합)
- Query-specific threshold 학습

### 3. 성능 최적화
- FAISS 인덱스로 Dense 검색 가속
- Quantization (INT8) 적용
- 분산 처리 (Ray, Dask)

### 4. 평가 고도화
- Recall@K, MRR, NDCG 추가 측정
- Error analysis 자동화
- A/B 테스트 프레임워크

---

## 📚 참고 문헌

1. **BGE-M3**: BAAI General Embedding, Multilingual, Multi-Granularity
2. **BM25**: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
3. **Cross-Encoder**: Reimers & Gurevych, "Sentence-BERT"
4. **Query Rewriting**: Nogueira et al., "Document Expansion by Query Prediction"

---

## 👥 기여자

- 시스템 설계: IR Team
- 구현: RAG Pipeline v2.0
- 평가: Top-3 Accuracy 79.09%

---

## 📄 라이선스

MIT License
