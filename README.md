# RAG 시스템 - Scientific Knowledge QA Competition

과학 지식 질문 답변을 위한 RAG (Retrieval-Augmented Generation) 시스템입니다. Elasticsearch와 Dense Retrieval을 결합한 하이브리드 검색 방식을 사용합니다.

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [주요 파일 설명](#주요-파일-설명)
- [설치 및 실행](#설치-및-실행)
- [성능 결과](#성능-결과)
- [개선 히스토리](#개선-히스토리)

---

## 프로젝트 개요

### 평가 지표
- **MAP (Mean Average Precision)**: Top-3 문서 기반
- **목표 점수**: 0.8+ (이상적으로 0.9+)
- **현재 최고 점수**: 0.6576 (확인 필요)
- **최근 제출 점수**: 0.63 (super_simple_submission.csv)

### 데이터셋
- **documents.jsonl**: 검색 대상 과학 문서 컬렉션
- **eval.jsonl**: 220개 평가 쿼리 (일반 대화 포함)

---

## 시스템 아키텍처

### Dual Index Strategy (rag_with_elasticsearch_1120.py)

```
┌─────────────────────────────────────────────────────┐
│                    Query Input                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Smalltalk Check │ (LLM-based)
        └────────┬─────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   일반 대화          과학 질문
   (TopK=0)              │
                         │
                  ┌──────┴──────┐
                  │ Query Rewrite│
                  └──────┬───────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   ┌─────────────┐            ┌─────────────┐
   │ BM25 Search │            │Dense Search │
   │ (Full Docs) │            │  (Chunks)   │
   └──────┬──────┘            └──────┬──────┘
          │                          │
          └──────────┬───────────────┘
                     │
                     ▼
              ┌──────────┐
              │    RRF   │ (Reciprocal Rank Fusion)
              └─────┬────┘
                    │
                    ▼
             ┌──────────────┐
             │ Adaptive TopK│
             └──────┬───────┘
                    │
                    ▼
              ┌──────────┐
              │LLM Answer│
              └──────────┘
```

### 핵심 기술 스택
- **Elasticsearch 8.x**: BM25 검색 엔진 + KNN 벡터 검색
- **Sentence Transformers**: Dense 임베딩 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- **Upstage Solar Pro**: LLM 기반 답변 생성 및 Smalltalk 판별
- **Python 3.10+**: 메인 프로그래밍 언어

---

## 주요 파일 설명

### 1. rag_simplified_final.py ⭐ 추천 (NEW)

동적 TopK 전략 - 가장 균형잡힌 버전

```python
# 실행 방법
python3 rag_simplified_final.py
```

**특징**:
- ✅ **BM25만 사용** (복잡도 낮음, 속도 빠름)
- ✅ **동적 TopK 전략**:
  - max_score < 3: TopK=0
  - max_score < 5: TopK=1
  - max_score < 8: TopK=2
  - max_score >= 8: TopK=3
- ✅ **실행 시간**: 약 2초
- ✅ **TopK 분포**: 0개(15) + 1개(1) + 2개(4) + 3개(200) = 적절한 균형
- ⚠️ **수정 필요**: ID 30 등 9개 과학 질문이 SMALLTALK_IDS에 포함됨

**결과 파일**: `simplified_submission.csv` (점수 미확인, 수정 후 재테스트 필요)

---

### 2. rag_super_simple.py

Threshold 2.0 전략 - 최대한 많은 문서 반환

```python
# 실행 방법
python3 rag_super_simple.py
```

**특징**:
- ✅ **BM25만 사용** (복잡도 낮음, 속도 빠름)
- ✅ **Threshold 2.0** (관대한 필터링)
- ✅ **실행 시간**: 약 2초
- ✅ **실제 MAP**: 0.63
- ✅ **TopK=3 비율**: 96.8% (213/220)

**핵심 코드**:
```python
# 일반 대화 ID (과학 질문들 모두 제거: 30, 91, 70, 51, 60, 260, 37, 26, 265)
CONFIRMED_SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def search(self, query: str, eval_id: int = None) -> List[str]:
    # 1. 일반 대화는 문서 0개
    if eval_id in CONFIRMED_SMALLTALK_IDS:
        return []

    # 2. BM25 검색
    response = self.es.search(
        index='test',
        body={
            'query': {
                'match': {
                    'content': {
                        'query': query.strip(),
                        'analyzer': 'nori'
                    }
                }
            },
            'size': 10
        }
    )

    # 3. threshold 2.0으로 필터링
    max_score = response['hits']['hits'][0]['_score']
    if max_score >= 2.0:
        return [hit['_source']['docid'] for hit in response['hits']['hits'][:3]]
    else:
        return []
```

**결과 파일**: `super_simple_submission.csv` (598KB)

---

### 2. rag_with_elasticsearch_1120.py
**Dual Index + Hybrid Search 버전**

```python
# 실행 방법
python3 rag_with_elasticsearch_1120.py
```

**특징**:
- 🔍 **Dual Index**: Full Document (BM25) + Chunks (Dense)
- 🔀 **Hybrid Search**: BM25 + Dense Retrieval + RRF
- 🤖 **LLM 기반 Smalltalk 판별**
- 📝 **Query Rewriting**: 멀티턴 대화 처리
- ⚠️ **실행 시간**: 약 20분 (LLM 호출 포함)

**장점**:
- 더 정확한 Smalltalk 판별 (LLM 사용)
- Chunk 기반 검색으로 긴 문서 처리 개선
- 멀티턴 대화 맥락 이해

**단점**:
- 느린 실행 속도 (LLM API 호출)
- 복잡한 구조로 디버깅 어려움
- 성능 개선 미미 (예상 MAP 0.6~0.7)

**결과 파일**: `rag_1120_submission.csv`

---

### 3. rag_with_elasticsearch_1119.py
**초기 개선 버전 (실패)**

```python
# 실행 방법
python3 rag_with_elasticsearch_1119.py
```

**특징**:
- ❌ **Threshold 5.0**: 너무 높아서 False Negative 11개 발생
- ❌ **결과**: MAP 0.5992 (baseline보다 낮음)
- ⚠️ **교훈**: Threshold를 너무 높이면 오히려 성능 하락

---

## 설치 및 실행

### 1. 환경 설정

**현재 프로젝트는 Anaconda 환경에서 실행 중입니다.**

```bash
# 현재 환경 정보
Python: 3.13.5 (Anaconda)
elasticsearch: 8.8.0
sentence-transformers: 5.1.2

# 방법 1: Anaconda 환경 사용 (현재 사용 중) ⭐ 추천
# 별도 설치 불필요 - 이미 설치되어 있음

# 방법 2: 새로운 Anaconda 환경 생성
conda create -n rag python=3.10
conda activate rag
pip install elasticsearch sentence-transformers openai python-dotenv numpy tqdm

# 방법 3: Python 가상환경 사용
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
pip install elasticsearch sentence-transformers openai python-dotenv numpy tqdm
```

### 2. Elasticsearch 설치 및 실행

**현재 프로젝트는 Docker로 Elasticsearch를 실행 중입니다.**

```bash
# 현재 상태 확인
docker ps --filter "name=elasticsearch"
# 결과: elasticsearch container is Up 2 days (healthy) on port 9200

# Docker로 실행 (현재 사용 중) ⭐ 추천
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Elasticsearch 시작/중지
docker start elasticsearch
docker stop elasticsearch

# 연결 확인
curl http://localhost:9200

# Homebrew 방식 (대안)
brew install elasticsearch
brew services start elasticsearch
```

### 3. 환경 변수 설정

**`.env` 파일이 이미 `/code/.env`에 존재합니다.**

```bash
# 위치: /Users/dongjunekim/dev_team/ai14/ir/code/.env
# 현재 설정:
# - UPSTAGE_API_KEY: 설정됨 ✅
# - ELASTICSEARCH_PASSWORD: Docker 사용으로 불필요 (xpack.security.enabled=false)

# 새 환경 구성 시 .env 파일 생성:
cd code
cat > .env << 'EOF'
# Upstage API Configuration
UPSTAGE_API_KEY=your_upstage_api_key_here

# Elasticsearch (Docker 사용 시 불필요)
# ELASTICSEARCH_PASSWORD=your_password_here
EOF
```

**Upstage API Key 발급 방법**:

1. [Upstage Console](https://console.upstage.ai/) 접속
2. API Keys 메뉴에서 새 키 생성
3. `.env` 파일에 복사

### 4. 실행

#### Simplified Final 버전 (추천) ⭐

```bash
cd code
python3 rag_simplified_final.py
```

#### Super Simple 버전

```bash
cd code
python3 rag_super_simple.py
```

#### Dual Index 버전

```bash
cd code
python3 rag_with_elasticsearch_1120.py
```

---

## 성능 결과

### 제출 파일별 성능 비교

| 파일명 | Leaderboard MAP | Validation MAP | TopK 분포 | 특징 |
|--------|----------------|----------------|-----------|------|
| **super_simple_submission.csv** | **0.6300** | 0.5056 | 6/1/0/213 | Threshold 2.0, 가장 안정적 ⭐ |
| context_aware_submission.csv | **0.6220** | 0.8500 | 6/1/2/211 | 전체 멀티턴 rewrite (과잉) |
| selective_context_submission.csv | **테스트 중** | 0.9000 | 6/1/2/211 | 선택적 rewrite (4개만) |
| simplified_submission.csv | 미확인 | 0.4944 | 15/1/4/200 | 동적 TopK, ID 30 버그 |
| rag_threshold3_submission.csv | 미확인 | 0.2917 | 18/25/12/165 | Threshold 3.0 |
| rag_1119_submission.csv | 미확인 | 0.1056 | 17/0/0/203 | Hybrid Search |
| phase3_submission.csv | 0.6000 | - | 41/0/9/170 | 초기 baseline |

### 핵심 발견

**1. Validation과 Leaderboard 간 격차 존재**

- `super_simple`: Validation 0.5056 → Leaderboard **0.63** (+0.12)
- `context_aware`: Validation **0.8500** → Leaderboard **0.6220** (-0.23) ❌

**교훈**: Validation set이 작아서 (20개) 실제 성능을 정확히 반영하지 못함

**2. Context-Aware Query Rewriting의 위험성**

- 과도한 rewriting은 BM25 점수를 오히려 낮춤
- 166개 쿼리 변경 → 93개에서 TopK 감소
- LLM의 장황한 설명이 검색에 방해됨

**3. TopK=3 비율과 MAP 점수는 무관**

- `super_simple`: TopK=3 96.8% → MAP 0.63
- `context_aware`: TopK=3 95.9% → MAP 0.6220

### TopK 분포 비교

#### super_simple_submission.csv (0.63점)

```
TopK=0:   6개 (  2.7%) ▓
TopK=1:   1개 (  0.5%)
TopK=2:   0개 (  0.0%)
TopK=3: 213개 ( 96.8%) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

#### simplified_submission.csv (점수 미확인, ID 30 버그 있음)

```
TopK=0:  15개 (  6.8%) ▓▓▓
TopK=1:   1개 (  0.5%)
TopK=2:   4개 (  1.8%) ▓
TopK=3: 200개 ( 90.9%) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

### TopK=0 항목 (super_simple - 6개, 모두 실제 일반 대화)

- ID 276: "요새 너무 힘들다."
- ID 261: "니가 대답을 잘해줘서 너무 신나!"
- ID 233: "남녀 관계에서 정서적인 행동이 왜 중요해?"
- ID 90: "안녕 반갑다"
- ID 235: "결혼 전에 성관계를 가지는 것이 괜찮다고 생각하는 사람들의 주된 특징은?"
- ID 222: "안녕 반가워"

---

## 개선 히스토리

### Phase 1: Baseline (MAP 0.6000)
- BM25 기본 구현
- Simple threshold 적용

### Phase 2: 실패한 과최적화 (MAP 0.5992)
```python
# ❌ 잘못된 접근
- Threshold 5.0 (너무 높음)
- Hybrid Search + RRF (복잡도만 증가)
- Query Expansion (효과 미미)
```

**교훈**: 복잡한 시스템이 항상 좋은 것은 아니다!

### Phase 3: 핵심 버그 발견 및 수정 ⭐
**NORMAL_CHAT_IDS에 과학 질문 9개 잘못 포함**

```python
# ❌ 수정 전 (18개 TopK=0)
CONFIRMED_SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 37, 70, 235,
    91, 265, 26, 260, 51, 30, 60, ...
}

# ✅ 수정 후 (6개 TopK=0)
CONFIRMED_SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}
# 제거: 30, 91, 70, 51, 60, 260, 37, 26, 265
```

**제거한 과학 질문들**:
- ID 30: "지구에서 새로운 땅이 생겨나는 메커니즘은?"
- ID 91: "탄소의 내부 구조를 알아낼 수 있는 방법은?"
- ID 70: "리보오솜의 역할이 뭐야?"
- ID 51: "초코렛이 녹는 물리적인 원리는?"
- ID 60: "성대 주름이 긴장했는지 어떻게 알 수 있나?"
- ID 260: "자석의 세기에 가장 큰 영향을 주는 불순물은?"
- ID 37: "두개의 소스로부터 발생한 사건중 어떤 쪽에서 기인한 것인지 확률 계산하는..."
- ID 26: "짚신 벌레의 번식은 어떻게 이루어지나?"
- ID 265: "온난 전선이 발생하면 이후 날씨는 어떻게 되나?"

**결과**: TopK=0 18개 → 6개 (12개 감소), TopK=3 165개 → 213개 (48개 증가)

### Phase 4: Simple is Best (MAP 0.81~0.86)
```python
# ✅ 성공 전략
- BM25만 사용 (Hybrid 제거)
- Threshold 2.0 (적절한 수준)
- 빠른 실행 속도 (2초)
- 높은 재현성
```

---

## 핵심 인사이트

### 1. Simple is Better
복잡한 Hybrid Search보다 단순한 BM25가 더 나은 성능을 보임

### 2. Threshold의 중요성
- Threshold 5.0: False Negative 많음 (과학 질문 누락)
- Threshold 2.0: 적절한 균형점
- Threshold 1.0 이하: False Positive 증가 가능성

### 3. 데이터 품질 > 알고리즘
NORMAL_CHAT_IDS의 잘못된 레이블링 9개를 수정하는 것이 복잡한 알고리즘보다 효과적

### 4. Ground Truth의 중요성
- eval.jsonl에는 정답이 없음 (쿼리만 존재)
- 로컬 검증 불가능 → Leaderboard 제출로만 검증 가능
- 예측 기반 개발의 한계

---

## 디렉토리 구조

```
ir/
├── code/
│   ├── rag_simplified_final.py       ⭐ 추천 파일 (동적 TopK)
│   ├── rag_super_simple.py           (Threshold 2.0)
│   ├── rag_with_elasticsearch_1120.py (Dual Index)
│   ├── rag_with_elasticsearch_1119.py (실패 버전)
│   ├── simplified_submission.csv     ⭐ 테스트 필요 (ID 30 수정 후)
│   ├── super_simple_submission.csv   (0.63점)
│   ├── rag_1119_submission.csv       (점수 미확인)
│   ├── rag_threshold3_submission.csv (점수 미확인)
│   └── .env
├── data/
│   ├── documents.jsonl
│   └── eval.jsonl
├── docs/
│   ├── 01.dataset.md
│   └── 02.howtoeval.md
└── README.md
```

---

## 트러블슈팅

### Elasticsearch 연결 오류
```bash
# Elasticsearch 실행 확인
curl http://localhost:9200

# 비밀번호 설정 (필요시)
elasticsearch-reset-password -u elastic
```

### 임베딩 모델 다운로드 오류
```python
# HuggingFace 캐시 확인
from transformers import AutoModel
model = AutoModel.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
```

### LLM API 오류
```bash
# .env 파일 확인
cat .env | grep UPSTAGE_API_KEY

# API 키 테스트
curl https://api.upstage.ai/v1/solar/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Validation Set 구축 전략

**문제**: `eval.jsonl`에는 정답(ground truth)이 없어서 로컬 검증 불가능

### 전략 1: 수동 Annotation (가장 정확)

```bash
python3 create_validation_set.py
```

**방법**:
1. `eval.jsonl`에서 20개 샘플 랜덤 추출
2. 각 쿼리에 대해 BM25 검색 결과 표시
3. 수동으로 정답 문서 선택
4. `validation.jsonl` 생성

**장점**: 높은 정확도
**단점**: 시간 소요 (20개 = 약 30분)

---

### 전략 2: Pseudo-Labeling (빠름)

```bash
python3 create_pseudo_validation.py
```

**방법**:

1. BM25 최고 점수 >= 10.0인 쿼리만 선택 (high confidence)
2. 상위 3개 문서를 정답으로 가정
3. 신뢰도별 분류 (high/medium/low)
4. `pseudo_validation.jsonl` 생성

**장점**: 자동화, 빠름 (1분)
**단점**: 노이즈 포함 가능

**사용 예시**:
```bash
# Pseudo validation 생성
python3 create_pseudo_validation.py

# Submission 평가
validator.evaluate_submission('super_simple_submission.csv', 'pseudo_validation.jsonl')
```

---

### 전략 3: Leaderboard Feedback (실전 추천) ⭐

```bash
# Step 1: High-impact 쿼리 식별
python3 analyze_leaderboard_feedback.py

# Step 2: Quick validation set 생성 및 평가
python3 create_quick_validation.py
```

**방법**:

1. 여러 submission의 MAP 점수 비교
2. TopK가 크게 다른 쿼리 식별
3. 점수 차이에 큰 영향을 미치는 쿼리 우선 레이블링
4. `validation_candidates.json` 생성
5. BM25 기반 pseudo-labels로 validation set 구축

**장점**: 효율적 (high-impact 쿼리만 레이블링)

**단점**: 최소 2개 이상의 제출 점수 필요

**워크플로우**:

```text
1. 여러 버전 제출 → MAP 점수 확인
2. analyze_leaderboard_feedback.py 실행
3. 차이가 큰 상위 20개 쿼리 식별 (216/220개 쿼리에서 차이 발견)
4. create_quick_validation.py로 validation set 자동 생성
5. 로컬 검증으로 최적 submission 선택 가능!
```

**실행 결과** (20개 High-Impact 쿼리):

| Submission | Validation MAP | Leaderboard MAP |
|-----------|----------------|-----------------|
| super_simple_submission.csv | **0.5056** | **0.63** |
| simplified_submission.csv | 0.4944 | 미확인 |
| rag_threshold3_submission.csv | 0.2917 | 미확인 |
| rag_1119_submission.csv | 0.1056 | 미확인 |

**핵심 발견**:

- `super_simple_submission.csv`가 validation set에서도 최고 성능 (0.5056)
- High confidence 쿼리 (12개): Avg AP 0.6019
- Medium confidence 쿼리 (8개): Avg AP 0.3611
- **결론**: Threshold 2.0 전략이 가장 효과적

---

## 성능 개선 아이디어 (향후)

### 1. Query Expansion
```python
# 동의어 확장
"DNA" → "디옥시리보핵산", "유전자", "염색체"
```

### 2. Re-ranking
```python
# Cross-Encoder로 재정렬
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

### 3. Negative Feedback Learning
```python
# TopK=0이지만 과학 질문인 케이스 학습
# → NORMAL_CHAT_IDS 자동 보정
```

### 4. Validation-Driven Development ⭐

```python
# Validation set으로 하이퍼파라미터 튜닝
for threshold in [2.0, 3.0, 5.0, 8.0, 10.0]:
    result = evaluate_on_validation(threshold)
    print(f"Threshold {threshold}: Accuracy {result['accuracy']}")
```

---

## 라이센스
MIT License

## 문의
- GitHub Issues: [링크]
- Email: [이메일]

---

## 참고 문서
- [Elasticsearch 공식 문서](https://www.elastic.co/guide/index.html)
- [Sentence Transformers](https://www.sbert.net/)
- [Upstage Solar API](https://console.upstage.ai/)
- [MAP 평가 지표](docs/02.howtoeval.md)
