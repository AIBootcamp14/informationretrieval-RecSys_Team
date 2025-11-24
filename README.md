# RAG 시스템 - Scientific Knowledge QA Competition

과학 지식 질문 답변을 위한 RAG (Retrieval-Augmented Generation) 시스템입니다. Elasticsearch BM25와 BGE-M3 Dense Retrieval을 결합한 Cascaded Reranking 전략을 사용합니다.

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [최고 성능 달성](#최고-성능-달성)
- [시스템 아키텍처](#시스템-아키텍처)
- [핵심 기술 스택](#핵심-기술-스택)
- [설치 및 실행](#설치-및-실행)
- [성능 결과](#성능-결과)
- [핵심 인사이트](#핵심-인사이트)

---

## 프로젝트 개요

### 평가 지표
- **MAP@3 (Mean Average Precision)**: Top-3 문서 기반
- **목표 점수**: 0.9
- **현재 최고 점수**: **0.8030** 🏆
- **베이스라인**: 0.7848

### 데이터셋
- **documents.jsonl**: 4,272개 한국어 과학 문서
- **eval.jsonl**: 220개 평가 쿼리 (과학 질문 + 일반 대화)

---

## 최고 성능 달성

### 🏆 cascaded_reranking_v1.py (MAP@3 0.8030)

**최종 제출 파일**: `cascaded_reranking_v1_full_submission_20251124_111913.csv`

```bash
cd code
export UPSTAGE_API_KEY=your_api_key
python3 generate_full_submission.py
```

### 성능 지표

| 지표 | 값 |
|------|-----|
| **MAP@3** | **0.8030** |
| **vs Baseline** | +2.32% |
| **총 샘플** | 220개 |
| **결과 포함** | 202개 (91.8%) |
| **Smalltalk** | 18개 (8.2%) |

### 3가지 핵심 성공 요인

#### 1. Nori Analyzer 재도입 ✨

**성능 영향**: +91.4% (0.3194 → 0.6111)

```python
settings = {
    'analysis': {
        'analyzer': {
            'nori': {
                'type': 'custom',
                'tokenizer': 'nori_tokenizer',
                'filter': ['nori_posfilter']
            }
        },
        'filter': {
            'nori_posfilter': {
                'type': 'nori_part_of_speech',
                'stoptags': ['E', 'IC', 'J', 'MAG', 'MAJ', 'MM',
                             'SP', 'SSC', 'SSO', 'SC', 'SE', 'XPN',
                             'XSA', 'XSN', 'XSV', 'UNA', 'NA', 'VSV']
            }
        }
    }
}
```

**Nori vs Standard 비교**:
```
쿼리: "광합성의 원리는 무엇인가요?"

Standard analyzer:
- "광합성", "의", "원리", "는", "무엇", "인가", "요"

Nori analyzer:
- "광합성" (N), "원리" (N), "무엇" (N)
```

#### 2. API Key 설정 문제 해결 🔑

**성능 영향**: +203.4% (0.2014 → 0.6111)

```bash
export UPSTAGE_API_KEY=your_upstage_api_key_here
```

#### 3. LLM 기반 Smalltalk 자동 분류 🤖

**변경 전**: 하드코딩된 11개 ID
**변경 후**: Hybrid 방식 (규칙 기반 90% + LLM 10%)

```python
def is_smalltalk(query, client=None):
    """
    하이브리드 방식:
    1. 규칙 기반 명확한 케이스 (90% 처리, 빠름)
    2. 애매한 경우만 LLM 호출 (10% 처리, 정확함)
    """
    # 1단계: 규칙 기반
    if len(query) < 5: return True
    if any(word in query for word in greetings): return True
    if any(marker in query for marker in question_markers): return False

    # 2단계: LLM 판단 (Solar Pro)
    response = client.chat.completions.create(
        model="solar-pro",
        messages=[{"role": "user", "content": f"과학질문 vs 일반대화 판단: {query}"}],
        temperature=0.0
    )
    return "SMALLTALK" in response.choices[0].message.content
```

**결과**: 18개 smalltalk 자동 감지 (기존 11개 대비 +7개)

---

## 시스템 아키텍처

### Cascaded Reranking v1 Pipeline

```
Query Input (msg)
    ↓
[1] Query Rewriting (Solar Pro LLM)
    - 멀티턴 대화 맥락 통합
    - 대명사 → 구체적 명사 변환
    ↓
[2] Smalltalk Classification (Hybrid)
    - Stage 1: Rule-based (90%)
    - Stage 2: LLM-based (10%)
    ↓ (if SCIENCE question)
[3] Hybrid Search (Top 30)
    - BM25 (Nori analyzer)
    - BGE-M3 (Dense + Sparse + ColBERT)
    - RRF Fusion (k=60)
    ↓
[4] Cascaded LLM Reranking
    - Stage 1: 30 → 10 (빠른 필터링)
    - Stage 2: 10 → 3 (정밀한 판단)
    ↓
Final Top-3 Documents
```

---

## 핵심 기술 스택

### 검색 엔진
- **Elasticsearch 8.x** with Nori Analyzer
  - BM25 lexical search
  - 한국어 형태소 분석

### 임베딩 모델
- **BGE-M3** (BAAI/bge-m3)
  - Multi-representation: Dense + Sparse + ColBERT
  - 다국어 지원
  - 8192 토큰 컨텍스트

### LLM
- **Upstage Solar Pro**
  - Query rewriting
  - Smalltalk classification
  - 2-stage cascaded reranking

### 개발 환경
- **Python 3.10+**
- **Docker** for Elasticsearch
- **Anaconda** environment

---

## 설치 및 실행

### 1. 환경 설정

```bash
# Anaconda 환경 생성
conda create -n rag python=3.10
conda activate rag

# 패키지 설치
cd code
pip install -r requirements.txt
```

**requirements.txt**:
```
elasticsearch>=8.8.0
sentence-transformers>=2.2.0
openai>=1.0.0
python-dotenv
numpy
tqdm
pandas
FlagEmbedding
```

### 2. Elasticsearch 설치 및 실행

#### Docker 방식 (권장)

```bash
# Elasticsearch 실행
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Nori plugin 설치
docker exec elasticsearch bin/elasticsearch-plugin install analysis-nori
docker restart elasticsearch

# 연결 확인
curl http://localhost:9200
```

### 3. 문서 인덱싱

```bash
cd code

# Nori analyzer로 인덱싱
python3 index_documents_nori.py
```

**출력**:
```
✅ 인덱싱 완료!
총 4272개 문서가 'test' 인덱스에 저장되었습니다.
```

### 4. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << 'EOF'
UPSTAGE_API_KEY=your_upstage_api_key_here
EOF
```

**Upstage API Key 발급**:
1. [Upstage Console](https://console.upstage.ai/) 접속
2. API Keys 메뉴에서 새 키 생성
3. `.env` 파일에 복사

### 5. 제출 파일 생성

```bash
export UPSTAGE_API_KEY=your_api_key
python3 generate_full_submission.py
```

**출력**:
```
================================================================================
Generating Full Submission File
================================================================================
Total samples: 220
Strategy: cascaded_reranking_v1 (LLM-based smalltalk classification)
================================================================================

Processing: 100%|██████████| 220/220

================================================================================
Full Submission Generated Successfully!
================================================================================
Output file: cascaded_reranking_v1_full_submission_20251124_111913.csv
Total samples: 220
Samples with results: 202
Empty results (smalltalk): 18
================================================================================
```

---

## 성능 결과

### 실험 결과 요약

| Task | 전략 | MAP@3 | vs Baseline | 상태 |
|------|------|-------|-------------|------|
| - | Baseline | 0.7848 | - | 대회 기준 |
| 3 | cascaded_reranking_v1 (Previous) | 0.7939 | +1.16% | ✅ 기존 최고 |
| 4 | cascaded_reranking_v2 | 0.7778 | -0.89% | ❌ 실패 |
| 5 | query_decomposition_v1 | 0.5278 | -32.74% | ❌ 실패 |
| 6 | document_context_expansion | - | - | ⛔ 불가능 |
| **7** | **cascaded_reranking_v1 (Final)** | **0.8030** | **+2.32%** | 🏆 **최고 성능** |

### 성능 향상 여정

```
0.7848 (Baseline)
  ↓ +1.16%
0.7939 (cascaded_reranking_v1 Previous)
  ↓ +1.15%
0.8030 (cascaded_reranking_v1 Final) 🏆
```

**총 향상**: +2.32% (0.7848 → 0.8030)

---

## 핵심 인사이트

### 1. 한글 처리의 중요성

**Nori analyzer가 BM25 검색 품질에 결정적 영향**

- Standard analyzer: MAP@3 0.3194
- Nori analyzer: MAP@3 0.6111
- **개선**: +0.2917 (+91.4%)

### 2. LLM 기능의 필수성

**API Key 활성화 시 얻는 기능**:
- Query rewriting (멀티턴 대화 맥락 통합)
- Smalltalk 자동 분류
- LLM Reranking (의미적 관련성 판단)

**성능 영향**: +203.4% (0.2014 → 0.6111)

### 3. Retrieval Recall이 병목

**Ultra Validation Set 분석 결과** (7개 실패 케이스):
- **Retrieval 단계 실패**: 6개 (85.7%) ← **병목**
- Reranking 단계 실패: 1개 (14.3%)

**결론**: Reranking은 이미 잘 작동하며, Retrieval 개선이 우선순위

### 4. 복잡도 증가는 역효과

**Task 4 (Cascaded v2)**: 3-stage reranking → 성능 하락 (-2.03%)
**Task 5 (Query Decomposition)**: 복잡한 쿼리 분해 → 성능 폭락 (-33.52%)

**교훈**: Simple is Better

### 5. 자동화의 가치

**하드코딩 문제점**:
- 새로운 평가 데이터에 대응 불가
- 수동 라벨링 필요
- 유지보수 어려움

**LLM 자동 분류 장점**:
- 일반화 능력
- 데이터 변경 자동 대응
- 확장성

---

## 디렉토리 구조

```
ir/
├── code/
│   ├── cascaded_reranking_v1.py              🏆 최고 성능 전략
│   ├── generate_full_submission.py           제출 파일 생성기
│   ├── index_documents_nori.py               Nori 인덱싱
│   ├── create_embeddings_bgem3_optimized.py  BGE-M3 임베딩
│   ├── auto_validate.py                      자동 검증
│   ├── cascaded_reranking_v1_full_submission_20251124_111913.csv  🏆
│   ├── docs/                                 실험 문서 (15개)
│   ├── archived/                             아카이브 (gitignored)
│   │   ├── embeddings/                       대용량 임베딩 파일
│   │   ├── submissions/                      이전 제출 파일 73개
│   │   └── experiments/                      실패한 실험 20개
│   ├── .env
│   ├── .gitignore
│   ├── requirements.txt
│   └── EXPERIMENT_SUMMARY_20251124.md        📊 종합 실험 보고서
├── data/
│   ├── documents.jsonl                       4,272 문서
│   └── eval.jsonl                            220 쿼리
├── docs/
│   ├── 01.dataset.md
│   └── 02.howtoeval.md
├── README.md                                 👈 현재 문서
└── ROADMAP_TO_0.9.md                        🎯 다음 단계 계획
```

---

## 트러블슈팅

### Elasticsearch 연결 오류

```bash
# Elasticsearch 실행 확인
curl http://localhost:9200

# Docker 컨테이너 상태 확인
docker ps --filter "name=elasticsearch"

# 로그 확인
docker logs elasticsearch
```

### Nori plugin 설치 오류

```bash
# Plugin 목록 확인
docker exec elasticsearch bin/elasticsearch-plugin list

# Plugin 재설치
docker exec elasticsearch bin/elasticsearch-plugin remove analysis-nori
docker exec elasticsearch bin/elasticsearch-plugin install analysis-nori
docker restart elasticsearch
```

### BGE-M3 임베딩 오류

```bash
# HuggingFace 캐시 확인
ls ~/.cache/huggingface/hub/

# 수동 다운로드
python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

### LLM API 오류

```bash
# .env 파일 확인
cat .env | grep UPSTAGE_API_KEY

# API 키 테스트
curl https://api.upstage.ai/v1/solar/chat/completions \
  -H "Authorization: Bearer $UPSTAGE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"solar-pro","messages":[{"role":"user","content":"안녕"}]}'
```

---

## 다음 단계

### 단기 (1-2주)
- **BM25 파라미터 튜닝** (k1, b 최적화)
  - 예상 성능 향상: +2-5%
- **Hybrid Weight 최적화** (BM25 vs BGE-M3 가중치)
  - 예상 성능 향상: +1-3%

### 중기 (3-4주)
- **BGE-M3 Fine-tuning** (과학 도메인 특화)
  - 예상 성능 향상: +3-7%
- **Prompt Engineering** (Reranking prompt 최적화)
  - 예상 성능 향상: +1-2%

### 장기 (1개월+)
- 앙상블 방법
- 새로운 임베딩 모델 실험
- Semantic Chunking 재시도

**상세 계획**: [ROADMAP_TO_0.9.md](ROADMAP_TO_0.9.md)

---

## 참고 문서

### 프로젝트 문서
- [EXPERIMENT_SUMMARY_20251124.md](code/EXPERIMENT_SUMMARY_20251124.md) - 종합 실험 보고서
- [ROADMAP_TO_0.9.md](ROADMAP_TO_0.9.md) - MAP@3 0.9 달성 로드맵

### 외부 문서
- [Elasticsearch 공식 문서](https://www.elastic.co/guide/index.html)
- [BGE-M3 GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [Upstage Solar API](https://console.upstage.ai/)
- [Nori Analyzer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori.html)

---

## 라이센스

MIT License

## 기여자

- AI Bootcamp 14기 RecSys Team
- Developed with Claude Code

---

**최종 업데이트**: 2025-11-24
**현재 최고 성능**: MAP@3 **0.8030** 🏆
**목표**: MAP@3 0.9
