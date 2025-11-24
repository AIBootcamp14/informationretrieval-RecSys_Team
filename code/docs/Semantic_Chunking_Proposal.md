# 과학지식처리 최적화 청킹 전략 제안서

**작성일**: 2025-11-21
**목표**: 현재 0.68 MAP 점수를 0.72-0.75로 향상
**근거**: Semantic Chunking 기법 적용 시 Recall 9% 향상 가능 (2025 연구 결과)

---

## 1. 현재 상태 분석

### 1.1 현재 Corpus 구조

```
총 문서 수: 4,272개
평균 길이: 315자
길이 분포:
  - 200자 미만: 433개 (10.1%)
  - 200-400자: 3,401개 (79.6%) ← 대부분
  - 400자 이상: 438개 (10.3%)
```

**분석 결과**:
- ✅ 이미 적절한 크기로 청킹됨 (200-400자 범위)
- ✅ 전통적인 크기 기반 재청킹 불필요
- ⚠️ 하지만 의미적(semantic) 경계 고려 없음

### 1.2 현재 검색 시스템 한계

```python
# solar_llm_optimized_solution.py:78-105
def search_bm25(query, top_k=3):
    response = es.search(
        index='test',
        body={
            'query': {
                'match': {
                    'content': {
                        'query': query,
                        'analyzer': 'nori'  # 단순 형태소 분석만
                    }
                }
            },
            'size': top_k
        }
    )
```

**문제점**:
1. 키워드 매칭만 수행 (의미적 유사도 미고려)
2. 문장 경계 무시 (중간에 잘린 문맥)
3. 과학 개념 간 관계성 반영 안 됨

---

## 2. 제안 전략: 3단계 접근법

### 전략 A: **문장 단위 Semantic Chunking** (즉시 적용 가능)

**아이디어**: 현재 문서를 의미적으로 완결된 문장 단위로 재구성

#### 구현 방법

```python
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 한국어 형태소 분석기
okt = Okt()

# 2. 의미 유사도 계산 모델
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def semantic_sentence_chunking(document, max_chunk_size=400, overlap=50):
    """
    문장 단위 의미적 청킹

    핵심:
    1. 문장 경계 감지
    2. 의미적 유사도 기반 그룹화
    3. 과학 개념 완결성 보장
    """
    # Step 1: 문장 분리
    sentences = split_sentences_korean(document)

    # Step 2: 각 문장 임베딩
    embeddings = embedding_model.encode(sentences)

    # Step 3: 의미적 유사도 기반 그룹화
    chunks = []
    current_chunk = []
    current_length = 0

    for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
        current_chunk.append(sent)
        current_length += len(sent)

        # 청크 크기 도달 시
        if current_length >= max_chunk_size:
            # 다음 문장과 유사도 계산
            if i + 1 < len(sentences):
                similarity = cosine_similarity(emb, embeddings[i+1])

                # 유사도 낮으면 (개념 전환점) → 청크 종료
                if similarity < 0.7:
                    chunks.append(' '.join(current_chunk))
                    # Overlap 적용 (마지막 1-2문장 보존)
                    overlap_sents = current_chunk[-1:] if overlap > 0 else []
                    current_chunk = overlap_sents
                    current_length = sum(len(s) for s in overlap_sents)

    # 마지막 청크
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def split_sentences_korean(text):
    """한국어 문장 분리 (과학 용어 고려)"""
    # KoNLPy로 형태소 분석 후 문장 경계 감지
    sentences = []
    current = []

    for word, pos in okt.pos(text):
        current.append(word)

        # 문장 종결 어미 감지
        if pos in ['SF', 'SE']:  # 마침표, 느낌표, 물음표
            sentences.append(''.join(current))
            current = []

    if current:
        sentences.append(''.join(current))

    return sentences

def cosine_similarity(emb1, emb2):
    """코사인 유사도 계산"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

#### 적용 효과 예측

**Before (현재)**:
```
문서: "DNA는 유전정보를 저장한다. DNA는 이중나선 구조이다. RNA는 단일가닥이다."
→ 1개 청크 (모든 내용 혼재)
```

**After (Semantic Chunking)**:
```
청크 1: "DNA는 유전정보를 저장한다. DNA는 이중나선 구조이다."
청크 2: "RNA는 단일가닥이다." (개념 전환)
→ 2개 청크 (DNA/RNA 분리)
```

**결과**: DNA 관련 쿼리 시 불필요한 RNA 정보 제외 → Precision 향상

---

### 전략 B: **Hybrid Retrieval with Dense Embeddings** (중기 적용)

**아이디어**: BM25(sparse) + Embedding(dense) 하이브리드 검색

#### 아키텍처

```
Query → [BM25 Search] → Top-10 (sparse)
      ↓
      [Embedding Rerank] → Top-3 (dense semantic)
      ↓
      [LLM Rerank] → Final Top-3
```

#### 구현 방법

```python
def hybrid_search_with_semantic_chunks(query, top_k=3):
    """
    Hybrid Search: BM25 + Semantic Embedding
    """
    # 1단계: BM25로 Top-10 검색 (기존 방식)
    bm25_results, max_score = search_bm25(query, top_k=10)

    # 2단계: 쿼리 임베딩
    query_embedding = embedding_model.encode(query)

    # 3단계: 각 문서 청크의 임베딩과 유사도 계산
    scored_results = []
    for doc in bm25_results:
        # 문서 내 문장별 임베딩
        chunks = semantic_sentence_chunking(doc['content'])
        chunk_embeddings = embedding_model.encode(chunks)

        # 가장 관련성 높은 청크 선택
        similarities = [
            cosine_similarity(query_embedding, chunk_emb)
            for chunk_emb in chunk_embeddings
        ]
        max_similarity = max(similarities)
        best_chunk_idx = similarities.index(max_similarity)

        scored_results.append({
            'docid': doc['docid'],
            'content': chunks[best_chunk_idx],  # 가장 관련성 높은 청크
            'semantic_score': max_similarity,
            'bm25_score': doc.get('score', 0)
        })

    # 4단계: Hybrid Scoring (BM25 30% + Semantic 70%)
    for result in scored_results:
        result['hybrid_score'] = (
            0.3 * (result['bm25_score'] / max_score) +
            0.7 * result['semantic_score']
        )

    # 5단계: Hybrid Score 기준 정렬 후 Top-5 선택
    scored_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    top5 = scored_results[:5]

    # 6단계: LLM Reranking (기존 방식)
    reranked_ids = llm_rerank_top3(query, top5)

    return reranked_ids, max_score
```

#### 효과 예측

**시나리오**: "DNA 복제 과정에서 효소 역할은?"

**Before (BM25만)**:
```
Top-3: ["DNA 복제 개요...", "효소 종류...", "세포 분열..."]
→ "DNA 복제" 키워드만 매칭
```

**After (Hybrid)**:
```
Top-3: ["DNA 복제에서 헬리카아제는...", "DNA 중합효소의 역할...", "프라이머효소..."]
→ "DNA 복제" + "효소" 의미적 관계 반영
```

**예상 개선**: Recall +9%, MAP +0.04-0.06 (0.68 → 0.72-0.74)

---

### 전략 C: **과학 개념 기반 Contextual Chunking** (장기 적용)

**아이디어**: 과학 개념 경계를 기준으로 지능형 청킹

#### 핵심 기법

```python
# 과학 개념 인식기
SCIENCE_CONCEPTS = {
    'DNA': ['유전자', '염기서열', '복제', '전사', '번역'],
    '광합성': ['엽록소', '광계', 'ATP', 'NADPH', '캘빈회로'],
    '세포호흡': ['미토콘드리아', 'TCA회로', '전자전달계', 'ATP'],
    # ... 과학 지식 그래프
}

def concept_aware_chunking(document, concepts=SCIENCE_CONCEPTS):
    """
    과학 개념 경계 기반 청킹

    예: "DNA 복제 설명 + 광합성 설명"
    → 2개 청크로 분리 (개념 단위)
    """
    # Step 1: 문서에서 개념 감지
    detected_concepts = []
    for concept, keywords in concepts.items():
        if any(kw in document for kw in keywords):
            detected_concepts.append(concept)

    # Step 2: 개념별 텍스트 분리
    chunks = []
    for concept in detected_concepts:
        concept_text = extract_text_for_concept(document, concept, concepts)
        chunks.append({
            'concept': concept,
            'text': concept_text,
            'keywords': concepts[concept]
        })

    return chunks

def extract_text_for_concept(document, concept, concept_map):
    """개념 관련 텍스트 추출"""
    keywords = concept_map[concept]

    sentences = split_sentences_korean(document)
    relevant_sentences = []

    for sent in sentences:
        # 개념 키워드가 포함된 문장만 추출
        if any(kw in sent for kw in keywords):
            relevant_sentences.append(sent)

    return ' '.join(relevant_sentences)
```

#### 효과

**문서 예시**:
```
"DNA는 이중나선 구조로 유전정보를 저장한다. DNA 복제는 반보존적으로 진행된다.
식물의 광합성은 엽록소에서 일어나며 빛 에너지를 화학 에너지로 변환한다."
```

**Before (단순 청킹)**:
```
청크 1: 전체 텍스트 (DNA + 광합성 혼재)
```

**After (Concept-Aware)**:
```
청크 1 [DNA]: "DNA는 이중나선 구조로 유전정보를 저장한다. DNA 복제는 반보존적으로 진행된다."
청크 2 [광합성]: "식물의 광합성은 엽록소에서 일어나며 빛 에너지를 화학 에너지로 변환한다."
```

**검색 효과**:
- "DNA 복제" 쿼리 → 청크 1만 반환 (광합성 노이즈 제거)
- Precision 대폭 향상

---

## 3. 구현 우선순위 및 로드맵

### Phase 1: 즉시 적용 (1-2일)

**전략 A: Sentence-Level Semantic Chunking**

```bash
# 1. 필요 라이브러리 설치
pip install konlpy sentence-transformers

# 2. 새 스크립트 작성
solar_semantic_chunking_v1.py

# 3. 기존 documents.jsonl 재처리
python reindex_with_semantic_chunks.py
```

**예상 효과**:
- 문장 경계 보존으로 문맥 완결성 향상
- Recall +3-5%
- MAP: 0.68 → **0.70-0.71**

**구현 코드**:
```python
# reindex_with_semantic_chunks.py
import json
from elasticsearch import Elasticsearch
from semantic_chunking import semantic_sentence_chunking

es = Elasticsearch(['http://localhost:9200'])

# 기존 인덱스 삭제 후 재생성
es.indices.delete(index='test', ignore=[404])
es.indices.create(index='test', body={
    'settings': {
        'analysis': {
            'analyzer': {
                'nori': {
                    'type': 'custom',
                    'tokenizer': 'nori_tokenizer'
                }
            }
        }
    }
})

# documents.jsonl 로드
with open('../data/documents.jsonl', 'r') as f:
    documents = [json.loads(line) for line in f]

# Semantic Chunking 후 재인덱싱
for doc in documents:
    chunks = semantic_sentence_chunking(doc['content'], max_chunk_size=400)

    for i, chunk in enumerate(chunks):
        es.index(
            index='test',
            body={
                'docid': f"{doc['docid']}_chunk{i}",
                'original_docid': doc['docid'],
                'content': chunk,
                'chunk_index': i
            }
        )

print(f"✅ 재인덱싱 완료: {len(documents)}개 문서 → {total_chunks}개 청크")
```

---

### Phase 2: 중기 적용 (3-5일)

**전략 B: Hybrid Retrieval (BM25 + Dense Embeddings)**

```bash
# 1. Elasticsearch에 벡터 검색 추가
# knn_vectors 필드 추가

# 2. 새 스크립트 작성
solar_hybrid_search_v1.py

# 3. 테스트 실행
python solar_hybrid_search_v1.py
```

**예상 효과**:
- BM25 키워드 + Semantic 의미 결합
- Recall +7-9%
- MAP: 0.68 → **0.72-0.75**

**핵심 변경**:
```python
# Elasticsearch 스키마 변경
{
    'mappings': {
        'properties': {
            'docid': {'type': 'keyword'},
            'content': {'type': 'text', 'analyzer': 'nori'},
            'embedding': {  # ← 추가
                'type': 'dense_vector',
                'dims': 768,  # ko-sroberta 임베딩 차원
                'index': True,
                'similarity': 'cosine'
            }
        }
    }
}
```

---

### Phase 3: 장기 적용 (1-2주)

**전략 C: Concept-Aware Chunking**

**준비 작업**:
1. 과학 개념 온톨로지 구축 (300-500개 개념)
2. 개념 간 관계 그래프 작성
3. LLM으로 개념 자동 추출

**예상 효과**:
- 과학 지식 그래프 기반 정밀 검색
- Precision +15-20%
- MAP: 0.68 → **0.75-0.78**

---

## 4. 비용-효과 분석

### 4.1 계산 비용

| 방법 | 초기 비용 | 쿼리당 비용 | 예상 MAP | 비용 대비 효과 |
|------|----------|------------|----------|--------------|
| **현재 (BM25 only)** | 0 | ~1초 | 0.68 | Baseline |
| **Phase 1 (Semantic Chunking)** | +30분 (재인덱싱) | ~1.5초 | 0.70-0.71 | ⭐⭐⭐⭐ 높음 |
| **Phase 2 (Hybrid Retrieval)** | +1시간 (임베딩 생성) | ~3초 | 0.72-0.75 | ⭐⭐⭐ 중간 |
| **Phase 3 (Concept-Aware)** | +5시간 (온톨로지 구축) | ~4초 | 0.75-0.78 | ⭐⭐ 낮음 |

### 4.2 권장 사항

**즉시 실행**: Phase 1 (Semantic Chunking)
- 비용 최소 (30분 재인덱싱만)
- 효과 검증 빠름 (+0.02-0.03 MAP)
- 실패 시 롤백 쉬움

**성공 시**: Phase 2 (Hybrid Retrieval)
- Phase 1 기반 추가 개선
- 벡터 검색으로 의미 유사도 강화
- +0.04-0.07 MAP 추가 향상 가능

**장기 목표**: Phase 3 (Concept-Aware)
- 과학 지식 특화 최적화
- 0.75+ MAP 목표
- 단, 구축 비용 고려 필요

---

## 5. 실험 계획

### 5.1 Phase 1 실험 설계

**날짜**: 2025-11-21 (오늘)
**소요 시간**: 2-3시간

#### Step 1: 환경 설정
```bash
# 라이브러리 설치
pip install konlpy sentence-transformers

# KoNLPy 한국어 사전 다운로드
python -c "from konlpy.tag import Okt; Okt()"
```

#### Step 2: Semantic Chunking 구현
```bash
# 새 파일 생성
code/semantic_chunking.py  # 청킹 로직
code/reindex_with_semantic_chunks.py  # 재인덱싱 스크립트
```

#### Step 3: 재인덱싱 실행
```bash
# Elasticsearch 실행 확인
docker start elasticsearch
sleep 10

# 재인덱싱 (기존 데이터 백업 후)
python code/reindex_with_semantic_chunks.py
```

#### Step 4: 새 검색 실행
```bash
# solar_llm_optimized_solution.py 기반 복사
cp code/solar_llm_optimized_solution.py code/solar_semantic_v1.py

# 실행
python code/solar_semantic_v1.py
```

#### Step 5: 결과 비교
```bash
# 제출 파일 생성
solar_semantic_v1_submission.csv

# 리더보드 제출 후 점수 확인
# 예상: 0.70-0.71 (현재 0.68 대비 +0.02-0.03)
```

### 5.2 성공 기준

**최소 성공 (Phase 1)**:
- MAP ≥ 0.70 (+0.02)
- TopK=3 비율 유지 (97%+)
- 처리 시간 ≤ 20분

**목표 성공 (Phase 2)**:
- MAP ≥ 0.73 (+0.05)
- TopK=3 비율 유지
- 처리 시간 ≤ 30분

**최대 성공 (Phase 3)**:
- MAP ≥ 0.75 (+0.07)
- TopK=3 비율 유지
- 처리 시간 ≤ 40분

---

## 6. 리스크 분석

### 6.1 Phase 1 리스크

**리스크 1**: 재인덱싱 후 기존 docid 불일치
- **완화**: `original_docid` 필드 추가로 매핑 유지
- **대응**: 롤백 스크립트 준비

**리스크 2**: 한국어 문장 분리 오류
- **완화**: KoNLPy 대신 규칙 기반 분리 병행
- **대응**: 수동 검증 샘플 100개 체크

**리스크 3**: 청킹으로 오히려 성능 저하
- **완화**: A/B 테스트로 비교
- **대응**: 즉시 원복 (백업 인덱스 사용)

### 6.2 Phase 2 리스크

**리스크 1**: 임베딩 생성 시간 과다
- **완화**: 배치 처리 (batch_size=32)
- **대응**: GPU 사용 (M1 Mac MPS)

**리스크 2**: Elasticsearch 벡터 검색 성능
- **완화**: HNSW 인덱스 사용
- **대응**: 벡터 차원 축소 (768 → 384)

---

## 7. 예상 타임라인

**오늘 (2025-11-21)**:
- [x] 청킹 전략 리서치 완료
- [x] 제안서 작성 완료
- [ ] Phase 1 구현 시작 (2-3시간)
- [ ] 재인덱싱 실행 (30분)
- [ ] 테스트 실행 및 결과 확인 (20분)

**내일 (2025-11-22)**:
- [ ] Phase 1 결과 분석
- [ ] Phase 2 구현 착수 (성공 시)
- [ ] Hybrid Retrieval 테스트

**다음 주**:
- [ ] Phase 2 결과 분석
- [ ] 최종 제출 결정

---

## 8. 결론 및 권장사항

### 8.1 핵심 인사이트

**발견 1**: 현재 문서는 크기는 적절하지만 의미적 경계 무시
- 해결: Sentence-level Semantic Chunking

**발견 2**: BM25는 키워드 매칭만, 의미 유사도 미반영
- 해결: Hybrid Retrieval (BM25 + Embeddings)

**발견 3**: 과학 개념 간 관계성 활용 안 됨
- 해결: Concept-Aware Chunking (장기)

### 8.2 즉시 실행 권장

**Phase 1: Semantic Sentence Chunking**
- 비용: 최소 (3시간 작업)
- 효과: +0.02-0.03 MAP (0.68 → 0.70-0.71)
- 리스크: 낮음 (롤백 가능)

**구현 우선순위**:
1. ✅ **즉시**: semantic_chunking.py 구현
2. ✅ **오늘**: 재인덱싱 실행
3. ✅ **오늘**: 테스트 및 결과 확인
4. ⏸️ **내일**: Phase 2 결정 (Phase 1 성공 시)

### 8.3 성공 가능성

**Phase 1**: ⭐⭐⭐⭐⭐ (90% 성공 예상)
- 문장 경계 보존은 검증된 기법
- 한국어 처리 경험 풍부 (KoNLPy)

**Phase 2**: ⭐⭐⭐⭐ (75% 성공 예상)
- Hybrid Retrieval은 2025 표준
- Elasticsearch 벡터 검색 성숙

**Phase 3**: ⭐⭐⭐ (60% 성공 예상)
- 과학 온톨로지 구축 비용 높음
- 도메인 지식 필요

---

**다음 단계**: Phase 1 구현 착수
**예상 완료**: 오늘 저녁 (3-4시간 소요)
**기대 결과**: MAP 0.70+ 달성

---

**작성자**: Claude Code
**참고 문헌**:
- "Semantic Chunking for RAG Systems" (2025)
- "Korean NLP Best Practices with KoNLPy"
- "Hybrid Dense-Sparse Retrieval" (ACL 2024)
