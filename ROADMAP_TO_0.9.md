# 🎯 MAP@3 0.9 달성 로드맵

## 현재 상황

- **현재 최고 점수**: **0.8030** 🏆
- **목표 점수**: 0.9
- **필요한 향상**: +0.097 (+12.1%)
- **베이스라인**: 0.7848

### 성능 향상 여정

```
0.7848 (Baseline)
  ↓ +1.16%
0.7939 (cascaded_reranking_v1 Previous)
  ↓ +1.15%
0.8030 (cascaded_reranking_v1 Final) 🏆 ← 현재 위치
  ↓ +12.1% (목표)
0.9000 (Target) 🎯
```

---

## 📊 병목 분석 결과

### Ultra Validation Set 실패 케이스 분석 (7개)

| 단계 | 실패 개수 | 비율 | 심각도 |
|------|----------|------|--------|
| **Retrieval (초기 검색)** | 6개 | 85.7% | 🔴 **HIGH** |
| Reranking (재정렬) | 1개 | 14.3% | 🟢 LOW |

**핵심 발견**:
- **Retrieval Recall이 병목**: Top-30에 정답이 없으면 Reranking도 무용지물
- Reranking은 이미 잘 작동 (6/7 성공률)
- **우선순위**: Retrieval 개선 >> Reranking 개선

### Retrieval 실패 사례 (6개)

**패턴 분석**:
1. **희귀 용어/고유명사**: 3개
   - "플랑크톤의 역할", "interferon", "bridge inverter"
2. **추상적 표현**: 2개
   - "달이 항상 같은 면만 보이는 이유"
3. **도메인 특화 용어**: 1개
   - "성대 주름 긴장"

---

## 🚀 3단계 개선 전략

### Phase 1: BM25 파라미터 튜닝 (예상 +2~5%) 🔴 최우선

**현재 상태**:
```python
# Elasticsearch 기본값 사용
k1 = 1.2  # Term frequency saturation
b = 0.75  # Length normalization
```

**문제점**:
- 한국어 과학 문서는 일반 문서보다 길이 편차가 큼
- 기본 파라미터는 영어 웹 문서에 최적화됨

**해결책**:
```python
# Grid Search로 최적값 찾기
for k1 in [0.8, 1.0, 1.2, 1.5, 2.0]:
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        map_score = evaluate(k1, b)
```

**예상 효과**:
- MAP@3 0.8030 → 0.82~0.84 (+2~5%)
- 실행 시간: 1~2시간

**구현 계획**:
1. `bm25_parameter_tuning.py` 작성
2. Ultra Validation Set으로 평가
3. 최적 파라미터 선택
4. 전체 데이터셋 제출

---

### Phase 2: Hybrid Weight 최적화 (예상 +1~3%) 🟡

**현재 상태**:
```python
# RRF Fusion (k=60)
# BM25와 BGE-M3 동등 가중치
```

**문제점**:
- BM25와 Dense의 상대적 중요도 미조정
- RRF k 값이 최적이 아닐 수 있음

**해결책**:
```python
# Weighted Hybrid
final_score = alpha * bm25_score + (1-alpha) * dense_score

# Grid Search
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for rrf_k in [30, 60, 90, 120]:
        map_score = evaluate(alpha, rrf_k)
```

**예상 효과**:
- MAP@3 0.84 → 0.85~0.86 (+1~3%)
- BM25 가중치 높일 것으로 예상 (alpha=0.7~0.8)

**구현 계획**:
1. `hybrid_weight_tuning.py` 작성
2. Ultra Validation Set으로 평가
3. 최적 가중치 선택

---

### Phase 3: BGE-M3 Fine-tuning (예상 +3~7%) 🟢

**현재 상태**:
```python
# Pre-trained BGE-M3 사용
# 일반 도메인 학습 모델
```

**문제점**:
- 과학 도메인 특화 학습 안 됨
- 한국어 과학 용어 임베딩 품질 낮음

**해결책**:
```python
# 1. Pseudo-labeling으로 학습 데이터 생성
# BM25 high-confidence 쿼리-문서 쌍 수집

# 2. Contrastive Learning
triplets = [
    (query, positive_doc, negative_doc)
    for each training sample
]

# 3. Fine-tuning
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
model.fit(triplets, epochs=3)
```

**예상 효과**:
- MAP@3 0.86 → 0.89~0.90 (+3~7%)
- 과학 용어 임베딩 품질 대폭 향상

**구현 계획**:
1. `create_training_data.py` - BM25 기반 pseudo-labeling
2. `finetune_bgem3.py` - Fine-tuning
3. `create_embeddings_finetuned.py` - 재생성
4. 전체 데이터셋 제출

---

## 📈 예상 최종 점수

| Phase | 개선 내용 | 예상 향상 | 누적 점수 | 난이도 |
|-------|----------|-----------|----------|--------|
| **현재** | cascaded_reranking_v1 Final | - | **0.8030** | - |
| **Phase 1** | BM25 파라미터 튜닝 | +2~5% | **0.82~0.84** | 🟢 LOW |
| **Phase 2** | Hybrid Weight 최적화 | +1~3% | **0.83~0.86** | 🟢 LOW |
| **Phase 3** | BGE-M3 Fine-tuning | +3~7% | **0.86~0.90** ✅ | 🔴 HIGH |

**총 예상 향상**: +6~15% (+0.05~0.12)
**목표 달성 가능성**: ✅ **HIGH**

---

## 🛠️ 단계별 실행 계획

### Step 1: BM25 파라미터 튜닝 (1~2일)

```bash
cd code

# 1. 튜닝 스크립트 작성
cat > bm25_parameter_tuning.py << 'EOF'
# Grid Search for BM25 parameters
# Ultra Validation Set으로 평가
EOF

# 2. 실행
python3 bm25_parameter_tuning.py

# 3. 최적 파라미터 적용
# index_documents_nori.py 수정
```

**기대 결과**:
- 최적 k1, b 값 발견
- Validation MAP: 0.8030 → 0.82~0.84

---

### Step 2: Hybrid Weight 최적화 (1일)

```bash
# 1. 튜닝 스크립트 작성
python3 hybrid_weight_tuning.py

# 2. 최적 가중치 적용
# cascaded_reranking_v1.py 수정
```

**기대 결과**:
- 최적 alpha, rrf_k 값 발견
- Validation MAP: 0.84 → 0.85~0.86

---

### Step 3: BGE-M3 Fine-tuning (3~5일)

```bash
# 1. 학습 데이터 생성
python3 create_training_data.py
# 출력: training_triplets.json (예상 1000~2000 쌍)

# 2. Fine-tuning
python3 finetune_bgem3.py
# 소요 시간: 2~4시간 (GPU 필요)

# 3. 임베딩 재생성
python3 create_embeddings_finetuned.py

# 4. 제출 파일 생성
python3 generate_full_submission.py
```

**기대 결과**:
- 과학 도메인 특화 임베딩
- Validation MAP: 0.86 → 0.89~0.90

---

## 💡 추가 최적화 아이디어

### 1. Prompt Engineering (예상 +1~2%)

**현재 Reranking Prompt**:
```python
prompt = f"""
다음 질문에 대해 문서가 관련이 있는지 판단하세요.
질문: {query}
문서: {doc}
"""
```

**개선 방향**:
```python
# 1. 과학 도메인 특화
# 2. Few-shot examples 추가
# 3. Chain-of-Thought reasoning
```

### 2. Query Expansion (예상 +1~2%)

```python
# 영어 키워드 한글 변환
"interferon" → "인터페론"

# 동의어 확장
"역할" → ["역할", "기능", "작용", "효과"]
```

### 3. Semantic Chunking 재시도 (예상 +2~4%)

**이전 실패 원인**:
- Task 6에서 데이터 구조 한계로 포기

**새로운 접근**:
```python
# Chunk 단위 검색 + Full Document 재구성
1. Chunk 검색으로 관련 문서 찾기
2. Chunk가 속한 Full Document 반환
3. LLM Reranking으로 정확도 향상
```

---

## 📋 체크리스트

### Phase 1: BM25 튜닝 (필수) ✅ 완료
- [x] bm25_parameter_tuning.py 작성 → `optimize_bm25.py`
- [x] Ultra Validation Set으로 평가
- [x] 최적 파라미터 선택 (k1, b) → **k1=0.9, b=0.5 (MAP@3 0.99)**
- [x] index_documents_nori.py 업데이트 → **BM25 k1=0.9, b=0.5 적용 완료**
- [x] 전체 인덱스 재생성 완료 (4272개 문서)
- [x] 전체 데이터셋 제출 → **`cascaded_reranking_v1_full_submission_20251124_201646.csv`**

### Phase 2: Hybrid Weight 최적화 (권장) ✅ 완료
- [x] hybrid_weight_tuning.py 작성 ✅
- [x] Grid Search 실행 (alpha, rrf_k) → **k=30 (MAP@3 0.99)**
- [x] cascaded_reranking_v1.py 확인 → **이미 k=30 사용 중** ✅
- [x] 전체 데이터셋 제출 → **`cascaded_reranking_v1_full_submission_20251124_201646.csv`**

### Phase 3: BGE-M3 Fine-tuning (선택)
- [ ] create_training_data.py 작성
- [ ] Pseudo-labeling으로 학습 데이터 생성
- [ ] finetune_bgem3.py 작성
- [ ] GPU 환경 확보
- [ ] Fine-tuning 실행 (2~4시간)
- [ ] 임베딩 재생성
- [ ] 전체 데이터셋 제출

### 추가 최적화 (선택)
- [ ] Prompt Engineering
- [ ] Query Expansion
- [ ] Semantic Chunking 재시도

---

## 🎯 핵심 메시지

**0.8030 → 0.9 달성은 충분히 가능합니다!**

### 성공 확률

| 시나리오 | 예상 점수 | 확률 |
|---------|----------|------|
| Phase 1만 완료 | **0.82~0.84** | 90% |
| Phase 1+2 완료 | **0.85~0.86** | 80% |
| Phase 1+2+3 완료 | **0.89~0.90** ✅ | 70% |

### 성공 요인

1. **BM25 파라미터가 최적화되지 않음** (가장 쉬운 개선)
2. **Hybrid Weight가 조정되지 않음** (빠른 개선)
3. **BGE-M3이 과학 도메인 학습 안 됨** (큰 개선 여지)

### 리스크

1. **Phase 3 GPU 필요**: Colab Pro 또는 AWS 사용
2. **Fine-tuning 실패 가능성**: Hyperparameter 조정 필요
3. **Overfitting 위험**: Ultra Validation Set 크기 작음 (8개)

---

## 📞 즉시 실행 가능한 작업

### 1. BM25 파라미터 튜닝 (오늘 시작 가능)

```bash
cd code
python3 bm25_parameter_tuning.py
```

**예상 소요 시간**: 1~2시간
**예상 성능 향상**: +2~5% (0.8030 → 0.82~0.84)

### 2. 문서 읽기

**필수 문서**:
- [EXPERIMENT_SUMMARY_20251124.md](code/EXPERIMENT_SUMMARY_20251124.md) - 전체 실험 과정
- [code/docs/TASK5_FAILURE_ANALYSIS.md](code/docs/TASK5_FAILURE_ANALYSIS.md) - 실패 분석

**참고 문서**:
- [Elasticsearch BM25 Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)
- [BGE-M3 Fine-tuning Guide](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)

---

## 🔬 실험 우선순위

| 순위 | 실험 | 난이도 | 예상 향상 | ROI |
|------|------|--------|----------|-----|
| 1 | BM25 파라미터 튜닝 | 🟢 LOW | +2~5% | ⭐⭐⭐⭐⭐ |
| 2 | Hybrid Weight 최적화 | 🟢 LOW | +1~3% | ⭐⭐⭐⭐ |
| 3 | Prompt Engineering | 🟡 MED | +1~2% | ⭐⭐⭐ |
| 4 | Query Expansion | 🟡 MED | +1~2% | ⭐⭐⭐ |
| 5 | BGE-M3 Fine-tuning | 🔴 HIGH | +3~7% | ⭐⭐⭐⭐⭐ |
| 6 | Semantic Chunking | 🔴 HIGH | +2~4% | ⭐⭐ |

**추천 순서**: 1 → 2 → 5 (Phase 1 → 2 → 3)

---

## 📅 타임라인

### Week 1-2 (현재)
- [x] Task 7 완료: MAP@3 0.8030 달성
- [x] 실험 결과 문서화
- [x] GitHub 푸시 완료
- [ ] BM25 파라미터 튜닝

### Week 3-4
- [ ] Hybrid Weight 최적화
- [ ] Prompt Engineering
- [ ] Query Expansion

### Week 5-6 (선택)
- [ ] BGE-M3 Fine-tuning
- [ ] Semantic Chunking 재시도
- [ ] 앙상블 방법 시도

---

**최종 업데이트**: 2025-11-24
**현재 최고 성능**: MAP@3 **0.8030** 🏆
**다음 마일스톤**: MAP@3 **0.85** (Phase 1+2 완료)
