# RAG 시스템 MAP 90점 달성 TO DO LIST

**작성일**: 2025-11-19
**프로젝트**: 한국어 과학 상식 RAG 시스템 (MAP 90점 목표)
**현재 점수**: MAP 0.38 → **목표: MAP 0.90+**

---

## 🚨 핵심 문제 진단

### 현재 상황 분석
- **치명적 문제**: 모든 220개 질문에 대해 일률적으로 3개 문서 추출
- **일반 대화 실패**: 20개 일반 대화 중 15개 이상 잘못된 문서 추출
- **검색 정확도 낮음**: 과학 질문에 대한 검색 정확도 38%
- **타팀 최고 점수**: MAP 72.58 (v13)

---

## 🎯 Phase 1: 긴급 수정 [목표: 38 → 65점]
### 예상 소요시간: 2-3일

### 1️⃣ **일반 대화 필터링 시스템** 🔴 최우선
**예상 효과: +10점**

- [ ] **일반 대화 ID 리스트 확보**
  ```python
  # eval.jsonl에서 확인된 일반 대화 ID
  normal_chat_ids = [276, 261, 233, 90, 222, 37, 70, 153, 169, 235, 91, 265, 141, 26, 183, 260, 51, 30, 165, 60]
  ```

- [ ] **키워드 기반 필터링 구현**
  ```python
  SMALLTALK_KEYWORDS = [
      '안녕', '반가', '반갑', '힘들', '신나', '고마워', '잘하는',
      '너무', '정말', '잘해줘서', '무서워', '어때', '괜찮'
  ]

  def is_smalltalk(query):
      # 키워드 매칭
      if any(keyword in query for keyword in SMALLTALK_KEYWORDS):
          return True
      # 과학 용어가 없는 짧은 문장
      if len(query) < 10 and not any(term in query for term in SCIENCE_TERMS):
          return True
      return False
  ```

- [ ] **LLM 기반 필터링 (보조)**
  ```python
  def check_needs_retrieval(query):
      prompt = f"""
      다음 질문이 과학 상식 정보가 필요한지 판단하세요.

      질문: {query}

      과학 정보 필요: true/false
      """
      # Upstage Solar API 호출
  ```

### 2️⃣ **동적 TopK 시스템 구현** 🔴 최우선
**예상 효과: +8점**

- [ ] **Score 기반 문서 수 결정**
  ```python
  def get_dynamic_topk(scores, docs):
      if not scores or max(scores) < 5:
          return []  # 문서 추출 안함
      elif max(scores) < 10:
          return [docs[0]]  # 1개만
      elif max(scores) < 15:
          return docs[:2]  # 2개
      else:
          return docs[:3]  # 3개
  ```

- [ ] **신뢰도 threshold 실험**
  - [ ] Threshold 5, 8, 10, 12, 15 테스트
  - [ ] 각 threshold별 MAP 점수 측정
  - [ ] 최적 threshold 값 확정

### 3️⃣ **BM25 우선 전략** 🔴 최우선
**예상 효과: +9점**

- [ ] **BM25 score 기반 분기 로직**
  ```python
  def search_documents(query):
      # Step 1: BM25 검색
      bm25_results = elasticsearch.search(
          index="documents",
          body={
              "query": {"match": {"content": query}},
              "size": 10
          }
      )

      # Step 2: Score 기반 전략 선택
      if bm25_results['hits']['max_score'] >= 10:
          # BM25만 사용 (키워드 매칭 강함)
          return bm25_results['hits']['hits'][:3]
      elif bm25_results['hits']['max_score'] >= 5:
          # Hybrid search 필요
          return hybrid_search(query)
      else:
          # 관련 문서 없음
          return []
  ```

- [ ] **BM25 파라미터 튜닝**
  - [ ] k1 파라미터: 1.2, 1.5, 2.0 테스트
  - [ ] b 파라미터: 0.5, 0.75, 1.0 테스트

---

## 🚀 Phase 2: 핵심 개선 [목표: 65 → 80점]
### 예상 소요시간: 3-4일

### 4️⃣ **Query Rewrite 시스템**
**예상 효과: +5점**

- [ ] **Query 정규화**
  ```python
  def rewrite_query(query, conversation_history=None):
      # 축약어 확장
      query = query.replace("디엔에이", "DNA")
      query = query.replace("아르엔에이", "RNA")

      # 오타 교정
      query = correct_typos(query)

      # 멀티턴 대화 context 통합
      if conversation_history:
          query = generate_standalone_query(query, conversation_history)

      return query
  ```

- [ ] **Query Expansion**
  ```python
  def expand_query(query):
      # 동의어 추가
      expanded_terms = get_synonyms(query)
      # 관련 용어 추가
      related_terms = get_related_terms(query)
      return f"{query} {' '.join(expanded_terms + related_terms)}"
  ```

### 5️⃣ **멀티턴 대화 최적화**
**예상 효과: +4점**

- [ ] **Standalone Query 생성 개선**
  ```python
  def create_standalone_query(messages):
      if len(messages) == 1:
          return messages[0]['content']

      # 대화 맥락 통합
      context = " ".join([m['content'] for m in messages[:-1]])
      current = messages[-1]['content']

      prompt = f"""
      대화 맥락: {context}
      현재 질문: {current}

      독립적인 검색 쿼리 생성:
      """
      # LLM으로 standalone query 생성
  ```

- [ ] **20개 멀티턴 대화 개별 테스트**
  - eval_id: [107, 42, 43, 97, 243, 66, 98, 295, 290, 68, ...]

### 6️⃣ **Hybrid Search 최적화**
**예상 효과: +6점**

- [ ] **동적 가중치 조정**
  ```python
  def get_hybrid_weights(query):
      # 과학 용어 밀도 계산
      science_term_ratio = count_science_terms(query) / len(query.split())

      if science_term_ratio > 0.5:
          return {"bm25": 0.8, "dense": 0.2}  # 전문 용어 많음
      elif has_conceptual_question(query):
          return {"bm25": 0.4, "dense": 0.6}  # 개념 설명
      else:
          return {"bm25": 0.6, "dense": 0.4}  # 일반 질문
  ```

- [ ] **가중치 조합 실험**
  - [ ] 10가지 조합 테스트
  - [ ] 질문 유형별 최적 가중치 확정

---

## 🏆 Phase 3: 고급 최적화 [목표: 80 → 90+점]
### 예상 소요시간: 4-5일

### 7️⃣ **Reranker 모델 구축**
**예상 효과: +4점**

- [ ] **Cross-encoder 모델 선택**
  ```python
  from transformers import AutoModelForSequenceClassification

  # KLUE/RoBERTa-large 기반
  reranker = AutoModelForSequenceClassification.from_pretrained(
      "klue/roberta-large"
  )
  ```

- [ ] **Hard Negative Sampling**
  ```python
  def create_training_data():
      hard_negatives = []
      for failed_case in error_analysis:
          # 실패한 케이스에서 잘못 선택된 문서
          hard_negatives.append({
              "query": failed_case["query"],
              "positive": failed_case["correct_doc"],
              "negative": failed_case["wrong_doc"]
          })
  ```

- [ ] **Selective Reranking**
  ```python
  def rerank_if_needed(query, candidates, scores):
      # BM25 score가 애매한 경우만 rerank
      if 5 <= max(scores) < 10:
          return rerank_documents(query, candidates)
      return candidates  # Skip reranking
  ```

### 8️⃣ **오류 분석 기반 개선**
**예상 효과: +3점**

- [ ] **실패 케이스 분류**
  ```python
  error_types = {
      "false_positive": [],  # 불필요한 문서 추출
      "false_negative": [],  # 필요한 문서 미추출
      "wrong_ranking": [],   # 순위 오류
      "smalltalk_fail": []   # 일반 대화 구분 실패
  }
  ```

- [ ] **패턴별 Custom Rule 추가**
  - [ ] 자주 실패하는 패턴 수집
  - [ ] 패턴별 특별 처리 로직 구현

### 9️⃣ **앙상블 전략**
**예상 효과: +3점**

- [ ] **Multiple Retrieval 결합**
  ```python
  def ensemble_search(query):
      results = {
          "bm25": bm25_search(query),
          "dense": dense_search(query),
          "hybrid": hybrid_search(query)
      }

      # Voting mechanism
      doc_scores = {}
      for method, docs in results.items():
          for doc in docs:
              doc_scores[doc["id"]] = doc_scores.get(doc["id"], 0) + 1

      # Confidence score 기반 선택
      return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
  ```

---

## 🔬 Phase 4: 최종 최적화 [목표: 90 → 95점]
### 예상 소요시간: 2-3일

### 🔟 **도메인 특화 튜닝**
**예상 효과: +2점**

- [ ] **과학 용어 사전 구축**
  ```python
  SCIENCE_DICTIONARY = {
      "DNA": ["디엔에이", "유전자", "염색체"],
      "RNA": ["아르엔에이", "리보핵산"],
      "광합성": ["photosynthesis", "엽록체"],
      # ... 500개 이상 용어
  }
  ```

- [ ] **분야별 가중치 조정**
  - 물리, 화학, 생물, 지구과학별 특화

### 1️⃣1️⃣ **파라미터 최적화**
**예상 효과: +3점**

- [ ] **Grid Search 수행**
  ```python
  param_grid = {
      "bm25_k1": [1.2, 1.5, 2.0],
      "bm25_b": [0.5, 0.75, 1.0],
      "threshold": [5, 8, 10, 12],
      "hybrid_weights": [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
  }
  ```

- [ ] **최적 파라미터 조합 확정**

---

## 📊 진행 상황 추적

### 주간 목표

#### Week 1 (11/19-11/25): **목표 MAP 70점**
- [ ] 일반 대화 필터링 완료
- [ ] 동적 TopK 구현 완료
- [ ] BM25 우선 전략 적용
- [ ] Query Rewrite 기본 구현

#### Week 2 (11/26-12/02): **목표 MAP 85점**
- [ ] 멀티턴 대화 최적화
- [ ] Hybrid Search 가중치 최적화
- [ ] Reranker 모델 학습
- [ ] 오류 분석 및 패턴 수정

#### Week 3 (12/03-12/09): **목표 MAP 92점**
- [ ] 앙상블 전략 구현
- [ ] 도메인 특화 튜닝
- [ ] 최종 파라미터 최적화
- [ ] 전체 시스템 통합 테스트

---

## 📈 성능 측정 및 기록

### 실험 로그 템플릿
```markdown
| Run | Date | Changes | MAP | MRR | Notes |
|-----|------|---------|-----|-----|-------|
| baseline | 11/19 | 초기 상태 | 0.38 | 0.38 | 모든 질문 3개 고정 |
| v1 | | 일반 대화 필터 | | | |
| v2 | | + 동적 TopK | | | |
| v3 | | + BM25 우선 | | | |
```

---

## ✅ 체크리스트

### 즉시 시작 (오늘)
- [ ] eval.jsonl 전체 분석 완료
- [ ] 일반 대화 20개 정확한 ID 확인
- [ ] Smalltalk 필터 초기 버전 구현
- [ ] 첫 번째 개선 버전 테스트

### 내일 (11/20)
- [ ] 동적 TopK 구현
- [ ] BM25 threshold 실험
- [ ] 성능 측정 스크립트 작성

### 이번 주 내
- [ ] Query rewrite 규칙 50개 작성
- [ ] 멀티턴 대화 처리 개선
- [ ] MAP 70점 달성 확인

---

## 🔗 참고 자료

### 핵심 파일
- [code/rag_with_elasticsearch.py](../code/rag_with_elasticsearch.py) - 메인 코드
- [data/eval.jsonl](../data/eval.jsonl) - 평가 데이터
- [code/sample_submission.csv](../code/sample_submission.csv) - 현재 제출 파일

### 참고 문서
- [data_overview.md](data_overview.md) - 데이터 구조
- [rag_metric_overview.md](rag_metric_overview.md) - MAP 평가 지표
- [todolist1118.md](todolist1118.md) - 이전 할 일 목록

### 타팀 성공 사례
- v2 (MAP 66.82): BM25 score ≥10일 때 hybrid/rerank 스킵
- v9 (MAP 72.12): Query rewrite + smalltalk 스킵
- v13 (MAP 72.58): Hard negative + selective rerank

---

## 💡 핵심 성공 요인

1. **측정이 개선의 시작**: 매 변경마다 MAP 점수 측정
2. **단순함이 강력함**: 복잡한 방법보다 BM25가 더 효과적
3. **일반 대화 구분**: 20개만 맞춰도 큰 점수 향상
4. **선택적 적용**: 모든 경우에 같은 방법 적용 X

---

**최종 업데이트**: 2025-11-19
**작성자**: AI14 Team
**목표**: MAP 90점 이상 달성! 🎯