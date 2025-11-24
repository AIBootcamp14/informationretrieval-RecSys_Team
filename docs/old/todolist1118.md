# RAG 시스템 개선 TO DO LIST

**작성일**: 2025-11-18
**프로젝트**: 한국어 과학 상식 RAG 시스템 (Elasticsearch + Upstage Solar API)

---

## 📋 Phase 1: 환경 설정 및 기본 실행

### 🎯 High Priority

- [x] **Upstage API 키 설정**
  - `.env` 파일에 `UPSTAGE_API_KEY` 설정 완료
  - `solar-pro-2` 모델 사용으로 변경 완료

- [ ] **Elasticsearch 설치 및 설정**
  - `install_elasticsearch.sh` 실행
  - 생성된 비밀번호를 `.env` 파일의 `ELASTICSEARCH_PASSWORD`에 입력
  - 참고: [code/README.md](../code/README.md)

- [ ] **의존성 패키지 설치**
  ```bash
  cd /Users/dongjunekim/dev_team/ai14/ir/code
  pip install -r requirements.txt
  ```
  - 필요 패키지: `sentence_transformers`, `elasticsearch`, `openai`, `python-dotenv`

- [ ] **Baseline 코드 실행 테스트**
  ```bash
  python rag_with_elasticsearch.py
  ```
  - [ ] 4,272개 문서 인덱싱 확인
  - [ ] 220개 평가 쿼리 처리 확인
  - [ ] `sample_submission.csv` 출력 확인

---

## 🔍 Phase 2: 시스템 개선 - 검색 성능 향상

### 🎯 High Priority

- [ ] **Hybrid Search 구현**
  - 현재: `sparse_retrieve`만 사용 중 ([rag_with_elasticsearch.py:244](../code/rag_with_elasticsearch.py#L244))
  - 목표: Sparse (역색인) + Dense (벡터) 검색 결합
  - 구현 방법:
    ```python
    sparse_results = sparse_retrieve(query, k)
    dense_results = dense_retrieve(query, k)
    # 두 결과를 가중합하여 최종 Top-K 선정
    ```
  - 가중치 조정 실험 필요 (예: 0.3 sparse + 0.7 dense)

### 🎯 Medium Priority

- [ ] **임베딩 모델 실험**
  - 현재: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
  - 대안 모델 후보:
    - [ ] Upstage `solar-embedding-1-large-query` (쿼리용)
    - [ ] Upstage `solar-embedding-1-large-passage` (문서용)
    - [ ] `jhgan/ko-sroberta-multitask`
    - [ ] `BM-K/KoSimCSE-roberta`
  - 성능 비교 후 최적 모델 선택

- [ ] **검색 파라미터 튜닝**
  - [ ] Top-K 개수 조정 (현재 3개)
    - 3, 5, 10 실험
  - [ ] KNN의 `num_candidates` 조정 (현재 100)
    - 50, 100, 200, 500 실험
  - [ ] Similarity metric 실험
    - 현재: L2 norm
    - 대안: cosine similarity, dot product

---

## 📊 Phase 3: 데이터 증강 (학습 데이터 활용)

### 참고 문서
- [how_to_use_training_data.md](how_to_use_training_data.md)

### 🎯 Medium Priority

- [ ] **ICT (Inverse Cloze Task) 데이터 생성**
  - 4,272개 문서에서 문장 분리
  - 각 문장을 pseudo-query로 사용
  - 나머지 문장을 evidence로 활용
  - Negative samples 추가 (다른 문서에서)
  - 출력: Query-Evidence 페어 데이터셋
  - 참고: https://arxiv.org/pdf/1906.00300.pdf

- [ ] **Question Generation (QG) 구현**
  - 방법 1: PORORO 라이브러리 활용
    ```python
    from pororo import Pororo
    qg = Pororo(task="qg", lang="ko")
    ```
  - 방법 2: Upstage Solar 모델로 질문 생성
    ```python
    # 프롬프트: "다음 문서를 읽고 관련 질문을 생성하세요"
    ```
  - [ ] 문서 → 질문 페어 자동 생성
  - [ ] 생성된 질문 품질 평가
  - [ ] 자연스러운 질문 필터링

### 🎯 Low Priority

- [ ] **생성된 데이터로 임베딩 모델 파인튜닝**
  - ICT/QG 데이터로 Retrieval 모델 학습
  - Contrastive Learning 적용
  - 학습된 모델로 문서 재임베딩
  - 성능 비교 평가

---

## 🤖 Phase 4: LLM 프롬프트 최적화

### 🎯 High Priority

- [ ] **Function Calling 프롬프트 개선**
  - 현재 프롬프트: [rag_with_elasticsearch.py:186-192](../code/rag_with_elasticsearch.py#L186-L192)
  - 개선 목표:
    - [ ] 과학 질문 vs 일반 대화 구분 정확도 향상
    - [ ] Few-shot examples 추가
    - [ ] 더 명확한 지시문 작성
  - 테스트:
    - "안녕 반갑다" → 검색 안 함 (정답)
    - "금성이 왜 밝아?" → 검색 수행 (정답)

### 🎯 Medium Priority

- [ ] **QA 프롬프트 개선**
  - 현재 프롬프트: [rag_with_elasticsearch.py:176-183](../code/rag_with_elasticsearch.py#L176-L183)
  - 개선 방향:
    - [ ] 검색된 문서 활용도 향상
    - [ ] 답변 품질 개선 (더 자세하고 정확하게)
    - [ ] 정보 부족 시 명확한 안내
    - [ ] Chain-of-Thought 프롬프팅 적용

- [ ] **멀티턴 대화 처리 개선**
  - 평가 데이터 중 20개는 멀티턴 대화
  - 대화 맥락을 고려한 standalone query 생성 개선
  - 예시:
    ```
    User: 기억 상실증 걸리면 너무 무섭겠다.
    Assistant: 네 맞습니다.
    User: 어떤 원인 때문에 발생하는지 궁금해.
    ```
    → "기억 상실증의 원인"으로 쿼리 변환

---

## 📈 Phase 5: 평가 및 성능 측정

### 참고 문서
- [rag_metric_overview.md](rag_metric_overview.md)

### 🎯 High Priority

- [ ] **MAP (Mean Average Precision) 평가 구현**
  - 문서 추출 정확도 측정
  - 변형된 MAP 지표 적용:
    - 문서 추출 불필요 시 + 추출 안함 = MAP 1
    - 문서 추출 불필요 시 + 추출함 = MAP 0
  - 복수 정답 문서 처리 평가

- [ ] **일반 대화 필터링 성능 평가**
  - 평가 데이터 중 20개는 일반 대화 (과학 지식 불필요)
  - 이 경우 문서 추출하지 않아야 함
  - 잘못 추출 시 패널티 받음
  - 예시: "안녕 반갑다", "너 정말 똑똑하다!", "너 잘하는 게 뭐야?"

### 🎯 Medium Priority

- [ ] **오류 분석 및 개선**
  - [ ] 실패 케이스 분석
    - 검색 실패: 관련 문서를 못 찾은 경우
    - 오검색: 관련 없는 문서를 가져온 경우
    - 답변 실패: 문서는 맞지만 답변이 부정확한 경우
  - [ ] 검색 실패 원인 파악
  - [ ] 답변 생성 품질 평가

- [ ] **성능 벤치마크**
  - [ ] Baseline 성능 기록
  - [ ] 각 개선사항별 성능 변화 추적
  - [ ] 최종 성능 비교표 작성

---

## 🚀 Phase 6: 고급 최적화

### 🎯 Low Priority

- [ ] **Reranking 모델 추가**
  - 검색 결과 재순위화
  - Cross-encoder 모델 활용
  - 후보 모델:
    - `cross-encoder/ms-marco-MiniLM-L-12-v2`
    - Korean Cross-encoder 모델

- [ ] **Query Expansion**
  - 쿼리 확장 기법 적용
  - 동의어, 관련어 추가
  - LLM 활용한 쿼리 재작성

- [ ] **Chunk 전략 개선**
  - 문서 분할 방식 최적화
  - 현재: 문서 전체를 하나의 단위로 사용
  - 개선: 문서를 여러 청크로 분할
  - Overlap 전략 실험

- [ ] **캐싱 및 성능 최적화**
  - 임베딩 결과 캐싱
  - Elasticsearch 인덱스 최적화
  - 배치 처리 최적화
  - 병렬 처리 적용

---

## 📝 Phase 7: 제출 준비

### 🎯 High Priority

- [ ] **최종 테스트**
  - [ ] 전체 평가 데이터 (220개) 처리 확인
  - [ ] 출력 포맷 검증
    ```jsonl
    {"eval_id": 0, "standalone_query": "...", "topk": ["docid1", "docid2", "docid3"], "answer": "...", "references": [...]}
    ```
  - [ ] 모든 eval_id가 포함되었는지 확인

- [ ] **결과 분석**
  - [ ] `sample_submission.csv` 검증
  - [ ] 각 eval_id별 결과 확인
  - [ ] topk 문서 관련도 체크
  - [ ] standalone_query 품질 체크
  - [ ] answer 품질 체크

- [ ] **문서화**
  - [ ] 구현 방법 정리
  - [ ] 실험 결과 기록
  - [ ] 성능 개선 내역 문서화
  - [ ] README 업데이트

---

## 🎯 즉시 착수 가능한 우선순위 작업

### ✅ **Completed**
1. ✅ Upstage API 키 설정 완료
2. ✅ `.env` 파일로 소프트코딩 완료
3. ✅ `solar-pro-2` 모델로 변경 완료
4. ✅ Hybrid Search 구현 완료 (Sparse + Dense)

### 🔥 **Next Steps (즉시 시작)**
1. Elasticsearch 설치 및 비밀번호 설정
2. 의존성 패키지 설치
3. Baseline 코드 실행 테스트

### 📊 **Medium Term (다음 단계)**
5. 임베딩 모델 실험
6. Function Calling 프롬프트 개선
7. MAP 평가 구현
8. 일반 대화 필터링 성능 개선

### 🔬 **Long Term (시간이 있을 때)**
9. ICT/QG 데이터 생성
10. 임베딩 모델 파인튜닝
11. Reranking 모델 추가
12. Query Expansion 및 고급 최적화

---

## 📚 참고 자료

- [data_overview.md](data_overview.md) - 데이터 구조 및 평가 방식
- [how_to_use_training_data.md](how_to_use_training_data.md) - ICT, QG 데이터 생성 방법
- [rag_metric_overview.md](rag_metric_overview.md) - MAP 평가 지표 설명
- [code/README.md](../code/README.md) - Baseline 실행 방법
- [code/rag_with_elasticsearch.py](../code/rag_with_elasticsearch.py) - 메인 코드

---

## 💡 추가 아이디어

- [ ] Elasticsearch의 `function_score`를 활용한 스코어 조정
- [ ] BM25 파라미터 튜닝 (k1, b 값)
- [ ] 문서별 가중치 부여 (출처별 신뢰도)
- [ ] 앙상블 방식: 여러 검색 전략 결합
- [ ] A/B 테스트: 다양한 설정 비교
- [ ] 로깅 및 모니터링 시스템 구축

---

**최종 업데이트**: 2025-11-18
**작성자**: AI14 Team
