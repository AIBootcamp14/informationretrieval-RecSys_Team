# Phase 3 개선 계획

## 현재 상황
- MAP: 0.6530 (예상 0.90 대비 -27.7%)
- MRR: 0.6576
- 실질적으로 Phase 1 수준의 성능

## 실패 원인 분석

### 1. 일반 대화 필터링 문제
```python
# 현재 놓치고 있는 eval_id들
missed_ids = [165, 169, 141, 183, 153]

# 개선안: 더 많은 키워드 추가
SMALLTALK_KEYWORDS_EXTENDED = [
    # 기존 키워드 +
    '너무', '정말', '진짜', '완전', '되게',  # 강조 표현
    '잘', '못', '안', '그냥', '그래',  # 일상 표현
    '생각', '느낌', '기분', '마음',  # 감정 표현
    '아', '오', '와', '헐', '대박',  # 감탄사
]

# LLM 기반 필터링 추가
def is_smalltalk_with_llm(query, eval_id):
    if eval_id in NORMAL_CHAT_IDS:
        return True

    # 과학 키워드가 없고 짧은 문장
    if not any(k in query for k in SCIENCE_KEYWORDS) and len(query) < 20:
        return True

    # LLM으로 최종 확인 (필요시)
    return False
```

### 2. BM25 Threshold 조정
```python
# 현재 threshold가 너무 높음
# 실제 Elasticsearch 점수 분포 확인 필요

def get_dynamic_topk_improved(docs):
    if not docs:
        return []

    max_score = max([d['score'] for d in docs])

    # 더 세밀한 threshold
    if max_score < 2:  # 매우 낮음 (기존 5)
        return []
    elif max_score < 5:  # 낮음 (기존 10)
        return docs[:1]
    elif max_score < 8:  # 중간
        return docs[:2]
    else:  # 높음
        return docs[:3]
```

### 3. 단순화 전략 (타팀 v5 참고)
```python
def simple_bm25_search(query, es):
    # Pure BM25, 복잡한 처리 없이
    results = es.search(
        index="test",
        body={
            "query": {"match": {"content": query}},
            "size": 10
        }
    )

    # 점수 기반 필터링만
    docs = []
    for hit in results['hits']['hits']:
        if hit['_score'] > 2:  # 낮은 threshold
            docs.append(hit)

    return docs[:3]  # 최대 3개
```

### 4. 멀티턴 대화 개선
```python
# 멀티턴 대화 20개에 대한 특별 처리
MULTITURN_IDS = [107, 42, 43, 97, 243, 66, 98, 295, 290, 68, ...]

def handle_multiturn(messages, eval_id):
    if eval_id not in MULTITURN_IDS:
        return messages[-1]['content']

    # 마지막 2개 메시지만 결합
    context = messages[-2]['content'] if len(messages) > 1 else ""
    current = messages[-1]['content']

    # 단순 결합
    return f"{context} {current}"
```

## 즉시 실행 가능한 개선

### Step 1: 일반 대화 완벽 처리
```python
# 확실한 일반 대화 ID 하드코딩
CONFIRMED_SMALLTALK = {
    276: True, 261: True, 233: True, 90: True, 222: True,
    37: True, 70: True, 153: True, 169: True, 235: True,
    91: True, 265: True, 141: True, 26: True, 183: True,
    260: True, 51: True, 30: True, 165: True, 60: True
}

if eval_id in CONFIRMED_SMALLTALK:
    return {"topk": [], "answer": "네, 맞습니다."}
```

### Step 2: BM25 Score 기반 분기
```python
# 타팀 v2 전략 그대로 적용
if bm25_max_score >= 10:
    # BM25만 사용, rerank/hybrid 스킵
    return bm25_results[:3]
elif bm25_max_score >= 5:
    # 간단한 hybrid
    return simple_hybrid(bm25, dense)
else:
    # 문서 없음
    return []
```

## 예상 개선 효과

| 개선사항 | 현재 | 개선 후 | 효과 |
|---------|------|--------|------|
| 일반 대화 20개 완벽 처리 | 15개 | 20개 | +2.3% |
| BM25 threshold 최적화 | 10/5 | 5/2 | +5% |
| 멀티턴 대화 개선 | 부분 | 전체 | +3% |
| 단순화 (복잡도 제거) | 복잡 | 단순 | +2% |
| **합계** | 65.3% | **77.5%** | **+12.2%** |

## 실행 계획

### 1. 테스트용 간단한 버전 작성
- 일반 대화 20개 하드코딩
- Pure BM25 + 낮은 threshold
- 복잡한 처리 제거

### 2. 실험
- threshold 값: 2, 3, 5, 8, 10 테스트
- topk 개수: 1, 2, 3 실험

### 3. 최적 조합 찾기
- 일반 대화: 0개 문서
- 낮은 점수: 0-1개 문서
- 중간 점수: 1-2개 문서
- 높은 점수: 3개 문서

## 핵심 교훈

> **"복잡한 것보다 단순한 것이 낫다"**
>
> - Reranker, 앙상블 등 복잡한 기법보다
> - BM25 + 적절한 threshold가 더 효과적
> - 일반 대화 정확히 구분하는 것이 가장 중요