"""
LLM 최대 활용 솔루션 - Maximum LLM Usage Strategy

핵심 전략: 모든 단계에서 LLM 적극 활용
1. LLM Intent Classification (Smalltalk 판별)
2. LLM Context-Aware Query Rewriting (멀티턴 대화)
3. LLM Query Enhancement (과학 용어 확장)
4. BM25 검색
5. LLM Document Relevance Scoring (관련도 점수화)
6. LLM Reranking (순위 조정)
7. LLM Final Validation (최종 검증)

목표: LLM 파워로 0.90+ MAP 달성
"""

import json
import os
from elasticsearch import Elasticsearch
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Solar API 클라이언트
upstage_api_key = os.getenv("UPSTAGE_API_KEY")
client = OpenAI(
    base_url="https://api.upstage.ai/v1/solar",
    api_key=upstage_api_key
)

def llm_intent_classification(msg):
    """
    LLM으로 질문 의도 분류

    Returns:
        'science': 과학 질문 (검색 필요)
        'smalltalk': 일반 대화 (검색 불필요)
    """
    # 쿼리 추출
    if isinstance(msg, list):
        query = msg[-1]['content']
        # 멀티턴인 경우 이전 맥락도 포함
        context = "\n".join([f"{m['role']}: {m['content']}" for m in msg[:-1]]) if len(msg) > 1 else ""
    else:
        query = msg
        context = ""

    prompt = f"""다음 질문이 과학 지식 검색이 필요한 질문인지 판단하세요.

{"이전 대화:" + context if context else ""}

질문: {query}

판단 기준:
- 과학 질문: 생물, 화학, 물리, 천문, 지구과학 등 과학 지식이 필요한 질문
- 일반 대화: 인사, 날씨, 감정, 일상 대화, 의견 묻기 등

출력: 'science' 또는 'smalltalk' 중 하나만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",  # 더 정확한 판단을 위해 pro 모델
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip().lower()

        if 'science' in result:
            return 'science'
        elif 'smalltalk' in result:
            return 'smalltalk'
        else:
            # 불명확한 경우 안전하게 science로 처리
            return 'science'

    except Exception as e:
        print(f"⚠️ Intent Classification 실패: {e}")
        # 실패 시 안전하게 science로 처리
        return 'science'

def llm_context_aware_rewriting(msg):
    """
    LLM으로 멀티턴 대화 맥락 통합하여 쿼리 재작성
    """
    if isinstance(msg, str):
        # 단일턴은 그대로 반환
        return msg

    if len(msg) == 1:
        return msg[0]['content']

    # 멀티턴 대화 처리
    current_query = msg[-1]['content']

    # 모호한 표현 확인
    ambiguous_terms = ['그 ', '그것', '이것', '이거', '저것', '저거', '왜', '어떻게', '이유', '방법']

    if not any(term in current_query for term in ambiguous_terms):
        return current_query

    # LLM으로 맥락 통합
    conversation_context = "\n".join([
        f"{m['role']}: {m['content']}" for m in msg[:-1]
    ])

    prompt = f"""다음은 이전 대화입니다:

{conversation_context}

현재 질문: {current_query}

이 질문을 이전 대화 맥락을 반영하여 독립적으로 이해 가능한 완전한 과학 질문으로 재작성하세요.
대명사(그것, 이것 등)를 구체적인 과학 용어로 바꾸세요.

재작성된 질문만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )

        rewritten = response.choices[0].message.content.strip()
        rewritten = rewritten.strip('"').strip("'")

        print(f"  [Context-Aware] {current_query} → {rewritten}")
        return rewritten

    except Exception as e:
        print(f"⚠️ Context Rewriting 실패: {e}")
        return current_query

def llm_query_enhancement(query):
    """
    LLM으로 쿼리 개선 및 확장

    핵심: 과학 용어 명확화 + 동의어 확장 + 관련 키워드 추가
    """
    prompt = f"""과학 지식 검색을 위한 쿼리 개선 및 확장:

원본 질문: {query}

다음 기준으로 검색 쿼리를 개선하고 확장하세요:
1. 핵심 과학 개념을 명확히
2. 동의어 추가 (한글 + 영어 + 한자어)
3. 관련 과학 용어 확장
4. 검색에 도움되는 키워드 추가
5. 불필요한 조사 제거

예시:
- "DNA 조각 결합하는 거" → "DNA 조각 연결 결합 효소 ligase 리가아제 리가제 유전공학 재조합"
- "식물 광합성 어떻게" → "식물 광합성 과정 메커니즘 엽록소 chloroplast 엽록체 명반응 암반응"
- "달 같은면" → "달 자전 공전 동기화 조석잠금 tidal locking 항상 같은면"

출력: 개선 및 확장된 검색 쿼리만 한 줄로 작성"""

    try:
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,  # 약간 높여서 다양한 키워드 생성
            max_tokens=150
        )

        enhanced = response.choices[0].message.content.strip()
        enhanced = enhanced.replace('"', '').replace("'", '').strip()

        print(f"  [Query Enhancement] {query[:30]}... → {enhanced[:50]}...")
        return enhanced

    except Exception as e:
        print(f"⚠️ Query Enhancement 실패: {e}")
        return query

def search_bm25(query, top_k=5):
    """BM25 검색 (더 많이 가져와서 LLM으로 필터링)"""
    response = es.search(
        index='test',
        body={
            'query': {
                'match': {
                    'content': {
                        'query': query,
                        'analyzer': 'nori'
                    }
                }
            },
            'size': top_k
        }
    )

    if not response['hits']['hits']:
        return [], 0.0

    results = []
    for hit in response['hits']['hits']:
        results.append({
            'docid': hit['_source']['docid'],
            'content': hit['_source']['content'][:1000],  # 더 많은 내용
            'bm25_score': hit['_score']
        })

    max_score = response['hits']['hits'][0]['_score']
    return results, max_score

def llm_document_relevance_scoring(query, documents):
    """
    LLM으로 각 문서의 관련도 점수화

    Returns:
        List of (docid, llm_score) tuples
    """
    if not documents:
        return []

    # 문서를 LLM에게 보여주고 관련도 점수 요청
    docs_text = ""
    for i, doc in enumerate(documents, 1):
        docs_text += f"\n[문서 {i}] (BM25: {doc['bm25_score']:.2f})\n{doc['content'][:800]}\n" + "-"*60

    prompt = f"""질문: {query}

다음은 BM25 검색으로 찾은 문서들입니다. 각 문서가 질문에 얼마나 관련이 있는지 0~100점으로 점수를 매기세요.

{docs_text}

점수 기준:
- 90-100: 질문에 직접적으로 답변하는 핵심 내용
- 70-89: 질문과 관련성 높음
- 50-69: 질문과 어느 정도 관련
- 30-49: 약간 관련
- 0-29: 거의 무관

출력 형식: 문서번호:점수 (예: 1:95, 2:80, 3:65, 4:45, 5:20)
점수만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",  # 정확한 판단을 위해 pro 모델
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()

        # 파싱: "1:95, 2:80, 3:65" → [(docid1, 95), (docid2, 80), ...]
        scores = []
        for pair in result.replace(' ', '').split(','):
            try:
                idx_str, score_str = pair.split(':')
                idx = int(idx_str)
                score = int(score_str)

                if 1 <= idx <= len(documents) and 0 <= score <= 100:
                    scores.append((documents[idx-1]['docid'], score, documents[idx-1]['bm25_score']))
            except:
                continue

        print(f"  [LLM Scoring] {len(scores)}개 문서 점수화 완료")
        return scores

    except Exception as e:
        print(f"⚠️ Document Scoring 실패: {e}")
        # 실패 시 BM25 점수 기반으로 반환
        return [(doc['docid'], 70, doc['bm25_score']) for doc in documents[:3]]

def llm_final_reranking(query, scored_documents):
    """
    LLM으로 최종 Top-3 선택 및 순위 조정

    scored_documents: List of (docid, llm_score, bm25_score)
    """
    if not scored_documents:
        return []

    # LLM 점수 높은 순으로 정렬
    scored_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)

    # 점수 70 이상인 문서만 선택
    high_score_docs = [doc for doc in scored_documents if doc[1] >= 70]

    if len(high_score_docs) == 0:
        # 70점 이상 없으면 빈 리스트 반환
        return []

    # Top-3 선택
    top3 = high_score_docs[:3]

    print(f"  [Final Selection] Top-3 선택: {[(d[0][:8], d[1]) for d in top3]}")

    return [doc[0] for doc in top3]

def llm_maximum_search(query, msg=None):
    """
    LLM 최대 활용 검색 파이프라인

    모든 단계에서 LLM 활용:
    1. Intent Classification
    2. Context-Aware Rewriting
    3. Query Enhancement
    4. BM25 Search
    5. Document Relevance Scoring
    6. Final Reranking
    """
    print(f"\n{'='*80}")
    print(f"Query: {query if isinstance(query, str) else query[-1]['content'] if isinstance(query, list) else query}")
    print(f"{'='*80}")

    # 1단계: Intent Classification
    intent = llm_intent_classification(msg if msg else query)
    print(f"  [1] Intent: {intent}")

    if intent == 'smalltalk':
        print(f"  → Smalltalk 감지, 검색 생략")
        return [], 0.0

    # 2단계: Context-Aware Rewriting (멀티턴인 경우)
    if isinstance(msg, list) and len(msg) > 1:
        query = llm_context_aware_rewriting(msg)
        print(f"  [2] Context-Aware Rewritten")
    else:
        query = query if isinstance(query, str) else (msg[0]['content'] if isinstance(msg, list) else query)
        print(f"  [2] Single-turn, skip context rewriting")

    # 3단계: Query Enhancement
    enhanced_query = llm_query_enhancement(query)
    print(f"  [3] Query Enhanced")

    # 4단계: BM25 Search (Top-5)
    combined_query = f"{enhanced_query} {query}"
    top5_docs, max_score = search_bm25(combined_query, top_k=5)
    print(f"  [4] BM25 Search: {len(top5_docs)}개 문서, max_score={max_score:.2f}")

    if not top5_docs:
        # 개선 쿼리로 실패 시 원본 시도
        top5_docs, max_score = search_bm25(query, top_k=5)
        print(f"      Retry with original query: {len(top5_docs)}개 문서")

    if not top5_docs or max_score < 2.0:
        print(f"  → 검색 결과 없음 또는 낮은 점수")
        return [], max_score

    # 5단계: LLM Document Relevance Scoring
    scored_docs = llm_document_relevance_scoring(query, top5_docs)
    print(f"  [5] Document Scoring: {len(scored_docs)}개 점수화")

    if not scored_docs:
        print(f"  → 점수화 실패")
        return [], max_score

    # 6단계: Final Reranking (Top-3 선택)
    final_topk = llm_final_reranking(query, scored_docs)
    print(f"  [6] Final Top-{len(final_topk)}: {[d[:8]+'...' for d in final_topk]}")

    return final_topk, max_score

def process_with_llm_maximum(eval_path, output_path):
    """LLM 최대 활용 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"LLM 최대 활용 RAG 실행")
    print(f"전략: 모든 단계에서 LLM 적극 활용")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc="LLM Maximum"):
        eval_id = item['eval_id']
        msg = item['msg']

        # LLM Maximum Search
        topk_docs, max_score = llm_maximum_search(msg, msg)

        results.append({
            'eval_id': eval_id,
            'topk': topk_docs
        })

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 통계
    topk_counts = {}
    for r in results:
        count = len(r['topk'])
        topk_counts[count] = topk_counts.get(count, 0) + 1

    print(f"\n✅ 완료: {output_path}")
    print(f"\nTopK 분포:")
    for k in sorted(topk_counts.keys()):
        print(f"  TopK={k}: {topk_counts[k]:3d}개 ({topk_counts[k]/len(results)*100:5.1f}%)")
    print(f"{'='*80}\n")

def main():
    print("=" * 80)
    print("LLM 최대 활용 솔루션")
    print("모든 단계에서 LLM 파워 극대화")
    print("=" * 80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    if not upstage_api_key:
        print("❌ UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    print("✅ Upstage Solar API Key 확인")

    # 처리 실행
    process_with_llm_maximum(
        eval_path='../data/eval.jsonl',
        output_path='llm_maximum_submission.csv'
    )

    print(f"\n{'='*80}")
    print(f"✅ LLM 최대 활용 완료")
    print(f"{'='*80}")
    print(f"\n생성된 파일:")
    print(f"  - llm_maximum_submission.csv")
    print(f"\nLLM 활용 단계:")
    print(f"  1. Intent Classification (solar-pro)")
    print(f"  2. Context-Aware Rewriting (solar-pro)")
    print(f"  3. Query Enhancement (solar-mini)")
    print(f"  4. BM25 Search")
    print(f"  5. Document Relevance Scoring (solar-pro)")
    print(f"  6. Final Reranking")
    print(f"\n기대 효과:")
    print(f"  - LLM의 언어 이해력 최대 활용")
    print(f"  - 정교한 의도 분류 및 문서 선택")
    print(f"  - 예상 MAP: 0.80~0.90")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
