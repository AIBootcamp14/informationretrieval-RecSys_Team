"""
Cascaded Reranking v1: 2-Stage LLM Reranking
- Stage 1: Top 30 → Top 10 (빠른 필터링)
- Stage 2: Top 10 → Top 3 (정밀한 Reranking)
- LLM 기반 일반대화 자동 분류 (하드코딩 제거)

목표: "Lost in the Middle" 문제 해결하며 Recall 향상
예상 성능: 0.81~0.82 MAP@3

업데이트 (2025-11-24):
- 하드코딩된 일반대화 ID 제거
- LLM 기반 하이브리드 분류로 변경 (규칙 + Solar Pro)
- 어떤 평가 데이터에도 대응 가능
"""

import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Solar API 초기화
upstage_api_key = os.environ.get('UPSTAGE_API_KEY')
client = None
if upstage_api_key:
    client = OpenAI(
        api_key=upstage_api_key,
        base_url="https://api.upstage.ai/v1/solar"
    )

# BGE-M3 모델 로드
print("BGE-M3 모델 로드 중...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ BGE-M3 모델 로드 완료")

# BGE-M3 최적화 임베딩 로드
print("\nBGE-M3 최적화 임베딩 로드 중...")
with open('embeddings_test_bgem3_optimized.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print(f"✅ {len(embeddings_dict)}개 문서 임베딩 로드 완료")

def is_smalltalk(query, client=None):
    """
    LLM 기반 일반대화 자동 분류

    하이브리드 방식:
    1. 규칙 기반으로 명확한 케이스 먼저 처리 (빠름)
    2. 애매한 경우만 LLM 호출 (정확함)

    Returns:
        True: 일반대화 (문서 검색 불필요)
        False: 과학질문 (문서 검색 필요)
    """
    # 1단계: 규칙 기반 명확한 케이스

    # 매우 짧은 문장 (5자 이하)
    if len(query) < 5:
        return True

    # 명확한 인사 (짧은 문장 + 인사 키워드)
    greetings = ['안녕', '반가워', '반갑', 'hi', 'hello', 'hey', '잘가', 'bye']
    if any(word in query for word in greetings) and len(query) < 15:
        return True

    # 명확한 감정 표현 (짧은 문장 + 감정 키워드)
    emotions = ['고마워', '감사', '수고', '미안', '죄송', '신나', '힘들', '슬프', '기뻐']
    if any(word in query for word in emotions) and len(query) < 20:
        return True

    # 명확한 질문 마커 (과학질문일 가능성 높음)
    question_markers = ['왜', '어떻게', '무엇', '뭐', '어디', '언제', '누구', '어느', '원리', '이유', '방법', '과정']
    if any(marker in query for marker in question_markers):
        return False

    # 질문 부호
    if '?' in query or '？' in query:
        return False

    # 2단계: 애매한 케이스만 LLM 호출
    if not client:
        # LLM이 없으면 보수적으로 과학질문으로 처리
        return False

    try:
        prompt = f"""다음 문장이 과학/기술/지식에 대한 질문인지, 일반적인 대화(인사, 감정표현, 잡담)인지 판단하세요.

문장: {query}

과학질문이면 "SCIENCE", 일반대화면 "SMALLTALK"로만 답하세요."""

        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip().upper()
        return "SMALLTALK" in result

    except Exception as e:
        # LLM 호출 실패 시 보수적으로 과학질문으로 처리
        print(f"⚠️  Smalltalk 분류 실패: {e}")
        return False

def rewrite_query_with_context(msg):
    """
    멀티턴 대화의 맥락을 통합하여 쿼리 재작성 (rag_with_context.py 기반)
    """
    if isinstance(msg, str):
        return msg

    if len(msg) == 1:
        return msg[0]['content']

    current_query = msg[-1]['content']

    # 대명사나 모호한 표현 확인
    ambiguous_terms = ['그 ', '그것', '이것', '이거', '저것', '저거', '왜', '어떻게', '이유']

    if not any(term in current_query for term in ambiguous_terms):
        return current_query

    if not client:
        return current_query

    # LLM으로 쿼리 재작성
    conversation_context = "\n".join([
        f"{m['role']}: {m['content']}" for m in msg[:-1]
    ])

    prompt = f"""다음은 이전 대화 내용입니다:

{conversation_context}

현재 사용자의 질문은 다음과 같습니다:
"{current_query}"

이 질문을 이전 대화의 맥락을 반영하여 독립적으로 이해 가능한 완전한 질문으로 재작성해주세요.
대명사(그것, 이것 등)를 구체적인 명사로 바꿔주세요.

재작성된 질문만 출력하세요. 다른 설명은 하지 마세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )

        rewritten = response.choices[0].message.content.strip()
        rewritten = rewritten.strip('"').strip("'")

        return rewritten

    except Exception as e:
        return current_query

def search_bm25(query, top_k=20):
    """BM25 검색"""
    fetch_size = top_k + 5

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
            'size': fetch_size
        }
    )

    if not response['hits']['hits']:
        return []

    # original_docid 기반 중복 제거
    seen_original_docids = set()
    results = []

    for hit in response['hits']['hits']:
        source = hit['_source']
        original_docid = source.get('original_docid', source['docid'])

        if original_docid in seen_original_docids:
            continue

        seen_original_docids.add(original_docid)
        results.append({
            'docid': original_docid,
            'content': source['content'],
            'score': hit['_score'],
            'source': 'bm25'
        })

        if len(results) >= top_k:
            break

    return results

def bgem3_hybrid_score(query_dense, query_sparse, query_colbert,
                       doc_dense, doc_sparse, doc_colbert,
                       w1=0.4, w2=0.3, w3=0.3):
    """BGE-M3 Hybrid Scoring"""
    # 1. Dense 유사도
    s_dense = np.dot(query_dense, doc_dense) / (
        np.linalg.norm(query_dense) * np.linalg.norm(doc_dense)
    )

    # 2. Sparse 유사도
    s_lex = 0.0
    if query_sparse and doc_sparse:
        common_tokens = set(query_sparse.keys()) & set(doc_sparse.keys())
        for token in common_tokens:
            s_lex += query_sparse[token] * doc_sparse[token]

    # 3. ColBERT 유사도
    s_mul = 0.0
    if query_colbert.shape[0] > 0 and doc_colbert.shape[0] > 0:
        query_colbert_norm = query_colbert / np.linalg.norm(query_colbert, axis=1, keepdims=True)
        doc_colbert_norm = doc_colbert / np.linalg.norm(doc_colbert, axis=1, keepdims=True)
        sim_matrix = np.dot(query_colbert_norm, doc_colbert_norm.T)
        s_mul = np.mean(np.max(sim_matrix, axis=1))

    hybrid_score = w1 * s_dense + w2 * s_lex + w3 * s_mul
    return hybrid_score

def search_bgem3_hybrid(query, embeddings_dict, top_k=20, max_length=128):
    """
    BGE-M3 Hybrid 검색 (쿼리 길이 최적화)
    max_length: 64 → 128 (Query Expansion으로 쿼리가 길어질 수 있음)
    """
    # BGE-M3 쿼리 임베딩 (최적화된 max_length)
    query_embedding = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
        max_length=max_length
    )

    query_dense = query_embedding['dense_vecs'][0]
    query_sparse = query_embedding['lexical_weights'][0]
    query_colbert = query_embedding['colbert_vecs'][0]

    # 모든 문서에 대해 Hybrid Score 계산
    scores = []
    for docid, doc_emb in embeddings_dict.items():
        score = bgem3_hybrid_score(
            query_dense, query_sparse, query_colbert,
            doc_emb['dense'], doc_emb['sparse'], doc_emb['colbert']
        )
        scores.append((docid, score))

    # 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    # ES에서 content 가져오기
    results = []
    for docid, score in scores[:top_k]:
        try:
            resp = es.search(
                index='test',
                body={
                    'query': {
                        'bool': {
                            'should': [
                                {'term': {'docid.keyword': docid}},
                                {'term': {'original_docid.keyword': docid}}
                            ]
                        }
                    },
                    'size': 1
                }
            )

            if resp['hits']['hits']:
                source = resp['hits']['hits'][0]['_source']
                results.append({
                    'docid': docid,
                    'content': source['content'],
                    'score': float(score),
                    'source': 'bgem3_hybrid'
                })
        except Exception as e:
            continue

    return results

def hybrid_search_rrf(query, embeddings_dict, top_k=30, k=60, query_max_length=128):
    """
    RRF로 BM25 + BGE-M3 Hybrid 결합
    top_k를 30으로 증가 (Cascaded Reranking용)
    """
    # BM25 검색
    bm25_results = search_bm25(query, top_k=top_k)

    # BGE-M3 Hybrid 검색 (최적화된 쿼리 길이)
    bgem3_results = search_bgem3_hybrid(query, embeddings_dict, top_k=top_k, max_length=query_max_length)

    # RRF 스코어 계산
    rrf_scores = {}
    doc_contents = {}

    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']

    for rank, doc in enumerate(bgem3_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']

    # 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for docid, score in sorted_docs:
        results.append({
            'docid': docid,
            'content': doc_contents[docid],
            'score': score
        })

    return results

def llm_stage1_filter(query, docs, top_k=10):
    """
    Stage 1: Top 30 → Top 10 필터링
    빠르고 간단한 프롬프트로 후보 압축
    """
    if not docs or len(docs) <= top_k or not client:
        return [i for i in range(min(top_k, len(docs)))]

    try:
        doc_list = []
        for i, doc in enumerate(docs[:30]):
            content_preview = doc['content'][:200]
            if len(doc['content']) > 200:
                content_preview += "..."
            doc_list.append(f"[{i}] {content_preview}")

        docs_text = "\n\n".join(doc_list)

        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 과학 지식 검색 시스템의 1차 필터링 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"""쿼리: {query}

문서들:
{docs_text}

이 질문과 관련성이 높은 상위 {top_k}개 문서 번호를 선택하세요.

출력: 번호만 콤마로 구분 (예: 0,2,4,5,7,8,10,12,14,16)
설명 없이 번호만 출력하세요."""
                }
            ],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()
        indices = [int(x.strip()) for x in result.split(',') if x.strip().isdigit()]

        # 유효한 인덱스만 반환
        valid_indices = [idx for idx in indices[:top_k] if 0 <= idx < len(docs)]

        # 부족하면 원래 순서로 채우기
        for i in range(len(docs)):
            if len(valid_indices) >= top_k:
                break
            if i not in valid_indices:
                valid_indices.append(i)

        return valid_indices[:top_k]

    except Exception as e:
        return [i for i in range(min(top_k, len(docs)))]

def llm_stage2_rerank(query, docs, top_k=3):
    """
    Stage 2: Top 10 → Top 3 정밀 Reranking
    기존 llm_rerank와 동일한 정밀한 프롬프트
    """
    if not docs or len(docs) <= top_k or not client:
        return [doc['docid'] for doc in docs[:top_k]]

    try:
        doc_list = []
        for i, doc in enumerate(docs):
            content_preview = doc['content'][:300]
            if len(doc['content']) > 300:
                content_preview += "..."
            doc_list.append(f"[{i}] {content_preview}")

        docs_text = "\n\n".join(doc_list)

        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 과학 지식 검색 시스템의 relevance 판단 전문가입니다. 주어진 쿼리에 가장 관련성 높은 문서를 정확히 선택하세요."
                },
                {
                    "role": "user",
                    "content": f"""쿼리: {query}

문서들:
{docs_text}

이 질문에 답하는 데 가장 관련성 높은 문서 {top_k}개의 번호를 선택하세요.
- 질문과 직접 관련된 내용을 포함하는가?
- 질문에 대한 답변을 제공하는가?
- 과학적으로 정확한가?

출력: 번호만 콤마로 구분 (예: 0,2,4)
설명 없이 번호만 출력하세요."""
                }
            ],
            temperature=0.0,
            max_tokens=30
        )

        result = response.choices[0].message.content.strip()
        indices = [int(x.strip()) for x in result.split(',') if x.strip().isdigit()]

        reranked_docids = []
        for idx in indices[:top_k]:
            if 0 <= idx < len(docs):
                reranked_docids.append(docs[idx]['docid'])

        # 부족하면 원래 순서로 채우기
        for doc in docs:
            if len(reranked_docids) >= top_k:
                break
            if doc['docid'] not in reranked_docids:
                reranked_docids.append(doc['docid'])

        return reranked_docids[:top_k]

    except Exception as e:
        return [doc['docid'] for doc in docs[:top_k]]

def llm_cascaded_rerank(query, docs, top_k=3):
    """
    Cascaded (2-Stage) Reranking

    Stage 1: Top 30 → Top 10 (빠른 필터링)
    Stage 2: Top 10 → Top 3 (정밀한 판단)
    """
    if not docs or len(docs) <= top_k:
        return [doc['docid'] for doc in docs[:top_k]]

    # Stage 1: Top 30 → Top 10 필터링
    stage1_indices = llm_stage1_filter(query, docs, top_k=10)

    # Stage 1에서 선택된 문서들
    stage1_docs = [docs[i] for i in stage1_indices if i < len(docs)]

    # Stage 2: Top 10 → Top 3 정밀 Reranking
    final_docids = llm_stage2_rerank(query, stage1_docs, top_k=top_k)

    return final_docids

def cascaded_reranking_strategy(eval_id, msg, embeddings_dict):
    """
    Cascaded Reranking v1 전략

    1. LLM 쿼리 재작성 (멀티턴 대화 맥락 통합)
    2. Hybrid Search (Top 30으로 확장)
    3. Cascaded LLM Reranking (30 → 10 → 3)
    """
    # Step 1: 쿼리 재작성
    rewritten_query = rewrite_query_with_context(msg)

    # 일반 대화 자동 분류 (LLM 기반 하이브리드)
    if is_smalltalk(rewritten_query, client):
        return []

    # Step 2: Hybrid Search (Top 30으로 확장)
    hybrid_results = hybrid_search_rrf(
        rewritten_query,
        embeddings_dict,
        top_k=30,  # ✅ 15 → 30으로 확장
        k=60,
        query_max_length=128
    )

    if not hybrid_results:
        return []

    # Step 3: Cascaded LLM Reranking (30 → 10 → 3)
    final_topk = llm_cascaded_rerank(rewritten_query, hybrid_results, top_k=3)

    return final_topk

def run_experiment():
    """Cascaded Reranking v1 실험 실행"""
    print("="*80)
    print("Cascaded Reranking v1 실험")
    print("="*80)
    print("Stage 1: Top 30 → Top 10 필터링")
    print("Stage 2: Top 10 → Top 3 정밀 Reranking")
    print("="*80)

    # Eval 데이터 로드
    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    print(f"\n📋 총 {len(eval_data)}개 쿼리 처리 시작\n")

    results = []

    for item in tqdm(eval_data, desc="Cascaded Reranking v1"):
        eval_id = item['eval_id']
        msg = item['msg']

        # Cascaded Reranking 전략 실행
        topk = cascaded_reranking_strategy(eval_id, msg, embeddings_dict)

        results.append({
            'eval_id': eval_id,
            'retrieve': topk
        })

    # 제출 파일 생성
    output_path = 'cascaded_reranking_v1_submission.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            json_obj = {
                'eval_id': r['eval_id'],
                'topk': r['retrieve']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"✅ 실험 완료")
    print(f"{'='*80}")
    print(f"💾 제출 파일: {output_path}")
    print(f"{'='*80}")

    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Cascaded Reranking v1: 2-Stage LLM Reranking")
    print("목표: Lost in the Middle 해결 + Recall 향상")
    print("예상 성능: 0.81~0.82 MAP@3")
    print("="*80)
    run_experiment()
