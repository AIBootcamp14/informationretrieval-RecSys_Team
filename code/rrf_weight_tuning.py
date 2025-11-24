"""
RRF Weight Tuning: query_expansion_v1 기반 가중치 최적화
목표: w_bm25, w_bgem3 조합을 테스트하여 0.9+ MAP@3 달성

실험:
1. w_bm25=0.4, w_bgem3=0.6 (BM25 증가)
2. w_bm25=0.2, w_bgem3=0.8 (BGE-M3 증가)
3. w_bm25=0.25, w_bgem3=0.75 (약간 BGE-M3 강조)
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

# 일반 대화 ID
SMALLTALK_IDS = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

# BGE-M3 최적화 임베딩 로드
print("\nBGE-M3 최적화 임베딩 로드 중...")
with open('embeddings_test_bgem3_optimized.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print(f"✅ {len(embeddings_dict)}개 문서 임베딩 로드 완료")

def rewrite_query_with_context(msg):
    """멀티턴 대화의 맥락을 통합하여 쿼리 재작성"""
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
    """BGE-M3 Hybrid 검색"""
    # BGE-M3 쿼리 임베딩
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

def hybrid_search_rrf(query, embeddings_dict, top_k=20, k=60,
                      query_max_length=128, w_bm25=0.3, w_bgem3=0.7):
    """
    RRF로 BM25 + BGE-M3 Hybrid 결합

    Args:
        w_bm25: BM25 가중치 (기본 0.3)
        w_bgem3: BGE-M3 가중치 (기본 0.7)
    """
    # BM25 검색
    bm25_results = search_bm25(query, top_k=top_k)

    # BGE-M3 Hybrid 검색
    bgem3_results = search_bgem3_hybrid(query, embeddings_dict, top_k=top_k, max_length=query_max_length)

    # RRF 스코어 계산 (가중치 적용)
    rrf_scores = {}
    doc_contents = {}

    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + w_bm25 / (k + rank)
        doc_contents[docid] = doc['content']

    for rank, doc in enumerate(bgem3_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + w_bgem3 / (k + rank)
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

def llm_rerank(query, docs, top_k=3):
    """Solar-pro로 Reranking"""
    if not docs or len(docs) <= top_k or not client:
        return [doc['docid'] for doc in docs[:top_k]]

    try:
        doc_list = []
        for i, doc in enumerate(docs[:15]):
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

def rrf_weight_strategy(eval_id, msg, embeddings_dict, w_bm25, w_bgem3):
    """RRF Weight Tuning 전략"""
    # 일반 대화는 빈 결과
    if eval_id in SMALLTALK_IDS:
        return []

    # Step 1: 쿼리 재작성
    rewritten_query = rewrite_query_with_context(msg)

    # Step 2: Hybrid Search (RRF 가중치 조정)
    hybrid_results = hybrid_search_rrf(
        rewritten_query,
        embeddings_dict,
        top_k=20,
        k=60,
        query_max_length=128,
        w_bm25=w_bm25,  # ✅ 가중치 조정
        w_bgem3=w_bgem3  # ✅ 가중치 조정
    )

    if not hybrid_results:
        return []

    # Step 3: LLM Reranking
    final_topk = llm_rerank(rewritten_query, hybrid_results, top_k=3)

    return final_topk

def run_rrf_weight_experiment(w_bm25, w_bgem3, experiment_name):
    """RRF Weight 실험 실행"""
    print("="*80)
    print(f"RRF Weight Tuning Experiment: {experiment_name}")
    print("="*80)
    print(f"w_bm25 = {w_bm25}")
    print(f"w_bgem3 = {w_bgem3}")
    print("="*80)

    # Eval 데이터 로드
    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    print(f"\n📋 총 {len(eval_data)}개 쿼리 처리 시작\n")

    results = []

    for item in tqdm(eval_data, desc=f"RRF w_bm25={w_bm25} w_bgem3={w_bgem3}"):
        eval_id = item['eval_id']
        msg = item['msg']

        # RRF Weight 전략 실행
        topk = rrf_weight_strategy(eval_id, msg, embeddings_dict, w_bm25, w_bgem3)

        results.append({
            'eval_id': eval_id,
            'retrieve': topk
        })

    # 제출 파일 생성
    output_path = f'rrf_w{w_bm25}_{w_bgem3}_submission.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            json_obj = {
                'eval_id': r['eval_id'],
                'topk': r['retrieve']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"✅ 실험 완료: {experiment_name}")
    print(f"{'='*80}")
    print(f"💾 제출 파일: {output_path}")
    print(f"{'='*80}")

    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RRF Weight Tuning: 3가지 가중치 조합 실험")
    print("="*80)
    print("현재 최고 점수: query_expansion_v1 = 0.7848")
    print("목표: 0.9+ MAP@3")
    print("="*80 + "\n")

    # Experiment 1: w_bm25=0.4, w_bgem3=0.6 (BM25 증가)
    print("\n" + "="*80)
    print("Experiment 1: w_bm25=0.4, w_bgem3=0.6 (BM25 영향력 증가)")
    print("="*80)
    run_rrf_weight_experiment(
        w_bm25=0.4,
        w_bgem3=0.6,
        experiment_name="BM25 Boost"
    )

    # Experiment 2: w_bm25=0.2, w_bgem3=0.8 (BGE-M3 증가)
    print("\n" + "="*80)
    print("Experiment 2: w_bm25=0.2, w_bgem3=0.8 (BGE-M3 영향력 증가)")
    print("="*80)
    run_rrf_weight_experiment(
        w_bm25=0.2,
        w_bgem3=0.8,
        experiment_name="BGE-M3 Boost"
    )

    # Experiment 3: w_bm25=0.25, w_bgem3=0.75 (약간 BGE-M3 강조)
    print("\n" + "="*80)
    print("Experiment 3: w_bm25=0.25, w_bgem3=0.75 (BGE-M3 약간 강조)")
    print("="*80)
    run_rrf_weight_experiment(
        w_bm25=0.25,
        w_bgem3=0.75,
        experiment_name="BGE-M3 Slight Emphasis"
    )

    print("\n" + "="*80)
    print("✅ 모든 RRF Weight Tuning 실험 완료")
    print("="*80)
    print("\n제출 파일:")
    print("  - rrf_w0.4_0.6_submission.csv")
    print("  - rrf_w0.2_0.8_submission.csv")
    print("  - rrf_w0.25_0.75_submission.csv")
    print("="*80)
