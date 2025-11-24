"""
BGE-M3 최적화 Hybrid Search 솔루션
세 가지 검색 모드 활용: Dense + Sparse + ColBERT
목표: 0.79+ MAP@3 달성
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

# 샘플 임베딩 구조 확인
sample_docid = list(embeddings_dict.keys())[0]
sample_emb = embeddings_dict[sample_docid]
print(f"   Dense 차원: {len(sample_emb['dense'])}")
print(f"   Sparse 토큰 수: {len(sample_emb['sparse'])}")
print(f"   ColBERT 벡터 수: {sample_emb['colbert'].shape[0]}")

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
    """
    BGE-M3 Hybrid Scoring

    Args:
        query_dense: 쿼리 dense 벡터 (1024,)
        query_sparse: 쿼리 sparse 가중치 dict {token_id: weight}
        query_colbert: 쿼리 ColBERT 벡터 (M, 1024)
        doc_dense: 문서 dense 벡터 (1024,)
        doc_sparse: 문서 sparse 가중치 dict {token_id: weight}
        doc_colbert: 문서 ColBERT 벡터 (N, 1024)
        w1, w2, w3: 가중치 (합=1.0)

    Returns:
        hybrid_score: 최종 스코어
    """
    # 1. Dense 유사도 (코사인)
    s_dense = np.dot(query_dense, doc_dense) / (
        np.linalg.norm(query_dense) * np.linalg.norm(doc_dense)
    )

    # 2. Sparse 유사도 (공통 토큰 가중치 곱의 합)
    s_lex = 0.0
    if query_sparse and doc_sparse:
        common_tokens = set(query_sparse.keys()) & set(doc_sparse.keys())
        for token in common_tokens:
            s_lex += query_sparse[token] * doc_sparse[token]

    # 3. ColBERT 유사도 (MaxSim)
    # MaxSim: 각 쿼리 벡터에 대해 가장 유사한 문서 벡터의 평균
    s_mul = 0.0
    if query_colbert.shape[0] > 0 and doc_colbert.shape[0] > 0:
        # 정규화
        query_colbert_norm = query_colbert / np.linalg.norm(query_colbert, axis=1, keepdims=True)
        doc_colbert_norm = doc_colbert / np.linalg.norm(doc_colbert, axis=1, keepdims=True)

        # 코사인 유사도 행렬 (M x N)
        sim_matrix = np.dot(query_colbert_norm, doc_colbert_norm.T)

        # MaxSim: 각 쿼리 벡터의 최대 유사도 평균
        s_mul = np.mean(np.max(sim_matrix, axis=1))

    # Hybrid Score
    hybrid_score = w1 * s_dense + w2 * s_lex + w3 * s_mul

    return hybrid_score

def search_bgem3_hybrid(query, embeddings_dict, top_k=20,
                        w1=0.4, w2=0.3, w3=0.3):
    """
    BGE-M3 Hybrid Search (Dense + Sparse + ColBERT)

    Args:
        query: 검색 쿼리
        embeddings_dict: BGE-M3 최적화 임베딩 딕셔너리
        top_k: 반환할 문서 수
        w1, w2, w3: Dense, Sparse, ColBERT 가중치
    """
    # 쿼리 임베딩 생성 (세 가지 모두)
    query_embedding = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
        max_length=64  # 쿼리는 짧게
    )

    query_dense = query_embedding['dense_vecs'][0]
    query_sparse = query_embedding['lexical_weights'][0]
    query_colbert = query_embedding['colbert_vecs'][0]

    # 모든 문서에 대해 Hybrid Score 계산
    scores = []
    for docid, doc_emb in embeddings_dict.items():
        try:
            hybrid_score = bgem3_hybrid_score(
                query_dense, query_sparse, query_colbert,
                doc_emb['dense'], doc_emb['sparse'], doc_emb['colbert'],
                w1, w2, w3
            )
            scores.append((docid, hybrid_score))
        except Exception as e:
            # 임베딩 오류 시 스킵
            continue

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

def hybrid_search_rrf_bgem3(query, embeddings_dict, top_k=20, k=60,
                            w1=0.4, w2=0.3, w3=0.3):
    """
    RRF Fusion: BM25 + BGE-M3 Hybrid

    Args:
        query: 검색 쿼리
        embeddings_dict: BGE-M3 최적화 임베딩
        top_k: 각 검색에서 가져올 문서 수
        k: RRF 파라미터
        w1, w2, w3: BGE-M3 가중치
    """
    # BM25 검색
    bm25_results = search_bm25(query, top_k=top_k)

    # BGE-M3 Hybrid 검색
    bgem3_results = search_bgem3_hybrid(query, embeddings_dict, top_k=top_k,
                                        w1=w1, w2=w2, w3=w3)

    # RRF 스코어 계산
    rrf_scores = {}
    doc_contents = {}

    # BM25 결과
    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']

    # BGE-M3 Hybrid 결과
    for rank, doc in enumerate(bgem3_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']

    # 스코어 순으로 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 결과 생성
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
        # 문서 목록 생성
        doc_list = []
        for i, doc in enumerate(docs[:15]):  # 상위 15개만 평가
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
        print(f"⚠️  Reranking 실패: {e}")
        return [doc['docid'] for doc in docs[:top_k]]

def optimized_bgem3_strategy(eval_id, query, embeddings_dict,
                             w1=0.4, w2=0.3, w3=0.3):
    """
    BGE-M3 최적화 전략

    Pipeline:
    1. BM25 + BGE-M3 Hybrid (Dense+Sparse+ColBERT) → RRF Fusion
    2. LLM Reranking

    Args:
        eval_id: 평가 ID
        query: 검색 쿼리
        embeddings_dict: BGE-M3 최적화 임베딩
        w1, w2, w3: BGE-M3 가중치
    """
    # 일반 대화는 빈 결과
    if eval_id in SMALLTALK_IDS:
        return []

    # Hybrid Search (RRF)
    hybrid_results = hybrid_search_rrf_bgem3(
        query, embeddings_dict,
        top_k=20, k=60,
        w1=w1, w2=w2, w3=w3
    )

    if not hybrid_results:
        return []

    # LLM Reranking
    final_topk = llm_rerank(query, hybrid_results, top_k=3)

    return final_topk

def run_optimized_experiment(w1=0.4, w2=0.3, w3=0.3, exp_name="default"):
    """
    BGE-M3 최적화 실험 실행

    Args:
        w1, w2, w3: BGE-M3 가중치 (Dense, Sparse, ColBERT)
        exp_name: 실험 이름
    """
    print("="*80)
    print(f"BGE-M3 최적화 Hybrid Search 실험: {exp_name}")
    print("="*80)
    print(f"전략: BM25 + BGE-M3 Hybrid (Dense+Sparse+ColBERT) + RRF + LLM")
    print(f"가중치: Dense={w1:.1f}, Sparse={w2:.1f}, ColBERT={w3:.1f}")
    print("="*80)

    # Eval 데이터 로드
    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    print(f"\n📋 총 {len(eval_data)}개 쿼리 처리 시작\n")

    results = []

    for item in tqdm(eval_data, desc=f"BGE-M3 최적화 ({exp_name})"):
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 최적화 전략 실행
        topk = optimized_bgem3_strategy(eval_id, query, embeddings_dict,
                                        w1=w1, w2=w2, w3=w3)

        results.append({
            'eval_id': eval_id,
            'retrieve': topk
        })

    # 제출 파일 생성
    output_path = f'hybrid_bgem3_optimized_{exp_name}_submission.csv'
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
    # Phase 1: 기본 가중치로 테스트
    print("\n🚀 Phase 1: 기본 가중치 (0.4, 0.3, 0.3)")
    run_optimized_experiment(w1=0.4, w2=0.3, w3=0.3, exp_name="w433")

    print("\n" + "="*80)
    print("✅ BGE-M3 최적화 실험 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("1. 결과 확인 후 가중치 튜닝 (Phase 2)")
    print("2. 최적 가중치 조합 탐색")
    print("   - 예: w1=0.3, w2=0.3, w3=0.4 (ColBERT 강조)")
    print("   - 예: w1=0.5, w2=0.2, w3=0.3 (Dense 강조)")
