"""
BGE-M3 임베딩을 활용한 Hybrid Search 솔루션
BM25 + Dense Retrieval (BGE-M3) + Solar LLM Reranking
목표: MAP@3 0.8+ 달성
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

# BGE-M3 임베딩 로드
print("\nBGE-M3 Dense 임베딩 로드 중...")
with open('embeddings_test_bgem3.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print(f"✅ {len(embeddings_dict)}개 문서 임베딩 로드 완료")
print(f"   임베딩 차원: {list(embeddings_dict.values())[0].shape[0]}")

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

def search_dense_bgem3(query, embeddings_dict, top_k=20):
    """BGE-M3 Dense Retrieval 검색"""
    # BGE-M3로 쿼리 임베딩 생성
    query_embedding = model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        max_length=512
    )
    query_emb = query_embedding['dense_vecs'][0]

    # 코사인 유사도 계산
    scores = []
    for docid, doc_emb in embeddings_dict.items():
        similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scores.append((docid, similarity))

    # 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    # ES에서 content 가져오기
    results = []
    for docid, score in scores[:top_k]:
        try:
            # ES에서 문서 조회
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
                    'source': 'dense_bgem3'
                })
        except Exception as e:
            continue

    return results

def hybrid_search_rrf(query, embeddings_dict, top_k=20, k=60):
    """
    Reciprocal Rank Fusion (RRF)로 BM25 + Dense (BGE-M3) 결합

    Args:
        query: 검색 쿼리
        embeddings_dict: BGE-M3 임베딩 딕셔너리
        top_k: 각 검색에서 가져올 문서 수
        k: RRF 파라미터 (기본값 60)

    Returns:
        RRF로 융합된 문서 리스트
    """
    # BM25 검색
    bm25_results = search_bm25(query, top_k=top_k)

    # Dense 검색 (BGE-M3)
    dense_results = search_dense_bgem3(query, embeddings_dict, top_k=top_k)

    # RRF 스코어 계산
    rrf_scores = {}
    doc_contents = {}

    # BM25 결과
    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']

    # Dense 결과
    for rank, doc in enumerate(dense_results, 1):
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

def hybrid_bgem3_strategy(eval_id, query, embeddings_dict):
    """BGE-M3 기반 Hybrid 전략"""
    # 일반 대화는 빈 결과
    if eval_id in SMALLTALK_IDS:
        return []

    # Hybrid Search (RRF)
    hybrid_results = hybrid_search_rrf(query, embeddings_dict, top_k=20, k=60)

    if not hybrid_results:
        return []

    # LLM Reranking
    final_topk = llm_rerank(query, hybrid_results, top_k=3)

    return final_topk

def run_bgem3_experiment():
    """BGE-M3 기반 실험 실행"""
    print("="*80)
    print("BGE-M3 Hybrid Search 실험")
    print("="*80)
    print("전략: BM25 + Dense (BGE-M3, 1024d) + Solar LLM Reranking")
    print("개선: ko-sroberta (768d) → BGE-M3 (1024d)")
    print("="*80)

    # Eval 데이터 로드
    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    print(f"\n📋 총 {len(eval_data)}개 쿼리 처리 시작\n")

    results = []

    for item in tqdm(eval_data, desc="BGE-M3 Hybrid Search"):
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # Hybrid 전략 실행
        topk = hybrid_bgem3_strategy(eval_id, query, embeddings_dict)

        results.append({
            'eval_id': eval_id,
            'retrieve': topk
        })

    # 제출 파일 생성
    output_path = 'hybrid_bgem3_submission.csv'
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
    run_bgem3_experiment()
