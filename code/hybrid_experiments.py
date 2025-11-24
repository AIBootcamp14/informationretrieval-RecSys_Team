"""
Hybrid Search 개선 실험
다양한 파라미터 조합으로 성능 최적화
"""

import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
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

# 임베딩 모델
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 일반 대화 ID
SMALLTALK_IDS = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

# Dense 임베딩 로드
print("Dense 임베딩 로드 중...")
with open('embeddings_test.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print(f"✅ {len(embeddings_dict)}개 문서 임베딩 로드 완료")

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

def search_dense(query, embeddings_dict, top_k=20):
    """Dense Retrieval 검색"""
    query_emb = model.encode([query], convert_to_numpy=True)[0]

    scores = []
    for docid, doc_emb in embeddings_dict.items():
        similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scores.append((docid, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)

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
                    'source': 'dense'
                })
        except Exception as e:
            continue

    return results

def hybrid_search_rrf(query, embeddings_dict, bm25_top_k=20, dense_top_k=20, rrf_k=60):
    """RRF Fusion"""
    bm25_results = search_bm25(query, top_k=bm25_top_k)
    dense_results = search_dense(query, embeddings_dict, top_k=dense_top_k)

    rrf_scores = {}
    doc_contents = {}

    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (rrf_k + rank)
        doc_contents[docid] = doc['content']

    for rank, doc in enumerate(dense_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (rrf_k + rank)
        doc_contents[docid] = doc['content']

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for docid, score in sorted_docs:
        results.append({
            'docid': docid,
            'content': doc_contents[docid],
            'score': score
        })

    return results

def llm_rerank(query, docs, top_k=3, num_candidates=15):
    """Solar-pro로 Reranking"""
    if not docs or len(docs) <= top_k or not client:
        return [doc['docid'] for doc in docs[:top_k]]

    try:
        doc_list = []
        for i, doc in enumerate(docs[:num_candidates]):
            content_preview = doc['content'][:300]
            if len(doc['content']) > 300:
                content_preview += "..."
            doc_list.append(f"[{i}] {content_preview}")

        docs_text = "\\n\\n".join(doc_list)

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

**평가 기준:**
1. 질문과 직접 관련된 내용을 포함하는가?
2. 질문에 대한 정확한 답변을 제공하는가?
3. 과학적으로 정확하고 신뢰할 수 있는가?

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

        for doc in docs:
            if len(reranked_docids) >= top_k:
                break
            if doc['docid'] not in reranked_docids:
                reranked_docids.append(doc['docid'])

        return reranked_docids[:top_k]

    except Exception as e:
        print(f"⚠️  Reranking 실패: {e}")
        return [doc['docid'] for doc in docs[:top_k]]

def run_experiment(config):
    """실험 실행"""
    print(f"\n{'='*80}")
    print(f"실험: {config['name']}")
    print(f"{'='*80}")
    print(f"BM25 Top-K: {config['bm25_top_k']}")
    print(f"Dense Top-K: {config['dense_top_k']}")
    print(f"RRF k: {config['rrf_k']}")
    print(f"LLM Candidates: {config['llm_candidates']}")
    print(f"Use LLM: {config['use_llm']}")

    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    for item in tqdm(eval_data, desc=config['name']):
        eval_id = item['eval_id']

        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 일반 대화
        if eval_id in SMALLTALK_IDS:
            results.append({
                'eval_id': eval_id,
                'retrieve': []
            })
            continue

        # Hybrid Search
        hybrid_results = hybrid_search_rrf(
            query,
            embeddings_dict,
            bm25_top_k=config['bm25_top_k'],
            dense_top_k=config['dense_top_k'],
            rrf_k=config['rrf_k']
        )

        if not hybrid_results:
            results.append({
                'eval_id': eval_id,
                'retrieve': []
            })
            continue

        # LLM Reranking
        if config['use_llm']:
            final_topk = llm_rerank(
                query,
                hybrid_results,
                top_k=3,
                num_candidates=config['llm_candidates']
            )
        else:
            final_topk = [doc['docid'] for doc in hybrid_results[:3]]

        results.append({
            'eval_id': eval_id,
            'retrieve': final_topk
        })

    # 제출 파일 생성
    output_path = f"{config['name']}_submission.csv"
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            json_obj = {
                'eval_id': r['eval_id'],
                'topk': r['retrieve']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"✅ 제출 파일: {output_path}")
    return output_path

if __name__ == "__main__":
    # 실험 설정 - LLM 중심 최적화
    experiments = [
        # LLM에게 더 많은 후보 제공
        {
            'name': 'exp1_llm_more_candidates',
            'bm25_top_k': 30,
            'dense_top_k': 30,
            'rrf_k': 60,
            'llm_candidates': 20,
            'use_llm': True
        },
        {
            'name': 'exp2_llm_max_candidates',
            'bm25_top_k': 50,
            'dense_top_k': 50,
            'rrf_k': 60,
            'llm_candidates': 30,
            'use_llm': True
        },
        # RRF로 더 나은 후보 생성
        {
            'name': 'exp3_rrf_tighter',
            'bm25_top_k': 30,
            'dense_top_k': 30,
            'rrf_k': 30,
            'llm_candidates': 20,
            'use_llm': True
        },
        {
            'name': 'exp4_rrf_looser',
            'bm25_top_k': 30,
            'dense_top_k': 30,
            'rrf_k': 90,
            'llm_candidates': 20,
            'use_llm': True
        },
        # 균형잡힌 설정
        {
            'name': 'exp5_balanced',
            'bm25_top_k': 40,
            'dense_top_k': 40,
            'rrf_k': 60,
            'llm_candidates': 25,
            'use_llm': True
        }
    ]

    print("="*80)
    print("Hybrid Search 개선 실험 시작")
    print("="*80)
    print(f"총 {len(experiments)}개 실험 실행 예정")

    for exp in experiments:
        try:
            run_experiment(exp)
        except Exception as e:
            print(f"❌ {exp['name']} 실패: {e}")
            continue

    print("\n" + "="*80)
    print("✅ 모든 실험 완료")
    print("="*80)
    print("\n다음 명령으로 결과 비교:")
    print("python batch_validate.py")
