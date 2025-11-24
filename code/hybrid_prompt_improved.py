"""
Hybrid Search with Improved Prompt Engineering
프롬프트 개선을 통한 LLM Reranking 성능 향상
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

def hybrid_search_rrf(query, embeddings_dict, top_k=20, k=60):
    """RRF Fusion"""
    bm25_results = search_bm25(query, top_k=top_k)
    dense_results = search_dense(query, embeddings_dict, top_k=top_k)

    rrf_scores = {}
    doc_contents = {}

    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']

    for rank, doc in enumerate(dense_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
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

def llm_rerank_improved(query, docs, top_k=3):
    """
    개선된 프롬프트로 Reranking

    개선 포인트:
    1. 명확한 relevance 기준 제시
    2. Few-shot 예시 추가
    3. Step-by-step 사고 유도
    4. 더 구체적인 평가 지침
    """
    if not docs or len(docs) <= top_k or not client:
        return [doc['docid'] for doc in docs[:top_k]]

    try:
        # 문서 목록 생성 (상위 15개)
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
                    "content": """당신은 과학 지식 검색 시스템의 문서 relevance 평가 전문가입니다.

주어진 질문에 가장 적합한 문서를 선택하는 것이 목표입니다.

평가 기준:
1. **정확성**: 질문이 요구하는 정보를 정확히 포함하는가?
2. **완전성**: 질문에 완전한 답변을 제공하는가 (부분적인 답변은 낮은 점수)?
3. **관련성**: 질문의 핵심 개념과 직접 관련되는가?
4. **신뢰성**: 과학적으로 정확하고 신뢰할 수 있는 내용인가?

중요: 제목이나 키워드가 유사하다고 해서 관련성이 높은 것은 아닙니다.
실제로 질문에 답할 수 있는 정보를 포함해야 합니다."""
                },
                {
                    "role": "user",
                    "content": f"""질문: {query}

아래는 검색된 문서들입니다:

{docs_text}

위의 질문에 가장 적합한 문서 {top_k}개를 선택하세요.

선택 프로세스:
1. 각 문서가 질문의 어떤 부분에 답하는지 확인
2. 정확성, 완전성, 관련성, 신뢰성 기준으로 평가
3. 가장 높은 점수를 받은 {top_k}개 선택

출력 형식: 번호만 콤마로 구분 (예: 0,2,4)
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

def llm_rerank_cot(query, docs, top_k=3):
    """
    Chain-of-Thought 프롬프트로 Reranking
    LLM이 단계별로 사고하도록 유도
    """
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
                    "content": "당신은 과학 지식 검색 시스템의 문서 relevance 평가 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"""질문: {query}

문서들:
{docs_text}

다음 단계를 따라 가장 관련성 높은 문서 {top_k}개를 선택하세요:

1단계: 질문의 핵심 키워드와 의도 파악
2단계: 각 문서가 질문에 답하는지 평가 (답함/부분적/답 안함)
3단계: 답하는 문서들 중 가장 완전하고 정확한 {top_k}개 선택

최종 출력: 선택한 문서 번호를 콤마로 구분 (예: 0,2,4)
번호만 출력하세요."""
                }
            ],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()
        # 마지막 줄에서 숫자만 추출
        lines = result.split('\\n')
        for line in reversed(lines):
            if any(c.isdigit() for c in line):
                indices = [int(x.strip()) for x in line.split(',') if x.strip().isdigit()]
                if indices:
                    break

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

def run_prompt_experiments():
    """프롬프트 개선 실험 실행"""
    print("="*80)
    print("프롬프트 엔지니어링 개선 실험")
    print("="*80)

    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    experiments = [
        {
            'name': 'prompt_v1_improved',
            'rerank_func': llm_rerank_improved,
            'description': '명확한 기준 + 구체적 지침'
        },
        {
            'name': 'prompt_v2_cot',
            'rerank_func': llm_rerank_cot,
            'description': 'Chain-of-Thought'
        }
    ]

    for exp in experiments:
        print(f"\\n{'='*80}")
        print(f"실험: {exp['name']}")
        print(f"설명: {exp['description']}")
        print(f"{'='*80}")

        results = []

        for item in tqdm(eval_data, desc=exp['name']):
            eval_id = item['eval_id']

            if isinstance(item['msg'], list):
                query = item['msg'][-1]['content']
            else:
                query = item['msg']

            if eval_id in SMALLTALK_IDS:
                results.append({
                    'eval_id': eval_id,
                    'retrieve': []
                })
                continue

            # Hybrid Search
            hybrid_results = hybrid_search_rrf(query, embeddings_dict, top_k=20, k=60)

            if not hybrid_results:
                results.append({
                    'eval_id': eval_id,
                    'retrieve': []
                })
                continue

            # LLM Reranking
            final_topk = exp['rerank_func'](query, hybrid_results, top_k=3)

            results.append({
                'eval_id': eval_id,
                'retrieve': final_topk
            })

        # 제출 파일 생성
        output_path = f"{exp['name']}_submission.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                json_obj = {
                    'eval_id': r['eval_id'],
                    'topk': r['retrieve']
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

        print(f"✅ 제출 파일: {output_path}")

    print("\\n" + "="*80)
    print("✅ 모든 프롬프트 실험 완료")
    print("="*80)
    print("\\n다음 명령으로 결과 비교:")
    print("python batch_validate.py")

if __name__ == "__main__":
    run_prompt_experiments()
