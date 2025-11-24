"""
Multi-Query Generation v2: Phase 2 실험
- LLM 쿼리 재작성 (Phase 1 활용)
- Multi-Query Generation: 단일 쿼리 → 3-5개 변형 생성
- RRF Fusion으로 모든 변형 쿼리 결과 통합
- BGE-M3 Hybrid Search

목표: 0.85+ MAP@3
"""

import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from collections import defaultdict

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
    """
    멀티턴 대화의 맥락을 통합하여 쿼리 재작성 (Phase 1)
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

def generate_multi_queries(query, num_variants=4):
    """
    Multi-Query Generation: 단일 쿼리를 여러 변형으로 생성

    Args:
        query: 원본 쿼리
        num_variants: 생성할 변형 개수 (기본 4개, 원본 포함 총 5개)

    Returns:
        List[str]: 원본 + 변형 쿼리들
    """
    if not client:
        return [query]

    prompt = f"""다음 과학 지식 질문에 대해 {num_variants}개의 서로 다른 변형 질문을 생성해주세요.

원본 질문: "{query}"

변형 질문은 다음 방법들을 활용하세요:
1. 동의어 치환 (예: "이유" → "원인", "까닭")
2. 표현 방식 변경 (예: 의문문 → 서술형)
3. 핵심 개념 강조 (예: 구체적 용어 추가)
4. 관련 개념 확장 (예: 유사한 과학 용어 포함)

각 변형은 원본 질문과 의미가 동일하지만 다른 관점이나 표현을 사용해야 합니다.

형식:
1. [첫 번째 변형]
2. [두 번째 변형]
3. [세 번째 변형]
4. [네 번째 변형]

번호와 변형 질문만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 다양성을 위해 temperature 높임
            max_tokens=400
        )

        variants_text = response.choices[0].message.content.strip()

        # 응답 파싱
        variants = [query]  # 원본 쿼리 포함

        for line in variants_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # "1. ", "2. " 등의 번호 제거
            if line[0].isdigit() and '. ' in line:
                variant = line.split('. ', 1)[1].strip()
                variant = variant.strip('"').strip("'")
                variants.append(variant)

        # num_variants+1개 (원본 포함)로 제한
        return variants[:num_variants+1]

    except Exception as e:
        print(f"  ⚠️ Multi-query generation 실패: {e}")
        return [query]

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

    # 3. ColBERT 유사도 (MaxSim)
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
    BGE-M3 Hybrid 검색
    """
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

    # 점수 내림차순 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    # Top-K 선택
    top_docids = [docid for docid, _ in scores[:top_k]]

    # ES에서 문서 내용 가져오기
    results = []
    if top_docids:
        response = es.search(
            index='test',
            body={
                'query': {
                    'terms': {
                        'docid': top_docids
                    }
                },
                'size': top_k
            }
        )

        # docid 순서 유지
        doc_map = {}
        for hit in response['hits']['hits']:
            source = hit['_source']
            doc_map[source['docid']] = source['content']

        for docid, score in scores[:top_k]:
            if docid in doc_map:
                results.append({
                    'docid': docid,
                    'content': doc_map[docid],
                    'score': score,
                    'source': 'bgem3'
                })

    return results

def rrf_fusion(results_list, k=60):
    """
    Reciprocal Rank Fusion (RRF)
    여러 검색 결과를 하나로 융합

    Args:
        results_list: List[List[Dict]] - 각 쿼리의 검색 결과
        k: RRF 상수 (기본 60)

    Returns:
        List[Dict]: 융합된 검색 결과
    """
    rrf_scores = defaultdict(float)
    doc_contents = {}

    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            docid = doc['docid']
            rrf_scores[docid] += 1.0 / (k + rank)

            # 문서 내용 저장 (첫 번째 것만)
            if docid not in doc_contents:
                doc_contents[docid] = doc['content']

    # RRF 점수로 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 결과 구성
    fused_results = []
    for docid, score in sorted_docs:
        fused_results.append({
            'docid': docid,
            'content': doc_contents[docid],
            'score': score,
            'source': 'rrf'
        })

    return fused_results

def llm_rerank(query, docs, top_k=3):
    """
    LLM 기반 재순위화
    """
    if not client or not docs:
        return [doc['docid'] for doc in docs[:top_k]]

    # 상위 15개 문서만 reranking
    rerank_candidates = docs[:15]

    # 문서 목록 생성
    doc_list = []
    for i, doc in enumerate(rerank_candidates):
        content_preview = doc['content'][:300]
        if len(doc['content']) > 300:
            content_preview += "..."
        doc_list.append(f"[{i}] {content_preview}")

    docs_text = "\n\n".join(doc_list)

    prompt = f"""다음은 과학 지식 질문에 대한 검색 결과입니다.

질문: "{query}"

검색된 문서들:
{docs_text}

위 문서들 중에서 질문과 가장 관련성이 높은 문서 {top_k}개를 선택하여 번호만 출력하세요.
관련성이 높은 순서대로 정렬해주세요.

형식: 0,5,12 (번호만 쉼표로 구분)"""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()

        # 번호 파싱
        selected_indices = []
        for num_str in result.split(','):
            num_str = num_str.strip()
            if num_str.isdigit():
                idx = int(num_str)
                if 0 <= idx < len(rerank_candidates):
                    selected_indices.append(idx)

        # 선택된 문서 ID 반환
        if selected_indices:
            return [rerank_candidates[i]['docid'] for i in selected_indices[:top_k]]

        # Fallback: 상위 top_k개
        return [doc['docid'] for doc in docs[:top_k]]

    except Exception as e:
        # 실패 시 상위 top_k개 반환
        return [doc['docid'] for doc in docs[:top_k]]

def process_query(eval_id, msg, use_multi_query=True, num_variants=4):
    """
    단일 쿼리 처리

    Args:
        eval_id: 평가 ID
        msg: 메시지 (str 또는 List[Dict])
        use_multi_query: Multi-Query Generation 사용 여부
        num_variants: 생성할 변형 개수
    """
    # 일반 대화 처리
    if eval_id in SMALLTALK_IDS:
        return {'eval_id': eval_id, 'topk': []}

    # 1. 쿼리 재작성 (멀티턴 대화 맥락 통합)
    rewritten_query = rewrite_query_with_context(msg)

    # 2. Multi-Query Generation
    if use_multi_query:
        queries = generate_multi_queries(rewritten_query, num_variants=num_variants)
    else:
        queries = [rewritten_query]

    # 3. 각 쿼리로 검색
    all_bm25_results = []
    all_bgem3_results = []

    for q in queries:
        # BM25 검색
        bm25_results = search_bm25(q, top_k=20)
        all_bm25_results.append(bm25_results)

        # BGE-M3 Hybrid 검색
        bgem3_results = search_bgem3_hybrid(q, embeddings_dict, top_k=20, max_length=128)
        all_bgem3_results.append(bgem3_results)

    # 4. RRF Fusion
    # 4-1. BM25 결과들 융합
    fused_bm25 = rrf_fusion(all_bm25_results, k=60)

    # 4-2. BGE-M3 결과들 융합
    fused_bgem3 = rrf_fusion(all_bgem3_results, k=60)

    # 4-3. BM25 + BGE-M3 최종 융합
    final_fused = rrf_fusion([fused_bm25[:20], fused_bgem3[:20]], k=60)

    # 5. LLM Reranking
    topk_docs = llm_rerank(rewritten_query, final_fused, top_k=3)

    return {'eval_id': eval_id, 'topk': topk_docs}

def run_experiment(use_multi_query=True, num_variants=4):
    """
    실험 실행

    Args:
        use_multi_query: Multi-Query Generation 사용 여부
        num_variants: 생성할 변형 개수
    """
    # 평가 데이터 로드
    with open('../data/eval.jsonl', 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"Multi-Query Generation v2 실험")
    print(f"Multi-Query: {use_multi_query}, Variants: {num_variants if use_multi_query else 1}")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc="Processing queries"):
        eval_id = item['eval_id']
        msg = item['msg']

        result = process_query(eval_id, msg, use_multi_query=use_multi_query, num_variants=num_variants)
        results.append(result)

    # 결과 저장
    if use_multi_query:
        output_path = f'multi_query_v2_variants{num_variants}_submission.csv'
    else:
        output_path = 'multi_query_v2_baseline_submission.csv'

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 통계
    topk_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for result in results:
        topk_counts[len(result['topk'])] += 1

    print(f"\n{'='*80}")
    print(f"✅ 완료: {output_path}")
    print(f"{'='*80}")
    print(f"\nTopK 분포:")
    print(f"  TopK=0: {topk_counts[0]:3d}개 ({topk_counts[0]/len(results)*100:5.1f}%)")
    print(f"  TopK=1: {topk_counts[1]:3d}개 ({topk_counts[1]/len(results)*100:5.1f}%)")
    print(f"  TopK=2: {topk_counts[2]:3d}개 ({topk_counts[2]/len(results)*100:5.1f}%)")
    print(f"  TopK=3: {topk_counts[3]:3d}개 ({topk_counts[3]/len(results)*100:5.1f}%)")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Elasticsearch 연결 확인
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        exit(1)

    print("✅ Elasticsearch 연결 성공")

    # Experiment 1: Multi-Query Generation (4 variants)
    print("\n" + "="*80)
    print("Experiment 1: Multi-Query (4 variants)")
    print("="*80)
    run_experiment(use_multi_query=True, num_variants=4)

    # Experiment 2: Multi-Query Generation (3 variants) - 더 빠름
    print("\n" + "="*80)
    print("Experiment 2: Multi-Query (3 variants)")
    print("="*80)
    run_experiment(use_multi_query=True, num_variants=3)
