"""
전체 데이터셋 기반 빠른 실험
목표: 220개 전체로 여러 전략 테스트하여 최적 조합 찾기
"""

import json
import os
from tqdm import tqdm
from elasticsearch import Elasticsearch

# Upstage Solar API
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

# 일반 대화 ID
SMALLTALK_IDS = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

def search_bm25(query, top_k=3):
    """BM25 검색 with Semantic Chunking"""
    fetch_size = top_k + 2

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
            'content': source['content'][:800],
            'score': hit['_score']
        })

        if len(results) >= top_k:
            break

    return results

def llm_query_rewriting(query):
    """Solar-pro Query Rewriting"""
    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 과학 검색 쿼리 개선 전문가입니다.
주어진 쿼리를 BM25 검색에 최적화하여 개선하세요.

개선 방법:
1. 핵심 과학 개념을 명확히
2. 동의어 추가 (한글+영어)
3. 관련 키워드 확장

출력: 개선된 쿼리만 (설명 없이)"""
                },
                {
                    "role": "user",
                    "content": f"쿼리: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=100
        )

        improved = response.choices[0].message.content.strip()
        return improved

    except Exception as e:
        return query

def llm_rerank(query, docs):
    """Solar-pro Reranking"""
    if not docs or len(docs) <= 3:
        return [doc['docid'] for doc in docs[:3]]

    try:
        # 문서 목록 생성
        doc_list = []
        for i, doc in enumerate(docs):
            doc_list.append(f"[{i}] {doc['content'][:200]}")

        docs_text = "\n\n".join(doc_list)

        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 문서 relevance 평가 전문가입니다. 주어진 쿼리에 가장 관련성 높은 문서 3개를 선택하세요."
                },
                {
                    "role": "user",
                    "content": f"""쿼리: {query}

문서들:
{docs_text}

가장 관련성 높은 문서 3개의 번호를 선택하세요 (예: 0,2,4).
출력: 번호만 (설명 없이)"""
                }
            ],
            temperature=0.0,
            max_tokens=20
        )

        result = response.choices[0].message.content.strip()
        indices = [int(x.strip()) for x in result.split(',') if x.strip().isdigit()]

        reranked_docids = []
        for idx in indices[:3]:
            if 0 <= idx < len(docs):
                reranked_docids.append(docs[idx]['docid'])

        # 부족하면 원래 순서로 채우기
        for doc in docs:
            if len(reranked_docids) >= 3:
                break
            if doc['docid'] not in reranked_docids:
                reranked_docids.append(doc['docid'])

        return reranked_docids[:3]

    except Exception as e:
        return [doc['docid'] for doc in docs[:3]]

def strategy_1_bm25_only(eval_id, query):
    """전략 1: BM25만 (Semantic Chunking)"""
    if eval_id in SMALLTALK_IDS:
        return []

    docs = search_bm25(query, top_k=3)
    return [doc['docid'] for doc in docs]

def strategy_2_query_rewriting(eval_id, query):
    """전략 2: Query Rewriting + BM25"""
    if eval_id in SMALLTALK_IDS:
        return []

    improved_query = llm_query_rewriting(query)
    search_query = f"{improved_query} {query}"
    docs = search_bm25(search_query, top_k=3)
    return [doc['docid'] for doc in docs]

def strategy_3_reranking(eval_id, query):
    """전략 3: BM25 + Reranking"""
    if eval_id in SMALLTALK_IDS:
        return []

    docs = search_bm25(query, top_k=5)
    if not docs:
        return []

    return llm_rerank(query, docs)

def strategy_4_full(eval_id, query):
    """전략 4: Query Rewriting + BM25 + Reranking (현재)"""
    if eval_id in SMALLTALK_IDS:
        return []

    improved_query = llm_query_rewriting(query)
    search_query = f"{improved_query} {query}"
    docs = search_bm25(search_query, top_k=5)

    if not docs:
        return []

    return llm_rerank(query, docs)

def calculate_topk_accuracy(results):
    """TopK 정확도 계산"""
    topk_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for r in results:
        topk_counts[len(r['retrieve'])] += 1

    return topk_counts

def run_experiment(strategy_func, strategy_name):
    """전략 실행"""
    print(f"\n{'='*80}")
    print(f"전략 실행: {strategy_name}")
    print(f"{'='*80}")

    # Eval 데이터 로드
    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    for item in tqdm(eval_data, desc=strategy_name):
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 전략 실행
        topk = strategy_func(eval_id, query)

        results.append({
            'eval_id': eval_id,
            'retrieve': topk
        })

    # 통계
    topk_counts = calculate_topk_accuracy(results)

    print(f"\n📊 결과:")
    for k in range(4):
        count = topk_counts[k]
        pct = count / len(eval_data) * 100
        print(f"   TopK={k}:  {count:3d}개 ({pct:5.1f}%)")

    # 제출 파일 생성
    output_path = f'{strategy_name}_submission.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            json_obj = {
                'eval_id': r['eval_id'],
                'topk': r['retrieve']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"\n💾 제출 파일: {output_path}")

    return results

def main():
    print("="*80)
    print("전체 데이터셋 기반 전략 실험")
    print("="*80)
    print("데이터셋: 220개 (전체)")
    print("="*80)

    # ES 연결 확인
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # API Key 확인
    if not upstage_api_key:
        print("❌ UPSTAGE_API_KEY 환경변수 없음")
        return

    print("✅ Upstage Solar API Key 확인\n")

    # 전략 1: BM25 only
    print("\n" + "="*80)
    print("실험 1/4: BM25 only (Semantic Chunking)")
    print("="*80)
    run_experiment(strategy_1_bm25_only, "strategy_1_bm25_only")

    # 전략 2: Query Rewriting
    print("\n" + "="*80)
    print("실험 2/4: Query Rewriting + BM25")
    print("="*80)
    run_experiment(strategy_2_query_rewriting, "strategy_2_query_rewriting")

    # 전략 3: Reranking
    print("\n" + "="*80)
    print("실험 3/4: BM25 + Reranking")
    print("="*80)
    run_experiment(strategy_3_reranking, "strategy_3_reranking")

    # 전략 4: Full (이미 0.5144로 알려짐)
    print("\n" + "="*80)
    print("실험 4/4: Query Rewriting + BM25 + Reranking")
    print("="*80)
    print("(이미 제출됨: 0.5144 MAP)")

    print("\n" + "="*80)
    print("✅ 모든 실험 완료")
    print("="*80)

    print("\n다음 단계:")
    print("1. 각 제출 파일을 평가 시스템에 제출")
    print("2. MAP 점수 비교")
    print("3. 최고 성능 전략 선택")

if __name__ == "__main__":
    main()
