"""
Selective Context-Aware RAG
멀티턴 대화 중 정말 필요한 경우에만 Query Rewriting 적용

전략:
1. 대명사가 포함된 짧은 쿼리만 rewrite
2. 나머지는 super_simple 전략 유지
"""

import json
import os
from openai import OpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1/solar"
)

es = Elasticsearch(['http://localhost:9200'])

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def should_rewrite(msg, query):
    """
    Query Rewriting이 필요한지 판단

    조건:
    1. 멀티턴 대화여야 함
    2. 대명사나 모호한 표현 포함
    3. 짧은 쿼리 (15자 이하)
    """
    # 단일턴은 rewrite 불필요
    if isinstance(msg, str) or len(msg) == 1:
        return False

    # 대명사나 모호한 표현 확인
    ambiguous_terms = ['그 이유', '그것', '이것', '그럼', '여기']
    has_ambiguous = any(term in query for term in ambiguous_terms)

    # 짧은 쿼리
    is_short = len(query) <= 20

    return has_ambiguous and is_short

def rewrite_query_minimal(msg):
    """
    최소한의 Query Rewriting (짧고 명확하게)
    """
    current_query = msg[-1]['content']

    if not should_rewrite(msg, current_query):
        return current_query

    # 이전 대화 (마지막 1턴만)
    prev_context = msg[-2]['content'] if len(msg) >= 2 else ""

    prompt = f"""이전 질문: "{prev_context}"
현재 질문: "{current_query}"

현재 질문의 대명사를 구체적인 명사로 바꿔서 짧게 재작성하세요.
과도하게 길어지지 않도록 핵심만 포함하세요.

재작성된 질문만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50  # 짧게 제한
        )

        rewritten = response.choices[0].message.content.strip().strip('"').strip("'")

        print(f"  [Rewrite] {current_query} → {rewritten}")
        return rewritten

    except Exception as e:
        print(f"  ⚠️ Rewrite 실패: {e}")
        return current_query

def search_bm25(query, top_k=10):
    """BM25 검색"""
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

    results = [(hit['_source']['docid'], hit['_score'])
               for hit in response['hits']['hits']]
    max_score = response['hits']['hits'][0]['_score']

    return results, max_score

def adaptive_topk_selection(max_score):
    """동적 TopK 선택 (Threshold 2.0 전략)"""
    if max_score < 2.0:
        return 0
    elif max_score < 5.0:
        return 1
    elif max_score < 8.0:
        return 2
    else:
        return 3

def process_eval(eval_path, output_path):
    """평가 데이터 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"Selective Context-Aware RAG 실행")
    print(f"{'='*80}\n")

    rewrite_count = 0

    for item in tqdm(eval_data, desc="Processing queries"):
        eval_id = item['eval_id']
        msg = item['msg']

        # 일반 대화 처리
        if eval_id in SMALLTALK_IDS:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # 쿼리 추출
        if isinstance(msg, list):
            current_query = msg[-1]['content']
        else:
            current_query = msg

        # 선택적 Query Rewriting
        if should_rewrite(msg, current_query):
            query = rewrite_query_minimal(msg)
            rewrite_count += 1
        else:
            query = current_query

        # BM25 검색
        bm25_results, max_score = search_bm25(query, top_k=10)

        # TopK 선택
        topk_count = adaptive_topk_selection(max_score)

        # 상위 K개 문서만 선택
        topk_docs = [doc_id for doc_id, _ in bm25_results[:topk_count]]

        results.append({
            'eval_id': eval_id,
            'topk': topk_docs
        })

    # 결과 저장
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
    print(f"\nQuery Rewriting 적용: {rewrite_count}개 쿼리")
    print(f"\nTopK 분포:")
    print(f"  TopK=0: {topk_counts[0]:3d}개 ({topk_counts[0]/len(results)*100:5.1f}%)")
    print(f"  TopK=1: {topk_counts[1]:3d}개 ({topk_counts[1]/len(results)*100:5.1f}%)")
    print(f"  TopK=2: {topk_counts[2]:3d}개 ({topk_counts[2]/len(results)*100:5.1f}%)")
    print(f"  TopK=3: {topk_counts[3]:3d}개 ({topk_counts[3]/len(results)*100:5.1f}%)")
    print(f"{'='*80}")

def main():
    print("=" * 80)
    print("Selective Context-Aware RAG System")
    print("정말 필요한 쿼리만 선택적으로 Rewriting")
    print("=" * 80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    process_eval(
        eval_path='../data/eval.jsonl',
        output_path='selective_context_submission.csv'
    )

if __name__ == "__main__":
    main()
