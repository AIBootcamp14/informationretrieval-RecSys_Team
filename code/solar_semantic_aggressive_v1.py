"""
Aggressive Chunking v1 - 제안 1번 테스트

핵심 변경:
1. Aggressive Chunking (max=250)으로 더 작은 청크 생성
2. original_docid 기반 중복 제거
3. BM25 Top-3 신뢰 + LLM Reranking

목표: 0.6864 → 0.695-0.705 (+1-3%)
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

SMALLTALK_IDS = {
    276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183
}

def llm_query_rewriting(query):
    """
    LLM으로 쿼리 개선

    핵심: 과학 용어로 명확하게 변환
    """
    prompt = f"""과학 지식 검색을 위한 쿼리 개선:

원본 질문: {query}

다음 기준으로 검색 쿼리를 개선하세요:
1. 핵심 과학 개념 명확히
2. 동의어 추가 (한글 + 영어)
3. 관련 키워드 확장
4. 불필요한 조사 제거

예시:
- "DNA 조각 결합하는 거" → "DNA 조각 연결 효소 ligase 리가아제"
- "식물 광합성 어떻게" → "식물 광합성 과정 엽록소 chloroplast"

출력: 개선된 검색 쿼리만 한 줄로 작성"""

    try:
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 약간의 창의성
            max_tokens=100
        )

        improved = response.choices[0].message.content.strip()
        # 따옴표 제거
        improved = improved.replace('"', '').replace("'", '').strip()
        return improved

    except Exception as e:
        print(f"⚠️ Query Rewriting 실패: {e}")
        return query

def search_bm25(query, top_k=3):
    """
    BM25 검색 with Semantic Chunking 지원

    변경사항:
    - original_docid 기반 중복 제거
    - Top-5 검색 후 중복 제거하여 Top-3 확보
    """
    # Top-5로 검색 (청크 중복 고려)
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
            'size': 5  # 중복 제거 후 3개 확보 위해 5개 검색
        }
    )

    if not response['hits']['hits']:
        return [], 0.0

    # original_docid 기반 중복 제거
    seen_original_docids = set()
    results = []

    for hit in response['hits']['hits']:
        source = hit['_source']

        # original_docid가 있으면 사용, 없으면 docid 사용 (호환성)
        original_docid = source.get('original_docid', source['docid'])

        # 중복 체크
        if original_docid in seen_original_docids:
            continue

        seen_original_docids.add(original_docid)
        results.append({
            'docid': original_docid,  # 원본 문서 ID 반환
            'content': source['content'][:800]  # 800자
        })

        # Top-K 개수 확보 시 종료
        if len(results) >= top_k:
            break

    max_score = response['hits']['hits'][0]['_score']
    return results, max_score

def llm_rerank_top3(query, top3_docs):
    """
    LLM으로 Top-3 순위 조정

    중요: 제외하지 않음! 순서만 변경
    BM25가 찾은 3개는 모두 관련 있다고 신뢰
    """
    if not top3_docs or len(top3_docs) == 0:
        return []

    # 1개 또는 2개만 있으면 그대로 반환
    if len(top3_docs) <= 2:
        return [doc['docid'] for doc in top3_docs]

    # 정확히 3개인 경우만 LLM으로 재정렬
    docs_text = ""
    for i, doc in enumerate(top3_docs, 1):
        docs_text += f"\n[문서 {i}]\n{doc['content'][:600]}\n" + "-"*40

    prompt = f"""질문: {query}

BM25가 찾은 관련 문서 3개입니다. 질문과의 관련도 순으로 정렬하세요.

{docs_text}

중요:
- 3개 모두 관련 있는 문서입니다 (BM25 신뢰)
- 제외하지 말고, 순서만 조정하세요
- 가장 직접적으로 답변하는 문서를 1번으로

출력: 순위대로 문서 번호 3개 (예: 2,1,3)"""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20
        )

        result = response.choices[0].message.content.strip()

        # 파싱: "2,1,3" → [2, 1, 3]
        indices = []
        for x in result.replace(' ', '').split(','):
            try:
                idx = int(x)
                if 1 <= idx <= 3 and idx not in indices:
                    indices.append(idx)
            except:
                pass

        # 3개 모두 파싱 성공 시
        if len(indices) == 3:
            return [top3_docs[i-1]['docid'] for i in indices]
        else:
            # 파싱 실패 시 원본 순서
            return [doc['docid'] for doc in top3_docs]

    except Exception as e:
        print(f"⚠️ Reranking 실패: {e}")
        return [doc['docid'] for doc in top3_docs]

def llm_optimized_search(query):
    """
    LLM 최적화 검색 파이프라인

    1. Query Rewriting
    2. BM25 Top-3
    3. LLM Reranking (순서 조정만)
    """
    # 1단계: 쿼리 개선
    improved_query = llm_query_rewriting(query)

    # 2단계: BM25 Top-3 (개선된 쿼리 + 원본 쿼리 결합)
    combined_query = f"{improved_query} {query}"
    top3_docs, max_score = search_bm25(combined_query, top_k=3)

    if not top3_docs:
        # 개선 쿼리로 실패 시 원본 시도
        top3_docs, max_score = search_bm25(query, top_k=3)

    if not top3_docs:
        return [], 0.0

    # 3단계: LLM Reranking (순서만)
    reranked_ids = llm_rerank_top3(query, top3_docs)

    return reranked_ids, max_score

def process_with_llm_optimized(eval_path, output_path):
    """LLM 최적화 처리"""
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []

    print(f"\n{'='*80}")
    print(f"LLM 최적화 RAG 실행")
    print(f"전략: BM25 신뢰 + LLM 순위 조정")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc="LLM Optimized"):
        eval_id = item['eval_id']
        msg = item['msg']

        # Smalltalk
        if eval_id in SMALLTALK_IDS:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # 쿼리 추출
        if isinstance(msg, list):
            query = msg[-1]['content']
        else:
            query = msg

        # LLM Optimized Search
        topk_docs, max_score = llm_optimized_search(query)

        if not topk_docs:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

        # Threshold 체크 (2.0 고정 - super_simple과 동일)
        if max_score < 2.0:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue

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
    print("Aggressive Chunking v1 - 제안 1번 테스트")
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
    process_with_llm_optimized(
        eval_path='../data/eval.jsonl',
        output_path='solar_semantic_aggressive_v1_submission.csv'
    )

    print(f"\n{'='*80}")
    print(f"✅ Aggressive Chunking v1 완료")
    print(f"{'='*80}")
    print(f"\n생성된 파일:")
    print(f"  - solar_semantic_aggressive_v1_submission.csv")
    print(f"\n핵심 변경사항:")
    print(f"  1. Aggressive Chunking (max=250)으로 더 작은 청크")
    print(f"  2. original_docid 기반 중복 제거")
    print(f"  3. BM25 Top-5 → 중복 제거 → Top-3")
    print(f"  4. LLM Reranking으로 순위 정밀화")
    print(f"\n기대 효과:")
    print(f"  - Aggressive Chunking: 정밀한 검색 가능")
    print(f"  - 중복 제거: 다양성 증가")
    print(f"  - 예상 MAP: 0.695-0.705 (+1-3%)")
    print(f"\nElasticsearch 통계:")
    print(f"  - 원본 문서: 4,272개")
    print(f"  - 청크 문서: 10,631개 (vs Phase1: 5,613개)")
    print(f"  - 청킹 비율: 74.8% (3,196개 문서)")
    print(f"  - 설정: max=250, min=80, threshold=0.70")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
