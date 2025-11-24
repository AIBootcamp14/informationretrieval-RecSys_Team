"""
빠른 실험 파이프라인 - CI/CD 방식

전략:
1. Validation Set (50개)로 빠른 실험
2. 여러 전략 순차적으로 테스트
3. 가장 좋은 전략을 찾아 점진적 개선
4. 최종 전략으로 전체 eval.jsonl 처리
"""

import json
import os
from elasticsearch import Elasticsearch
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import time

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

# ============================================================================
# 핵심 함수들 (solar_semantic_v1.py에서 가져옴)
# ============================================================================

def llm_query_rewriting(query, model="solar-mini", temperature=0.3, max_tokens=100):
    """LLM으로 쿼리 개선"""
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
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        improved = response.choices[0].message.content.strip()
        improved = improved.replace('"', '').replace("'", '').strip()
        return improved
        
    except Exception as e:
        return query

def search_bm25(query, top_k=3):
    """BM25 검색 with Semantic Chunking 지원"""
    fetch_size = top_k + 2  # 중복 제거 고려
    
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
        return [], 0.0
    
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
            'content': source['content'][:800]
        })
        
        if len(results) >= top_k:
            break
    
    max_score = response['hits']['hits'][0]['_score']
    return results, max_score

def llm_rerank_top3(query, top3_docs, model="solar-pro"):
    """LLM으로 Top-3 순위 조정"""
    if not top3_docs or len(top3_docs) <= 2:
        return [doc['docid'] for doc in top3_docs]
    
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
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
        
        if len(indices) == 3:
            return [top3_docs[i-1]['docid'] for i in indices]
        else:
            return [doc['docid'] for doc in top3_docs]
            
    except Exception as e:
        return [doc['docid'] for doc in top3_docs]

# ============================================================================
# 실험 전략 정의
# ============================================================================

STRATEGIES = [
    {
        'name': '1. Baseline (BM25 Only)',
        'description': 'BM25 검색만 사용 (Top-3)',
        'use_rewriting': False,
        'use_reranking': False,
        'top_k': 3,
        'threshold': 2.0
    },
    {
        'name': '2. Query Rewriting (Solar-mini)',
        'description': 'Solar-mini로 쿼리 개선',
        'use_rewriting': True,
        'rewriting_model': 'solar-mini',
        'use_reranking': False,
        'top_k': 3,
        'threshold': 2.0
    },
    {
        'name': '3. Query Rewriting (Solar-pro)',
        'description': 'Solar-pro로 쿼리 개선',
        'use_rewriting': True,
        'rewriting_model': 'solar-pro',
        'use_reranking': False,
        'top_k': 3,
        'threshold': 2.0
    },
    {
        'name': '4. BM25 + Reranking',
        'description': 'BM25 Top-3 + Solar-pro Reranking',
        'use_rewriting': False,
        'use_reranking': True,
        'reranking_model': 'solar-pro',
        'top_k': 3,
        'threshold': 2.0
    },
    {
        'name': '5. Full Pipeline (mini)',
        'description': 'Solar-mini Rewriting + Reranking',
        'use_rewriting': True,
        'rewriting_model': 'solar-mini',
        'use_reranking': True,
        'reranking_model': 'solar-pro',
        'top_k': 3,
        'threshold': 2.0
    },
    {
        'name': '6. Full Pipeline (pro)',
        'description': 'Solar-pro Rewriting + Reranking',
        'use_rewriting': True,
        'rewriting_model': 'solar-pro',
        'use_reranking': True,
        'reranking_model': 'solar-pro',
        'top_k': 3,
        'threshold': 2.0
    },
    {
        'name': '7. Top-5 + Reranking',
        'description': 'BM25 Top-5 → Solar-pro Reranking → Top-3',
        'use_rewriting': False,
        'use_reranking': True,
        'reranking_model': 'solar-pro',
        'top_k': 5,
        'threshold': 2.0
    },
    {
        'name': '8. Full Pipeline Top-5 (pro)',
        'description': 'Solar-pro Rewriting + Top-5 + Reranking',
        'use_rewriting': True,
        'rewriting_model': 'solar-pro',
        'use_reranking': True,
        'reranking_model': 'solar-pro',
        'top_k': 5,
        'threshold': 2.0
    }
]

# ============================================================================
# 실험 실행 함수
# ============================================================================

def run_experiment(eval_path, strategy):
    """
    하나의 전략으로 실험 실행
    
    Returns:
        results (list): 검색 결과
        stats (dict): 통계 정보
    """
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]
    
    results = []
    start_time = time.time()
    
    for item in tqdm(eval_data, desc=strategy['name']):
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
        
        # 1단계: Query Rewriting (옵션)
        if strategy.get('use_rewriting'):
            model = strategy.get('rewriting_model', 'solar-mini')
            improved_query = llm_query_rewriting(query, model=model)
            search_query = f"{improved_query} {query}"
        else:
            search_query = query
        
        # 2단계: BM25 검색
        top_k = strategy.get('top_k', 3)
        top_docs, max_score = search_bm25(search_query, top_k=top_k)
        
        if not top_docs:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue
        
        # Threshold 체크
        threshold = strategy.get('threshold', 2.0)
        if max_score < threshold:
            results.append({
                'eval_id': eval_id,
                'topk': []
            })
            continue
        
        # 3단계: Reranking (옵션)
        if strategy.get('use_reranking'):
            model = strategy.get('reranking_model', 'solar-pro')
            top3_docs = top_docs[:3]  # 항상 Top-3만 반환
            reranked_ids = llm_rerank_top3(query, top3_docs, model=model)
            final_docs = reranked_ids
        else:
            final_docs = [doc['docid'] for doc in top_docs[:3]]
        
        results.append({
            'eval_id': eval_id,
            'topk': final_docs
        })
    
    elapsed_time = time.time() - start_time
    
    # 통계 계산
    topk_counts = {}
    for r in results:
        count = len(r['topk'])
        topk_counts[count] = topk_counts.get(count, 0) + 1
    
    stats = {
        'elapsed_time': elapsed_time,
        'topk_counts': topk_counts,
        'total': len(results)
    }
    
    return results, stats

# ============================================================================
# 메인 파이프라인
# ============================================================================

def main():
    print("="*80)
    print("빠른 실험 파이프라인 - CI/CD 방식")
    print("="*80)
    
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return
    
    print("✅ Elasticsearch 연결 성공")
    
    if not upstage_api_key:
        print("❌ UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    print("✅ Upstage Solar API Key 확인")
    
    # Validation set 경로
    val_path = '../data/val.jsonl'
    
    if not os.path.exists(val_path):
        print(f"❌ Validation set이 없습니다: {val_path}")
        print("   create_validation_set_solar.py를 먼저 실행하세요")
        return
    
    print(f"✅ Validation set 확인: {val_path}")
    
    # 모든 전략 실행
    print(f"\n{'='*80}")
    print(f"순차적 실험 시작 (총 {len(STRATEGIES)}개 전략)")
    print(f"{'='*80}\n")
    
    experiment_results = []
    
    for idx, strategy in enumerate(STRATEGIES, 1):
        print(f"\n[{idx}/{len(STRATEGIES)}] {strategy['name']}")
        print(f"설명: {strategy['description']}")
        print("-"*80)
        
        # 실험 실행
        results, stats = run_experiment(val_path, strategy)
        
        # 결과 출력
        print(f"\n⏱️  소요 시간: {stats['elapsed_time']:.1f}초")
        print(f"📊 TopK 분포:")
        for k in sorted(stats['topk_counts'].keys()):
            count = stats['topk_counts'][k]
            pct = count / stats['total'] * 100
            print(f"   TopK={k}: {count:3d}개 ({pct:5.1f}%)")
        
        # 결과 저장
        experiment_results.append({
            'strategy': strategy,
            'results': results,
            'stats': stats
        })
        
        print("="*80)
    
    # 최종 요약
    print(f"\n{'='*80}")
    print(f"✅ 전체 실험 완료")
    print(f"{'='*80}")
    
    print(f"\n📊 전략별 요약:")
    print(f"\n{'전략':40s} {'시간':>8s} {'TopK=3':>10s}")
    print("-"*80)
    
    for exp in experiment_results:
        strategy_name = exp['strategy']['name']
        elapsed = exp['stats']['elapsed_time']
        top3_count = exp['stats']['topk_counts'].get(3, 0)
        top3_pct = top3_count / exp['stats']['total'] * 100
        
        print(f"{strategy_name:40s} {elapsed:7.1f}s {top3_count:3d}개 ({top3_pct:5.1f}%)")
    
    print("="*80)
    
    print(f"\n💡 다음 단계:")
    print(f"  1. 위 결과를 분석하여 가장 좋은 전략 선택")
    print(f"  2. 선택한 전략으로 전체 eval.jsonl 처리")
    print(f"  3. 제출 파일 생성 및 평가")
    print("="*80)

if __name__ == "__main__":
    main()
