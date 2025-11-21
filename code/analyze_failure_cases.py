"""
실패 케이스 분석 도구
- 어떤 쿼리에서 점수를 잃고 있는지 분석
- Ground truth가 없으므로 submission 간 비교 + 수동 검증
"""

import json
from collections import defaultdict
from elasticsearch import Elasticsearch

def load_submission(path):
    """Submission 파일 로드"""
    with open(path, 'r') as f:
        return {json.loads(line)['eval_id']: json.loads(line) for line in f}

def load_eval():
    """Eval 데이터 로드"""
    with open('../data/eval.jsonl', 'r') as f:
        return {json.loads(line)['eval_id']: json.loads(line) for line in f}

def analyze_low_confidence_queries(es):
    """
    낮은 BM25 점수를 받은 쿼리 분석
    - 이런 쿼리들이 점수를 크게 깎아먹을 가능성이 높음
    """
    eval_data = load_eval()

    # 일반 대화 제외
    smalltalk_ids = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

    low_score_queries = []

    for eval_id, item in eval_data.items():
        if eval_id in smalltalk_ids:
            continue

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # BM25 검색
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
                'size': 10
            }
        )

        if not response['hits']['hits']:
            max_score = 0
            top_docs = []
        else:
            max_score = response['hits']['hits'][0]['_score']
            top_docs = [hit['_source']['docid'] for hit in response['hits']['hits'][:5]]

        # 낮은 점수 케이스
        if max_score < 10.0:
            low_score_queries.append({
                'eval_id': eval_id,
                'query': query,
                'max_score': max_score,
                'top_docs': top_docs,
                'msg': item['msg']
            })

    # 점수 순 정렬
    low_score_queries.sort(key=lambda x: x['max_score'])

    return low_score_queries

def categorize_query_types(low_score_queries):
    """
    실패 쿼리를 타입별로 분류
    """
    categories = {
        'context_dependent': [],  # 이전 대화 맥락 필요
        'ambiguous': [],           # 모호한 쿼리
        'specific_entity': [],     # 특정 개체명 포함
        'abstract': [],            # 추상적 개념
        'other': []
    }

    for item in low_score_queries:
        query = item['query']
        msg = item['msg']

        # 이전 대화 맥락이 필요한지 확인
        if isinstance(msg, list) and len(msg) > 1:
            categories['context_dependent'].append(item)
        # "그 이유", "그것", "이것" 등 대명사 포함
        elif any(word in query for word in ['그 ', '그것', '이것', '이거', '저것', '저거']):
            categories['ambiguous'].append(item)
        # 고유명사 포함 (대문자로 시작하는 영어 단어)
        elif any(word[0].isupper() for word in query.split() if word[0].isalpha()):
            categories['specific_entity'].append(item)
        # 짧은 쿼리 (추상적일 가능성)
        elif len(query) < 15:
            categories['abstract'].append(item)
        else:
            categories['other'].append(item)

    return categories

def print_analysis(categories):
    """분석 결과 출력"""
    print(f"\n{'='*80}")
    print(f"실패 케이스 분석 결과")
    print(f"{'='*80}\n")

    total = sum(len(items) for items in categories.values())
    print(f"총 낮은 점수 쿼리: {total}개\n")

    for cat_name, items in categories.items():
        if not items:
            continue

        print(f"\n{'='*80}")
        print(f"📊 {cat_name.upper()} ({len(items)}개)")
        print(f"{'='*80}\n")

        for i, item in enumerate(items[:5], 1):  # 각 카테고리 상위 5개만
            print(f"[{i}] ID {item['eval_id']}: {item['query']}")
            print(f"    Max Score: {item['max_score']:.2f}")

            # 멀티턴 대화인 경우
            if isinstance(item['msg'], list) and len(item['msg']) > 1:
                print(f"    이전 대화:")
                for msg in item['msg'][:-1]:
                    role = "👤" if msg['role'] == 'user' else "🤖"
                    print(f"      {role} {msg['content'][:50]}...")

            print(f"    Top docs: {item['top_docs'][:2]}")
            print()

def suggest_solutions(categories):
    """해결책 제안"""
    print(f"\n{'='*80}")
    print(f"💡 해결책 제안")
    print(f"{'='*80}\n")

    solutions = []

    if categories['context_dependent']:
        solutions.append({
            'problem': f"Context-Dependent 쿼리 ({len(categories['context_dependent'])}개)",
            'solution': "멀티턴 대화 컨텍스트를 쿼리에 통합",
            'priority': 'HIGH',
            'expected_gain': '+0.1~0.15'
        })

    if categories['ambiguous']:
        solutions.append({
            'problem': f"Ambiguous 쿼리 ({len(categories['ambiguous'])}개)",
            'solution': "대명사를 구체적 명사로 치환 (Query Rewriting)",
            'priority': 'HIGH',
            'expected_gain': '+0.05~0.10'
        })

    if categories['specific_entity']:
        solutions.append({
            'problem': f"Specific Entity 쿼리 ({len(categories['specific_entity'])}개)",
            'solution': "Entity 기반 검색 강화 (정확한 매칭)",
            'priority': 'MEDIUM',
            'expected_gain': '+0.03~0.05'
        })

    if categories['abstract']:
        solutions.append({
            'problem': f"Abstract 쿼리 ({len(categories['abstract'])}개)",
            'solution': "Query Expansion (동의어, 관련 키워드)",
            'priority': 'MEDIUM',
            'expected_gain': '+0.02~0.05'
        })

    # 우선순위 순 정렬
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    solutions.sort(key=lambda x: priority_order[x['priority']])

    for i, sol in enumerate(solutions, 1):
        print(f"{i}. [{sol['priority']}] {sol['problem']}")
        print(f"   해결책: {sol['solution']}")
        print(f"   예상 점수 향상: {sol['expected_gain']}")
        print()

    # 총 예상 향상
    total_gain = sum(float(s['expected_gain'].split('~')[1]) for s in solutions)
    current_score = 0.63
    expected_score = current_score + total_gain

    print(f"\n{'='*80}")
    print(f"📈 예상 최종 점수")
    print(f"{'='*80}")
    print(f"현재 점수: {current_score}")
    print(f"예상 향상: +{total_gain:.2f}")
    print(f"예상 최종: {expected_score:.2f}")

    if expected_score >= 0.9:
        print(f"✅ 목표 0.9점 달성 가능!")
    else:
        gap = 0.9 - expected_score
        print(f"⚠️  목표까지 {gap:.2f}점 추가 필요")
        print(f"💡 추가 전략: Dense Retrieval, Hybrid Search 고도화")

def main():
    print("=" * 80)
    print("실패 케이스 분석 도구")
    print("=" * 80)

    # Elasticsearch 연결
    es = Elasticsearch(['http://localhost:9200'])
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공\n")

    # 낮은 점수 쿼리 분석
    print("🔍 BM25 낮은 점수 쿼리 분석 중...")
    low_score_queries = analyze_low_confidence_queries(es)

    print(f"✅ 총 {len(low_score_queries)}개 쿼리 발견 (max_score < 10.0)\n")

    # 카테고리별 분류
    categories = categorize_query_types(low_score_queries)

    # 결과 출력
    print_analysis(categories)

    # 해결책 제안
    suggest_solutions(categories)

    # 저장
    output_path = 'failure_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_low_score': len(low_score_queries),
            'categories': {k: [{'eval_id': item['eval_id'],
                              'query': item['query'],
                              'max_score': item['max_score']}
                             for item in v]
                          for k, v in categories.items()},
            'queries': low_score_queries
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 분석 결과 저장: {output_path}")

if __name__ == "__main__":
    main()
