"""
High-Impact Query Validation Tool
validation_candidates.json에서 식별된 상위 20개 쿼리에 대해 수동 레이블링
"""

import json
from elasticsearch import Elasticsearch

def load_high_impact_queries():
    """validation_candidates.json에서 high-impact 쿼리 로드"""
    with open('validation_candidates.json', 'r') as f:
        candidates = json.load(f)

    # eval.jsonl에서 전체 정보 로드
    with open('../data/eval.jsonl', 'r') as f:
        eval_data = {json.loads(line)['eval_id']: json.loads(line)
                     for line in f}

    # Candidate에 전체 정보 추가
    for candidate in candidates:
        eval_id = candidate['eval_id']
        if eval_id in eval_data:
            candidate['msg'] = eval_data[eval_id]['msg']

    return candidates

def search_and_display(es, query, top_k=10):
    """검색 결과를 보여주고 정답 선택"""
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

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")

    results = []
    for i, hit in enumerate(response['hits']['hits'], 1):
        docid = hit['_source']['docid']
        score = hit['_score']
        content = hit['_source']['content'][:300]

        print(f"\n[{i}] Score: {score:.2f}")
        print(f"DocID: {docid}")
        print(f"Content: {content}...")

        results.append({
            'docid': docid,
            'score': score,
            'content': hit['_source']['content']
        })

    return results

def show_submission_comparison(candidate):
    """각 submission의 TopK 비교 표시"""
    print(f"\n{'='*80}")
    print(f"기존 Submission들의 답변:")
    print(f"{'='*80}")

    for name, topk in candidate.get('topks', {}).items():
        print(f"\n{name}: {len(topk)}개 문서")
        for i, docid in enumerate(topk[:3], 1):
            print(f"  [{i}] {docid}")

def annotate_high_impact_queries(candidates, es):
    """High-impact 쿼리에 정답 레이블 추가"""
    annotated = []

    print(f"\n{'#'*80}")
    print(f"총 {len(candidates)}개의 High-Impact 쿼리에 대해 레이블링을 진행합니다")
    print(f"{'#'*80}")

    for idx, candidate in enumerate(candidates, 1):
        eval_id = candidate['eval_id']
        query = candidate['query']
        variation = candidate.get('variation', 0)

        print(f"\n\n{'#'*80}")
        print(f"[{idx}/{len(candidates)}] Eval ID: {eval_id}")
        print(f"Variation: {variation}가지 (높을수록 submission간 차이가 큼)")
        print(f"{'#'*80}")

        # 기존 submission 비교
        show_submission_comparison(candidate)

        # 검색 결과 표시
        results = search_and_display(es, query, top_k=10)

        # 정답 입력
        print(f"\n{'='*80}")
        print(f"정답 문서 번호를 입력하세요:")
        print(f"  - 1-10: 해당 문서 선택 (여러 개는 콤마로 구분, 예: 1,2,3)")
        print(f"  - 0: 일반 대화 (문서 없음)")
        print(f"  - s: 건너뛰기")
        print(f"{'='*80}")
        answer = input("> ").strip()

        if answer.lower() == 's':
            print("⏭️  건너뜀")
            continue

        if answer == "0":
            ground_truth = []
        else:
            try:
                indices = [int(x.strip())-1 for x in answer.split(',')]
                ground_truth = [results[i]['docid'] for i in indices
                              if 0 <= i < len(results)]
            except (ValueError, IndexError) as e:
                print(f"⚠️ 입력 오류: {e}. 건너뜀")
                continue

        annotated.append({
            'eval_id': eval_id,
            'query': query,
            'msg': candidate.get('msg', query),
            'ground_truth': ground_truth,
            'variation': variation,
            'confidence': 'manual'
        })

        print(f"✅ 저장: {len(ground_truth)}개 문서")

    return annotated

def save_validation_set(annotated, output_path):
    """validation set 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"✅ Validation set 저장 완료: {output_path}")
    print(f"총 {len(annotated)}개 샘플 레이블링 완료")
    print(f"{'='*80}")

def main():
    print("=" * 80)
    print("High-Impact Query Validation Tool")
    print("=" * 80)

    # Elasticsearch 연결
    es = Elasticsearch(['http://localhost:9200'])
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공\n")

    # High-impact 쿼리 로드
    candidates = load_high_impact_queries()
    print(f"📋 {len(candidates)}개 High-Impact 쿼리 로드됨")
    print(f"💡 이 쿼리들은 여러 submission에서 결과가 가장 많이 다른 쿼리들입니다")

    # Annotation
    annotated = annotate_high_impact_queries(candidates, es)

    # 저장
    if annotated:
        save_validation_set(annotated, 'validation_high_impact.jsonl')
    else:
        print("\n⚠️ 레이블링된 쿼리가 없습니다")

if __name__ == "__main__":
    main()
