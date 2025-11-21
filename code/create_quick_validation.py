"""
Quick Validation Set Generator
High-impact 쿼리에 대해 BM25 기반 pseudo-labels 생성
수동 작업 없이 빠르게 validation set 구축
"""

import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

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

def create_pseudo_validation(candidates, es, confidence_threshold=10.0, top_k=3):
    """
    High-impact 쿼리에 대해 pseudo validation 생성
    - BM25 상위 K개를 정답으로 가정
    - 신뢰도 레벨 부여
    """
    # 일반 대화 ID
    smalltalk_ids = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

    validation_set = []
    stats = {
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0,
        'smalltalk': 0,
        'total': len(candidates)
    }

    print(f"\n{'='*80}")
    print(f"Pseudo Validation 생성 중...")
    print(f"{'='*80}\n")

    for candidate in tqdm(candidates, desc="Processing queries"):
        eval_id = candidate['eval_id']
        query = candidate['query']
        variation = candidate.get('variation', 0)

        # 일반 대화
        if eval_id in smalltalk_ids:
            validation_set.append({
                'eval_id': eval_id,
                'query': query,
                'msg': candidate.get('msg', query),
                'ground_truth': [],
                'confidence': 'smalltalk',
                'max_score': 0.0,
                'variation': variation
            })
            stats['smalltalk'] += 1
            continue

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
            continue

        max_score = response['hits']['hits'][0]['_score']
        top_docs = [hit['_source']['docid'] for hit in response['hits']['hits'][:top_k]]

        # 신뢰도 레벨
        if max_score >= confidence_threshold:
            confidence = 'high'
            stats['high_confidence'] += 1
        elif max_score >= 5.0:
            confidence = 'medium'
            stats['medium_confidence'] += 1
        else:
            confidence = 'low'
            stats['low_confidence'] += 1

        validation_set.append({
            'eval_id': eval_id,
            'query': query,
            'msg': candidate.get('msg', query),
            'ground_truth': top_docs,
            'confidence': confidence,
            'max_score': max_score,
            'variation': variation
        })

    # 통계 출력
    print(f"\n{'='*80}")
    print(f"✅ Pseudo Validation Set 생성 완료")
    print(f"{'='*80}")
    print(f"통계:")
    print(f"  - Total queries: {stats['total']}")
    print(f"  - High confidence (score >= {confidence_threshold}): {stats['high_confidence']}")
    print(f"  - Medium confidence (score >= 5.0): {stats['medium_confidence']}")
    print(f"  - Low confidence (score < 5.0): {stats['low_confidence']}")
    print(f"  - Smalltalk: {stats['smalltalk']}")
    print(f"{'='*80}")

    return validation_set

def evaluate_submission(submission_path, validation_path):
    """
    Submission 파일을 pseudo validation set으로 평가
    """
    # Load validation set
    with open(validation_path, 'r') as f:
        val_data = [json.loads(line) for line in f]

    # Load submission
    with open(submission_path, 'r') as f:
        sub_data = {json.loads(line)['eval_id']: json.loads(line)
                   for line in f}

    # 신뢰도별 평가
    results = {
        'high': {'total': 0, 'ap_sum': 0, 'overlap_sum': 0},
        'medium': {'total': 0, 'ap_sum': 0, 'overlap_sum': 0},
        'low': {'total': 0, 'ap_sum': 0, 'overlap_sum': 0},
        'smalltalk': {'total': 0, 'correct': 0},
        'overall': {'total': 0, 'map_sum': 0}
    }

    for val_item in val_data:
        eval_id = val_item['eval_id']
        if eval_id not in sub_data:
            continue

        confidence = val_item['confidence']
        ground_truth = val_item['ground_truth']
        predicted = sub_data[eval_id]['topk']

        results['overall']['total'] += 1

        if confidence == 'smalltalk':
            results['smalltalk']['total'] += 1
            if len(predicted) == 0:
                results['smalltalk']['correct'] += 1
        else:
            results[confidence]['total'] += 1

            # Calculate Average Precision for this query
            if ground_truth and predicted:
                ap = 0.0
                hits = 0
                for i, pred_doc in enumerate(predicted, 1):
                    if pred_doc in ground_truth:
                        hits += 1
                        precision_at_i = hits / i
                        ap += precision_at_i

                if hits > 0:
                    ap /= min(len(ground_truth), len(predicted))

                results[confidence]['ap_sum'] += ap
                results['overall']['map_sum'] += ap

                # Overlap 계산
                overlap = len(set(predicted[:3]) & set(ground_truth))
                results[confidence]['overlap_sum'] += overlap

    # 결과 출력
    print(f"\n{'='*80}")
    print(f"Validation 결과: {submission_path}")
    print(f"{'='*80}")

    # Overall MAP
    if results['overall']['total'] > 0:
        overall_map = results['overall']['map_sum'] / results['overall']['total']
        print(f"\n📊 Overall MAP (on validation set): {overall_map:.4f}")

    # 신뢰도별
    for conf in ['high', 'medium', 'low']:
        if results[conf]['total'] > 0:
            avg_ap = results[conf]['ap_sum'] / results[conf]['total']
            avg_overlap = results[conf]['overlap_sum'] / results[conf]['total']

            print(f"\n{conf.upper()} Confidence ({results[conf]['total']}개):")
            print(f"  - Avg AP: {avg_ap:.4f}")
            print(f"  - Avg Overlap (Top-3): {avg_overlap:.2f}")

    # Smalltalk
    if results['smalltalk']['total'] > 0:
        acc = results['smalltalk']['correct'] / results['smalltalk']['total'] * 100
        print(f"\nSmalltalk ({results['smalltalk']['total']}개):")
        print(f"  - Accuracy: {acc:.1f}%")

    print(f"{'='*80}")

    return results

def main():
    print("=" * 80)
    print("Quick Validation Set Generator")
    print("=" * 80)

    # Elasticsearch 연결
    es = Elasticsearch(['http://localhost:9200'])
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # High-impact 쿼리 로드
    candidates = load_high_impact_queries()
    print(f"📋 {len(candidates)}개 High-Impact 쿼리 로드")

    # Pseudo validation set 생성
    validation_set = create_pseudo_validation(
        candidates, es,
        confidence_threshold=10.0,
        top_k=3
    )

    # 저장
    output_path = 'validation_high_impact.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in validation_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 저장 완료: {output_path}")

    # 기존 submission 평가
    print(f"\n{'='*80}")
    print(f"기존 Submissions 평가")
    print(f"{'='*80}")

    submissions = [
        'super_simple_submission.csv',
        'simplified_submission.csv',
        'rag_1119_submission.csv',
        'rag_threshold3_submission.csv'
    ]

    best_score = 0
    best_submission = None

    for sub_file in submissions:
        try:
            result = evaluate_submission(sub_file, output_path)
            if result['overall']['total'] > 0:
                map_score = result['overall']['map_sum'] / result['overall']['total']
                if map_score > best_score:
                    best_score = map_score
                    best_submission = sub_file
        except FileNotFoundError:
            print(f"\n⚠️ {sub_file} 파일 없음")
        except Exception as e:
            print(f"\n⚠️ {sub_file} 평가 실패: {e}")

    if best_submission:
        print(f"\n{'='*80}")
        print(f"🏆 Best Submission on Validation Set:")
        print(f"   {best_submission} - MAP {best_score:.4f}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
