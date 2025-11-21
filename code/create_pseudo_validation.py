"""
전략 2: Pseudo-Labeling
- BM25 상위 K개 문서를 정답으로 가정
- 빠르게 validation set 생성
- Cross-validation으로 다른 방법 테스트
"""

import json
import random
from elasticsearch import Elasticsearch
from tqdm import tqdm

class PseudoValidator:
    def __init__(self, threshold_score=10.0, top_k=3):
        self.es = Elasticsearch(['http://localhost:9200'])
        self.threshold_score = threshold_score
        self.top_k = top_k

        if not self.es.ping():
            raise ConnectionError("Elasticsearch 연결 실패")
        print("✅ Elasticsearch 연결 성공")

    def create_pseudo_labels(self, eval_path, output_path, confidence_threshold=10.0):
        """
        고신뢰도 pseudo labels 생성

        전략:
        1. BM25 최고 점수가 threshold 이상인 쿼리만 선택 (high confidence)
        2. 상위 K개를 정답으로 가정
        3. 낮은 점수 쿼리는 제외 (노이즈 방지)
        """
        with open(eval_path, 'r') as f:
            eval_data = [json.loads(line) for line in f]

        # 일반 대화 ID
        smalltalk_ids = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

        validation_set = []
        stats = {
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'smalltalk': 0
        }

        for item in tqdm(eval_data, desc="Generating pseudo labels"):
            eval_id = item['eval_id']

            # 쿼리 추출
            if isinstance(item['msg'], list):
                query = item['msg'][-1]['content']
            else:
                query = item['msg']

            # 일반 대화
            if eval_id in smalltalk_ids:
                validation_set.append({
                    'eval_id': eval_id,
                    'query': query,
                    'msg': item['msg'],
                    'ground_truth': [],
                    'confidence': 'smalltalk',
                    'max_score': 0.0
                })
                stats['smalltalk'] += 1
                continue

            # BM25 검색
            response = self.es.search(
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
            top_docs = [hit['_source']['docid'] for hit in response['hits']['hits'][:self.top_k]]

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
                'msg': item['msg'],
                'ground_truth': top_docs,
                'confidence': confidence,
                'max_score': max_score
            })

        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in validation_set:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n{'='*80}")
        print(f"✅ Pseudo Validation Set 생성 완료: {output_path}")
        print(f"{'='*80}")
        print(f"통계:")
        print(f"  - High confidence (score >= {confidence_threshold}): {stats['high_confidence']}")
        print(f"  - Medium confidence (score >= 5.0): {stats['medium_confidence']}")
        print(f"  - Low confidence (score < 5.0): {stats['low_confidence']}")
        print(f"  - Smalltalk: {stats['smalltalk']}")
        print(f"  - Total: {len(validation_set)}")
        print(f"{'='*80}")

        return validation_set

    def evaluate_submission(self, submission_path, validation_path):
        """
        Submission 파일을 pseudo validation set으로 평가
        """
        # Load validation set
        with open(validation_path, 'r') as f:
            val_data = {json.loads(line)['eval_id']: json.loads(line)
                       for line in f}

        # Load submission
        with open(submission_path, 'r') as f:
            sub_data = {json.loads(line)['eval_id']: json.loads(line)
                       for line in f}

        # 신뢰도별 평가
        results = {
            'high': {'total': 0, 'correct': 0, 'precision': 0, 'recall': 0},
            'medium': {'total': 0, 'correct': 0, 'precision': 0, 'recall': 0},
            'low': {'total': 0, 'correct': 0, 'precision': 0, 'recall': 0},
            'smalltalk': {'total': 0, 'correct': 0}
        }

        for eval_id, val_item in val_data.items():
            if eval_id not in sub_data:
                continue

            confidence = val_item['confidence']
            ground_truth = set(val_item['ground_truth'])
            predicted = set(sub_data[eval_id]['topk'][:3])  # Top-3만

            if confidence == 'smalltalk':
                results['smalltalk']['total'] += 1
                if len(predicted) == 0:  # Correctly identified as smalltalk
                    results['smalltalk']['correct'] += 1
            else:
                results[confidence]['total'] += 1

                # Precision & Recall
                if predicted and ground_truth:
                    intersection = predicted & ground_truth
                    precision = len(intersection) / len(predicted)
                    recall = len(intersection) / len(ground_truth)

                    results[confidence]['precision'] += precision
                    results[confidence]['recall'] += recall

                    if precision > 0.5:  # 50% 이상 overlap
                        results[confidence]['correct'] += 1

        # 결과 출력
        print(f"\n{'='*80}")
        print(f"Pseudo Validation 결과: {submission_path}")
        print(f"{'='*80}")

        for conf in ['high', 'medium', 'low']:
            if results[conf]['total'] > 0:
                acc = results[conf]['correct'] / results[conf]['total'] * 100
                avg_precision = results[conf]['precision'] / results[conf]['total']
                avg_recall = results[conf]['recall'] / results[conf]['total']

                print(f"\n{conf.upper()} Confidence ({results[conf]['total']}개):")
                print(f"  - Accuracy: {acc:.1f}%")
                print(f"  - Avg Precision: {avg_precision:.3f}")
                print(f"  - Avg Recall: {avg_recall:.3f}")

        if results['smalltalk']['total'] > 0:
            acc = results['smalltalk']['correct'] / results['smalltalk']['total'] * 100
            print(f"\nSmalltalk ({results['smalltalk']['total']}개):")
            print(f"  - Accuracy: {acc:.1f}%")

        print(f"{'='*80}")

def main():
    print("=" * 80)
    print("Pseudo Validation Set 생성")
    print("=" * 80)

    validator = PseudoValidator(confidence_threshold=10.0, top_k=3)

    # Pseudo validation set 생성
    validator.create_pseudo_labels(
        eval_path='../data/eval.jsonl',
        output_path='pseudo_validation.jsonl',
        confidence_threshold=10.0
    )

    # 기존 submission 평가 (optional)
    print("\n기존 submission 파일을 평가하시겠습니까? (y/n)")
    choice = input("> ").strip().lower()

    if choice == 'y':
        print("\nSubmission 파일 경로를 입력하세요:")
        sub_path = input("> ").strip()

        if sub_path:
            validator.evaluate_submission(sub_path, 'pseudo_validation.jsonl')

if __name__ == "__main__":
    main()
