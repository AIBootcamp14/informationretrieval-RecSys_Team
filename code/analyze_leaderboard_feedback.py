"""
전략 3: Leaderboard Feedback 분석
- 여러 submission의 MAP 점수 비교
- 어떤 쿼리가 점수에 큰 영향을 미치는지 역추적
- High-impact 쿼리를 우선 수동 레이블링
"""

import json
import pandas as pd
from typing import List, Dict

class LeaderboardAnalyzer:
    def __init__(self):
        self.submissions = {}

    def load_submission(self, name: str, path: str, map_score: float):
        """Submission 파일과 점수 저장"""
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]

        self.submissions[name] = {
            'data': data,
            'map_score': map_score,
            'path': path
        }

        print(f"✅ {name} 로드 완료: MAP {map_score}")

    def compare_submissions(self):
        """여러 submission 비교"""
        if len(self.submissions) < 2:
            print("⚠️ 비교하려면 최소 2개의 submission이 필요합니다")
            return

        print(f"\n{'='*80}")
        print(f"Submission 비교 분석")
        print(f"{'='*80}")

        # 모든 submission의 TopK 추출
        all_topks = {}
        for name, sub in self.submissions.items():
            for item in sub['data']:
                eval_id = item['eval_id']
                if eval_id not in all_topks:
                    all_topks[eval_id] = {}
                all_topks[eval_id][name] = item['topk'][:3]

        # 차이 분석
        differences = []
        for eval_id, topks in all_topks.items():
            if len(topks) < 2:
                continue

            # 모든 submission이 같은 결과를 냈는지 확인
            topk_sets = [set(topk) for topk in topks.values()]
            if not all(s == topk_sets[0] for s in topk_sets):
                differences.append({
                    'eval_id': eval_id,
                    'topks': topks
                })

        print(f"\n차이가 있는 쿼리: {len(differences)}개")

        # 점수 차이가 큰 submission 쌍 찾기
        sorted_subs = sorted(self.submissions.items(), key=lambda x: x[1]['map_score'], reverse=True)

        if len(sorted_subs) >= 2:
            best = sorted_subs[0]
            worst = sorted_subs[-1]

            print(f"\n최고 점수: {best[0]} (MAP {best[1]['map_score']})")
            print(f"최저 점수: {worst[0]} (MAP {worst[1]['map_score']})")
            print(f"점수 차이: {best[1]['map_score'] - worst[1]['map_score']:.4f}")

            # 차이 분석
            self._analyze_difference(best[0], worst[0], differences)

        return differences

    def _analyze_difference(self, best_name: str, worst_name: str, differences: List[Dict]):
        """두 submission의 차이 분석"""
        print(f"\n{'='*80}")
        print(f"차이 분석: {best_name} vs {worst_name}")
        print(f"{'='*80}")

        print(f"\nTopK가 다른 쿼리 ({len(differences)}개):")

        for i, diff in enumerate(differences[:10], 1):  # 상위 10개만
            eval_id = diff['eval_id']
            topks = diff['topks']

            print(f"\n[{i}] Eval ID: {eval_id}")

            if best_name in topks and worst_name in topks:
                best_topk = topks[best_name]
                worst_topk = topks[worst_name]

                print(f"  {best_name}: {len(best_topk)}개 - {best_topk[:3]}")
                print(f"  {worst_name}: {len(worst_topk)}개 - {worst_topk[:3]}")

                # Overlap 계산
                overlap = set(best_topk) & set(worst_topk)
                print(f"  Overlap: {len(overlap)}개")

    def suggest_validation_queries(self, top_n=20):
        """Validation set에 포함할 쿼리 제안"""
        differences = self.compare_submissions()

        if not differences:
            print("\n⚠️ 차이가 없습니다")
            return []

        print(f"\n{'='*80}")
        print(f"Validation Set 추천 쿼리 (상위 {top_n}개)")
        print(f"{'='*80}")

        # 차이가 큰 쿼리 우선
        recommendations = []
        for diff in differences[:top_n]:
            eval_id = diff['eval_id']
            topks = diff['topks']

            # 각 submission의 TopK 개수
            topk_counts = {name: len(topk) for name, topk in topks.items()}

            recommendations.append({
                'eval_id': eval_id,
                'topk_variation': len(set(tuple(topk) for topk in topks.values())),
                'topk_counts': topk_counts
            })

        # 출력
        for i, rec in enumerate(recommendations, 1):
            print(f"\n[{i}] Eval ID: {rec['eval_id']}")
            print(f"  Variation: {rec['topk_variation']}가지")
            print(f"  TopK counts: {rec['topk_counts']}")

        # 저장
        output_path = 'validation_candidates.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 추천 쿼리 저장: {output_path}")

        return recommendations

def main():
    print("=" * 80)
    print("Leaderboard Feedback 분석 도구")
    print("=" * 80)

    analyzer = LeaderboardAnalyzer()

    # 기존 submission 파일들 로드
    submissions = [
        ('super_simple', 'super_simple_submission.csv', 0.63),
        ('simplified', 'simplified_submission.csv', None),  # 점수 미확인
        ('rag_1119', 'rag_1119_submission.csv', None),
        ('threshold3', 'rag_threshold3_submission.csv', None),
    ]

    print("\nSubmission 파일 로드 중...")
    for name, path, score in submissions:
        try:
            if score is None:
                print(f"\n{name}의 MAP 점수를 입력하세요 (또는 Enter로 건너뛰기):")
                score_input = input("> ").strip()
                score = float(score_input) if score_input else 0.0

            analyzer.load_submission(name, path, score)
        except FileNotFoundError:
            print(f"⚠️ {path} 파일 없음")
        except Exception as e:
            print(f"⚠️ {name} 로드 실패: {e}")

    # 비교 분석
    print("\n분석 시작...")
    differences = analyzer.compare_submissions()

    # Validation 쿼리 제안
    analyzer.suggest_validation_queries(top_n=20)

if __name__ == "__main__":
    main()
