"""
Auto Validation CICD Pipeline

Automatically tests any RAG strategy on ultra validation set and compares with baseline.

Usage:
    python auto_validate.py <strategy_module> <strategy_function>

Example:
    python auto_validate.py cascaded_reranking_v1 cascaded_reranking_strategy
"""

import json
import os
import sys
import importlib
from datetime import datetime

# Baseline performance
BASELINE_MAP = 0.7848
BASELINE_NAME = "query_expansion_v1"

# Competition best
COMPETITION_BEST = 0.7939
COMPETITION_BEST_NAME = "cascaded_reranking_v1"

def load_ultra_validation(path='ultra_validation_solar.jsonl'):
    """Load ultra validation samples"""
    validation_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                validation_data.append(json.loads(line))
    return validation_data

def calculate_ap_at_3(ground_truth, predicted):
    """Calculate Average Precision @ 3"""
    if not ground_truth or not predicted:
        return 0.0

    ground_truth_set = set(ground_truth)
    num_hits = 0
    sum_precisions = 0.0

    for i, doc_id in enumerate(predicted[:3], 1):
        if doc_id in ground_truth_set:
            num_hits += 1
            precision_at_i = num_hits / i
            sum_precisions += precision_at_i

    ap = sum_precisions / len(ground_truth) if ground_truth else 0.0
    return ap

def count_hits(ground_truth, predicted):
    """Count hits"""
    ground_truth_set = set(ground_truth)
    return sum(1 for doc in predicted[:3] if doc in ground_truth_set)

def evaluate_strategy(strategy_func, validation_data, embeddings_dict):
    """Evaluate strategy on validation set"""
    results = []

    print(f"\n{'='*80}")
    print(f"Testing strategy on {len(validation_data)} validation samples...")
    print(f"{'='*80}\n")

    for idx, val_item in enumerate(validation_data, 1):
        eval_id = val_item['eval_id']
        query = val_item['query']
        ground_truth = val_item['ground_truth']

        print(f"[{idx}/{len(validation_data)}] eval_id={eval_id}...", end=' ')

        try:
            # Run strategy
            predicted_topk = strategy_func(eval_id, query, embeddings_dict)

            # Calculate metrics
            ap = calculate_ap_at_3(ground_truth, predicted_topk)
            hits = count_hits(ground_truth, predicted_topk)

            # Extract query text
            query_text = query[-1]['content'] if isinstance(query, list) else query

            results.append({
                'eval_id': eval_id,
                'query_text': query_text,
                'ground_truth': ground_truth,
                'predicted': predicted_topk,
                'ap': ap,
                'hits': hits
            })

            print(f"AP@3={ap:.4f}, Hits={hits}/3")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'eval_id': eval_id,
                'query_text': '',
                'ground_truth': ground_truth,
                'predicted': [],
                'ap': 0.0,
                'hits': 0,
                'error': str(e)
            })

    return results

def generate_report(strategy_name, results, output_path):
    """Generate validation report"""
    map_score = sum(r['ap'] for r in results) / len(results) if results else 0.0
    perfect_matches = sum(1 for r in results if r['ap'] == 1.0)
    partial_matches = sum(1 for r in results if 0 < r['ap'] < 1.0)
    failures = sum(1 for r in results if r['ap'] == 0.0)

    # Calculate delta vs baseline
    delta_baseline = map_score - BASELINE_MAP
    pct_baseline = (delta_baseline / BASELINE_MAP * 100) if BASELINE_MAP > 0 else 0

    # Calculate delta vs competition best
    delta_best = map_score - COMPETITION_BEST
    pct_best = (delta_best / COMPETITION_BEST * 100) if COMPETITION_BEST > 0 else 0

    report = {
        'strategy_name': strategy_name,
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'num_samples': len(results),
            'map_score': map_score,
            'perfect_matches': perfect_matches,
            'partial_matches': partial_matches,
            'failures': failures
        },
        'comparison': {
            'baseline_map': BASELINE_MAP,
            'baseline_name': BASELINE_NAME,
            'delta_vs_baseline': delta_baseline,
            'pct_vs_baseline': pct_baseline,
            'competition_best': COMPETITION_BEST,
            'competition_best_name': COMPETITION_BEST_NAME,
            'delta_vs_best': delta_best,
            'pct_vs_best': pct_best
        }
    }

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"VALIDATION REPORT: {strategy_name}")
    print(f"{'='*80}")
    print(f"Samples: {len(results)}")
    print(f"MAP@3: {map_score:.4f}")
    print(f"  - Perfect (3/3): {perfect_matches}")
    print(f"  - Partial (1-2/3): {partial_matches}")
    print(f"  - Failed (0/3): {failures}")
    print(f"\nComparison:")
    print(f"  vs Baseline ({BASELINE_NAME}={BASELINE_MAP:.4f})")
    print(f"    Δ {delta_baseline:+.4f} ({pct_baseline:+.2f}%)")
    print(f"  vs Competition Best ({COMPETITION_BEST_NAME}={COMPETITION_BEST:.4f})")
    print(f"    Δ {delta_best:+.4f} ({pct_best:+.2f}%)")

    # Recommendation
    print(f"\n{'='*80}")
    if map_score > COMPETITION_BEST:
        print(f"🎉 NEW BEST! Submit to competition!")
    elif map_score > BASELINE_MAP:
        print(f"✅ Better than baseline. Consider submitting.")
    elif abs(map_score - BASELINE_MAP) < 0.02:
        print(f"~ Similar to baseline.")
    else:
        print(f"❌ Worse than baseline. Need improvement.")
    print(f"{'='*80}\n")

    print(f"Report saved: {output_path}")

    return report

def main():
    if len(sys.argv) < 3:
        print("Usage: python auto_validate.py <strategy_module> <strategy_function>")
        print("Example: python auto_validate.py cascaded_reranking_v1 cascaded_reranking_strategy")
        sys.exit(1)

    module_name = sys.argv[1]
    function_name = sys.argv[2]

    print(f"\n{'='*80}")
    print(f"AUTO VALIDATION PIPELINE")
    print(f"{'='*80}")
    print(f"Strategy Module: {module_name}")
    print(f"Strategy Function: {function_name}")
    print(f"{'='*80}\n")

    # Change to code directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Load validation data
    print("Loading ultra validation set...")
    validation_data = load_ultra_validation()
    print(f"✓ Loaded {len(validation_data)} samples\n")

    # Import strategy
    print(f"Importing strategy from {module_name}...")
    try:
        strategy_module = importlib.import_module(module_name)
        strategy_func = getattr(strategy_module, function_name)
        embeddings_dict = getattr(strategy_module, 'embeddings_dict', {})
        print(f"✓ Strategy loaded\n")
    except Exception as e:
        print(f"ERROR loading strategy: {e}")
        sys.exit(1)

    # Run evaluation
    results = evaluate_strategy(strategy_func, validation_data, embeddings_dict)

    # Generate report
    output_path = f'validation_report_{module_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    report = generate_report(module_name, results, output_path)

    # Return exit code based on performance
    if report['summary']['map_score'] > COMPETITION_BEST:
        sys.exit(0)  # Success - new best
    elif report['summary']['map_score'] > BASELINE_MAP:
        sys.exit(0)  # Success - better than baseline
    else:
        sys.exit(1)  # Failure - worse than baseline

if __name__ == "__main__":
    main()
