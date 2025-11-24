"""
Generate Competition Submission File

Uses cascaded_reranking_v1 with LLM-based smalltalk classification
to generate submission CSV for the entire eval dataset.

Requirements:
- Elasticsearch must be running on localhost:9200
- UPSTAGE_API_KEY environment variable must be set

Usage:
    python generate_submission.py

Output:
    cascaded_reranking_v1_submission_{timestamp}.csv
"""

import json
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Import strategy
from cascaded_reranking_v1 import cascaded_reranking_strategy, embeddings_dict

def load_eval_dataset(path='../data/eval.jsonl'):
    """Load evaluation dataset"""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset

def generate_submission(dataset, embeddings_dict, output_path):
    """
    Generate submission CSV

    Args:
        dataset: List of eval samples
        embeddings_dict: BGE-M3 embeddings dictionary
        output_path: Output CSV path

    Returns:
        DataFrame with submission results
    """
    results = []

    print(f"\n{'='*80}")
    print(f"Generating Submission File")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset)}")
    print(f"Strategy: cascaded_reranking_v1 (LLM-based smalltalk classification)")
    print(f"{'='*80}\n")

    for item in tqdm(dataset, desc="Processing"):
        eval_id = item['eval_id']
        msg = item['msg']

        try:
            # Run strategy
            topk_docs = cascaded_reranking_strategy(eval_id, msg, embeddings_dict)

            results.append({
                'eval_id': eval_id,
                'topk_docs': topk_docs
            })

        except Exception as e:
            print(f"\n⚠️  Error on eval_id={eval_id}: {e}")
            results.append({
                'eval_id': eval_id,
                'topk_docs': []
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"Submission Generated Successfully!")
    print(f"{'='*80}")
    print(f"Output file: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Samples with results: {sum(df['topk_docs'].apply(lambda x: len(x) > 0))}")
    print(f"Empty results (smalltalk): {sum(df['topk_docs'].apply(lambda x: len(x) == 0))}")
    print(f"{'='*80}\n")

    return df

def validate_submission(df):
    """Validate submission format"""
    print("Validating submission format...")

    # Check columns
    required_columns = ['eval_id', 'topk_docs']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Check eval_id uniqueness
    if df['eval_id'].duplicated().any():
        raise ValueError("Duplicate eval_ids found")

    # Check topk_docs format
    for idx, row in df.iterrows():
        topk = row['topk_docs']
        if not isinstance(topk, list):
            raise ValueError(f"eval_id {row['eval_id']}: topk_docs must be a list")
        if len(topk) > 3:
            raise ValueError(f"eval_id {row['eval_id']}: topk_docs has {len(topk)} items (max 3)")

    print("✅ Validation passed!")
    return True

def print_statistics(df):
    """Print submission statistics"""
    print(f"\n{'='*80}")
    print(f"Submission Statistics")
    print(f"{'='*80}")

    # Distribution of topk lengths
    topk_lengths = df['topk_docs'].apply(len)
    print(f"\nDistribution of topk_docs lengths:")
    print(f"  0 docs (smalltalk): {sum(topk_lengths == 0)} samples")
    print(f"  1 doc: {sum(topk_lengths == 1)} samples")
    print(f"  2 docs: {sum(topk_lengths == 2)} samples")
    print(f"  3 docs: {sum(topk_lengths == 3)} samples")

    print(f"\nTotal samples: {len(df)}")
    print(f"{'='*80}\n")

def main():
    print("\n" + "="*80)
    print("Competition Submission Generator")
    print("="*80)
    print("Strategy: cascaded_reranking_v1")
    print("Features:")
    print("  - LLM-based smalltalk classification (no hardcoding)")
    print("  - Query rewriting for multi-turn conversations")
    print("  - Hybrid Search (BM25 + BGE-M3)")
    print("  - 2-Stage Cascaded Reranking (30 → 10 → 3)")
    print("="*80 + "\n")

    # Check Elasticsearch
    print("Checking Elasticsearch connection...")
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(['http://localhost:9200'])
        es.ping()
        print("✅ Elasticsearch is running\n")
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        print("\nPlease start Elasticsearch first:")
        print("  Option 1: ./start_elasticsearch.sh")
        print("  Option 2: ./start_elasticsearch_docker.sh")
        print()
        return

    # Check API key
    if not os.environ.get('UPSTAGE_API_KEY'):
        print("⚠️  Warning: UPSTAGE_API_KEY not set")
        print("   LLM features will be disabled")
        print()

    # Load dataset
    print("Loading eval dataset...")
    dataset = load_eval_dataset()
    print(f"✅ Loaded {len(dataset)} samples\n")

    # Generate submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'cascaded_reranking_v1_submission_{timestamp}.csv'

    df = generate_submission(dataset, embeddings_dict, output_path)

    # Validate
    validate_submission(df)

    # Statistics
    print_statistics(df)

    print(f"🎉 Submission file ready: {output_path}")
    print(f"📊 Expected MAP@3: ~0.8333 (based on ultra validation)")
    print(f"🚀 Ready to submit to competition!")
    print()

if __name__ == "__main__":
    main()
