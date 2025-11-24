"""
BM25 Parameter Optimization using Grid Search

Tests different k1 and b values to find optimal BM25 configuration.
"""

import json
import numpy as np
from elasticsearch import Elasticsearch
from tqdm import tqdm
import itertools

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Validation set
VALIDATION_FILE = '../data/synthetic_validation.jsonl'

def load_validation_set(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_ap_at_3(ground_truth, predicted):
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
            
    return sum_precisions / len(ground_truth) if ground_truth else 0.0

def search_bm25_custom(query, k1, b, top_k=3):
    """BM25 search with custom parameters"""
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
            'size': top_k + 5  # Fetch extra to handle duplicates
        }
    )
    
    if not response['hits']['hits']:
        return []
    
    # Deduplicate by original_docid
    seen = set()
    results = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        docid = source.get('original_docid', source['docid'])
        if docid not in seen:
            seen.add(docid)
            results.append(docid)
            if len(results) >= top_k:
                break
    
    return results

def update_bm25_settings(k1, b):
    """Update BM25 similarity settings in Elasticsearch"""
    # Close index to update settings
    es.indices.close(index='test')
    
    # Update similarity settings
    es.indices.put_settings(
        index='test',
        body={
            'index': {
                'similarity': {
                    'custom_bm25': {
                        'type': 'BM25',
                        'k1': k1,
                        'b': b
                    }
                }
            }
        }
    )
    
    # Reopen index
    es.indices.open(index='test')

def evaluate_bm25_config(k1, b, validation_data):
    """Evaluate a specific BM25 configuration"""
    print(f"\nTesting k1={k1}, b={b}")
    
    # Update settings
    update_bm25_settings(k1, b)
    
    # Wait a bit for settings to apply
    import time
    time.sleep(2)
    
    # Evaluate
    total_ap = 0.0
    for item in validation_data:
        query = item['query']
        ground_truth = item['ground_truth']
        
        predicted = search_bm25_custom(query, k1, b)
        ap = calculate_ap_at_3(ground_truth, predicted)
        total_ap += ap
    
    map_score = total_ap / len(validation_data) if validation_data else 0.0
    print(f"  MAP@3: {map_score:.4f}")
    
    return map_score

def main():
    print("="*80)
    print("BM25 Parameter Optimization")
    print("="*80)
    
    # Load validation set
    validation_data = load_validation_set(VALIDATION_FILE)
    print(f"Loaded {len(validation_data)} validation samples\n")
    
    # Grid search parameters
    k1_values = [0.9, 1.2, 1.5, 1.8]
    b_values = [0.5, 0.75, 0.9]
    
    results = []
    
    for k1, b in itertools.product(k1_values, b_values):
        map_score = evaluate_bm25_config(k1, b, validation_data)
        results.append({
            'k1': k1,
            'b': b,
            'map': map_score
        })
    
    # Sort by MAP score
    results.sort(key=lambda x: x['map'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Rank':<6}{'k1':<10}{'b':<10}{'MAP@3':<10}")
    print("-"*80)
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['k1']:<10}{r['b']:<10}{r['map']:<10.4f}")
    
    # Best configuration
    best = results[0]
    print("\n" + "="*80)
    print(f"BEST CONFIGURATION: k1={best['k1']}, b={best['b']}, MAP@3={best['map']:.4f}")
    print("="*80)
    
    # Save results
    with open('bm25_optimization_results.json', 'w') as f:
        json.dump({
            'best': best,
            'all_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to bm25_optimization_results.json")
    
    # Apply best settings
    print(f"\nApplying best settings...")
    update_bm25_settings(best['k1'], best['b'])
    print("✅ Done")

if __name__ == "__main__":
    main()
