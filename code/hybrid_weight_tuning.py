"""
Hybrid Weight Tuning: RRF k Parameter Optimization

Tests different RRF k values to find optimal balance between BM25 and BGE-M3.
"""

import json
import pickle
import numpy as np
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import itertools

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# BGE-M3 로드
print("Loading BGE-M3 model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ Model loaded")

print("Loading embeddings...")
with open('embeddings_test_bgem3_optimized.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print(f"✅ Loaded {len(embeddings_dict)} embeddings")

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

def search_bm25(query, top_k=30):
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
            'size': top_k + 5
        }
    )
    
    if not response['hits']['hits']:
        return []
    
    seen = set()
    results = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        docid = source.get('original_docid', source['docid'])
        if docid not in seen:
            seen.add(docid)
            results.append({
                'docid': docid,
                'content': source['content'],
                'score': hit['_score'],
                'source': 'bm25'
            })
            if len(results) >= top_k:
                break
    
    return results

def bgem3_hybrid_score(query_dense, query_sparse, query_colbert,
                       doc_dense, doc_sparse, doc_colbert,
                       w1=0.4, w2=0.3, w3=0.3):
    # Dense
    s_dense = np.dot(query_dense, doc_dense) / (
        np.linalg.norm(query_dense) * np.linalg.norm(doc_dense)
    )
    
    # Sparse
    s_lex = 0.0
    if query_sparse and doc_sparse:
        common_tokens = set(query_sparse.keys()) & set(doc_sparse.keys())
        for token in common_tokens:
            s_lex += query_sparse[token] * doc_sparse[token]
    
    # ColBERT
    s_mul = 0.0
    if query_colbert.shape[0] > 0 and doc_colbert.shape[0] > 0:
        query_colbert_norm = query_colbert / np.linalg.norm(query_colbert, axis=1, keepdims=True)
        doc_colbert_norm = doc_colbert / np.linalg.norm(doc_colbert, axis=1, keepdims=True)
        sim_matrix = np.dot(query_colbert_norm, doc_colbert_norm.T)
        s_mul = np.mean(np.max(sim_matrix, axis=1))
    
    return w1 * s_dense + w2 * s_lex + w3 * s_mul

def search_bgem3_hybrid(query, embeddings_dict, top_k=30):
    query_embedding = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
        max_length=128
    )
    
    query_dense = query_embedding['dense_vecs'][0]
    query_sparse = query_embedding['lexical_weights'][0]
    query_colbert = query_embedding['colbert_vecs'][0]
    
    scores = []
    for docid, doc_emb in embeddings_dict.items():
        score = bgem3_hybrid_score(
            query_dense, query_sparse, query_colbert,
            doc_emb['dense'], doc_emb['sparse'], doc_emb['colbert']
        )
        scores.append((docid, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for docid, score in scores[:top_k]:
        try:
            resp = es.search(
                index='test',
                body={
                    'query': {
                        'bool': {
                            'should': [
                                {'term': {'docid.keyword': docid}},
                                {'term': {'original_docid.keyword': docid}}
                            ]
                        }
                    },
                    'size': 1
                }
            )
            
            if resp['hits']['hits']:
                source = resp['hits']['hits'][0]['_source']
                results.append({
                    'docid': docid,
                    'content': source['content'],
                    'score': float(score),
                    'source': 'bgem3_hybrid'
                })
        except:
            continue
    
    return results

def hybrid_search_rrf(query, embeddings_dict, k=60, top_k=30):
    """RRF with custom k parameter"""
    bm25_results = search_bm25(query, top_k=top_k)
    bgem3_results = search_bgem3_hybrid(query, embeddings_dict, top_k=top_k)
    
    rrf_scores = {}
    doc_contents = {}
    
    for rank, doc in enumerate(bm25_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']
    
    for rank, doc in enumerate(bgem3_results, 1):
        docid = doc['docid']
        rrf_scores[docid] = rrf_scores.get(docid, 0) + 1 / (k + rank)
        doc_contents[docid] = doc['content']
    
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for docid, score in sorted_docs:
        results.append({'docid': docid, 'content': doc_contents[docid], 'score': score})
    
    return results

def evaluate_rrf_k(k_value, validation_data):
    """Evaluate specific RRF k value"""
    print(f"\nTesting RRF k={k_value}")
    
    total_ap = 0.0
    for item in tqdm(validation_data, desc=f"k={k_value}"):
        query = item['query']
        ground_truth = item['ground_truth']
        
        # Get top docs with this k
        results = hybrid_search_rrf(query, embeddings_dict, k=k_value, top_k=3)
        predicted = [r['docid'] for r in results[:3]]
        
        ap = calculate_ap_at_3(ground_truth, predicted)
        total_ap += ap
    
    map_score = total_ap / len(validation_data) if validation_data else 0.0
    print(f"  MAP@3: {map_score:.4f}")
    
    return map_score

def main():
    print("="*80)
    print("Hybrid Weight Tuning: RRF k Optimization")
    print("="*80)
    
    validation_data = load_validation_set(VALIDATION_FILE)
    print(f"Loaded {len(validation_data)} validation samples\n")
    
    # Test different k values
    k_values = [30, 60, 90, 120, 150]
    
    results = []
    for k in k_values:
        map_score = evaluate_rrf_k(k, validation_data)
        results.append({'k': k, 'map': map_score})
    
    # Sort by MAP
    results.sort(key=lambda x: x['map'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Rank':<6}{'RRF k':<10}{'MAP@3':<10}")
    print("-"*80)
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['k']:<10}{r['map']:<10.4f}")
    
    best = results[0]
    print("\n" + "="*80)
    print(f"BEST CONFIGURATION: RRF k={best['k']}, MAP@3={best['map']:.4f}")
    print("="*80)
    
    # Save results
    with open('hybrid_weight_optimization_results.json', 'w') as f:
        json.dump({
            'best': best,
            'all_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to hybrid_weight_optimization_results.json")
    print(f"\n📝 Next: Update cascaded_reranking_v1.py with k={best['k']}")

if __name__ == "__main__":
    main()
