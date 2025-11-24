"""
Generate Full Competition Submission File with All Features

Includes:
- eval_id
- standalone_query (rewritten query)
- topk (document IDs)
- answer (Korean answer text)
- references (full document contents)

Requirements:
- Elasticsearch must be running on localhost:9200
- UPSTAGE_API_KEY environment variable must be set

Usage:
    export UPSTAGE_API_KEY=your_key
    python generate_full_submission.py

Output:
    cascaded_reranking_v1_full_submission_{timestamp}.csv
"""

import json
import os
from datetime import datetime
from tqdm import tqdm
from elasticsearch import Elasticsearch

# Import strategy
from cascaded_reranking_v1 import cascaded_reranking_strategy, embeddings_dict, rewrite_query_with_context, is_smalltalk, client

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

def load_eval_dataset(path='../data/eval.jsonl'):
    """Load evaluation dataset"""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset

def get_document_content(docid):
    """Elasticsearch에서 문서 내용 가져오기"""
    try:
        resp = es.search(
            index='test',
            body={
                'query': {
                    'term': {'docid': docid}
                },
                'size': 1
            }
        )

        if resp['hits']['hits']:
            return resp['hits']['hits'][0]['_source']['content']
        return ""
    except Exception as e:
        print(f"⚠️  Error fetching document {docid}: {e}")
        return ""

def generate_full_submission(dataset, embeddings_dict, output_path):
    """
    Generate full submission CSV with all features

    Args:
        dataset: List of eval samples
        embeddings_dict: BGE-M3 embeddings dictionary
        output_path: Output CSV path
    """
    results = []

    print(f"\n{'='*80}")
    print(f"Generating Full Submission File")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset)}")
    print(f"Strategy: cascaded_reranking_v1 (LLM-based smalltalk classification)")
    print(f"{'='*80}\n")

    for item in tqdm(dataset, desc="Processing"):
        eval_id = item['eval_id']
        msg = item['msg']

        try:
            # Step 1: Rewrite query
            standalone_query = rewrite_query_with_context(msg)

            # Step 2: Check if smalltalk
            if is_smalltalk(standalone_query, client):
                # Smalltalk - no documents
                result = {
                    'eval_id': eval_id,
                    'standalone_query': standalone_query,
                    'topk': [],
                    'answer': "관련 문서를 찾을 수 없습니다.",
                    'references': []
                }
            else:
                # Step 3: Run strategy
                topk_docs = cascaded_reranking_strategy(eval_id, msg, embeddings_dict)

                # Step 4: Get document contents
                references = []
                for docid in topk_docs:
                    content = get_document_content(docid)
                    if content:
                        references.append({
                            'docid': docid,
                            'content': content
                        })

                # Step 5: Generate answer
                if topk_docs:
                    answer = f"검색 결과 {len(topk_docs)}개 문서를 찾았습니다."
                else:
                    answer = "관련 문서를 찾을 수 없습니다."

                result = {
                    'eval_id': eval_id,
                    'standalone_query': standalone_query,
                    'topk': topk_docs,
                    'answer': answer,
                    'references': references
                }

            results.append(result)

        except Exception as e:
            print(f"\n⚠️  Error on eval_id={eval_id}: {e}")
            results.append({
                'eval_id': eval_id,
                'standalone_query': msg if isinstance(msg, str) else msg[-1]['content'],
                'topk': [],
                'answer': "관련 문서를 찾을 수 없습니다.",
                'references': []
            })

    # Save to CSV (JSONL format, each line is a JSON object)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"\n{'='*80}")
    print(f"Full Submission Generated Successfully!")
    print(f"{'='*80}")
    print(f"Output file: {output_path}")
    print(f"Total samples: {len(results)}")

    # Statistics
    with_results = sum(1 for r in results if len(r['topk']) > 0)
    without_results = len(results) - with_results

    print(f"Samples with results: {with_results}")
    print(f"Empty results (smalltalk): {without_results}")
    print(f"{'='*80}\n")

    return results

def main():
    print("\n" + "="*80)
    print("Full Competition Submission Generator")
    print("="*80)
    print("Strategy: cascaded_reranking_v1")
    print("Features:")
    print("  - LLM-based smalltalk classification")
    print("  - Query rewriting for multi-turn conversations")
    print("  - Hybrid Search (BM25 + BGE-M3)")
    print("  - 2-Stage Cascaded Reranking (30 → 10 → 3)")
    print("  - Full document references with Korean content")
    print("="*80 + "\n")

    # Check Elasticsearch
    print("Checking Elasticsearch connection...")
    try:
        es.ping()
        print("✅ Elasticsearch is running\n")
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
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
    output_path = f'cascaded_reranking_v1_full_submission_{timestamp}.csv'

    results = generate_full_submission(dataset, embeddings_dict, output_path)

    print(f"🎉 Full submission file ready: {output_path}")
    print(f"📊 Expected MAP@3: ~0.8333 (based on ultra validation)")
    print(f"🚀 Ready to submit to competition!")
    print()

if __name__ == "__main__":
    main()
