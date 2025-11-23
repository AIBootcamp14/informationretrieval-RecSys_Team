"""
Failure Case Analysis Tool

Analyzes partial match cases to identify patterns and root causes
"""

import json

# Load validation results from auto_validate.py
with open('validation_report_cascaded_reranking_v1_20251123_234822.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

# Load ultra validation data
validation_data = []
with open('ultra_validation_solar.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            validation_data.append(json.loads(line))

print("="*80)
print("FAILURE CASE ANALYSIS")
print("="*80)

# Find partial match cases (0 < AP < 1.0)
partial_cases = [r for r in report['results'] if 0 < r['ap'] < 1.0]

print(f"\nTotal partial matches: {len(partial_cases)}")
print(f"Perfect matches: {len([r for r in report['results'] if r['ap'] == 1.0])}")
print(f"Complete failures: {len([r for r in report['results'] if r['ap'] == 0.0])}")

for case in partial_cases:
    eval_id = case['eval_id']
    query_text = case['query_text']
    predicted = case['predicted']
    ground_truth = case['ground_truth']
    ap = case['ap']
    hits = case['hits']

    # Find validation entry
    val_entry = next((e for e in validation_data if e['eval_id'] == eval_id), None)

    print(f"\n{'='*80}")
    print(f"eval_id={eval_id} | AP@3={ap:.4f} | Hits={hits}/3")
    print(f"Query: {query_text}")
    print(f"{'='*80}")

    # Show what we got right and wrong
    print("\nGround Truth (Solar Pro validated):")
    for i, doc in enumerate(ground_truth, 1):
        if val_entry:
            score_info = next((s for s in val_entry['detailed_scores'] if s['docid'] == doc), None)
            if score_info:
                status = "✓" if doc in predicted else "✗ MISSED"
                print(f"  {i}. [{score_info['score']}/5] {doc} {status}")
                if doc not in predicted:
                    print(f"     Rationale: {score_info['rationale'][:150]}...")

    print("\nPredicted (what our system returned):")
    for i, doc in enumerate(predicted, 1):
        status = "✓" if doc in ground_truth else "✗ WRONG"
        # Find score if available
        if val_entry and doc in [s['docid'] for s in val_entry['detailed_scores']]:
            score_info = next((s for s in val_entry['detailed_scores'] if s['docid'] == doc), None)
            if score_info:
                print(f"  {i}. [{score_info['score']}/5] {doc} {status}")
            else:
                print(f"  {i}. [?/5] {doc} {status}")
        else:
            print(f"  {i}. [NOT SCORED] {doc} {status}")

    # Analysis
    print("\nROOT CAUSE ANALYSIS:")
    missed_docs = [d for d in ground_truth if d not in predicted]
    wrong_docs = [d for d in predicted if d not in ground_truth]

    if missed_docs:
        print(f"  - Missed {len(missed_docs)} correct documents")
        if val_entry:
            for doc in missed_docs:
                score_info = next((s for s in val_entry['detailed_scores'] if s['docid'] == doc), None)
                if score_info:
                    print(f"    • {doc} [Score: {score_info['score']}/5]")
                    print(f"      Evidence: {score_info.get('evidence', 'N/A')[:100]}...")

    if wrong_docs:
        print(f"  - Returned {len(wrong_docs)} incorrect documents")
        for doc in wrong_docs:
            if val_entry and doc in [s['docid'] for s in val_entry['detailed_scores']]:
                score_info = next((s for s in val_entry['detailed_scores'] if s['docid'] == doc), None)
                if score_info:
                    print(f"    • {doc} [Solar Pro Score: {score_info['score']}/5]")
                    print(f"      This doc was scored but ranked outside top-3 by Solar Pro")
            else:
                print(f"    • {doc} [NOT in Solar Pro's scored set - completely wrong]")

print("\n" + "="*80)
print("SUMMARY & PATTERNS")
print("="*80)

# Pattern analysis
print("\n1. RANKING ERRORS:")
print("   Documents that were in the scored set but ranked incorrectly")

print("\n2. RETRIEVAL ERRORS:")
print("   Documents completely outside the scored set")

print("\n3. RERANKING FAILURES:")
print("   Cascaded reranking moved correct docs out of top-3")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
Based on the failure analysis:

1. INCREASE CASCADED STAGES:
   Current: 30→10→3
   Proposed: 50→30→20→10→3
   Rationale: More stages = better filtering, less information loss

2. IMPROVE RETRIEVAL RECALL:
   - Check if missed documents were even in top-30 initial retrieval
   - If not: BM25/BGE-M3 retrieval is failing
   - If yes: Reranking is pushing good docs down

3. OPTIMIZE RERANKING PROMPTS:
   - Add explicit instructions for edge cases
   - Include query type detection (factual vs conceptual)
   - Emphasize evidence matching

4. QUERY ENHANCEMENT:
   - For queries with low initial recall, expand with synonyms
   - Add domain-specific terms based on query analysis
""")
