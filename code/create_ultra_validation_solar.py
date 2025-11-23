"""
Ultra High-Quality Validation Set Creation using Solar Pro
5단계 검증 파이프라인으로 최고 품질의 밸리데이션 세트 생성

목표: 120~150개 ultra-high-quality validation samples
신뢰도: 95%+
"""

import json
import os
import pickle
from collections import Counter
from tqdm import tqdm
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
import numpy as np

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Solar API 초기화
upstage_api_key = os.environ.get('UPSTAGE_API_KEY')
if not upstage_api_key:
    raise ValueError("UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다")

client = OpenAI(
    api_key=upstage_api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

# BGE-M3 모델 로드
print("BGE-M3 모델 로드 중...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ BGE-M3 모델 로드 완료")

# 일반 대화 ID
SMALLTALK_IDS = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

# BGE-M3 최적화 임베딩 로드
print("\nBGE-M3 최적화 임베딩 로드 중...")
with open('embeddings_test_bgem3_optimized.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print(f"✅ {len(embeddings_dict)}개 문서 임베딩 로드 완료")


def rewrite_query_with_context(msg):
    """멀티턴 대화의 맥락을 통합하여 쿼리 재작성"""
    if isinstance(msg, str):
        return msg

    if len(msg) == 1:
        return msg[0]['content']

    current_query = msg[-1]['content']

    # 대명사나 모호한 표현 확인
    ambiguous_terms = ['그 ', '그것', '이것', '이거', '저것', '저거', '왜', '어떻게', '이유']

    if not any(term in current_query for term in ambiguous_terms):
        return current_query

    # LLM으로 쿼리 재작성
    conversation_context = "\n".join([
        f"{m['role']}: {m['content']}" for m in msg[:-1]
    ])

    prompt = f"""다음은 이전 대화 내용입니다:

{conversation_context}

현재 사용자의 질문은 다음과 같습니다:
"{current_query}"

이 질문을 이전 대화의 맥락을 반영하여 독립적으로 이해 가능한 완전한 질문으로 재작성해주세요.
대명사(그것, 이것 등)를 구체적인 명사로 바꿔주세요.

재작성된 질문만 출력하세요. 다른 설명은 하지 마세요."""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )

        rewritten = response.choices[0].message.content.strip()
        rewritten = rewritten.strip('"').strip("'")

        return rewritten

    except Exception as e:
        return current_query


def search_bm25(query, top_k=20):
    """BM25 검색"""
    fetch_size = top_k + 5

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
            'size': fetch_size
        }
    )

    if not response['hits']['hits']:
        return []

    # original_docid 기반 중복 제거
    seen_original_docids = set()
    results = []

    for hit in response['hits']['hits']:
        source = hit['_source']
        original_docid = source.get('original_docid', source['docid'])

        if original_docid in seen_original_docids:
            continue

        seen_original_docids.add(original_docid)
        results.append({
            'docid': original_docid,
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
    """BGE-M3 Hybrid Scoring"""
    # 1. Dense 유사도
    s_dense = np.dot(query_dense, doc_dense) / (
        np.linalg.norm(query_dense) * np.linalg.norm(doc_dense)
    )

    # 2. Sparse 유사도
    s_lex = 0.0
    if query_sparse and doc_sparse:
        common_tokens = set(query_sparse.keys()) & set(doc_sparse.keys())
        for token in common_tokens:
            s_lex += query_sparse[token] * doc_sparse[token]

    # 3. ColBERT 유사도
    s_mul = 0.0
    if query_colbert.shape[0] > 0 and doc_colbert.shape[0] > 0:
        query_colbert_norm = query_colbert / np.linalg.norm(query_colbert, axis=1, keepdims=True)
        doc_colbert_norm = doc_colbert / np.linalg.norm(doc_colbert, axis=1, keepdims=True)
        sim_matrix = np.dot(query_colbert_norm, doc_colbert_norm.T)
        s_mul = np.mean(np.max(sim_matrix, axis=1))

    hybrid_score = w1 * s_dense + w2 * s_lex + w3 * s_mul
    return hybrid_score


def search_bgem3_hybrid(query, embeddings_dict, top_k=20, max_length=128):
    """BGE-M3 Hybrid 검색"""
    # BGE-M3 쿼리 임베딩
    query_embedding = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
        max_length=max_length
    )

    query_dense = query_embedding['dense_vecs'][0]
    query_sparse = query_embedding['lexical_weights'][0]
    query_colbert = query_embedding['colbert_vecs'][0]

    # 모든 문서에 대해 Hybrid Score 계산
    scores = []
    for docid, doc_emb in embeddings_dict.items():
        score = bgem3_hybrid_score(
            query_dense, query_sparse, query_colbert,
            doc_emb['dense'], doc_emb['sparse'], doc_emb['colbert']
        )
        scores.append((docid, score))

    # 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    # ES에서 content 가져오기
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
        except Exception as e:
            continue

    return results


def hybrid_search_rrf(query, embeddings_dict, top_k=20, k=60, query_max_length=128):
    """RRF로 BM25 + BGE-M3 Hybrid 결합"""
    # BM25 검색
    bm25_results = search_bm25(query, top_k=top_k)

    # BGE-M3 Hybrid 검색
    bgem3_results = search_bgem3_hybrid(query, embeddings_dict, top_k=top_k, max_length=query_max_length)

    # RRF 스코어 계산
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

    # 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for docid, score in sorted_docs:
        results.append({
            'docid': docid,
            'content': doc_contents[docid],
            'score': score
        })

    return results


# ========== Phase 1: Multi-Pass Self-Consistency ==========

def solar_label_docs(query, docs, temperature=0.0, top_k=5):
    """Solar Pro로 문서 라벨링 (단일 호출)"""
    if not docs:
        return []

    try:
        # 문서 목록 작성
        doc_list = []
        for i, doc in enumerate(docs[:20]):  # Top 20만 평가
            content_preview = doc['content'][:250]
            if len(doc['content']) > 250:
                content_preview += "..."
            doc_list.append(f"[{i}] {content_preview}")

        docs_text = "\n\n".join(doc_list)

        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 과학 지식 검색 시스템의 정답 라벨링 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"""쿼리: {query}

문서들:
{docs_text}

이 질문에 답하기 위해 가장 관련성이 높은 문서 {top_k}개의 번호를 선택하세요.
관련성이 높은 순서대로 나열하세요.

출력: 번호만 콤마로 구분 (예: 0,2,4,5,7)
설명 없이 번호만 출력하세요."""
                }
            ],
            temperature=temperature,
            max_tokens=50
        )

        # 파싱
        result_text = response.choices[0].message.content.strip()

        # 번호 추출
        import re
        numbers = re.findall(r'\d+', result_text)
        selected_indices = [int(n) for n in numbers if int(n) < len(docs)][:top_k]

        return selected_indices

    except Exception as e:
        print(f"  ⚠️ Solar labeling 실패: {e}")
        return list(range(min(top_k, len(docs))))


def phase1_multipass_consensus(query, docs):
    """
    Phase 1: Multi-Pass Self-Consistency
    3번 호출하여 2/3 합의된 문서만 선택
    """
    if not docs or len(docs) < 3:
        return [doc['docid'] for doc in docs]

    # 3번 호출 (temp 0.0, 0.3, 0.0)
    temperatures = [0.0, 0.3, 0.0]
    all_selections = []

    for temp in temperatures:
        indices = solar_label_docs(query, docs, temperature=temp, top_k=5)
        selected_docids = [docs[i]['docid'] for i in indices if i < len(docs)]
        all_selections.append(selected_docids)

    # 2/3 이상 선택된 문서만 추출
    docid_counts = Counter()
    for selection in all_selections:
        for docid in selection:
            docid_counts[docid] += 1

    # 2번 이상 선택된 문서
    consensus_docs = [docid for docid, count in docid_counts.items() if count >= 2]

    return consensus_docs


# ========== Phase 2: Detailed Scoring with Rationale ==========

def phase2_detailed_scoring(query, docs, candidate_docids):
    """
    Phase 2: 상세 점수화 및 근거 생성
    각 문서에 대해 1-5점 평가 + 근거
    """
    if not candidate_docids:
        return []

    # candidate_docids에 해당하는 문서만 추출
    candidate_docs = [doc for doc in docs if doc['docid'] in candidate_docids]

    detailed_scores = []

    for doc in candidate_docids[:10]:  # 최대 10개까지만 상세 평가
        # 해당 docid의 content 찾기
        doc_content = None
        for d in docs:
            if d['docid'] == doc:
                doc_content = d['content']
                break

        if not doc_content:
            continue

        content_preview = doc_content[:500]
        if len(doc_content) > 500:
            content_preview += "..."

        try:
            response = client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 엄격한 문서 평가 전문가입니다."
                    },
                    {
                        "role": "user",
                        "content": f"""쿼리: {query}

문서:
{content_preview}

이 문서가 쿼리에 답하는 정도를 1-5점으로 평가하세요.

평가 기준:
5점: 쿼리에 직접적이고 완전한 답변 포함
4점: 관련성 높으나 일부 정보 부족
3점: 관련은 있으나 불충분
2점: 약간 관련
1점: 무관

출력 형식:
점수: X
근거: [구체적 이유 1-2문장]
증거: [문서의 관련 부분 인용 1문장]

위 형식대로만 출력하세요."""
                    }
                ],
                temperature=0.0,
                max_tokens=300
            )

            result_text = response.choices[0].message.content.strip()

            # 점수 추출
            import re
            score_match = re.search(r'점수:\s*(\d)', result_text)
            if not score_match:
                continue

            score = int(score_match.group(1))

            # 근거와 증거 추출
            rationale_match = re.search(r'근거:\s*(.+?)(?=증거:|$)', result_text, re.DOTALL)
            evidence_match = re.search(r'증거:\s*(.+?)$', result_text, re.DOTALL)

            rationale = rationale_match.group(1).strip() if rationale_match else "N/A"
            evidence = evidence_match.group(1).strip() if evidence_match else "N/A"

            # 3점 이상만 채택 (확장된 validation set 생성을 위해 임계값 낮춤)
            if score >= 3:
                detailed_scores.append({
                    'docid': doc,
                    'score': score,
                    'rationale': rationale,
                    'evidence': evidence
                })

        except Exception as e:
            print(f"  ⚠️ Detailed scoring 실패 for {doc}: {e}")
            continue

    return detailed_scores


# ========== Phase 3: Confidence-Based Filtering ==========

def phase3_confidence_check(query, detailed_scores):
    """
    Phase 3: 전체 정답 세트에 대한 신뢰도 평가
    1-5점, 4점 이상만 통과
    """
    if not detailed_scores:
        return None

    # 상위 3개 문서 정보
    top3_docs = detailed_scores[:3]
    doc_summary = "\n".join([
        f"문서 {i+1}: {doc['rationale'][:100]}"
        for i, doc in enumerate(top3_docs)
    ])

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 정답 세트 신뢰도 평가 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"""쿼리: {query}

정답 후보 문서들:
{doc_summary}

이 정답 세트가 쿼리에 대한 올바른 답변인지 신뢰도를 평가하세요.

신뢰도 척도:
5점 (Certain): 100% 확신, 명확한 정답
4점 (High): 90% 확신, 높은 관련성
3점 (Medium): 70% 확신, 어느 정도 관련
2점 (Low): 50% 확신, 불확실
1점 (Uncertain): 확신 없음

출력 형식:
신뢰도: X
이유: [1-2문장]

위 형식대로만 출력하세요."""
                }
            ],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()

        # 신뢰도 점수 추출
        import re
        conf_match = re.search(r'신뢰도:\s*(\d)', result_text)
        if not conf_match:
            return None

        confidence_score = int(conf_match.group(1))

        # 이유 추출
        reason_match = re.search(r'이유:\s*(.+?)$', result_text, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "N/A"

        return {
            'score': confidence_score,
            'reason': reason
        }

    except Exception as e:
        print(f"  ⚠️ Confidence check 실패: {e}")
        return None


# ========== Phase 4: Multi-Turn Query Enhancement ==========

def phase4_multiturn_rewrites(msg):
    """
    Phase 4: 멀티턴 쿼리 3가지 재작성 버전 생성
    """
    if isinstance(msg, str) or len(msg) == 1:
        return None  # 단일턴은 스킵

    current_query = msg[-1]['content']

    # 대명사가 없으면 스킵
    ambiguous_terms = ['그 ', '그것', '이것', '이거', '저것', '저거', '왜', '어떻게', '이유']
    if not any(term in current_query for term in ambiguous_terms):
        return None

    conversation_context = "\n".join([
        f"{m['role']}: {m['content']}" for m in msg[:-1]
    ])

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 쿼리 재작성 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"""다음은 이전 대화 내용입니다:

{conversation_context}

현재 사용자의 질문:
"{current_query}"

이 질문을 이전 대화 맥락을 반영하여 3가지 다른 방식으로 재작성해주세요.
각 재작성은 독립적으로 이해 가능해야 하며, 대명사를 구체적 명사로 바꿔야 합니다.

출력 형식:
1. [첫 번째 재작성]
2. [두 번째 재작성]
3. [세 번째 재작성]

위 형식대로만 출력하세요."""
                }
            ],
            temperature=0.5,  # 다양성을 위해 temperature 증가
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # 재작성 추출
        import re
        rewrites = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', result_text, re.DOTALL)
        rewrites = [r.strip() for r in rewrites if r.strip()]

        return rewrites[:3] if len(rewrites) >= 3 else None

    except Exception as e:
        print(f"  ⚠️ Multi-turn rewrites 실패: {e}")
        return None


# ========== Phase 5: Adversarial Self-Validation ==========

def phase5_adversarial_validation(query, detailed_scores):
    """
    Phase 5: Solar Pro가 자신의 정답을 비판적으로 재검증
    """
    if not detailed_scores:
        return "FAIL"

    # 정답 문서 요약
    answer_summary = "\n".join([
        f"문서 {i+1}: {doc['rationale'][:150]}"
        for i, doc in enumerate(detailed_scores[:3])
    ])

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 엄격한 품질 검증 전문가입니다. 정답 후보를 비판적으로 평가하세요."
                },
                {
                    "role": "user",
                    "content": f"""쿼리: {query}

정답 후보:
{answer_summary}

이 정답 후보가 정말 올바른지 비판적으로 검토하세요.

가능한 문제점:
- 쿼리와 실제로 무관한가?
- 일부만 답하고 핵심을 놓쳤는가?
- 오해의 소지가 있는가?
- 너무 일반적이거나 모호한가?

문제가 없으면 "PASS", 문제가 있으면 "FAIL"을 출력하고 이유를 1-2문장으로 설명하세요.

출력 형식:
판정: PASS 또는 FAIL
이유: [이유]

위 형식대로만 출력하세요."""
                }
            ],
            temperature=0.0,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # PASS/FAIL 추출
        if "PASS" in result_text:
            return "PASS"
        else:
            return "FAIL"

    except Exception as e:
        print(f"  ⚠️ Adversarial validation 실패: {e}")
        return "FAIL"


# ========== 메인 파이프라인 ==========

def create_ultra_validation_set(eval_path, output_path):
    """Ultra 고품질 밸리데이션 세트 생성"""
    # 데이터 로드
    with open(eval_path, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    validation_set = []
    rejected_set = []

    stats = {
        'total': 0,
        'smalltalk_skipped': 0,
        'phase1_failed': 0,
        'phase2_failed': 0,
        'phase3_failed': 0,
        'phase5_failed': 0,
        'passed': 0
    }

    print(f"\n{'='*80}")
    print(f"Ultra High-Quality Validation Set Creation")
    print(f"{'='*80}\n")

    for item in tqdm(eval_data, desc="Processing queries"):
        stats['total'] += 1
        eval_id = item['eval_id']
        msg = item['msg']

        # 일반 대화 스킵
        if eval_id in SMALLTALK_IDS:
            stats['smalltalk_skipped'] += 1
            continue

        # 쿼리 재작성
        query = rewrite_query_with_context(msg)

        # 문서 검색 (Top 20)
        docs = hybrid_search_rrf(query, embeddings_dict, top_k=20)

        if len(docs) < 3:
            stats['phase1_failed'] += 1
            rejected_set.append({
                'eval_id': eval_id,
                'query': query,
                'reason': 'insufficient_docs',
                'phase': 'initial_search'
            })
            continue

        # Phase 1: Multi-Pass Consensus
        consensus_docids = phase1_multipass_consensus(query, docs)

        if len(consensus_docids) < 3:
            stats['phase1_failed'] += 1
            rejected_set.append({
                'eval_id': eval_id,
                'query': query,
                'reason': 'no_consensus',
                'phase': 'phase1',
                'consensus_count': len(consensus_docids)
            })
            continue

        # Phase 2: Detailed Scoring
        detailed_scores = phase2_detailed_scoring(query, docs, consensus_docids)

        if len(detailed_scores) < 3:
            stats['phase2_failed'] += 1
            rejected_set.append({
                'eval_id': eval_id,
                'query': query,
                'reason': 'low_scores',
                'phase': 'phase2',
                'scored_count': len(detailed_scores)
            })
            continue

        # Phase 3: Confidence Check
        confidence = phase3_confidence_check(query, detailed_scores)

        if not confidence or confidence['score'] < 4:
            stats['phase3_failed'] += 1
            rejected_set.append({
                'eval_id': eval_id,
                'query': query,
                'reason': 'low_confidence',
                'phase': 'phase3',
                'confidence': confidence['score'] if confidence else 0
            })
            continue

        # Phase 4: Multi-Turn Rewrites (optional, 검증용)
        rewrites = None
        if isinstance(msg, list) and len(msg) > 1:
            rewrites = phase4_multiturn_rewrites(msg)

        # Phase 5: Adversarial Validation
        validation_result = phase5_adversarial_validation(query, detailed_scores)

        if validation_result != "PASS":
            stats['phase5_failed'] += 1
            rejected_set.append({
                'eval_id': eval_id,
                'query': query,
                'reason': 'failed_adversarial',
                'phase': 'phase5'
            })
            continue

        # 최종 승인!
        stats['passed'] += 1

        # 점수 높은 순으로 정렬하여 상위 3개를 ground truth로 선택
        sorted_scores = sorted(detailed_scores, key=lambda x: x['score'], reverse=True)

        validation_entry = {
            'eval_id': eval_id,
            'query': query if isinstance(msg, str) else msg,
            'rewritten_query': query,
            'ground_truth': [s['docid'] for s in sorted_scores[:3]],
            'detailed_scores': detailed_scores,
            'confidence': confidence,
            'rewrites': rewrites,
            'validation': 'PASS',
            'quality': 'ultra_high'
        }

        validation_set.append(validation_entry)

    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in validation_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open('validation_rejected.jsonl', 'w', encoding='utf-8') as f:
        for entry in rejected_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 통계 저장
    with open('validation_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 통계 출력
    print(f"\n{'='*80}")
    print(f"✅ Ultra Validation Set 생성 완료")
    print(f"{'='*80}")
    print(f"\n📊 통계:")
    print(f"  전체 쿼리: {stats['total']}")
    print(f"  일반 대화 제외: {stats['smalltalk_skipped']}")
    print(f"  Phase 1 실패 (합의 부족): {stats['phase1_failed']}")
    print(f"  Phase 2 실패 (낮은 점수): {stats['phase2_failed']}")
    print(f"  Phase 3 실패 (낮은 신뢰도): {stats['phase3_failed']}")
    print(f"  Phase 5 실패 (검증 실패): {stats['phase5_failed']}")
    print(f"  ✅ 최종 통과: {stats['passed']} samples")
    print(f"\n  신뢰도: {stats['passed']/(stats['total']-stats['smalltalk_skipped'])*100:.1f}%")
    print(f"{'='*80}")

    return validation_set, rejected_set, stats


def main():
    print("="*80)
    print("Ultra High-Quality Validation Set Creation")
    print("Solar Pro 5-Phase Pipeline")
    print("="*80)

    # Elasticsearch 연결 확인
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # 처리
    validation_set, rejected_set, stats = create_ultra_validation_set(
        eval_path='../data/eval.jsonl',
        output_path='ultra_validation_solar.jsonl'
    )

    print(f"\n💾 저장 완료:")
    print(f"  - ultra_validation_solar.jsonl: {len(validation_set)}개")
    print(f"  - validation_rejected.jsonl: {len(rejected_set)}개")
    print(f"  - validation_stats.json")


if __name__ == "__main__":
    main()
