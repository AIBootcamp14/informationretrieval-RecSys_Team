"""
LLM 기반 자동 Ground Truth 생성
Solar-pro가 과학 전문가 역할로 정답 문서 선택
"""

import json
import os
from tqdm import tqdm
from elasticsearch import Elasticsearch
from openai import OpenAI

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# Solar API 초기화
upstage_api_key = os.environ.get('UPSTAGE_API_KEY')
client = OpenAI(
    api_key=upstage_api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

# 일반 대화 ID
SMALLTALK_IDS = {276, 261, 233, 90, 222, 235, 165, 153, 169, 141, 183}

def search_bm25(query, top_k=10):
    """BM25 검색으로 후보 문서 가져오기"""
    fetch_size = top_k + 2

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
            'score': hit['_score']
        })

        if len(results) >= top_k:
            break

    return results

def llm_judge_relevance(query, documents):
    """
    LLM이 문서들을 평가하여 정답 선택
    """
    if not documents:
        return []

    # 프롬프트 구성
    prompt = f"""당신은 과학 지식 검색 시스템의 평가 전문가입니다.

**질문**: {query}

다음은 검색된 문서 목록입니다. 이 질문에 답하는 데 **가장 적합한 문서**를 선택하세요.

**선택 기준**:
1. 질문과 직접적으로 관련된 내용을 포함하는가?
2. 질문에 대한 답변을 제공하는가?
3. 과학적으로 정확하고 신뢰할 수 있는가?

**문서 목록**:
"""

    for i, doc in enumerate(documents):
        # 문서 내용 최대 400자까지만 표시
        content_preview = doc['content'][:400]
        if len(doc['content']) > 400:
            content_preview += "..."
        prompt += f"\n[{i}] {content_preview}\n"

    prompt += """
**출력 형식**:
- 정답이 될 수 있는 문서의 번호를 선택하세요 (최대 3개)
- 번호만 콤마로 구분하여 출력하세요 (예: 0,2,5)
- 적합한 문서가 없으면 "없음"이라고 출력하세요
- 설명이나 다른 텍스트는 출력하지 마세요

출력:"""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 과학 문서 검색 시스템의 평가 전문가입니다. 정확하고 객관적으로 문서를 평가합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # 일관성 중요
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()

        # "없음" 또는 빈 응답
        if not result or result == "없음" or result.lower() == "none":
            return []

        # 숫자 파싱
        selected_indices = []
        for part in result.split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(documents):
                    selected_indices.append(idx)

        # docid 반환
        return [documents[i]['docid'] for i in selected_indices[:3]]

    except Exception as e:
        print(f"⚠️  LLM 판단 실패: {e}")
        return []

def build_ground_truth():
    """220개 전체에 대해 Ground Truth 생성"""
    print("="*80)
    print("LLM 기반 자동 Ground Truth 생성")
    print("="*80)

    # ES 연결 확인
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공")

    # API Key 확인
    if not upstage_api_key:
        print("❌ UPSTAGE_API_KEY 환경변수 없음")
        return

    print("✅ Solar API Key 확인\n")

    # eval.jsonl 로드
    with open('../data/eval.jsonl', 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    print(f"📋 총 {len(eval_data)}개 쿼리 처리 시작\n")

    ground_truth = []
    stats = {'with_answers': 0, 'no_answers': 0, 'smalltalk': 0}

    for item in tqdm(eval_data, desc="LLM 레이블링"):
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 일반 대화는 건너뛰기
        if eval_id in SMALLTALK_IDS:
            ground_truth.append({
                'eval_id': eval_id,
                'query': query,
                'ground_truth': [],
                'type': 'smalltalk'
            })
            stats['smalltalk'] += 1
            continue

        # Top-10 문서 검색
        docs = search_bm25(query, top_k=10)

        if not docs:
            ground_truth.append({
                'eval_id': eval_id,
                'query': query,
                'ground_truth': [],
                'type': 'no_docs_found'
            })
            stats['no_answers'] += 1
            continue

        # LLM으로 정답 선택
        selected_docids = llm_judge_relevance(query, docs)

        ground_truth.append({
            'eval_id': eval_id,
            'query': query,
            'ground_truth': selected_docids,
            'type': 'science',
            'num_candidates': len(docs)
        })

        if selected_docids:
            stats['with_answers'] += 1
        else:
            stats['no_answers'] += 1

        # 중간 저장 (50개마다)
        if len(ground_truth) % 50 == 0:
            checkpoint_path = f'ground_truth_checkpoint_{len(ground_truth)}.jsonl'
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                for gt in ground_truth:
                    f.write(json.dumps(gt, ensure_ascii=False) + '\n')
            print(f"\n💾 체크포인트 저장: {checkpoint_path}")

    # 최종 저장
    output_path = 'ground_truth_solar_auto.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for gt in ground_truth:
            f.write(json.dumps(gt, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"✅ Ground Truth 생성 완료")
    print(f"{'='*80}")
    print(f"💾 저장 위치: {output_path}")
    print(f"\n📊 통계:")
    print(f"   총 쿼리:       {len(ground_truth)}개")
    print(f"   과학 질문:     {len(ground_truth) - stats['smalltalk']}개")
    print(f"   - 정답 있음:   {stats['with_answers']}개 ({stats['with_answers']/(len(ground_truth)-stats['smalltalk'])*100:.1f}%)")
    print(f"   - 정답 없음:   {stats['no_answers']}개")
    print(f"   일반 대화:     {stats['smalltalk']}개")
    print(f"{'='*80}")

if __name__ == "__main__":
    build_ground_truth()
