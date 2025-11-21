"""
AI가 직접 작성하는 Complete Validation Set

전략:
1. LLM으로 각 쿼리가 과학 질문인지 일반 대화인지 판단
2. 과학 질문이면 BM25 Top-10 결과를 LLM에게 보여주고 정답 선택
3. 완전 자동화된 validation set 생성
"""

import json
import os
from openai import OpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1/solar"
)

es = Elasticsearch(['http://localhost:9200'])

def classify_query_type(query, msg):
    """
    LLM으로 쿼리 타입 분류
    - 과학 질문 vs 일반 대화
    """
    # 멀티턴인 경우 컨텍스트 포함
    if isinstance(msg, list) and len(msg) > 1:
        context = "\n".join([f"{m['role']}: {m['content']}" for m in msg[:-1]])
        full_query = f"이전 대화:\n{context}\n\n현재 질문: {query}"
    else:
        full_query = query

    prompt = f"""다음 질문을 분류해주세요.

질문: "{full_query}"

분류 기준:
- "science": 과학적 사실, 자연 현상, 기술, 교육, 사회 현상 등에 대한 지식 질문
- "general": 일반 대화, 감정 표현, 메타 질문, 의견 질문

예시:
"DNA의 역할은?" → science
"너 뭘 잘해?" → general
"오늘 너무 즐거웠어" → general
"광합성 과정을 설명해줘" → science
"강아지가 사회화되는 행동의 사례는?" → general (행동학은 있지만 사례 나열 요청)

정답만 출력하세요 (science 또는 general):"""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        result = response.choices[0].message.content.strip().lower()

        if 'science' in result:
            return 'science'
        elif 'general' in result:
            return 'general'
        else:
            # 애매한 경우 휴리스틱으로 판단
            if any(word in query for word in ['너', '나', '우리', '어떻게 생각', '~할까', '~있을까']):
                return 'general'
            return 'science'

    except Exception as e:
        print(f"  ⚠️ Classification failed: {e}")
        # Fallback to heuristic
        if any(word in query for word in ['너', '나', '우리', '힘들', '즐거', '신나']):
            return 'general'
        return 'science'

def select_relevant_documents(query, search_results):
    """
    LLM으로 관련 문서 선택

    BM25 Top-10 중에서 실제로 질문에 답변할 수 있는 문서를 선택
    """
    if not search_results:
        return []

    # 문서 정보 구성
    docs_info = []
    for i, hit in enumerate(search_results, 1):
        doc_content = hit['_source']['content'][:500]  # 처음 500자만
        docs_info.append(f"[{i}] {doc_content}...")

    docs_text = "\n\n".join(docs_info)

    prompt = f"""다음 질문에 답변하는 데 도움이 되는 문서를 선택하세요.

질문: "{query}"

문서 목록:
{docs_text}

지침:
1. 질문에 직접적으로 답변할 수 있는 문서만 선택
2. 부분적으로라도 관련된 내용이 있으면 선택
3. 관련 없는 문서는 제외
4. 최소 0개, 최대 3개까지 선택
5. 문서 번호만 출력 (예: 1,3,5 또는 없으면 "none")

선택된 문서 번호:"""

    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        result = response.choices[0].message.content.strip()

        if 'none' in result.lower() or not result:
            return []

        # 숫자 추출
        selected = []
        for num_str in result.replace(',', ' ').split():
            try:
                num = int(num_str)
                if 1 <= num <= len(search_results):
                    selected.append(num - 1)  # 0-indexed
            except ValueError:
                continue

        # 상위 3개만
        selected = selected[:3]

        return [search_results[i]['_source']['docid'] for i in selected]

    except Exception as e:
        print(f"  ⚠️ Document selection failed: {e}")
        # Fallback: Top-3
        return [hit['_source']['docid'] for hit in search_results[:3]]

def search_documents(query, top_k=10):
    """BM25 검색"""
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
            'size': top_k
        }
    )

    return response['hits']['hits']

def create_ai_validation():
    """
    AI가 직접 작성하는 validation set
    """
    print("=" * 80)
    print("AI 자동 생성 Validation Set")
    print("=" * 80)

    # Load eval data
    with open('../data/eval.jsonl', 'r') as f:
        eval_data = [json.loads(line) for line in f]

    validation_set = []

    print(f"\n총 {len(eval_data)}개 쿼리 처리 중...\n")

    for item in tqdm(eval_data, desc="Processing"):
        eval_id = item['eval_id']

        # 쿼리 추출
        if isinstance(item['msg'], list):
            query = item['msg'][-1]['content']
        else:
            query = item['msg']

        # 1. 쿼리 타입 분류
        query_type = classify_query_type(query, item['msg'])

        if query_type == 'general':
            # 일반 대화
            validation_set.append({
                'eval_id': eval_id,
                'query': query,
                'msg': item['msg'],
                'ground_truth': [],
                'confidence': 'ai_certain',
                'query_type': 'general',
                'source': 'ai'
            })
            continue

        # 2. 과학 질문 - 문서 검색
        search_results = search_documents(query, top_k=10)

        if not search_results:
            # 검색 결과 없음
            validation_set.append({
                'eval_id': eval_id,
                'query': query,
                'msg': item['msg'],
                'ground_truth': [],
                'confidence': 'ai_no_result',
                'query_type': 'science',
                'source': 'ai'
            })
            continue

        # 3. 관련 문서 선택
        ground_truth = select_relevant_documents(query, search_results)

        max_score = search_results[0]['_score']

        # 난이도 판정
        if max_score >= 15:
            difficulty = 'easy'
        elif max_score >= 8:
            difficulty = 'medium'
        elif max_score >= 3:
            difficulty = 'hard'
        else:
            difficulty = 'very_hard'

        validation_set.append({
            'eval_id': eval_id,
            'query': query,
            'msg': item['msg'],
            'ground_truth': ground_truth,
            'confidence': 'ai_confident',
            'query_type': 'science',
            'difficulty': difficulty,
            'max_score': max_score,
            'source': 'ai'
        })

    # 저장
    output_path = 'ai_validation.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in validation_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 통계
    science_count = sum(1 for item in validation_set if item['query_type'] == 'science')
    general_count = len(validation_set) - science_count

    with_docs = sum(1 for item in validation_set if item['ground_truth'])
    without_docs = len(validation_set) - with_docs

    print(f"\n{'='*80}")
    print(f"✅ AI Validation Set 생성 완료: {output_path}")
    print(f"{'='*80}")
    print(f"총 {len(validation_set)}개 쿼리")
    print(f"\nQuery Type:")
    print(f"  - 과학 질문: {science_count}개")
    print(f"  - 일반 대화: {general_count}개")
    print(f"\nGround Truth:")
    print(f"  - 문서 있음: {with_docs}개")
    print(f"  - 문서 없음: {without_docs}개")
    print(f"{'='*80}")

def main():
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    print("✅ Elasticsearch 연결 성공\n")

    create_ai_validation()

if __name__ == "__main__":
    main()
