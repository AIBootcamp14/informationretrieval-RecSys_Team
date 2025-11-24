"""
Dense Retrieval을 위한 문서 임베딩 생성 및 인덱싱
ko-sroberta 모델 사용
"""

import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pickle

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# 임베딩 모델 로드 (이미 semantic_chunking.py에서 사용 중인 모델)
print("임베딩 모델 로드 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print(f"✅ 모델 로드 완료: {model.get_sentence_embedding_dimension()}차원")

def generate_embeddings_for_index(index_name='test', batch_size=32):
    """
    ES 인덱스의 모든 문서에 대해 임베딩 생성

    Args:
        index_name: Elasticsearch 인덱스명
        batch_size: 배치 크기
    """
    print(f"\n{'='*80}")
    print(f"Dense Indexing: {index_name}")
    print(f"{'='*80}")

    # 전체 문서 수 확인
    count_response = es.count(index=index_name)
    total_docs = count_response['count']
    print(f"총 문서 수: {total_docs}개")

    # 모든 문서 가져오기 (scroll API 사용)
    documents = []
    scroll_resp = es.search(
        index=index_name,
        scroll='5m',
        size=1000,
        body={
            'query': {'match_all': {}},
            '_source': ['docid', 'content', 'original_docid']
        }
    )

    scroll_id = scroll_resp['_scroll_id']
    documents.extend(scroll_resp['hits']['hits'])

    # 스크롤로 나머지 문서 가져오기
    while len(scroll_resp['hits']['hits']) > 0:
        scroll_resp = es.scroll(scroll_id=scroll_id, scroll='5m')
        documents.extend(scroll_resp['hits']['hits'])
        print(f"진행: {len(documents)}/{total_docs}개 문서 로드됨")

    es.clear_scroll(scroll_id=scroll_id)

    print(f"\n✅ {len(documents)}개 문서 로드 완료")

    # 임베딩 생성
    print(f"\n임베딩 생성 시작 (배치 크기: {batch_size})")

    embeddings_dict = {}
    contents = []
    docids = []

    for doc in documents:
        source = doc['_source']
        docid = source['docid']
        content = source['content']

        docids.append(docid)
        contents.append(content)

    # 배치로 임베딩 생성
    all_embeddings = []
    for i in tqdm(range(0, len(contents), batch_size), desc="임베딩 생성"):
        batch = contents[i:i+batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings)

    # Dictionary 생성
    for docid, embedding in zip(docids, all_embeddings):
        embeddings_dict[docid] = embedding

    # 저장
    output_file = f'embeddings_{index_name}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"\n{'='*80}")
    print(f"✅ 임베딩 생성 완료")
    print(f"{'='*80}")
    print(f"💾 저장 위치: {output_file}")
    print(f"📊 통계:")
    print(f"   문서 수: {len(embeddings_dict)}개")
    print(f"   임베딩 차원: {all_embeddings[0].shape[0]}차원")
    print(f"   파일 크기: {len(pickle.dumps(embeddings_dict)) / (1024*1024):.1f} MB")
    print(f"{'='*80}")

    return embeddings_dict

def test_dense_search(embeddings_dict, query, top_k=10):
    """
    테스트용 Dense 검색

    Args:
        embeddings_dict: {docid: embedding} 딕셔너리
        query: 검색 쿼리
        top_k: 반환할 문서 수
    """
    # 쿼리 임베딩
    query_emb = model.encode([query], convert_to_numpy=True)[0]

    # 코사인 유사도 계산
    scores = []
    for docid, doc_emb in embeddings_dict.items():
        similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scores.append((docid, similarity))

    # 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]

if __name__ == "__main__":
    # 임베딩 생성
    embeddings = generate_embeddings_for_index('test', batch_size=64)

    # 간단한 테스트
    print("\n" + "="*80)
    print("Dense Search 테스트")
    print("="*80)

    test_queries = [
        "광합성의 원리는 무엇인가?",
        "DNA 복제 과정을 설명해주세요",
        "태양계 행성의 특징은?"
    ]

    for query in test_queries:
        print(f"\n쿼리: {query}")
        results = test_dense_search(embeddings, query, top_k=3)
        for i, (docid, score) in enumerate(results, 1):
            print(f"  {i}. {docid} (유사도: {score:.4f})")

    print("\n" + "="*80)
    print("✅ 테스트 완료")
    print("="*80)
