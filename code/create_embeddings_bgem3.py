"""
BGE-M3로 Dense 임베딩 재생성
최신 멀티링구얼 임베딩 모델 사용
"""

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

# BGE-M3 모델 로드
print("BGE-M3 모델 로드 중...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ 모델 로드 완료")

# ES에서 모든 문서 가져오기
print("\nElasticsearch에서 문서 로드 중...")

# Scroll API로 모든 문서 가져오기
query = {
    "query": {"match_all": {}},
    "size": 1000
}

# 초기 검색
response = es.search(index='test', body=query, scroll='5m')
scroll_id = response['_scroll_id']
hits = response['hits']['hits']

all_docs = {}
for hit in hits:
    source = hit['_source']
    # original_docid가 있으면 사용, 없으면 docid 사용
    docid = source.get('original_docid', source['docid'])
    content = source['content']

    # 중복 제거 (original_docid 기준)
    if docid not in all_docs:
        all_docs[docid] = content

# 나머지 문서들 스크롤로 가져오기
while len(hits) > 0:
    response = es.scroll(scroll_id=scroll_id, scroll='5m')
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']

    for hit in hits:
        source = hit['_source']
        docid = source.get('original_docid', source['docid'])
        content = source['content']

        if docid not in all_docs:
            all_docs[docid] = content

# 스크롤 정리
es.clear_scroll(scroll_id=scroll_id)

print(f"✅ {len(all_docs)}개 문서 로드 완료")

# BGE-M3로 임베딩 생성
print("\nBGE-M3로 임베딩 생성 중...")
embeddings_dict = {}

batch_size = 32  # 배치 처리로 속도 향상
docids = list(all_docs.keys())
contents = [all_docs[docid] for docid in docids]

for i in tqdm(range(0, len(docids), batch_size), desc="임베딩 생성"):
    batch_docids = docids[i:i+batch_size]
    batch_contents = contents[i:i+batch_size]

    try:
        # BGE-M3 임베딩 (Dense만 사용)
        embeddings = model.encode(
            batch_contents,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=512  # 최대 길이 제한
        )

        # 결과 저장
        dense_vecs = embeddings['dense_vecs']
        for j, docid in enumerate(batch_docids):
            embeddings_dict[docid] = dense_vecs[j]

    except Exception as e:
        print(f"\n⚠️  배치 {i//batch_size} 실패: {e}")
        # 실패한 배치는 개별 처리
        for docid, content in zip(batch_docids, batch_contents):
            try:
                embedding = model.encode(
                    [content],
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                    max_length=512
                )
                embeddings_dict[docid] = embedding['dense_vecs'][0]
            except Exception as e2:
                print(f"⚠️  문서 {docid} 실패: {e2}")
                # 빈 벡터로 대체
                embeddings_dict[docid] = np.zeros(1024, dtype=np.float32)

print(f"\n✅ {len(embeddings_dict)}개 임베딩 생성 완료")

# 저장
output_path = 'embeddings_test_bgem3.pkl'
print(f"\n임베딩 저장 중: {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f"✅ 저장 완료!")

# 통계 출력
sample_embedding = list(embeddings_dict.values())[0]
print(f"\n📊 임베딩 통계:")
print(f"  - 총 문서 수: {len(embeddings_dict)}")
print(f"  - 임베딩 차원: {len(sample_embedding)}")
print(f"  - 데이터 타입: {sample_embedding.dtype}")
print(f"  - 파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
