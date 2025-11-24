"""
Simple Document Indexing Script for Elasticsearch

Indexes documents from documents.jsonl into Elasticsearch 'test' index.
"""

import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

# ES 연결 (HTTP, 보안 없음)
es = Elasticsearch(['http://localhost:9200'])

print("Elasticsearch 연결 확인...")
print(es.info())

# 인덱스 설정 (standard analyzer 사용)
settings = {
    'number_of_shards': 1,
    'number_of_replicas': 0
}

mappings = {
    'properties': {
        'docid': {'type': 'keyword'},
        'content': {
            'type': 'text',
            'analyzer': 'standard'
        },
        'src': {'type': 'keyword'}
    }
}

# 기존 인덱스 삭제
if es.indices.exists(index='test'):
    print("기존 'test' 인덱스 삭제 중...")
    es.indices.delete(index='test')
    print("✅ 삭제 완료")

# 새 인덱스 생성
print("새 'test' 인덱스 생성 중...")
es.indices.create(index='test', settings=settings, mappings=mappings)
print("✅ 인덱스 생성 완료")

# 문서 로드
print("\n문서 로드 중...")
documents = []
with open('../data/documents.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line.strip())
        documents.append(doc)

print(f"✅ {len(documents)}개 문서 로드 완료")

# 문서 인덱싱
print(f"\n문서 인덱싱 시작...")
for doc in tqdm(documents, desc="인덱싱"):
    es.index(
        index='test',
        id=doc['docid'],
        document={
            'docid': doc['docid'],
            'content': doc['content'],
            'src': doc.get('src', '')
        }
    )

# Refresh
es.indices.refresh(index='test')

print(f"\n✅ 인덱싱 완료!")
print(f"총 {len(documents)}개 문서가 'test' 인덱스에 저장되었습니다.")

# 확인
count = es.count(index='test')
print(f"\n인덱스 문서 수: {count['count']}")
