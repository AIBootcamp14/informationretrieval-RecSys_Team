"""
Document Indexing Script with Nori Analyzer for Elasticsearch

Indexes documents from documents.jsonl into Elasticsearch 'test' index with nori analyzer.
"""

import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

# ES 연결 (HTTP, 보안 없음)
es = Elasticsearch(['http://localhost:9200'])

print("Elasticsearch 연결 확인...")
print(es.info())

# 인덱스 설정 (nori analyzer + optimized BM25)
settings = {
    'number_of_shards': 1,
    'number_of_replicas': 0,
    'analysis': {
        'analyzer': {
            'nori': {
                'type': 'custom',
                'tokenizer': 'nori_tokenizer',
                'filter': ['nori_posfilter']
            }
        },
        'filter': {
            'nori_posfilter': {
                'type': 'nori_part_of_speech',
                'stoptags': ['E', 'IC', 'J', 'MAG', 'MAJ', 'MM', 'SP', 'SSC', 'SSO', 'SC', 'SE', 'XPN', 'XSA', 'XSN', 'XSV', 'UNA', 'NA', 'VSV']
            }
        }
    },
    'similarity': {
        'custom_bm25': {
            'type': 'BM25',
            'k1': 0.9,
            'b': 0.5
        }
    }
}

mappings = {
    'properties': {
        'docid': {'type': 'keyword'},
        'content': {
            'type': 'text',
            'analyzer': 'nori',
            'similarity': 'custom_bm25'
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
print("새 'test' 인덱스 생성 중 (nori analyzer)...")
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

# Analyzer 테스트
print("\n" + "="*80)
print("Nori Analyzer 테스트")
print("="*80)

test_text = "광합성의 원리는 무엇인가요?"
analyze_result = es.indices.analyze(
    index='test',
    body={
        'analyzer': 'nori',
        'text': test_text
    }
)

print(f"\n테스트 텍스트: {test_text}")
print(f"토큰화 결과:")
for token in analyze_result['tokens']:
    print(f"  - {token['token']} (type: {token['type']})")

print("\n✅ Nori analyzer가 정상적으로 작동합니다!")
