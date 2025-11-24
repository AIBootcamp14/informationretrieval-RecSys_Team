"""
BGE-M3 최적화 임베딩 생성
세 가지 검색 모드 모두 활성화: Dense + Sparse + ColBERT
목표: 0.79+ MAP@3 달성을 위한 최적화
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

# BGE-M3로 최적화된 임베딩 생성
print("\n" + "="*80)
print("BGE-M3 최적화 임베딩 생성")
print("="*80)
print("최적화 설정:")
print("  - return_dense=True      (1024d 벡터)")
print("  - return_sparse=True     (학습된 어휘 가중치, BM25보다 우수)")
print("  - return_colbert_vecs=True (다중 벡터, 정밀한 관련성)")
print("  - max_length=1024        (과학 문서에 최적화)")
print("  - batch_size=12          (메모리 고려)")
print("="*80 + "\n")

embeddings_dict = {}

batch_size = 12  # ColBERT 벡터로 인한 메모리 사용량 고려
docids = list(all_docs.keys())
contents = [all_docs[docid] for docid in docids]

for i in tqdm(range(0, len(docids), batch_size), desc="임베딩 생성"):
    batch_docids = docids[i:i+batch_size]
    batch_contents = contents[i:i+batch_size]

    try:
        # BGE-M3 최적화 임베딩 (세 가지 모드 모두 활성화)
        embeddings = model.encode(
            batch_contents,
            return_dense=True,        # ✅ Dense 벡터 (1024d)
            return_sparse=True,       # ✅ Sparse 가중치 (BM25 대체)
            return_colbert_vecs=True, # ✅ ColBERT 다중 벡터 (정밀 매칭)
            max_length=1024,          # ✅ 과학 문서에 최적 (기존 512→1024)
            batch_size=12
        )

        # 결과 저장 (세 가지 유형 모두)
        dense_vecs = embeddings['dense_vecs']
        lexical_weights = embeddings['lexical_weights']
        colbert_vecs = embeddings['colbert_vecs']

        for j, docid in enumerate(batch_docids):
            embeddings_dict[docid] = {
                'dense': dense_vecs[j],        # numpy array (1024,)
                'sparse': lexical_weights[j],  # dict {token_id: weight}
                'colbert': colbert_vecs[j]     # numpy array (N, 1024)
            }

    except Exception as e:
        print(f"\n⚠️  배치 {i//batch_size} 실패: {e}")
        # 실패한 배치는 개별 처리
        for docid, content in zip(batch_docids, batch_contents):
            try:
                embedding = model.encode(
                    [content],
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                    max_length=1024
                )
                embeddings_dict[docid] = {
                    'dense': embedding['dense_vecs'][0],
                    'sparse': embedding['lexical_weights'][0],
                    'colbert': embedding['colbert_vecs'][0]
                }
            except Exception as e2:
                print(f"⚠️  문서 {docid} 실패: {e2}")
                # 빈 벡터로 대체
                embeddings_dict[docid] = {
                    'dense': np.zeros(1024, dtype=np.float32),
                    'sparse': {},
                    'colbert': np.zeros((1, 1024), dtype=np.float32)
                }

print(f"\n✅ {len(embeddings_dict)}개 임베딩 생성 완료")

# 저장
output_path = 'embeddings_test_bgem3_optimized.pkl'
print(f"\n임베딩 저장 중: {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f"✅ 저장 완료!")

# 통계 출력
sample_docid = list(embeddings_dict.keys())[0]
sample_emb = embeddings_dict[sample_docid]

print(f"\n📊 임베딩 통계:")
print(f"  - 총 문서 수: {len(embeddings_dict)}")
print(f"  - Dense 벡터 차원: {len(sample_emb['dense'])}")
print(f"  - Sparse 토큰 수 (샘플): {len(sample_emb['sparse'])}")
print(f"  - ColBERT 벡터 수 (샘플): {sample_emb['colbert'].shape[0]}")
print(f"  - ColBERT 벡터 차원: {sample_emb['colbert'].shape[1]}")
print(f"  - 데이터 타입: {sample_emb['dense'].dtype}")
print(f"  - 파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

print("\n" + "="*80)
print("✅ BGE-M3 최적화 임베딩 생성 완료")
print("="*80)
print("다음 단계: hybrid_bgem3_optimized.py 구현")
print("  - Dense + Sparse + ColBERT 하이브리드 스코어링")
print("  - 가중치 조합: w1·s_dense + w2·s_lex + w3·s_mul")
print("="*80)
