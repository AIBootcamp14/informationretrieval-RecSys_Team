"""
Elasticsearch 재인덱싱 with Semantic Chunking

목적:
1. 기존 documents.jsonl을 Semantic Chunking으로 재처리
2. test 인덱스 재생성
3. 의미적으로 완결된 청크로 재인덱싱
"""

import json
import time
from elasticsearch import Elasticsearch
from tqdm import tqdm
from semantic_chunking import semantic_sentence_chunking

# ES 연결
es = Elasticsearch(['http://localhost:9200'])

def backup_existing_index():
    """기존 인덱스 백업"""
    print(f"\n{'='*80}")
    print("기존 인덱스 백업")
    print(f"{'='*80}")

    if not es.indices.exists(index='test'):
        print("⚠️ 기존 'test' 인덱스 없음 (백업 불필요)")
        return

    # test_backup으로 복사
    if es.indices.exists(index='test_backup'):
        print("🗑️  기존 백업 삭제 중...")
        es.indices.delete(index='test_backup')

    print("💾 백업 생성 중... (test → test_backup)")
    es.reindex(
        body={
            'source': {'index': 'test'},
            'dest': {'index': 'test_backup'}
        },
        wait_for_completion=True
    )

    backup_count = es.count(index='test_backup')['count']
    print(f"✅ 백업 완료: {backup_count}개 문서")

def create_new_index():
    """새 인덱스 생성 (Nori 분석기 포함)"""
    print(f"\n{'='*80}")
    print("새 인덱스 생성")
    print(f"{'='*80}")

    # 기존 인덱스 삭제
    if es.indices.exists(index='test'):
        print("🗑️  기존 'test' 인덱스 삭제 중...")
        es.indices.delete(index='test')

    # 새 인덱스 생성
    print("🏗️  새 'test' 인덱스 생성 중...")
    es.indices.create(
        index='test',
        body={
            'settings': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
                'analysis': {
                    'analyzer': {
                        'nori': {
                            'type': 'custom',
                            'tokenizer': 'nori_tokenizer',
                            'filter': ['nori_part_of_speech']
                        }
                    }
                }
            },
            'mappings': {
                'properties': {
                    'docid': {'type': 'keyword'},
                    'original_docid': {'type': 'keyword'},  # 원본 문서 ID
                    'chunk_index': {'type': 'integer'},  # 청크 순서
                    'total_chunks': {'type': 'integer'},  # 전체 청크 수
                    'content': {
                        'type': 'text',
                        'analyzer': 'nori'
                    }
                }
            }
        }
    )

    print("✅ 인덱스 생성 완료")

def reindex_with_semantic_chunking(documents_path='../data/documents.jsonl'):
    """Semantic Chunking 기반 재인덱싱"""
    print(f"\n{'='*80}")
    print("Semantic Chunking 재인덱싱")
    print(f"{'='*80}")

    # 문서 로드
    print(f"\n📂 문서 로드 중: {documents_path}")
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f]

    print(f"✅ 로드 완료: {len(documents)}개 문서")

    # 통계
    total_chunks = 0
    chunk_distribution = {}  # 청크 수별 문서 개수
    total_original_docs = 0
    total_kept_original = 0  # 청킹되지 않은 문서 (1개 청크)
    total_chunked = 0  # 청킹된 문서 (2개 이상)

    # 재인덱싱
    print(f"\n🔄 재인덱싱 중...")
    for doc in tqdm(documents, desc="Semantic Chunking"):
        docid = doc['docid']
        content = doc['content']
        total_original_docs += 1

        # Semantic Chunking with Overlap
        chunks = semantic_sentence_chunking(
            content,
            max_chunk_size=400,
            min_chunk_size=100,
            similarity_threshold=0.65,
            overlap_sentences=1  # 1문장 overlap
        )

        # 통계 수집
        num_chunks = len(chunks)
        chunk_distribution[num_chunks] = chunk_distribution.get(num_chunks, 0) + 1
        total_chunks += num_chunks

        if num_chunks == 1:
            total_kept_original += 1
        else:
            total_chunked += 1

        # 각 청크 인덱싱
        for i, chunk in enumerate(chunks):
            # DocID 생성 규칙:
            # - 1개 청크: 원본 docid 그대로 (호환성)
            # - 2개 이상: docid_chunk0, docid_chunk1, ...
            if num_chunks == 1:
                chunk_docid = docid
            else:
                chunk_docid = f"{docid}_chunk{i}"

            es.index(
                index='test',
                body={
                    'docid': chunk_docid,
                    'original_docid': docid,
                    'chunk_index': i,
                    'total_chunks': num_chunks,
                    'content': chunk
                }
            )

    # 결과 출력
    print(f"\n{'='*80}")
    print("재인덱싱 완료")
    print(f"{'='*80}")

    print(f"\n📊 통계:")
    print(f"  원본 문서: {total_original_docs}개")
    print(f"  전체 청크: {total_chunks}개")
    print(f"  평균 청크 수: {total_chunks/total_original_docs:.2f}개/문서")

    print(f"\n📈 청크 분포:")
    for num_chunks in sorted(chunk_distribution.keys()):
        count = chunk_distribution[num_chunks]
        pct = count / total_original_docs * 100
        print(f"  {num_chunks}개 청크: {count:4d}개 문서 ({pct:5.1f}%)")

    print(f"\n🔍 청킹 효과:")
    print(f"  청킹 안 됨 (1개): {total_kept_original:4d}개 ({total_kept_original/total_original_docs*100:5.1f}%)")
    print(f"  청킹 됨 (2개+): {total_chunked:4d}개 ({total_chunked/total_original_docs*100:5.1f}%)")

    # Elasticsearch 문서 수 확인
    es.indices.refresh(index='test')
    time.sleep(2)
    indexed_count = es.count(index='test')['count']

    print(f"\n✅ Elasticsearch 인덱싱 완료: {indexed_count}개 문서")

    if indexed_count != total_chunks:
        print(f"⚠️ 경고: 예상 청크 수({total_chunks})와 인덱싱 수({indexed_count}) 불일치")
    else:
        print(f"✅ 검증 성공: 청크 수 일치")

def verify_index():
    """인덱스 검증"""
    print(f"\n{'='*80}")
    print("인덱스 검증")
    print(f"{'='*80}")

    # 샘플 검색
    response = es.search(
        index='test',
        body={
            'query': {'match': {'content': 'DNA'}},
            'size': 3
        }
    )

    print(f"\n🔍 샘플 검색 (query='DNA'):")
    print(f"  매칭 문서: {response['hits']['total']['value']}개")

    print(f"\n  Top-3:")
    for i, hit in enumerate(response['hits']['hits'], 1):
        source = hit['_source']
        print(f"\n  [{i}] {source['docid']}")
        print(f"      원본: {source['original_docid']}")
        print(f"      청크: {source['chunk_index']+1}/{source['total_chunks']}")
        print(f"      내용: {source['content'][:80]}...")

def main():
    print("="*80)
    print("Elasticsearch Overlap Chunking 재인덱싱")
    print("="*80)

    # 1. ES 연결 확인
    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        print("   docker start elasticsearch 를 실행하세요")
        return

    print("✅ Elasticsearch 연결 성공")

    # 2. 기존 인덱스 백업
    backup_existing_index()

    # 3. 새 인덱스 생성
    create_new_index()

    # 4. Semantic Chunking 재인덱싱
    reindex_with_semantic_chunking()

    # 5. 검증
    verify_index()

    print(f"\n{'='*80}")
    print("✅ 전체 프로세스 완료")
    print(f"{'='*80}")

    print(f"\n💡 다음 단계:")
    print(f"  1. solar_semantic_v1.py 실행으로 새 인덱스 테스트")
    print(f"  2. 결과가 좋으면 유지, 나쁘면 롤백:")
    print(f"     롤백 명령: python rollback_index.py")

if __name__ == "__main__":
    main()
