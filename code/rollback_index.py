"""
Elasticsearch 인덱스 롤백

목적: Semantic Chunking 결과가 좋지 않을 경우 원복
"""

from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])

def rollback():
    print("="*80)
    print("Elasticsearch 인덱스 롤백")
    print("="*80)

    if not es.ping():
        print("❌ Elasticsearch 연결 실패")
        return

    if not es.indices.exists(index='test_backup'):
        print("❌ 백업 인덱스 없음 (test_backup)")
        print("   롤백 불가능합니다.")
        return

    # 현재 인덱스 삭제
    print("\n🗑️  현재 'test' 인덱스 삭제 중...")
    if es.indices.exists(index='test'):
        es.indices.delete(index='test')

    # 백업에서 복원
    print("💾 백업에서 복원 중... (test_backup → test)")
    es.reindex(
        body={
            'source': {'index': 'test_backup'},
            'dest': {'index': 'test'}
        },
        wait_for_completion=True
    )

    restored_count = es.count(index='test')['count']
    print(f"✅ 롤백 완료: {restored_count}개 문서 복원")

if __name__ == "__main__":
    rollback()
