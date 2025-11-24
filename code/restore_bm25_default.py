"""
BM25 설정을 기본값으로 복원
"""

from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])

def restore_default_bm25():
    # Close index
    es.indices.close(index='test')
    
    # Restore to default BM25 settings
    es.indices.put_settings(
        index='test',
        body={
            'index': {
                'similarity': {
                    'custom_bm25': {
                        'type': 'BM25',
                        'k1': 1.2,  # Default
                        'b': 0.75   # Default
                    }
                }
            }
        }
    )
    
    # Reopen index
    es.indices.open(index='test')
    print("✅ BM25 설정을 기본값으로 복원했습니다 (k1=1.2, b=0.75)")

if __name__ == "__main__":
    restore_default_bm25()
