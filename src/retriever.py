"""
Hybrid retrieval module combining sparse (BM25) and dense (vector) search
"""
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from loguru import logger
import config.config as cfg
from src.embedder import get_embedder


class ElasticsearchRetriever:
    """Enhanced Elasticsearch retriever with hybrid search"""

    def __init__(
        self,
        host: str = cfg.ES_HOST,
        username: str = cfg.ES_USERNAME,
        password: str = cfg.ES_PASSWORD,
        index_name: str = cfg.ES_INDEX_NAME,
        ca_certs: str = cfg.ES_CA_CERTS
    ):
        """Initialize Elasticsearch client"""
        self.index_name = index_name

        logger.info(f"Connecting to Elasticsearch at {host}")
        logger.info(f"Using CA certificate: {ca_certs}")
        logger.info(f"Username: {username}")
        logger.info(f"Password: {'*' * len(password) if password else 'NOT SET'}")

        try:
            self.es = Elasticsearch(
                [host],
                basic_auth=(username, password),
                ca_certs=ca_certs,
                request_timeout=30
            )

            # Verify connection
            if self.es.ping():
                logger.info("Successfully connected to Elasticsearch")
                logger.info(f"Cluster info: {self.es.info()}")
            else:
                raise ConnectionError("Failed to connect to Elasticsearch - ping failed")
        except Exception as e:
            logger.error(f"Elasticsearch connection error: {type(e).__name__}: {str(e)}")
            logger.error(f"Host: {host}")
            logger.error(f"CA cert path: {ca_certs}")
            logger.error(f"CA cert exists: {Path(ca_certs).exists() if ca_certs else False}")
            raise ConnectionError(f"Failed to connect to Elasticsearch: {str(e)}")

        self.embedder = get_embedder()

    def create_index(self):
        """Create index with optimized settings"""
        if self.es.indices.exists(index=self.index_name):
            logger.warning(f"Index '{self.index_name}' already exists. Deleting...")
            self.es.indices.delete(index=self.index_name)

        logger.info(f"Creating index '{self.index_name}'")
        self.es.indices.create(
            index=self.index_name,
            settings=cfg.ES_SETTINGS,
            mappings=cfg.ES_MAPPINGS
        )
        logger.info("Index created successfully")

    def delete_index(self):
        """Delete index"""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logger.info(f"Deleted index '{self.index_name}'")

    def bulk_index(self, documents: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Bulk index documents with embeddings

        Args:
            documents: List of documents with 'content', 'docid', 'src' fields

        Returns:
            Tuple of (success_count, failed_count)
        """
        logger.info(f"Indexing {len(documents)} documents...")

        # Generate embeddings for all documents
        contents = [doc['content'] for doc in documents]
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode_batch(contents, show_progress=True)

        # Prepare bulk actions
        actions = []
        for doc, embedding in zip(documents, embeddings):
            doc['embeddings'] = embedding.tolist()
            actions.append({
                '_index': self.index_name,
                '_source': doc
            })

        # Bulk insert
        logger.info("Bulk indexing to Elasticsearch...")
        success, failed = helpers.bulk(self.es, actions, raise_on_error=False)

        logger.info(f"Indexed {success} documents, {len(failed)} failed")
        return success, len(failed)

    def sparse_retrieve(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """
        BM25-based sparse retrieval

        Args:
            query: Search query
            size: Number of results

        Returns:
            List of documents with scores
        """
        es_query = {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or"
                }
            }
        }

        response = self.es.search(
            index=self.index_name,
            query=es_query,
            size=size,
            _source=["docid", "content", "src"]
        )

        results = []
        for hit in response['hits']['hits']:
            results.append({
                'docid': hit['_source']['docid'],
                'content': hit['_source']['content'],
                'src': hit['_source'].get('src', ''),
                'score': hit['_score'],
                'method': 'sparse'
            })

        return results

    def dense_retrieve(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """
        Vector-based dense retrieval using KNN

        Args:
            query: Search query
            size: Number of results

        Returns:
            List of documents with scores
        """
        # Encode query (returns 2D array [1, 768], we need 1D [768])
        query_embedding = self.embedder.encode(query)

        # Ensure 1D vector for Elasticsearch
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]

        # KNN search
        knn = {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": size,
            "num_candidates": min(size * 10, 1000)  # Adaptive num_candidates
        }

        response = self.es.search(
            index=self.index_name,
            knn=knn,
            _source=["docid", "content", "src"],
            size=size
        )

        results = []
        for hit in response['hits']['hits']:
            results.append({
                'docid': hit['_source']['docid'],
                'content': hit['_source']['content'],
                'src': hit['_source'].get('src', ''),
                'score': hit['_score'],
                'method': 'dense'
            })

        return results

    def hybrid_retrieve(
        self,
        query: str,
        size: int = 10,
        sparse_weight: float = cfg.SPARSE_WEIGHT,
        dense_weight: float = cfg.DENSE_WEIGHT
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining sparse and dense methods using RRF

        Args:
            query: Search query
            size: Final number of results
            sparse_weight: Weight for BM25 scores
            dense_weight: Weight for vector scores

        Returns:
            List of documents with combined scores
        """
        # Retrieve more candidates for better fusion
        retrieve_size = size * 2

        # Get results from both methods
        sparse_results = self.sparse_retrieve(query, retrieve_size)
        dense_results = self.dense_retrieve(query, retrieve_size)

        # Combine using Reciprocal Rank Fusion (RRF)
        combined_scores = {}

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            docid = result['docid']
            rrf_score = sparse_weight / (60 + rank)  # k=60 is common
            if docid not in combined_scores:
                combined_scores[docid] = {
                    'docid': docid,
                    'content': result['content'],
                    'src': result['src'],
                    'score': 0,
                    'sparse_rank': rank,
                    'dense_rank': None
                }
            combined_scores[docid]['score'] += rrf_score

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            docid = result['docid']
            rrf_score = dense_weight / (60 + rank)
            if docid not in combined_scores:
                combined_scores[docid] = {
                    'docid': docid,
                    'content': result['content'],
                    'src': result['src'],
                    'score': 0,
                    'sparse_rank': None,
                    'dense_rank': rank
                }
            else:
                combined_scores[docid]['dense_rank'] = rank
            combined_scores[docid]['score'] += rrf_score

        # Sort by combined score
        final_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:size]

        logger.debug(f"Hybrid search: {len(sparse_results)} sparse, {len(dense_results)} dense -> {len(final_results)} combined")

        return final_results

    def get_document_by_id(self, docid: str) -> Dict[str, Any]:
        """Get document by docid"""
        query = {
            "term": {
                "docid": docid
            }
        }

        response = self.es.search(
            index=self.index_name,
            query=query,
            size=1
        )

        if response['hits']['total']['value'] > 0:
            hit = response['hits']['hits'][0]
            return {
                'docid': hit['_source']['docid'],
                'content': hit['_source']['content'],
                'src': hit['_source'].get('src', '')
            }
        return None

    def count_documents(self) -> int:
        """Count total documents in index"""
        return self.es.count(index=self.index_name)['count']
