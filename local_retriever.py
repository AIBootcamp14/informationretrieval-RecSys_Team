"""
Local file-based hybrid retrieval (BM25 + BGE-M3) without Elasticsearch.
Maintains BM25/dense scores and token overlaps for downstream evidence gating.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

import numpy as np
from loguru import logger
from whoosh import scoring
from whoosh.analysis import Tokenizer, Token, LowercaseFilter, Filter
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import MultifieldParser, OrGroup
from transformers import AutoTokenizer

from src.embedder import get_embedder
import config.config as cfg


class BGETokenizer(Tokenizer):
    """
    Custom Whoosh tokenizer using BGE-M3's tokenizer.
    This ensures BM25 and dense retrieval use the same tokenization.
    """

    def __init__(self, tokenizer: Optional[AutoTokenizer] = None, model_name: str = cfg.EMBEDDING_MODEL):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized BGE tokenizer from {self.tokenizer.name_or_path}")

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, mode='', **kwargs):
        """Tokenize text using BGE-M3 tokenizer."""
        tokens = self.tokenizer.tokenize(value)

        pos = start_pos
        for token_text in tokens:
            if token_text.startswith('[') and token_text.endswith(']'):
                continue
            if token_text in ('', ' ', '\n', '\t'):
                continue

            # Normalize token by removing subword prefixes
            normalized = self._normalize_token(token_text)
            if not normalized:
                continue

            t = Token()
            t.text = normalized.lower()
            t.pos = pos
            t.stopped = False

            yield t
            pos += 1

    def _normalize_token(self, token_text: str) -> str:
        """
        Normalize token by removing subword prefixes (▁, ##, Ġ).
        This helps with compound noun matching.
        """
        # Remove common subword prefixes
        if token_text.startswith('▁'):
            token_text = token_text[1:]
        elif token_text.startswith('##'):
            token_text = token_text[2:]
        elif token_text.startswith('Ġ'):
            token_text = token_text[1:]

        return token_text.strip()


class CharNGramFilter(Filter):
    """
    NEW: Character n-gram filter for handling compound nouns.

    This filter generates character-level n-grams (2-3 characters) to help
    match compound nouns with/without spaces. For example:
    - "평형분극비율" → ["평형", "형분", "분극", "극비", "비율", ...]
    - "평형 분극 비율" → ["평형", "분극", "비율", ...]

    This allows partial matching even when spacing differs.
    """

    def __init__(self, minsize=2, maxsize=3):
        self.minsize = minsize
        self.maxsize = maxsize

    def __call__(self, tokens):
        """Generate character n-grams for each token."""
        for t in tokens:
            # First yield the original token
            yield t

            # Then yield character n-grams
            text = t.text
            if len(text) >= self.minsize:
                for n in range(self.minsize, min(self.maxsize + 1, len(text) + 1)):
                    for i in range(len(text) - n + 1):
                        ngram_token = Token()
                        ngram_token.text = text[i:i + n]
                        ngram_token.pos = t.pos
                        ngram_token.stopped = False
                        yield ngram_token


def _build_analyzer(tokenizer: AutoTokenizer, use_chargram: bool = False):
    """
    Build analyzer with BGE tokenizer and optional character n-grams.

    Args:
        tokenizer: BGE-M3 tokenizer
        use_chargram: If True, add character n-gram filter for compound noun handling
    """
    analyzer = BGETokenizer(tokenizer=tokenizer) | LowercaseFilter()

    if use_chargram:
        # Add character n-gram filter (2-3 chars)
        analyzer = analyzer | CharNGramFilter(minsize=2, maxsize=3)

    return analyzer


class LocalHybridRetriever:
    """
    Local file-based hybrid retriever combining BM25 and BGE-M3.
    Keeps metadata (bm25, dense, overlap terms) for evidence gating.
    """

    def __init__(self):
        logger.info("Initializing Local Hybrid Retriever...")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)

        # BM25 index
        self.bm25_index = None
        self.chunks: List[Dict] = []
        self.chunk_docids: List[str] = []
        self.chunk_ids: List[str] = []
        self.chunkid_to_text: Dict[str, str] = {}

        # Token caches
        self.doc_tokens: Dict[str, Set[str]] = {}
        self.domain_vocab: Set[str] = set()

        # BGE-M3 embedder
        self.embedder = get_embedder()
        self.embeddings: Optional[np.ndarray] = None
        self.docid_to_indices: Dict[str, List[int]] = {}

        logger.info("Local Hybrid Retriever initialized")

    # -------------------------
    # Index building
    # -------------------------
    def build_bm25_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks with optional character n-gram support.

        If USE_CHAR_NGRAM is enabled, this adds a 'chargram' field to help
        match compound nouns with spacing variations.
        """
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        # Build analyzers
        standard_analyzer = _build_analyzer(self.tokenizer, use_chargram=False)

        # Schema fields
        schema_fields = {
            "chunk_id": ID(stored=True, unique=True),
            "docid": ID(stored=True),
            "title": TEXT(stored=True, analyzer=standard_analyzer, field_boost=1.3),
            "section": TEXT(stored=True, analyzer=standard_analyzer, field_boost=1.15),
            "content": TEXT(stored=True, analyzer=standard_analyzer),
        }

        # Add character n-gram field if enabled
        if cfg.USE_CHAR_NGRAM:
            chargram_analyzer = _build_analyzer(self.tokenizer, use_chargram=True)
            schema_fields["chargram"] = TEXT(stored=False, analyzer=chargram_analyzer, field_boost=0.5)
            logger.info("Character n-gram field enabled for compound noun handling")

        schema = Schema(**schema_fields)

        storage = RamStorage()
        idx = storage.create_index(schema)
        writer = idx.writer(limitmb=512)

        for chunk in chunks:
            doc_data = {
                "chunk_id": chunk["chunk_id"],
                "docid": chunk["docid"],
                "title": chunk.get("title") or "",
                "section": chunk.get("section") or "",
                "content": chunk.get("text") or "",
            }

            # Add character n-gram field
            if cfg.USE_CHAR_NGRAM:
                # Combine all text for n-gram indexing
                chargram_text = " ".join([
                    chunk.get("title") or "",
                    chunk.get("section") or "",
                    chunk.get("text") or ""
                ]).strip()
                doc_data["chargram"] = chargram_text

            writer.add_document(**doc_data)

        writer.commit()
        self.bm25_index = idx
        logger.info("BM25 index built successfully")

    def _tokenize_for_overlap(self, text: str) -> List[str]:
        """Tokenize text for overlap checks using the same tokenizer as BM25."""
        tokens = self.tokenizer.tokenize(text or "")
        return [tok.lower() for tok in tokens if tok and not tok.startswith('[')]

    def index_chunks(self, chunks: List[Dict]):
        """Index chunks for both BM25 and dense retrieval."""
        self.chunks = chunks
        self.chunkid_to_text = {c["chunk_id"]: c.get("text", "") for c in chunks}

        # Build token caches for overlap and vocab
        token_map: Dict[str, Set[str]] = defaultdict(set)
        for chunk in chunks:
            docid = chunk["docid"]
            for tok in self._tokenize_for_overlap(chunk.get("text", "")):
                token_map[docid].add(tok)
        self.doc_tokens = {k: set(v) for k, v in token_map.items()}
        self.domain_vocab = set().union(*self.doc_tokens.values()) if self.doc_tokens else set()
        logger.info(f"Domain vocab collected: {len(self.domain_vocab)} unique tokens")

        # Build BM25 index
        self.build_bm25_index(chunks)

        # Build dense index (BGE-M3)
        logger.info(f"Building dense index for {len(chunks)} chunks...")

        texts = [c["text"] for c in chunks]
        self.chunk_docids = [c["docid"] for c in chunks]
        self.chunk_ids = [c["chunk_id"] for c in chunks]

        raw_embeddings = self.embedder.encode_batch(texts, show_progress=True)
        self.embeddings = self._normalize(raw_embeddings)

        # Build docid mapping
        self.docid_to_indices.clear()
        for idx, docid in enumerate(self.chunk_docids):
            self.docid_to_indices.setdefault(docid, []).append(idx)

        logger.info("Dense index built successfully")

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    # -------------------------
    # Retrieval primitives
    # -------------------------
    def bm25_search(self, query: str, topk: int = 50) -> List[Dict]:
        """
        BM25-based sparse retrieval with optional character n-gram support.

        If USE_CHAR_NGRAM is enabled, this queries both standard fields
        (title, section, content) and the character n-gram field (chargram).
        """
        if not self.bm25_index:
            logger.warning("BM25 index not built")
            return []

        # Build field list and boosts
        fields = ["title", "section", "content"]
        fieldboosts = {"title": 1.3, "section": 1.15, "content": 1.0}

        # Add chargram field if enabled
        if cfg.USE_CHAR_NGRAM:
            fields.append("chargram")
            fieldboosts["chargram"] = 0.5  # Lower boost for n-grams

        parser = MultifieldParser(
            fields,
            schema=self.bm25_index.schema,
            group=OrGroup,
            fieldboosts=fieldboosts,
        )
        parsed_query = parser.parse(query)

        results: List[Dict] = []
        with self.bm25_index.searcher(weighting=scoring.BM25F(B=0.3, K1=1.6)) as searcher:
            for hit in searcher.search(parsed_query, limit=topk):
                results.append(
                    {
                        "docid": hit["docid"],
                        "chunk_id": hit["chunk_id"],
                        "score": float(hit.score),
                    }
                )

        return results

    def dense_search(self, query: str, topk: int = 50) -> List[Dict]:
        """Dense retrieval using BGE-M3."""
        if self.embeddings is None or not len(self.embeddings):
            logger.warning("Dense index not built")
            return []

        q_emb = self.embedder.encode(query)
        if len(q_emb.shape) > 1:
            q_emb = q_emb[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(-scores)[:topk]

        results: List[Dict] = []
        for idx in top_indices:
            results.append(
                {
                    "docid": self.chunk_docids[idx],
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(scores[idx]),
                }
            )

        return results

    # -------------------------
    # Hybrid retrieval with metadata
    # -------------------------
    def _init_candidate(self, docid: str) -> Dict:
        return {
            "docid": docid,
            "text": self._get_content_by_docid(docid),
            "bm25": 0.0,
            "dense": -1.0,
            "overlap": 0,
            "overlap_terms": [],
            "doc_tokens": self.doc_tokens.get(docid, set()),
            "score": 0.0,
            "best_chunk_id": None,
            "source_queries": set(),
        }

    def _update_overlap(self, candidate: Dict, query_tokens: Set[str]):
        doc_tokens = candidate.get("doc_tokens", set())
        matched = sorted(list(doc_tokens & query_tokens))
        if len(matched) > candidate.get("overlap", 0):
            candidate["overlap"] = len(matched)
            candidate["overlap_terms"] = matched

    def hybrid_retrieve_chunks(
        self,
        queries: Union[str, List[str]],
        n_sparse: int = 100,
        n_dense: int = 100,
        n_pool: int = 200,
    ) -> List[Dict]:
        """
        NEW: Hybrid retrieval that maintains chunk-level candidates.

        This method does NOT merge by docid early. Instead, it returns individual
        chunks from both BM25 and dense retrieval, taking the union and limiting
        to n_pool total chunks.

        Args:
            queries: Search query or list of queries
            n_sparse: Number of BM25 chunks to retrieve per query
            n_dense: Number of dense chunks to retrieve per query
            n_pool: Maximum number of unique chunks to return (after union)

        Returns:
            List of chunk candidates with fields:
                - docid: Parent document ID
                - chunk_id: Unique chunk identifier
                - chunk_text: Text content of the chunk
                - bm25: BM25 score (0.0 if not from BM25 retrieval)
                - dense: Dense similarity score (-1.0 if not from dense retrieval)
                - overlap: Number of overlapping tokens with query
                - overlap_terms: List of overlapping tokens
        """
        query_list = [q for q in (queries if isinstance(queries, list) else [queries]) if q]
        if not query_list:
            return []

        # Collect chunk candidates (keyed by chunk_id to avoid duplicates)
        chunk_candidates: Dict[str, Dict] = {}

        for query in query_list:
            sparse_results = self.bm25_search(query, n_sparse)
            dense_results = self.dense_search(query, n_dense)
            query_tokens = set(self._tokenize_for_overlap(query))

            # Process sparse results (chunk-level)
            for result in sparse_results:
                chunk_id = result["chunk_id"]
                docid = result["docid"]

                if chunk_id not in chunk_candidates:
                    chunk_candidates[chunk_id] = {
                        "docid": docid,
                        "chunk_id": chunk_id,
                        "chunk_text": self._get_chunk_text(chunk_id),
                        "bm25": result["score"],
                        "dense": -1.0,
                        "overlap": 0,
                        "overlap_terms": [],
                    }
                else:
                    # Update if this query gave a better BM25 score
                    if result["score"] > chunk_candidates[chunk_id]["bm25"]:
                        chunk_candidates[chunk_id]["bm25"] = result["score"]

                # Update overlap
                chunk_text = chunk_candidates[chunk_id]["chunk_text"]
                chunk_tokens = set(self._tokenize_for_overlap(chunk_text))
                matched = sorted(list(chunk_tokens & query_tokens))
                if len(matched) > chunk_candidates[chunk_id]["overlap"]:
                    chunk_candidates[chunk_id]["overlap"] = len(matched)
                    chunk_candidates[chunk_id]["overlap_terms"] = matched

            # Process dense results (chunk-level)
            for result in dense_results:
                chunk_id = result["chunk_id"]
                docid = result["docid"]

                if chunk_id not in chunk_candidates:
                    chunk_candidates[chunk_id] = {
                        "docid": docid,
                        "chunk_id": chunk_id,
                        "chunk_text": self._get_chunk_text(chunk_id),
                        "bm25": 0.0,
                        "dense": result["score"],
                        "overlap": 0,
                        "overlap_terms": [],
                    }
                else:
                    # Update if this query gave a better dense score
                    if result["score"] > chunk_candidates[chunk_id]["dense"]:
                        chunk_candidates[chunk_id]["dense"] = result["score"]

                # Update overlap
                chunk_text = chunk_candidates[chunk_id]["chunk_text"]
                chunk_tokens = set(self._tokenize_for_overlap(chunk_text))
                matched = sorted(list(chunk_tokens & query_tokens))
                if len(matched) > chunk_candidates[chunk_id]["overlap"]:
                    chunk_candidates[chunk_id]["overlap"] = len(matched)
                    chunk_candidates[chunk_id]["overlap_terms"] = matched

        # Convert to list and limit to n_pool
        chunk_list = list(chunk_candidates.values())

        # Sort by best available score (prioritize chunks with both scores)
        def score_chunk(chunk):
            has_bm25 = chunk["bm25"] > 0
            has_dense = chunk["dense"] > -0.5
            # Prioritize chunks with both scores, then by individual scores
            return (
                int(has_bm25 and has_dense),  # Both scores
                chunk["bm25"],
                chunk["dense"]
            )

        chunk_list.sort(key=score_chunk, reverse=True)
        chunk_list = chunk_list[:n_pool]

        logger.debug(
            f"Chunk-level retrieval: {len(chunk_list)} chunks from {len(query_list)} queries "
            f"(n_sparse={n_sparse}, n_dense={n_dense}, limited to n_pool={n_pool})"
        )

        return chunk_list

    def hybrid_retrieve(
        self,
        queries: Union[str, List[str]],
        size: int = 30,
        sparse_weight: float = 0.7,
        dense_weight: float = 0.3,
    ) -> List[Dict]:
        """
        Hybrid retrieval combining BM25 and BGE-M3 using RRF across multiple queries.

        NOTE: This method merges by docid early (legacy behavior).
        For chunk-level retrieval, use hybrid_retrieve_chunks() instead.

        Args:
            queries: Search query or list of queries
            size: Number of final results
            sparse_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores

        Returns:
            List of documents with combined scores and metadata
        """
        query_list = [q for q in (queries if isinstance(queries, list) else [queries]) if q]
        if not query_list:
            return []

        retrieve_size = size * 2
        combined_scores: Dict[str, Dict] = {}

        for query in query_list:
            sparse_results = self.bm25_search(query, retrieve_size)
            dense_results = self.dense_search(query, retrieve_size)
            query_tokens = set(self._tokenize_for_overlap(query))

            # Process sparse results
            for rank, result in enumerate(sparse_results, 1):
                docid = result["docid"]
                rrf_score = sparse_weight / (60 + rank)
                cand = combined_scores.setdefault(docid, self._init_candidate(docid))
                cand["score"] += rrf_score
                cand["source_queries"].add(query)
                if result["score"] > cand["bm25"]:
                    cand["bm25"] = result["score"]
                    cand["best_chunk_id"] = result["chunk_id"]
                    cand["text"] = self._get_chunk_text(result["chunk_id"]) or cand["text"]
                self._update_overlap(cand, query_tokens)

            # Process dense results
            for rank, result in enumerate(dense_results, 1):
                docid = result["docid"]
                rrf_score = dense_weight / (60 + rank)
                cand = combined_scores.setdefault(docid, self._init_candidate(docid))
                cand["score"] += rrf_score
                cand["source_queries"].add(query)
                if result["score"] > cand["dense"]:
                    cand["dense"] = result["score"]
                    if not cand["text"]:
                        cand["text"] = self._get_chunk_text(result["chunk_id"])
                self._update_overlap(cand, query_tokens)

        # Sort by combined score
        final_results = sorted(
            combined_scores.values(), key=lambda x: x["score"], reverse=True
        )[:size]

        # Convert source_queries sets to lists for serialization friendliness
        for cand in final_results:
            cand["source_queries"] = list(cand.get("source_queries", []))

        logger.debug(
            f"Hybrid search (multi-query): {len(final_results)} combined from {len(query_list)} queries"
        )

        return final_results

    # -------------------------
    # Helpers
    # -------------------------
    def _get_chunk_text(self, chunk_id: str) -> str:
        return self.chunkid_to_text.get(chunk_id, "")

    def _get_content_by_docid(self, docid: str) -> str:
        for chunk in self.chunks:
            if chunk["docid"] == docid:
                return chunk.get("text", "")
        return ""

    def tokenize_for_overlap(self, text: str) -> List[str]:
        """Public helper to reuse the same tokenizer for overlap checks."""
        return self._tokenize_for_overlap(text)

    def get_domain_vocab(self) -> Set[str]:
        """Expose domain vocabulary built from indexed documents."""
        return set(self.domain_vocab)

    def sparse_retrieve(self, query: str, size: int = 10) -> List[Dict]:
        """BM25-only retrieval (for compatibility)."""
        results = self.bm25_search(query, size)

        formatted = []
        for res in results:
            content = self._get_chunk_text(res["chunk_id"]) or self._get_content_by_docid(res["docid"])
            formatted.append(
                {
                    "docid": res["docid"],
                    "content": content,
                    "src": "",
                    "score": res["score"],
                    "method": "sparse",
                }
            )

        return formatted


# Singleton instance
_retriever_instance: Optional[LocalHybridRetriever] = None


def get_local_retriever() -> LocalHybridRetriever:
    """Get singleton local retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LocalHybridRetriever()
    return _retriever_instance
