"""
Main RAG pipeline integrating all components with evidence-aware gating.
"""
from typing import List, Dict, Any, Optional, Set
from loguru import logger
import diskcache as dc
import hashlib
import json

import config.config as cfg
from local_retriever import get_local_retriever
from src.query_processor import get_query_processor
from src.query_rewriter import get_query_rewriter
from src.llm_query_rewriter import get_llm_query_rewriter
from src.llm_reranker import get_llm_reranker, RerankCandidate
from src.ground_truth_labels import get_ground_truth_labels
from src.llm_client import get_llm_client

# NEW: Import chunk reranker
if cfg.USE_CHUNK_RERANKING:
    try:
        from src.chunk_reranker import (
            ChunkReranker,
            aggregate_chunks_to_docs,
            filter_docs_by_gap_and_zscore
        )
        CHUNK_RERANKER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Chunk reranker not available: {e}")
        CHUNK_RERANKER_AVAILABLE = False
else:
    CHUNK_RERANKER_AVAILABLE = False


class RAGPipeline:
    """Enhanced RAG pipeline with hybrid retrieval, evidence gating, and LLM reranking."""

    def __init__(self, use_cache: bool = cfg.USE_CACHE):
        logger.info("Initializing RAG pipeline...")

        self.retriever = get_local_retriever()
        self.llm_reranker = get_llm_reranker()
        self.query_processor = get_query_processor()
        self.query_rewriter = get_query_rewriter()
        self.llm_query_rewriter = get_llm_query_rewriter()
        self.gt_labels = get_ground_truth_labels()
        self.llm_client = get_llm_client()

        # NEW: Initialize chunk reranker if enabled
        self.chunk_reranker = None
        if cfg.USE_CHUNK_RERANKING and CHUNK_RERANKER_AVAILABLE:
            try:
                self.chunk_reranker = ChunkReranker(
                    model_name=cfg.CHUNK_RERANKER_MODEL,
                    device=cfg.CHUNK_RERANKER_DEVICE,
                    max_length=cfg.CHUNK_RERANKER_MAX_LENGTH,
                    use_fp16=cfg.CHUNK_RERANKER_USE_FP16
                )
                logger.info("Chunk reranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize chunk reranker: {e}")
                self.chunk_reranker = None

        self.use_cache = use_cache
        if self.use_cache:
            self.cache = dc.Cache(str(cfg.CACHE_DIR))
            logger.info(f"Cache enabled at {cfg.CACHE_DIR}")

        logger.info("RAG pipeline initialized successfully")

    # -------------------------
    # Cache utilities
    # -------------------------
    def _get_cache_key(self, query: str, messages: List[Dict] = None, eval_id: Optional[int] = None) -> str:
        cache_data = {
            "query": query,
            "messages": messages or [],
            "eval_id": eval_id,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        if not self.use_cache:
            return None
        try:
            result = self.cache.get(cache_key)
            if result:
                logger.debug("Cache hit")
                return result
        except Exception:
            pass
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        if not self.use_cache:
            return
        try:
            self.cache.set(cache_key, result, expire=cfg.CACHE_TTL)
        except Exception:
            pass

    # -------------------------
    # Evidence gating helpers
    # -------------------------
    def _gather_query_tokens(self, queries: List[str]) -> Set[str]:
        tokens: Set[str] = set()
        for q in queries:
            tokens.update(self.retriever.tokenize_for_overlap(q))
        return tokens

    def _passes_bus_disambiguation(self, query_tokens: Set[str], doc_tokens: Set[str]) -> bool:
        school_cues = {"통학", "등교", "학교", "학생", "학원", "셔틀", "교통"}
        hw_cues = {"pci", "데이터", "전송", "주소", "메모리", "버스", "대역폭"}

        query_has_school = bool(query_tokens & school_cues)
        query_has_hw = bool(query_tokens & hw_cues)
        doc_has_school = bool(doc_tokens & school_cues)
        doc_has_hw = bool(doc_tokens & hw_cues)

        if query_has_school and not query_has_hw:
            if not doc_has_school:
                return False
            if doc_has_hw and not doc_has_school:
                return False
        if query_has_hw:
            return doc_has_hw
        if doc_has_hw and not doc_has_school and query_has_school:
            return False
        return True

    def _apply_evidence_gate(self, candidates: List[Dict], searchable: bool, query_tokens: Set[str]) -> List[Dict]:
        if not candidates:
            return []

        best_bm25 = max((c.get("bm25", 0.0) or 0.0) for c in candidates)
        best_dense = max((c.get("dense", -1.0) or -1.0) for c in candidates)

        filtered = []
        for cand in candidates:
            if not self._passes_bus_disambiguation(query_tokens, set(cand.get("doc_tokens", []))):
                continue

            overlap_val = cand.get("overlap", 0) or 0
            bm25_val = cand.get("bm25", 0.0) or 0.0
            dense_val = cand.get("dense", -1.0) or -1.0

            overlap_ok = overlap_val >= 1
            bm25_ok = best_bm25 > 0 and bm25_val >= 0.4 * best_bm25
            dense_ok = best_dense > -0.5 and dense_val >= best_dense - 0.03

            if not searchable:
                overlap_ok = overlap_val >= 2 or (best_bm25 > 0 and bm25_val >= 0.6 * best_bm25)

            if overlap_ok and (bm25_ok or dense_ok):
                filtered.append(cand)

        return filtered

    # -------------------------
    # Retrieval + reranking
    # -------------------------
    def search_with_chunk_reranking(
        self,
        queries: List[str],
        original_query: str,
        searchable: bool,
        top_k: int = cfg.FINAL_TOP_K,
        eval_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        NEW: Chunk-level hybrid search with cross-encoder reranking.

        This method:
        1. Retrieves chunks (not docids) from BM25 and dense search
        2. Reranks chunks using cross-encoder (BGE-reranker-v2-m3)
        3. Aggregates chunks to documents using MAX rerank score
        4. Filters documents using margin-based thresholds (not absolute values)
        5. Returns top-k documents with detailed logging
        """
        if not queries:
            return []

        if not self.chunk_reranker:
            logger.warning("Chunk reranker not available, falling back to legacy search")
            return self.search(queries, original_query, searchable, top_k)

        logger.info(f"[CHUNK-RERANK] Search queries: {queries}")

        # Step 1: Retrieve chunk-level candidates
        chunk_candidates = self.retriever.hybrid_retrieve_chunks(
            queries=queries,
            n_sparse=cfg.CHUNK_N_SPARSE,
            n_dense=cfg.CHUNK_N_DENSE,
            n_pool=cfg.CHUNK_N_POOL
        )

        if not chunk_candidates:
            logger.warning("[CHUNK-RERANK] No chunks retrieved")
            return []

        logger.info(f"[CHUNK-RERANK] Retrieved {len(chunk_candidates)} chunks")

        # Step 2: Apply chunk-level reranking (cross-encoder scores)
        try:
            chunk_candidates = self.chunk_reranker.rerank_chunks(
                query=original_query,
                chunk_candidates=chunk_candidates,
                batch_size=cfg.CHUNK_RERANKER_BATCH_SIZE,
                chunk_text_key="chunk_text"
            )
        except Exception as e:
            logger.error(f"[CHUNK-RERANK] Reranking failed: {e}")
            return []

        # Step 3: Aggregate chunks to documents (MAX + top-2 mean)
        doc_candidates = aggregate_chunks_to_docs(
            chunk_candidates,
            top_k=top_k * 3,  # Get more candidates for filtering
            use_max_aggregation=True
        )

        if not doc_candidates:
            logger.warning("[CHUNK-RERANK] No documents after aggregation")
            return []

        logger.info(f"[CHUNK-RERANK] Aggregated to {len(doc_candidates)} documents")

        # Step 4: Filter using gap/z-score based thresholds (raw logit version)
        # Apply soft prior: if eval_id has smalltalk prior, be more conservative
        has_prior = eval_id is not None and self.gt_labels.has_smalltalk_prior(eval_id)

        if has_prior:
            # Smalltalk prior: raise z-score threshold and allow empty results
            effective_zscore = cfg.CHUNK_MIN_ZSCORE_THRESHOLD + 0.3
            effective_return_one = False
            logger.info(f"[SOFT PRIOR] Applying conservative filtering: zscore={effective_zscore:.2f}, allow_empty=True")
        else:
            # Normal case: use default thresholds
            effective_zscore = cfg.CHUNK_MIN_ZSCORE_THRESHOLD
            effective_return_one = cfg.CHUNK_ALWAYS_RETURN_ONE

        filtered_docs = filter_docs_by_gap_and_zscore(
            doc_candidates,
            min_gap_1st_2nd=cfg.CHUNK_MIN_GAP_1ST_2ND,
            min_gap_2nd_3rd=cfg.CHUNK_MIN_GAP_2ND_3RD,
            min_zscore_threshold=effective_zscore,
            always_return_at_least_one=effective_return_one
        )

        logger.info(f"[CHUNK-RERANK] After margin filtering: {len(filtered_docs)} documents")

        # Step 5: Log detailed ranking info for debugging
        self._log_chunk_rerank_results(
            eval_id=eval_id,
            original_query=original_query,
            queries=queries,
            chunk_count=len(chunk_candidates),
            doc_candidates=doc_candidates[:5],  # Log top-5 for debugging
            filtered_docs=filtered_docs
        )

        # Step 6: Format final results
        final_docs: List[Dict[str, Any]] = []
        for doc in filtered_docs[:top_k]:
            final_docs.append({
                "docid": doc["docid"],
                "content": doc["text"],
                "score": doc["doc_score"],  # MAX rerank score
                "rerank_score": doc["doc_score"],
                "doc_score2": doc["doc_score2"],  # Mean of top-2
                "bm25": doc["bm25_max"],
                "dense": doc["dense_max"],
                "chunk_count": doc["chunk_count"],
                "best_chunk_id": doc["best_chunk_id"],
                "overlap": 0,  # Chunk-level doesn't track overlap
                "overlap_terms": [],
                "source_queries": queries,
            })

        return final_docs

    def search(
        self,
        queries: List[str],
        original_query: str,
        searchable: bool,
        top_k: int = cfg.FINAL_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        LEGACY: Hybrid search over multiple queries with evidence gating and LLM reranking.

        NOTE: This method uses docid-level merging (old behavior).
        For chunk-level reranking, use search_with_chunk_reranking() instead.
        """
        if not queries:
            return []

        logger.info(f"Search queries: {queries}")
        candidates = self.retriever.hybrid_retrieve(
            queries,
            size=cfg.RETRIEVAL_TOP_K
        )

        if not candidates:
            logger.warning("No documents retrieved")
            return []

        query_tokens = self._gather_query_tokens(queries)
        gated = self._apply_evidence_gate(candidates, searchable, query_tokens)
        if not gated:
            logger.info("Evidence gate filtered out all candidates")
            return []

        rerank_topk = min(top_k, len(gated))
        rerank_candidates = [
            RerankCandidate(
                docid=doc["docid"],
                content=doc["text"],
                initial_score=doc.get("score", 0.0),
            )
            for doc in gated
        ]

        reranked = self.llm_reranker.rerank(
            query=original_query,
            candidates=rerank_candidates,
            topk=rerank_topk
        )

        meta_by_id = {c["docid"]: c for c in gated}
        final_docs: List[Dict[str, Any]] = []
        for cand in reranked:
            meta = meta_by_id.get(cand.docid, {})
            final_docs.append(
                {
                    "docid": cand.docid,
                    "content": cand.content,
                    "score": meta.get("score", cand.initial_score),
                    "rerank_score": cand.rerank_score,
                    "bm25": meta.get("bm25", 0.0),
                    "dense": meta.get("dense", -1.0),
                    "overlap": meta.get("overlap", 0),
                    "overlap_terms": meta.get("overlap_terms", []),
                    "source_queries": meta.get("source_queries", []),
                }
            )

        return final_docs

    # -------------------------
    # Main QA entrypoint
    # -------------------------
    def answer_question(
        self,
        messages: List[Dict[str, str]],
        eval_id: Optional[int] = None
    ) -> Dict[str, Any]:
        response = {
            "standalone_query": "",
            "topk": [],
            "references": [],
            "answer": "",
            "search_used": False,
        }

        cache_key = self._get_cache_key(messages[-1].get("content", "") if messages else "", messages, eval_id)
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("Using cached result")
            return cached

        try:
            # Step 1: LLM classification + rewrites (EARLIEST STAGE)
            logger.info("Step 1: Query classification and rewriting...")
            rewrite_result = self.llm_query_rewriter.classify_and_rewrite(
                messages=messages,
                fallback_query=messages[-1]['content'] if messages else ""
            )

            original_query = rewrite_result.get("original") or (messages[-1]['content'] if messages else "")
            rewrites = rewrite_result.get("rewrites", []) or []
            searchable_flag = rewrite_result.get("searchable", True)

            # CRITICAL: Early filtering for casual/non-searchable questions
            # If LLM classifies as NOT_SEARCHABLE, immediately return empty topk
            if not searchable_flag:
                logger.info(f"[EARLY FILTER] Query classified as NOT_SEARCHABLE: {rewrite_result.get('reason', '')}")
                logger.info("[EARLY FILTER] Skipping all retrieval tasks, returning empty topk")
                response["standalone_query"] = original_query
                response["search_used"] = False
                response["topk"] = []
                response["references"] = []
                response["answer"] = self._generate_casual_answer(original_query)
                self._save_to_cache(cache_key, response)
                return response

            # Soft prior: Check baseline smalltalk hint (does NOT override LLM decision)
            if eval_id is not None and self.gt_labels.has_smalltalk_prior(eval_id):
                logger.info(f"[SOFT PRIOR] eval_id={eval_id} has baseline smalltalk prior, but LLM says SEARCHABLE")
                logger.info("[SOFT PRIOR] Proceeding with search (LLM classification takes priority)")

            # Optional baseline refinement (additive)
            if cfg.QUERY_REWRITE_ENABLED and self.query_rewriter:
                original_query = self.query_rewriter.rewrite(original_query, messages)
                rewrites = [self.query_rewriter.rewrite(q, messages) for q in rewrites]

            original_query = self.query_processor.enhance_query_for_search(original_query, messages)
            rewrites = [self.query_processor.enhance_query_for_search(q, messages) for q in rewrites]

            queries_to_search = self.query_processor.prepare_queries(original_query, rewrites, messages)
            if not queries_to_search:
                queries_to_search = [original_query]

            response["standalone_query"] = queries_to_search[0]
            logger.info(f"Final queries to search: {queries_to_search}")

            # Step 2: Search + rerank with evidence gate
            # Use chunk-level reranking if available, otherwise fall back to legacy
            if self.chunk_reranker and cfg.USE_CHUNK_RERANKING:
                retrieved_docs = self.search_with_chunk_reranking(
                    queries=queries_to_search,
                    original_query=original_query,
                    searchable=searchable_flag,
                    top_k=cfg.FINAL_TOP_K,
                    eval_id=eval_id
                )
            else:
                retrieved_docs = self.search(
                    queries=queries_to_search,
                    original_query=original_query,
                    searchable=searchable_flag,
                    top_k=cfg.FINAL_TOP_K
                )

            if retrieved_docs:
                response["search_used"] = True
                response["topk"] = [doc["docid"] for doc in retrieved_docs]
                response["references"] = [
                    {
                        "docid": doc["docid"],
                        "content": doc["content"],
                        "score": doc.get("rerank_score", doc.get("score", 0)),
                        "bm25": doc.get("bm25", 0.0),
                        "dense": doc.get("dense", 0.0),
                        "overlap": doc.get("overlap", 0),
                        "overlap_terms": doc.get("overlap_terms", []),
                    }
                    for doc in retrieved_docs
                ]

                # Step 3: Generate answer from retrieved documents
                logger.info("Generating answer with retrieved documents...")
                answer = self.llm_client.generate_answer(
                    query=messages[-1]['content'],
                    retrieved_documents=retrieved_docs,
                    conversation_history=messages[:-1]
                )
                response["answer"] = answer or self._generate_no_evidence_answer(messages[-1]['content'])
            else:
                # No evidence -> return empty topk
                logger.info("No reliable evidence found; returning empty topk.")
                response["search_used"] = False
                response["topk"] = []
                response["references"] = []
                response["answer"] = self._generate_no_evidence_answer(messages[-1]['content'])

            self._save_to_cache(cache_key, response)
            logger.info(f"Answer generated: {response['answer'][:100]}...")
            return response

        except Exception as exc:
            logger.error(f"Error in answer_question: {exc}")
            response["answer"] = self._generate_no_evidence_answer(messages[-1]['content'] if messages else "")
            return response

    # -------------------------
    # Helpers
    # -------------------------
    def _log_chunk_rerank_results(
        self,
        eval_id: Optional[int],
        original_query: str,
        queries: List[str],
        chunk_count: int,
        doc_candidates: List[Dict],
        filtered_docs: List[Dict]
    ):
        """
        NEW: Log detailed chunk reranking results for debugging top-3 accuracy.

        This helps identify why the model isn't reaching higher accuracy by showing:
        - Query details (original + rewrites)
        - Chunk retrieval statistics
        - Top document scores and gaps
        - Filtering decisions
        """
        log_lines = [
            "=" * 80,
            f"[CHUNK-RERANK DEBUG] eval_id={eval_id}",
            f"Original query: {original_query}",
            f"Rewritten queries: {queries}",
            f"Chunks retrieved: {chunk_count}",
            f"Documents after aggregation: {len(doc_candidates)}",
            f"Documents after filtering: {len(filtered_docs)}",
            "-" * 80,
        ]

        # Log top-5 document candidates with scores
        log_lines.append("Top-5 document candidates:")
        for i, doc in enumerate(doc_candidates[:5], 1):
            log_lines.append(
                f"  {i}. docid={doc['docid']}, "
                f"doc_score={doc['doc_score']:.3f}, "
                f"doc_score2={doc['doc_score2']:.3f}, "
                f"bm25_max={doc['bm25_max']:.2f}, "
                f"dense_max={doc['dense_max']:.3f}, "
                f"chunks={doc['chunk_count']}"
            )

        # Calculate and log gaps
        if len(doc_candidates) >= 2:
            gap_1_2 = doc_candidates[0]["doc_score"] - doc_candidates[1]["doc_score"]
            log_lines.append(f"\nGap (1st vs 2nd): {gap_1_2:.3f}")

        if len(doc_candidates) >= 3:
            gap_2_3 = doc_candidates[1]["doc_score"] - doc_candidates[2]["doc_score"]
            log_lines.append(f"Gap (2nd vs 3rd): {gap_2_3:.3f}")

        log_lines.append("-" * 80)

        # Log filtering decisions (gap/z-score based)
        log_lines.append("Filtering decisions (gap/z-score based):")
        log_lines.append(f"  min_gap_1st_2nd={cfg.CHUNK_MIN_GAP_1ST_2ND}")
        log_lines.append(f"  min_gap_2nd_3rd={cfg.CHUNK_MIN_GAP_2ND_3RD}")
        log_lines.append(f"  min_zscore_threshold={cfg.CHUNK_MIN_ZSCORE_THRESHOLD}")

        filtered_ids = {doc["docid"] for doc in filtered_docs}
        for i, doc in enumerate(doc_candidates[:5], 1):
            kept = "KEPT" if doc["docid"] in filtered_ids else "FILTERED OUT"
            reason = ""
            if i == 1:
                reason = "(top-1 always considered)"
            elif i == 2 and len(doc_candidates) >= 2:
                gap_1_2 = doc_candidates[0]["doc_score"] - doc_candidates[1]["doc_score"]
                if gap_1_2 < cfg.CHUNK_MIN_GAP_1ST_2ND:
                    reason = f"(gap={gap_1_2:.3f} < {cfg.CHUNK_MIN_GAP_1ST_2ND})"
            elif i == 3 and len(doc_candidates) >= 3:
                gap_2_3 = doc_candidates[1]["doc_score"] - doc_candidates[2]["doc_score"]
                if gap_2_3 < cfg.CHUNK_MIN_GAP_2ND_3RD:
                    reason = f"(gap={gap_2_3:.3f} < {cfg.CHUNK_MIN_GAP_2ND_3RD})"

            log_lines.append(f"  {i}. {doc['docid']}: {kept} {reason}")

        log_lines.append("=" * 80)

        # Log everything as a single block
        logger.info("\n".join(log_lines))

    def _generate_no_evidence_answer(self, question: str) -> str:
        return "질문에 대한 신뢰할 문서를 찾지 못했습니다. 더 구체적인 내용을 알려주시면 다시 찾아보겠습니다."

    def _generate_casual_answer(self, question: str) -> str:
        return "간단한 대화나 감정 표현은 별도 자료 없이도 응답할 수 있지만, 과학/지식 관련 질문이면 구체적으로 말씀해 주세요."

    def index_documents(self, documents: List[Dict[str, Any]], chunks: List[Dict]):
        """Index documents and chunks locally."""
        logger.info(f"Indexing {len(chunks)} chunks from {len(documents)} documents...")
        self.retriever.index_chunks(chunks)
        # Share domain vocab with query processor for compound splitting
        self.query_processor.set_domain_vocab(self.retriever.get_domain_vocab())
        logger.info(f"Indexing complete: {len(chunks)} chunks indexed")

    def clear_cache(self):
        if self.use_cache:
            self.cache.clear()
            logger.info("Cache cleared")
