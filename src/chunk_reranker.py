"""
Cross-Encoder Chunk Reranker using BGE Reranker v2-m3

This module provides chunk-level reranking using a cross-encoder model that directly
scores query-passage pairs with high accuracy. By default, the model outputs raw logits
for better relative ranking. Sigmoid transformation is available but not recommended
as it inflates scores to 0.7-0.9 range, making threshold-based filtering ineffective.

Installation Requirements:
    pip install torch transformers numpy

    Optional for GPU acceleration:
    pip install torch --index-url https://download.pytorch.org/whl/cu118

Recommended Models:
    - BAAI/bge-reranker-v2-m3 (multilingual, high quality)
    - dragonkue/bge-reranker-v2-m3-ko (Korean-optimized)

Usage:
    reranker = ChunkReranker(model_name="BAAI/bge-reranker-v2-m3", device="cuda")
    scores = reranker.score_pairs(query="질문", passages=["청크1", "청크2"], return_raw_scores=True)
    # Returns: [2.35, -0.82] (raw logits, better for gap-based filtering)
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers or torch not available. Please install: "
        "pip install torch transformers"
    )


# Model constants
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
ALTERNATIVE_MODEL = "dragonkue/bge-reranker-v2-m3-ko"

# Default parameters
DEFAULT_MAX_LENGTH = 512  # Maximum token length for truncation
DEFAULT_BATCH_SIZE = 32   # Batch size for inference


class ChunkReranker:
    """
    Cross-encoder reranker for chunk-level relevance scoring.

    This reranker uses a transformer model to directly score the relevance
    between a query and passage pairs. Unlike bi-encoders that encode
    query and passage separately, cross-encoders process them together,
    resulting in more accurate relevance scores.

    Attributes:
        model_name: Name of the reranker model to use
        device: Device to run inference on ('cuda' or 'cpu')
        max_length: Maximum sequence length for tokenization
        tokenizer: HuggingFace tokenizer instance
        model: HuggingFace model instance
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_fp16: bool = True
    ):
        """
        Initialize the chunk reranker.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum token length for truncation
            use_fp16: Use float16 for GPU inference (faster, less memory)

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If model loading fails
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for ChunkReranker. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine dtype based on device and use_fp16
        if self.device == "cuda" and use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"Loading reranker model: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("Reranker model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise RuntimeError(f"Could not load model {model_name}: {e}")

    def score_pairs(
        self,
        query: str,
        passages: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        return_raw_scores: bool = False
    ) -> List[float]:
        """
        Score query-passage pairs and return relevance scores.

        This method tokenizes query-passage pairs, runs inference in batches,
        and returns normalized relevance scores between 0 and 1.

        Args:
            query: The query string
            passages: List of passage/chunk strings to score
            batch_size: Number of pairs to process in each batch
            return_raw_scores: If True, return raw logits instead of sigmoid scores

        Returns:
            List of relevance scores (0 to 1) for each passage, in the same order

        Example:
            >>> reranker = ChunkReranker()
            >>> scores = reranker.score_pairs(
            ...     query="What is RAM?",
            ...     passages=["RAM is memory", "CPU is processor", "Disk is storage"]
            ... )
            >>> # scores might be: [0.92, 0.15, 0.08]
        """
        if not passages:
            return []

        # Prepare pairs: [(query, passage1), (query, passage2), ...]
        pairs = [(query, passage) for passage in passages]

        all_scores = []

        # Process in batches for efficiency
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = self._score_batch(batch_pairs, return_raw_scores)
            all_scores.extend(batch_scores)

        return all_scores

    def _score_batch(
        self,
        pairs: List[Tuple[str, str]],
        return_raw_scores: bool = False
    ) -> List[float]:
        """
        Score a batch of query-passage pairs.

        Args:
            pairs: List of (query, passage) tuples
            return_raw_scores: If True, return raw logits

        Returns:
            List of relevance scores for the batch
        """
        # Tokenize the batch
        # pairs is [(q1, p1), (q2, p2), ...] which needs to be unpacked
        queries = [pair[0] for pair in pairs]
        passages = [pair[1] for pair in pairs]

        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference without gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (batch_size, num_labels)

            # Extract scores (typically the first column for binary classification)
            if logits.shape[-1] == 1:
                # Single output logit
                raw_scores = logits.squeeze(-1)
            else:
                # Multiple logits, take the first one
                raw_scores = logits[:, 0]

            if return_raw_scores:
                scores = raw_scores.cpu().float().numpy().tolist()
            else:
                # Apply sigmoid to get [0, 1] range
                scores = torch.sigmoid(raw_scores).cpu().float().numpy().tolist()

        return scores

    def rerank_chunks(
        self,
        query: str,
        chunk_candidates: List[dict],
        batch_size: int = DEFAULT_BATCH_SIZE,
        chunk_text_key: str = "chunk_text",
        use_raw_scores: bool = True
    ) -> List[dict]:
        """
        Rerank chunk candidates and add rerank scores.

        This is a convenience method that takes chunk candidate dictionaries,
        extracts their text, scores them, and adds the rerank score back to
        each candidate.

        Args:
            query: The query string
            chunk_candidates: List of chunk candidate dicts with text field
            batch_size: Batch size for inference
            chunk_text_key: Key name for the chunk text in candidate dicts
            use_raw_scores: If True, use raw logits (better for relative ranking)

        Returns:
            The same chunk_candidates list with 'rerank_score' added to each

        Example:
            >>> chunks = [
            ...     {"chunk_id": "doc1_0", "chunk_text": "RAM is memory", "bm25": 5.2},
            ...     {"chunk_id": "doc1_1", "chunk_text": "CPU is processor", "bm25": 3.1}
            ... ]
            >>> reranked = reranker.rerank_chunks("What is RAM?", chunks)
            >>> # Each chunk now has 'rerank_score' field (raw logit)
        """
        if not chunk_candidates:
            return chunk_candidates

        # Extract passages
        passages = [chunk.get(chunk_text_key, "") for chunk in chunk_candidates]

        # Score all passages with raw logits (no sigmoid)
        scores = self.score_pairs(query, passages, batch_size, return_raw_scores=use_raw_scores)

        # Add scores back to candidates
        for chunk, score in zip(chunk_candidates, scores):
            chunk["rerank_score"] = score

        return chunk_candidates


def aggregate_chunks_to_docs(
    chunk_candidates: List[dict],
    top_k: int = 3,
    use_max_aggregation: bool = True
) -> List[dict]:
    """
    Aggregate chunk-level candidates to document-level results.

    This function groups chunks by their parent docid and aggregates scores
    to produce final document rankings. The aggregation strategy prioritizes
    documents that contain at least one highly relevant chunk.

    Aggregation Strategy:
        1. Primary score (doc_score): MAX rerank score among all chunks
        2. Secondary score (doc_score2): MEAN of top-2 rerank scores
        3. Tertiary scores: MAX bm25 and MAX dense scores

    Sorting Priority:
        1. doc_score (descending)
        2. doc_score2 (descending)
        3. bm25_max (descending)
        4. dense_max (descending)

    Args:
        chunk_candidates: List of chunk dicts with 'docid', 'rerank_score', etc.
        top_k: Number of top documents to return
        use_max_aggregation: If True, use MAX aggregation; else use MEAN

    Returns:
        List of aggregated document candidates sorted by relevance

    Example:
        >>> chunks = [
        ...     {"docid": "doc1", "chunk_id": "doc1_0", "rerank_score": 0.9, "bm25": 5.0},
        ...     {"docid": "doc1", "chunk_id": "doc1_1", "rerank_score": 0.3, "bm25": 2.0},
        ...     {"docid": "doc2", "chunk_id": "doc2_0", "rerank_score": 0.7, "bm25": 4.0}
        ... ]
        >>> docs = aggregate_chunks_to_docs(chunks, top_k=2)
        >>> # Returns doc1 (max=0.9) and doc2 (max=0.7)
    """
    from collections import defaultdict

    # Group chunks by docid
    doc_chunks = defaultdict(list)
    for chunk in chunk_candidates:
        docid = chunk.get("docid")
        if docid:
            doc_chunks[docid].append(chunk)

    # Aggregate for each document
    doc_results = []
    for docid, chunks in doc_chunks.items():
        # Sort chunks by rerank score
        chunks_sorted = sorted(
            chunks,
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True
        )

        # Extract scores
        rerank_scores = [c.get("rerank_score", 0.0) for c in chunks_sorted]
        bm25_scores = [c.get("bm25", 0.0) for c in chunks]
        dense_scores = [c.get("dense", -1.0) for c in chunks]

        # Primary: MAX rerank score (best chunk wins)
        doc_score = max(rerank_scores) if rerank_scores else 0.0

        # Secondary: MEAN of top-2 rerank scores (for tie-breaking)
        top2_scores = rerank_scores[:2] if len(rerank_scores) >= 2 else rerank_scores
        doc_score2 = np.mean(top2_scores) if top2_scores else 0.0

        # Tertiary: MAX bm25 and dense
        bm25_max = max(bm25_scores) if bm25_scores else 0.0
        dense_max = max(dense_scores) if dense_scores else -1.0

        # Get the best chunk for text representation
        best_chunk = chunks_sorted[0] if chunks_sorted else {}

        # Create aggregated document candidate
        doc_candidate = {
            "docid": docid,
            "doc_score": doc_score,           # MAX rerank
            "doc_score2": doc_score2,         # MEAN top-2 rerank
            "bm25_max": bm25_max,
            "dense_max": dense_max,
            "chunk_count": len(chunks),
            "best_chunk_id": best_chunk.get("chunk_id", ""),
            "text": best_chunk.get("chunk_text", best_chunk.get("text", "")),
            "top_chunks": chunks_sorted[:3],  # Keep top-3 chunks for reference
        }

        doc_results.append(doc_candidate)

    # Sort documents by aggregated scores
    doc_results.sort(
        key=lambda x: (
            x["doc_score"],      # Primary: max rerank score
            x["doc_score2"],     # Secondary: mean top-2 rerank
            x["bm25_max"],       # Tertiary: max bm25
            x["dense_max"]       # Quaternary: max dense
        ),
        reverse=True
    )

    # Return top-k documents
    return doc_results[:top_k]


def filter_docs_by_gap_and_zscore(
    doc_candidates: List[dict],
    min_gap_1st_2nd: float = 0.5,
    min_gap_2nd_3rd: float = 0.3,
    min_zscore_threshold: float = -0.5,
    always_return_at_least_one: bool = True
) -> List[dict]:
    """
    Filter documents using gap-based and z-score thresholds (raw logit version).

    Unlike absolute thresholds (e.g., 0.8), this approach uses:
    1. Gap between top-1 and top-2 (significance of winner)
    2. Gap between top-2 and top-3 (significance of runner-up)
    3. Z-score of top-1 relative to all candidates (statistical significance)

    This prevents the "sigmoid inflated scores" problem where everything is 0.7-0.9.

    Filtering Rules:
        1. If top-1 z-score < threshold AND always_return_at_least_one=False:
           return empty (no confident match)
        2. If gap(1st-2nd) < min_gap, keep only top-1 (others too close)
        3. If gap(2nd-3rd) < min_gap, keep top-1 and top-2 only
        4. Otherwise keep top-3

    Args:
        doc_candidates: List of document candidates sorted by doc_score (descending)
        min_gap_1st_2nd: Minimum gap between 1st and 2nd to include 2nd
        min_gap_2nd_3rd: Minimum gap between 2nd and 3rd to include 3rd
        min_zscore_threshold: Minimum z-score for top-1 to be considered valid
        always_return_at_least_one: If True, always return at least 1 document
                                    If False, allow empty result when z-score is low

    Returns:
        Filtered list of document candidates (0 to 3 documents)

    Example (raw logit scores):
        >>> docs = [
        ...     {"docid": "d1", "doc_score": 2.5},   # Clear winner
        ...     {"docid": "d2", "doc_score": 0.8},   # Gap = 1.7 > 0.5 ✓
        ...     {"docid": "d3", "doc_score": 0.6}    # Gap = 0.2 < 0.3 ✗
        ... ]
        >>> filtered = filter_docs_by_gap_and_zscore(docs)
        >>> # Returns: [d1, d2] (d3 filtered due to small gap)
    """
    if not doc_candidates:
        return []

    # Extract scores
    scores = [doc.get("doc_score", 0.0) for doc in doc_candidates]

    # Calculate z-score for top-1
    if len(scores) >= 3:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 0:
            zscore_top1 = (scores[0] - mean_score) / std_score
        else:
            zscore_top1 = 0.0
    else:
        zscore_top1 = 1.0  # Not enough data for z-score

    logger.debug(f"[FILTER] Top-1 z-score: {zscore_top1:.2f}")

    # Check if top-1 is statistically significant
    if zscore_top1 < min_zscore_threshold and not always_return_at_least_one:
        logger.info(f"[FILTER] Top-1 z-score {zscore_top1:.2f} below threshold {min_zscore_threshold}, returning empty")
        return []

    # Always keep top-1
    filtered = [doc_candidates[0]]

    # Check gap between 1st and 2nd
    if len(doc_candidates) >= 2:
        gap_1_2 = scores[0] - scores[1]
        logger.debug(f"[FILTER] Gap(1st-2nd): {gap_1_2:.2f}")

        if gap_1_2 >= min_gap_1st_2nd:
            filtered.append(doc_candidates[1])

            # Check gap between 2nd and 3rd
            if len(doc_candidates) >= 3:
                gap_2_3 = scores[1] - scores[2]
                logger.debug(f"[FILTER] Gap(2nd-3rd): {gap_2_3:.2f}")

                if gap_2_3 >= min_gap_2nd_3rd:
                    filtered.append(doc_candidates[2])
                else:
                    logger.info(f"[FILTER] Gap(2nd-3rd) {gap_2_3:.2f} < {min_gap_2nd_3rd}, excluding 3rd")
        else:
            logger.info(f"[FILTER] Gap(1st-2nd) {gap_1_2:.2f} < {min_gap_1st_2nd}, excluding 2nd and 3rd")

    return filtered


def filter_docs_by_margin(
    doc_candidates: List[dict],
    min_score_3rd: float = 0.15,
    min_score_2nd: float = 0.12,
    always_return_at_least_one: bool = True
) -> List[dict]:
    """
    DEPRECATED: Use filter_docs_by_gap_and_zscore() for raw logit scores.

    Filter documents based on margin-based thresholds instead of absolute values.

    This function implements intelligent filtering that considers the distribution
    of rerank scores rather than using arbitrary absolute thresholds (like 0.8).
    This prevents the "all scores clustered around 0.8" problem.

    Filtering Rules:
        1. Always keep the top-1 document
        2. Keep 2nd document if doc_score >= min_score_2nd
        3. Keep 3rd document if doc_score >= min_score_3rd
        4. If always_return_at_least_one=True, always return at least 1 doc

    Args:
        doc_candidates: List of document candidates sorted by doc_score (descending)
        min_score_3rd: Minimum rerank score for 3rd place document
        min_score_2nd: Minimum rerank score for 2nd place document
        always_return_at_least_one: If True, always return at least 1 document

    Returns:
        Filtered list of document candidates (1 to 3 documents)

    Example:
        >>> docs = [
        ...     {"docid": "d1", "doc_score": 0.85},
        ...     {"docid": "d2", "doc_score": 0.20},
        ...     {"docid": "d3", "doc_score": 0.10}
        ... ]
        >>> filtered = filter_docs_by_margin(docs, min_score_3rd=0.15, min_score_2nd=0.12)
        >>> # Returns: [d1, d2] (d3 filtered out because 0.10 < 0.15)
    """
    if not doc_candidates:
        return []

    # Always keep at least 1 document
    if always_return_at_least_one and len(doc_candidates) >= 1:
        filtered = [doc_candidates[0]]
    else:
        filtered = []

    # Check 2nd place
    if len(doc_candidates) >= 2:
        doc2_score = doc_candidates[1].get("doc_score", 0.0)
        if doc2_score >= min_score_2nd:
            if len(filtered) == 0:
                filtered = [doc_candidates[0], doc_candidates[1]]
            else:
                filtered.append(doc_candidates[1])

    # Check 3rd place
    if len(doc_candidates) >= 3:
        doc3_score = doc_candidates[2].get("doc_score", 0.0)
        if doc3_score >= min_score_3rd:
            if len(filtered) < 2 and len(doc_candidates) >= 2:
                # Add 2nd if not already added
                filtered.append(doc_candidates[1])
            filtered.append(doc_candidates[2])

    return filtered


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ChunkReranker Demo")
    print("=" * 60)

    # Initialize reranker
    reranker = ChunkReranker(
        model_name=DEFAULT_MODEL,
        device="cpu",  # Change to "cuda" if GPU available
        use_fp16=False
    )

    # Example query and chunks
    query = "RAM의 역할은 무엇인가?"
    chunk_candidates = [
        {
            "docid": "doc1",
            "chunk_id": "doc1_0",
            "chunk_text": "RAM은 컴퓨터의 주기억장치로 데이터를 임시 저장합니다.",
            "bm25": 5.2,
            "dense": 0.85
        },
        {
            "docid": "doc1",
            "chunk_id": "doc1_1",
            "chunk_text": "CPU는 연산을 담당하는 중앙처리장치입니다.",
            "bm25": 2.1,
            "dense": 0.45
        },
        {
            "docid": "doc2",
            "chunk_id": "doc2_0",
            "chunk_text": "하드디스크는 영구 저장장치입니다.",
            "bm25": 1.8,
            "dense": 0.32
        }
    ]

    # Rerank chunks
    print(f"\nQuery: {query}\n")
    reranked_chunks = reranker.rerank_chunks(query, chunk_candidates)

    print("Chunk-level scores:")
    for chunk in reranked_chunks:
        print(f"  {chunk['chunk_id']}: rerank={chunk['rerank_score']:.3f}, "
              f"bm25={chunk['bm25']:.2f}, dense={chunk['dense']:.2f}")

    # Aggregate to documents
    doc_results = aggregate_chunks_to_docs(reranked_chunks, top_k=3)

    print("\nDocument-level aggregation:")
    for doc in doc_results:
        print(f"  {doc['docid']}: doc_score={doc['doc_score']:.3f}, "
              f"doc_score2={doc['doc_score2']:.3f}, chunks={doc['chunk_count']}")

    # Filter by margin
    filtered_docs = filter_docs_by_margin(doc_results)

    print("\nFiltered results:")
    for doc in filtered_docs:
        print(f"  {doc['docid']}: doc_score={doc['doc_score']:.3f}")

    print("\n" + "=" * 60)
