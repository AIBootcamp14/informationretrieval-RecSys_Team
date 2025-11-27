"""2-Stage Cascaded LLM Reranker using GPT-4o-mini"""
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
import openai
import os
import json


@dataclass
class RerankCandidate:
    """Candidate document for reranking"""
    docid: str
    content: str
    initial_score: float = 0.0
    rerank_score: float = 0.0


class LLMReranker:
    """
    2-Stage Cascaded LLM Reranker
    Stage 1: Coarse filtering (30 -> 10)
    Stage 2: Fine-grained ranking (10 -> 3)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_candidates: int = 30
    ):
        self.model = model
        self.temperature = temperature
        self.max_candidates = max_candidates
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        logger.info(f"Initialized 2-Stage LLM Reranker with model: {model}")

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        topk: int = 3
    ) -> List[RerankCandidate]:
        """
        2-Stage cascaded reranking

        Args:
            query: Search query
            candidates: List of candidate documents
            topk: Final number of documents (default: 3)

        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []

        # Limit candidates
        candidates = candidates[:self.max_candidates]

        logger.info(f"Starting 2-stage reranking: {len(candidates)} candidates -> {topk} final")

        # Stage 1: Coarse filtering (30 -> 10)
        stage1_topk = min(10, len(candidates))
        stage1_results = self._stage1_rerank(query, candidates, stage1_topk)

        logger.info(f"Stage 1 complete: {len(stage1_results)} candidates")

        if len(stage1_results) <= topk:
            return stage1_results

        # Stage 2: Fine-grained ranking (10 -> 3)
        stage2_results = self._stage2_rerank(query, stage1_results, topk)

        logger.info(f"Stage 2 complete: {len(stage2_results)} final candidates")

        return stage2_results

    def _stage1_rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        topk: int
    ) -> List[RerankCandidate]:
        """
        Stage 1: Coarse filtering using simple relevance check
        Fast, focuses on removing clearly irrelevant documents
        """
        system_prompt = """You are a document relevance classifier for scientific questions.

Your task: Given a query and multiple documents, identify which documents are relevant.

Output format: JSON object with "indices" key containing a list of relevant document indices
Example: {"indices": [0, 2, 5, 7, 9]}

If none of the documents can answer the question, return {"indices": []}.
Be generous in Stage 1, but still allow an empty list when nothing matches."""

        # Build document list
        doc_list = []
        for i, cand in enumerate(candidates):
            doc_preview = cand.content[:200] + "..." if len(cand.content) > 200 else cand.content
            doc_list.append(f"[{i}] {doc_preview}")

        user_prompt = f"""Query: {query}

Documents:
{chr(10).join(doc_list)}

Select top {topk} most relevant documents. Return as JSON with "indices" key:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"Stage 1 LLM response: {content}")

            result = json.loads(content)

            # Extract indices with fallback keys
            indices = []
            if isinstance(result, dict):
                indices = result.get("indices", result.get("relevant", result.get("documents", [])))
            elif isinstance(result, list):
                indices = result

            if not indices:
                logger.info("Stage1 LLM chose no relevant documents.")
                return []

            filtered = [candidates[i] for i in indices if 0 <= i < len(candidates)]

            for rank, cand in enumerate(filtered):
                cand.rerank_score = 1.0 - (rank / max(len(filtered), 1))

            return filtered[:topk]

        except Exception as e:
            logger.error(f"Stage 1 reranking failed: {e}")
            # Fallback: return top candidates by initial score
            return sorted(candidates, key=lambda x: x.initial_score, reverse=True)[:topk]

    def _stage2_rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        topk: int
    ) -> List[RerankCandidate]:
        """
        Stage 2: Fine-grained ranking with detailed analysis
        Slower, focuses on precise ranking of relevant documents
        """
        system_prompt = """You are a document ranking expert for scientific questions.

Your task: Rank documents by relevance to the query with detailed reasoning.

If none of the provided documents can answer, return an empty rankings list.

Ranking criteria:
1. Direct answer to the question
2. Completeness of information
3. Scientific accuracy
4. Clarity and detail

Output format: JSON list of rankings
Example:
{
  "rankings": [
    {"index": 2, "score": 0.95, "reason": "Directly answers with detailed explanation"},
    {"index": 0, "score": 0.80, "reason": "Relevant but lacks details"},
    {"index": 1, "score": 0.60, "reason": "Partially relevant"}
  ]
}"""

        # Build document list with more detail
        doc_list = []
        for i, cand in enumerate(candidates):
            doc_preview = cand.content[:500] + "..." if len(cand.content) > 500 else cand.content
            doc_list.append(f"[{i}] {doc_preview}")

        user_prompt = f"""Query: {query}

Documents:
{chr(10).join(doc_list)}

Rank these documents by relevance. Return top {topk} with scores and reasons:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            rankings = result.get("rankings", [])
            if not rankings:
                logger.info("Stage2 LLM returned no relevant documents.")
                return []

            reranked = []
            for rank_info in rankings[:topk]:
                idx = rank_info.get("index", -1)
                score = rank_info.get("score", 0.0)
                if 0 <= idx < len(candidates):
                    cand = candidates[idx]
                    cand.rerank_score = score
                    reranked.append(cand)

            return reranked

        except Exception as e:
            logger.error(f"Stage 2 reranking failed: {e}")
            # Fallback: return top candidates by stage1 score
            return candidates[:topk]


# Singleton instance
_llm_reranker_instance = None


def get_llm_reranker() -> LLMReranker:
    """Get singleton LLM reranker instance"""
    global _llm_reranker_instance
    if _llm_reranker_instance is None:
        _llm_reranker_instance = LLMReranker()
    return _llm_reranker_instance
