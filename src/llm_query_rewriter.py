"""LLM-based multi-turn query rewriter with SEARCHABLE classification."""
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import json
import openai
import os
import config.config as cfg


class LLMQueryRewriter:
    """
    Multi-turn query rewriting using GPT-4o-mini.
    - Resolves pronouns and adds necessary context from conversation history
    - Classifies queries as SEARCHABLE/NOT_SEARCHABLE (search is never skipped)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_rewrites: int = 3
    ):
        self.model = model
        self.temperature = temperature
        self.max_rewrites = max_rewrites
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        logger.info(f"Initialized LLM Query Rewriter with model: {model}")

    def classify_and_rewrite(
        self,
        messages: List[Dict[str, str]],
        fallback_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify query as SEARCHABLE/NOT_SEARCHABLE and return rewrites.

        Returns:
            {
                "original": str,
                "rewrites": List[str],
                "searchable": bool,
                "reason": str
            }
        """
        result: Dict[str, Any] = {
            "original": fallback_query or "",
            "rewrites": [],
            "searchable": True,
            "reason": ""
        }

        if not messages:
            return result

        current_query = messages[-1].get("content", "").strip() or (fallback_query or "")
        result["original"] = current_query
        if not current_query:
            return result

        try:
            searchable, reason = self._classify_searchable(current_query, messages)
            result["searchable"] = searchable
            result["reason"] = reason
            logger.info(f"Query classified as: {'SEARCHABLE' if searchable else 'NOT_SEARCHABLE'} ({reason})")

            context = self._build_context(messages[:-1]) if len(messages) > 1 else ""
            rewrites = self._rewrite_with_llm(current_query, context)
            if rewrites:
                result["rewrites"] = rewrites
            return result

        except Exception as exc:  # pragma: no cover - network/API errors
            logger.error(f"Query classification/rewriting failed: {exc}")
            result["searchable"] = True
            if not result["rewrites"]:
                result["rewrites"] = []
            return result

    def rewrite(
        self,
        messages: List[Dict[str, str]],
        fallback_query: Optional[str] = None
    ) -> List[str]:
        """
        Backward compatibility wrapper for rewrite method.

        Args:
            messages: Conversation history
            fallback_query: Fallback query if rewriting fails

        Returns:
            List of rewritten queries
        """
        result = self.classify_and_rewrite(messages, fallback_query)
        if result.get("rewrites"):
            return result["rewrites"]
        if result.get("original"):
            return [result["original"]]
        return []

    def _classify_searchable(self, query: str, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Classify whether the query is likely answerable from documents.

        Returns:
            (is_searchable, short_reason)
        """
        system_prompt = """You decide if a user query can likely be answered using a document collection.

Labels:
- SEARCHABLE: factual/educational/how-to/definition/look-up style questions that a document could answer.
- NOT_SEARCHABLE: smalltalk, opinions, feelings, personal requests, or tasks that are not in documents.

Rules:
- Never skip search. This label only controls how strict later evidence checks should be.
- Prefer SEARCHABLE when the query mentions concrete entities, methods, events, or domain keywords.
- Keep the output strictly in JSON."""

        context = ""
        if len(messages) > 1:
            context = f"Recent context:\\n{self._build_context(messages[-3:-1])}\\n"

        user_prompt = f"""{context}Current query: {query}

Return JSON: {{"label": "SEARCHABLE|NOT_SEARCHABLE", "reason": "short justification"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=120,
                response_format={"type": "json_object"}
            )

            payload = response.choices[0].message.content.strip()
            try:
                data = json.loads(payload)
            except Exception:
                data = {}

            label = str(data.get("label", "")).upper()
            reason = str(data.get("reason", "")).strip()
            is_searchable = label != "NOT_SEARCHABLE"

            return is_searchable, reason

        except Exception as exc:  # pragma: no cover - network/API errors
            logger.error(f"Classification API call failed: {exc}")
            return True, "fallback-searchable"

    def _build_context(self, messages: List[Dict[str, str]]) -> str:
        """Build context string from previous messages."""
        context_parts = []
        for msg in messages[-3:]:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if content:
                context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)

    def _rewrite_with_llm(self, query: str, context: str) -> List[str]:
        """Rewrite query using LLM and return up to max_rewrites variants."""
        system_prompt = """You rewrite user queries so they are standalone and search-friendly.

Requirements:
- Preserve rare keywords and proper nouns; never delete them.
- Resolve vague references using the given context.
- Produce 1-3 concise Korean queries suitable for retrieval.
- Keep the output strictly in JSON with a 'rewrites' list."""

        user_prompt = f"""Context:
{context or '(none)'}

Current query: {query}

Return JSON like {{"rewrites": ["variant1", "variant2"]}}. If the query is already clear, include it as-is."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
            except Exception:
                data = {}

            rewrites = data.get("rewrites", [])
            if isinstance(rewrites, str):
                rewrites = [rewrites]
            rewrites = [r.strip() for r in rewrites if isinstance(r, str) and r.strip()]

            if not rewrites:
                rewrites = [query]

            seen = set()
            unique_rewrites = []
            for candidate in rewrites:
                if candidate not in seen:
                    unique_rewrites.append(candidate)
                    seen.add(candidate)
                if len(unique_rewrites) >= self.max_rewrites:
                    break

            return unique_rewrites

        except Exception as exc:  # pragma: no cover - network/API errors
            logger.error(f"LLM API call failed: {exc}")
            return [query]


# Singleton instance
_llm_rewriter_instance: Optional[LLMQueryRewriter] = None


def get_llm_query_rewriter() -> LLMQueryRewriter:
    """Get singleton LLM query rewriter instance."""
    global _llm_rewriter_instance
    if _llm_rewriter_instance is None:
        _llm_rewriter_instance = LLMQueryRewriter()
    return _llm_rewriter_instance
