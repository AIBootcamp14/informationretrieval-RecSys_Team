"""
Query processing module: cleaning, expansion (OR-style), and compound token breaking.
"""
from typing import Dict, List, Optional, Set
import re
from loguru import logger


class QueryProcessor:
    """Query processor with additive expansions and vocab-based word breaking."""

    def __init__(self):
        self.domain_vocab: Set[str] = set()
        self.min_compound_length = 6
        # Additive expansions (never remove the original token)
        self.synonym_map: Dict[str, List[str]] = {
            "키우": ["재배", "재배법", "기르기"],
            "재배": ["재배법", "재배 방법"],
            "노하우": ["방법", "요령", "팁"],
            "병해충": ["병충해", "해충", "질병 관리"],
            "관리": ["유지 관리", "관리 요령"],
        }

    # -------------------------
    # Basic text utilities
    # -------------------------
    def set_domain_vocab(self, vocab: Set[str]) -> None:
        """Store domain vocabulary for compound splitting."""
        self.domain_vocab = set(vocab or [])
        logger.info(f"Domain vocab size set to {len(self.domain_vocab)}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lightweight tokenizer for overlap/expansion."""
        return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())

    def clean_query(self, query: str) -> str:
        """Normalize whitespace and strip trailing punctuation."""
        query = " ".join((query or "").split())
        return query.rstrip("?!.,")

    # -------------------------
    # Expansion helpers
    # -------------------------
    def _compound_parts(self, token: str) -> List[str]:
        """Split long token into vocab-backed parts (2-3 pieces)."""
        if not self.domain_vocab or len(token) < self.min_compound_length:
            return []
        if token in self.domain_vocab:
            return []

        # Try 2-part split
        for i in range(2, len(token) - 1):
            left, right = token[:i], token[i:]
            if left in self.domain_vocab and right in self.domain_vocab:
                return [left, right]

        # Try 3-part split
        for i in range(2, len(token) - 3):
            for j in range(i + 1, len(token) - 1):
                parts = [token[:i], token[i:j], token[j:]]
                if all(part in self.domain_vocab for part in parts):
                    return parts

        return []

    def _additive_expansions(self, base_query: str, tokens: List[str]) -> List[str]:
        """Generate additive expansions by appending related tokens."""
        expansions: List[str] = []
        for tok in tokens:
            for key, extras in self.synonym_map.items():
                if key in tok or tok in key:
                    for extra in extras:
                        if extra and extra not in base_query:
                            expansions.append(f"{base_query} {extra}")

            parts = self._compound_parts(tok)
            if parts:
                compound_phrase = " ".join(parts)
                expansions.append(f"{base_query} {compound_phrase}")

        seen = set()
        unique = []
        for exp in expansions:
            if exp not in seen:
                unique.append(exp)
                seen.add(exp)
        return unique

    def build_query_variants(self, base_query: str) -> List[str]:
        """Return base query plus additive variants (no replacements)."""
        cleaned = self.clean_query(base_query)
        if not cleaned:
            return []

        tokens = self._tokenize(cleaned)
        variants = [cleaned]
        variants.extend(self._additive_expansions(cleaned, tokens))

        # Deduplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for q in variants:
            if q not in seen:
                ordered.append(q)
                seen.add(q)
        return ordered

    # -------------------------
    # Public API
    # -------------------------
    def enhance_query_for_search(self, query: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Clean query (conversation context handled upstream by rewriters)."""
        return self.clean_query(query)

    def prepare_queries(
        self,
        original_query: str,
        rewrites: List[str],
        messages: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Build final list of queries to issue (original preserved, expansions are additive).
        """
        seeds: List[str] = []
        if original_query:
            seeds.append(original_query)
        seeds.extend(rewrites or [])

        prepared: List[str] = []
        for seed in seeds:
            enhanced = self.enhance_query_for_search(seed, messages)
            prepared.extend(self.build_query_variants(enhanced))

        # Deduplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for q in prepared:
            if q not in seen:
                ordered.append(q)
                seen.add(q)
        return ordered


# Singleton instance
_query_processor_instance: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """Get singleton query processor instance."""
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor()
    return _query_processor_instance
