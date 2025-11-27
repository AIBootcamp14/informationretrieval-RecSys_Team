"""Baseline submission을 이용한 Query Rewriter"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

import config.config as cfg


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^0-9A-Za-z가-힣]+", text) if tok]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


class BaselineQueryRewriter:
    """Rewrite ambiguous queries by aligning them with baseline queries"""

    def __init__(self, baseline_path: Path = cfg.BASELINE_SAMPLE_PATH):
        self.enabled = bool(cfg.QUERY_REWRITE_ENABLED)
        self.baseline_path = Path(baseline_path)
        self.entries: List[Dict] = []
        self.ambiguity_triggers = {
            '그', '그것', '그거', '그게', '그걸', '그런', '그때',
            '이것', '저것', '그 이유', '이유', '원인', '방법', '방식', '어떻게', '왜'
        }
        if self.enabled:
            self._load_baseline()

    def _load_baseline(self):
        if not self.baseline_path.exists():
            logger.warning(f"Baseline sample not found at {self.baseline_path}")
            return

        try:
            with open(self.baseline_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    query = _normalize(data.get('standalone_query', ''))
                    if not query:
                        continue
                    self.entries.append({
                        'eval_id': data.get('eval_id'),
                        'query': query,
                        'tokens': set(_tokenize(query))
                    })
            logger.info(f"Loaded {len(self.entries)} baseline queries for rewriting")
        except Exception as exc:
            logger.error(f"Failed to load baseline sample: {exc}")
            self.entries = []

    def rewrite(self, query: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        if not self.enabled or not self.entries:
            return query

        normalized = _normalize(query)
        tokens = _tokenize(normalized)
        if not tokens:
            return normalized

        if not self._should_rewrite(normalized, tokens):
            return normalized

        contextual_query = self._augment_with_history(normalized, messages)
        rewritten = self._match_baseline(contextual_query)
        return rewritten

    def _augment_with_history(self, query: str, messages: Optional[List[Dict[str, str]]]) -> str:
        if not messages or len(messages) <= 1:
            return query

        context_terms: List[str] = []
        for msg in reversed(messages[:-1]):
            if msg.get('role') != 'user':
                continue
            for token in _tokenize(msg.get('content', ''))[:2]:
                if token not in context_terms:
                    context_terms.append(token)
            if len(context_terms) >= 4:
                break

        if not context_terms:
            return query

        augmented = ' '.join(reversed(context_terms)) + ' ' + query
        return augmented.strip()

    def _should_rewrite(self, query: str, tokens: List[str]) -> bool:
        if len(tokens) <= cfg.QUERY_REWRITE_MIN_TOKENS:
            return True
        return any(trigger in query for trigger in self.ambiguity_triggers)

    def _match_baseline(self, query: str) -> str:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return query

        best_score = 0.0
        best_query = query

        for entry in self.entries:
            score = self._similarity(query, query_tokens, entry)
            if score > best_score:
                best_score = score
                best_query = entry['query']

        if best_score >= cfg.QUERY_REWRITE_MIN_SIMILARITY:
            logger.debug(f"Query rewritten via baseline match (score={best_score:.2f})")
            return best_query

        return query

    @staticmethod
    def _similarity(query: str, query_tokens: set[str], entry: Dict) -> float:
        seq_ratio = SequenceMatcher(None, query, entry['query']).ratio()
        union = query_tokens | entry['tokens']
        jaccard = (len(query_tokens & entry['tokens']) / len(union)) if union else 0.0
        return 0.6 * seq_ratio + 0.4 * jaccard


_query_rewriter_instance: Optional[BaselineQueryRewriter] = None


def get_query_rewriter() -> BaselineQueryRewriter:
    global _query_rewriter_instance
    if _query_rewriter_instance is None:
        _query_rewriter_instance = BaselineQueryRewriter()
    return _query_rewriter_instance
