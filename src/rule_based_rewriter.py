"""Rule-based query rewriting targeting zero-hit (0/3) error patterns."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from loguru import logger

import config.config as cfg


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for cleaner query strings."""
    return " ".join(text.split())


def _to_lower_set(values: Sequence[str]) -> Set[str]:
    """Convert a sequence of strings into a lowercase set, skipping empty values."""
    return {value.lower() for value in values if value}


@dataclass
class ReplacementRule:
    """Single replacement instruction."""

    pattern: str
    replacement: str = ""
    is_regex: bool = False
    ignore_case: bool = True

    def apply(self, text: str) -> str:
        """Apply replacement to the given text."""
        if not self.pattern:
            return text

        flags = re.IGNORECASE if self.ignore_case else 0
        if self.is_regex or self.ignore_case:
            try:
                regex = re.compile(self.pattern, flags)
            except re.error:
                logger.warning(f"Invalid regex pattern in rule: {self.pattern}")
                return text
            return regex.sub(self.replacement, text)

        return text.replace(self.pattern, self.replacement)


@dataclass
class RuleActions:
    """All supported actions for a rule."""

    prepend: List[str] = field(default_factory=list)
    append: List[str] = field(default_factory=list)
    ensure_phrases: List[str] = field(default_factory=list)
    remove_phrases: List[str] = field(default_factory=list)
    replacements: List[ReplacementRule] = field(default_factory=list)

    def apply(self, text: str) -> str:
        """Apply configured actions to the text."""
        for replacement in self.replacements:
            text = replacement.apply(text)

        for phrase in self.remove_phrases:
            if not phrase:
                continue
            text = text.replace(phrase, " ")

        for phrase in self.prepend:
            if phrase and phrase not in text:
                text = f"{phrase} {text}"

        for phrase in self.ensure_phrases:
            if phrase and phrase not in text:
                text = f"{text} {phrase}"

        for phrase in self.append:
            if phrase and phrase not in text:
                text = f"{text} {phrase}"

        return _normalize_whitespace(text)


@dataclass
class Rule:
    """Represents a single configurable rule."""

    name: str
    description: str = ""
    enabled: bool = True
    min_length: int = 0
    keywords_any: Set[str] = field(default_factory=set)
    keywords_all: Set[str] = field(default_factory=set)
    exclude_keywords: Set[str] = field(default_factory=set)
    regex_patterns: List[re.Pattern] = field(default_factory=list)
    eval_ids: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    actions: RuleActions = field(default_factory=RuleActions)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Rule":
        """Create a Rule instance from dict payload."""
        match_config = payload.get("match", {})
        action_config = payload.get("actions", {})

        regex_patterns: List[re.Pattern] = []
        for pattern in match_config.get("regex", []):
            if not pattern:
                continue
            try:
                regex_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                logger.warning(f"Invalid regex in rule '{payload.get('name', 'unknown')}': {pattern}")

        replacements = [
            ReplacementRule(
                pattern=entry.get("pattern", ""),
                replacement=entry.get("replacement", ""),
                is_regex=entry.get("regex", False),
                ignore_case=entry.get("ignore_case", True),
            )
            for entry in action_config.get("replace", [])
        ]

        return cls(
            name=payload.get("name", "unnamed"),
            description=payload.get("description", ""),
            enabled=payload.get("enabled", True),
            min_length=match_config.get("min_length", 0),
            keywords_any=_to_lower_set(match_config.get("any_keywords", [])),
            keywords_all=_to_lower_set(match_config.get("all_keywords", [])),
            exclude_keywords=_to_lower_set(match_config.get("exclude_keywords", [])),
            regex_patterns=regex_patterns,
            eval_ids={str(eid) for eid in match_config.get("eval_ids", []) if eid is not None},
            categories=_to_lower_set(match_config.get("categories", [])),
            actions=RuleActions(
                prepend=action_config.get("prepend", []),
                append=action_config.get("append", []),
                ensure_phrases=action_config.get("ensure_phrases", []),
                remove_phrases=action_config.get("remove_phrases", []),
                replacements=replacements,
            ),
        )

    def matches(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the rule should apply to the given query/metadata."""
        if not self.enabled:
            return False

        if self.min_length and len(query) < self.min_length:
            return False

        meta = metadata or {}
        if self.eval_ids:
            eval_id = meta.get("eval_id")
            if eval_id is None or str(eval_id) not in self.eval_ids:
                return False

        if self.categories:
            query_categories = _to_lower_set(meta.get("categories", []))
            if query_categories and not (self.categories & query_categories):
                return False

        query_lower = query.lower()

        if self.exclude_keywords and any(excl in query_lower for excl in self.exclude_keywords):
            return False

        if self.keywords_all and not all(keyword in query_lower for keyword in self.keywords_all):
            return False

        triggered = False
        if self.keywords_any:
            triggered = any(keyword in query_lower for keyword in self.keywords_any)

        if self.regex_patterns:
            regex_hit = any(pattern.search(query) for pattern in self.regex_patterns)
            triggered = triggered or regex_hit

        if not self.keywords_any and not self.regex_patterns and not self.keywords_all and self.eval_ids:
            # Eval-specific rule without textual triggers
            triggered = True

        return triggered

    def apply(self, query: str) -> str:
        """Apply rule actions."""
        return self.actions.apply(query)


class ErrorDrivenRuleEngine:
    """Loads and applies rule-based query rewriting logic."""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = Path(rules_path or cfg.ERROR_RULE_PATH)
        self.rules: List[Rule] = self._load_rules()
        logger.info(f"Loaded {len(self.rules)} rule-based query fixes from {self.rules_path}")

    def _load_rules(self) -> List[Rule]:
        """Load rules from JSON configuration."""
        if not self.rules_path.exists():
            logger.warning(f"Rule configuration file not found at {self.rules_path}")
            return []

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load rule configuration: {exc}")
            return []

        raw_rules = payload if isinstance(payload, list) else payload.get("rules", [])
        rules: List[Rule] = []
        for raw_rule in raw_rules:
            try:
                rules.append(Rule.from_dict(raw_rule))
            except Exception as exc:
                logger.error(f"Failed to parse rule: {exc}")
        return rules

    def apply(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Apply all matching rules to the query."""
        if not query or not self.rules:
            return query

        rewritten = query
        applied_rules: List[str] = []

        for rule in self.rules:
            if not rule.matches(rewritten, metadata):
                continue

            updated = rule.apply(rewritten)
            if updated != rewritten:
                applied_rules.append(rule.name)
                rewritten = updated

        if applied_rules:
            logger.info(
                f"Rule-based rewrite applied ({', '.join(applied_rules)}): '{query}' -> '{rewritten}'"
            )

        return rewritten


_rule_engine_instance: Optional[ErrorDrivenRuleEngine] = None


def get_rule_engine() -> ErrorDrivenRuleEngine:
    """Singleton accessor to avoid repeated file IO."""
    global _rule_engine_instance
    if _rule_engine_instance is None:
        _rule_engine_instance = ErrorDrivenRuleEngine()
    return _rule_engine_instance
