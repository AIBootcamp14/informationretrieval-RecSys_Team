"""Ground truth label helper using baseline submission to detect smalltalk queries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set

from loguru import logger

import config.config as cfg


class GroundTruthLabels:
    """Soft prior for smalltalk detection from baseline sample submission.

    Baseline 파일에서 `topk`가 비어 있는 eval_id들은 "smalltalk 가능성이 높다"는 prior로만 사용한다.
    실제 empty 판정은 LLM 분류 + evidence gate에서 결정한다.
    """

    def __init__(self, baseline_path: Path = cfg.BASELINE_SAMPLE_PATH):
        self.baseline_path = Path(baseline_path)
        self.smalltalk_prior_ids: Set[str] = set()
        self._load()

    def _load(self):
        if not self.baseline_path.exists():
            logger.warning(
                f"Baseline sample not found at {self.baseline_path}. No prior information available."
            )
            return

        try:
            with open(self.baseline_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line.strip())
                    eval_id = row.get('eval_id')
                    if eval_id is None:
                        continue
                    if not row.get('topk'):
                        self.smalltalk_prior_ids.add(str(eval_id))
        except Exception as exc:
            logger.error(f"Failed to parse baseline sample for smalltalk prior: {exc}")
            self.smalltalk_prior_ids.clear()
            return

        logger.info(
            f"Loaded {len(self.smalltalk_prior_ids)} smalltalk prior IDs from baseline (used as soft hints only)"
        )

    def has_smalltalk_prior(self, eval_id: Optional[int]) -> bool:
        """Check if eval_id has smalltalk prior (soft hint, not hard constraint)."""
        if eval_id is None:
            return False
        return str(eval_id) in self.smalltalk_prior_ids

    def requires_search(self, eval_id: Optional[int]) -> bool:
        """Deprecated: Use has_smalltalk_prior() instead for soft prior logic."""
        # For backward compatibility, return True (always allow search)
        return True


_gt_labels_instance: Optional[GroundTruthLabels] = None


def get_ground_truth_labels() -> GroundTruthLabels:
    global _gt_labels_instance
    if _gt_labels_instance is None:
        _gt_labels_instance = GroundTruthLabels()
    return _gt_labels_instance
