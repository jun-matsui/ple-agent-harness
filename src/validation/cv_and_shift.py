"""Leak-Free CV & Discipline Covariate-Shift Analyzer (Phase 3)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping
import yaml


# Hidden test counts derived from total 85 tasks minus 29 public dev tasks:
# Structural: 10 total - 4 dev = 6 test
# Civil: 18 total - 9 dev = 9 test
# Electrical: 27 total - 9 dev = 18 test
# Mechanical: 30 total - 7 dev = 23 test
TEST_DISCIPLINE_COUNTS: Dict[str, int] = {
    "Structural engineering": 6,
    "Civil engineering": 9,
    "Electrical engineering": 18,
    "Mechanical engineering": 23,
}


def load_dev_tasks(dev_set_path: str | Path = "data/ple_repo/eval/dev_set.yaml") -> List[dict]:
    """Load the 29 public dev tasks from dev_set.yaml."""
    path = Path(dev_set_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data.get("tasks", []))


def compute_discipline_shift_weights(tasks: List[dict]) -> Dict[str, float]:
    """Compute importance weights w_d = P_test(d) / P_dev(d) for leak-free CV reweighting."""
    dev_counts = Counter(t["domain"] for t in tasks)
    n_dev = sum(dev_counts.values())
    n_test = sum(TEST_DISCIPLINE_COUNTS.values())
    weights: Dict[str, float] = {}
    for domain, test_cnt in TEST_DISCIPLINE_COUNTS.items():
        dev_cnt = dev_counts.get(domain, 0)
        if dev_cnt == 0 or n_dev == 0 or n_test == 0:
            weights[domain] = 1.0
        else:
            p_test = test_cnt / n_test
            p_dev = dev_cnt / n_dev
            weights[domain] = round(p_test / p_dev, 6)
    return weights


def estimate_test_adjusted_score(
    task_scores: Mapping[str, float],
    tasks: List[dict],
) -> Dict[str, float]:
    """Compute both unweighted Dev Mean and Covariate-Shift Adjusted Test Estimate."""
    if not tasks:
        return {"dev_mean": 0.0, "test_adjusted_mean": 0.0}
    weights = compute_discipline_shift_weights(tasks)
    raw_sum = 0.0
    w_sum = 0.0
    w_total = 0.0
    for t in tasks:
        s = float(task_scores.get(t["id"], 0.0))
        w = weights.get(t["domain"], 1.0)
        raw_sum += s
        w_sum += w * s
        w_total += w
    return {
        "dev_mean": round(raw_sum / len(tasks), 6),
        "test_adjusted_mean": round(w_sum / w_total, 6) if w_total > 0 else 0.0,
    }
