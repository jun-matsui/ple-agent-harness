"""Official PLE-Bench evaluation metric implementation (Phase 1)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip01(value: float) -> float:
    if not _is_finite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def compute_normalized_objective(objective: float, objective_naive: float = 0.0) -> float:
    """Compute O' = clip((O - O_naive) / (1 - O_naive), 0, 1)."""
    if not _is_finite(objective) or not _is_finite(objective_naive):
        return 0.0
    o_val = float(objective)
    o_naive = float(objective_naive)
    if o_naive >= 1.0 - 1e-12:
        return 1.0 if o_val >= 1.0 - 1e-12 else 0.0
    return _clip01((o_val - o_naive) / (1.0 - o_naive))


def score_item(
    soft_fraction: float,
    hard_gate: bool | int | float,
    objective: float,
    objective_naive: float = 0.0,
) -> float:
    """Compute PLE item score: s = 0.4 * p + 0.6 * G * O'.

    Any NaN/Inf or failed hard gate forces G = 0, capping score at 0.40,
    and non-finite soft_fraction invalidates the item (0.0).
    """
    if not _is_finite(soft_fraction):
        return 0.0
    p = _clip01(soft_fraction)
    if isinstance(hard_gate, bool):
        g = 1.0 if hard_gate else 0.0
    elif _is_finite(hard_gate):
        g = 1.0 if float(hard_gate) >= 1.0 - 1e-12 else 0.0
    else:
        g = 0.0
    o_prime = compute_normalized_objective(objective, objective_naive)
    return 0.4 * p + 0.6 * g * o_prime


def score_task(items: Sequence[Mapping[str, float | bool]]) -> float:
    """Compute mean score across all deliverable items of a task."""
    if not items:
        return 0.0
    total = 0.0
    for item in items:
        total += score_item(
            soft_fraction=float(item.get("p", 0.0)),
            hard_gate=item.get("G", False),
            objective=float(item.get("O", 0.0)),
            objective_naive=float(item.get("O_naive", 0.0)),
        )
    return total / len(items)


def score_run(
    task_scores: Mapping[str, float],
    expected_task_ids: Iterable[str],
) -> float:
    """Compute overall run score; missing or non-finite tasks score 0.0."""
    expected = list(expected_task_ids)
    if not expected:
        return 0.0
    total = 0.0
    for tid in expected:
        val = task_scores.get(tid, 0.0)
        total += _clip01(val)
    return total / len(expected)


def _rankdata(values: Sequence[float]) -> list[float]:
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx <= 1e-15 or dy <= 1e-15:
        return 0.0
    return num / (dx * dy)


def correlation_cv_lb(cv_scores: Sequence[float], lb_scores: Sequence[float]) -> Dict[str, float]:
    """Return Pearson and Spearman rank correlations between local CV and LB."""
    if len(cv_scores) != len(lb_scores) or len(cv_scores) < 2:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": round(_pearson(cv_scores, lb_scores), 6),
        "spearman": round(_pearson(_rankdata(cv_scores), _rankdata(lb_scores)), 6),
    }
