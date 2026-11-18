"""Unit tests for the official PLE evaluation metric (Phase 1)."""

import math
import unittest

from src.metrics.official_metric import (
    compute_normalized_objective,
    correlation_cv_lb,
    score_item,
    score_run,
    score_task,
)


class TestOfficialMetric(unittest.TestCase):
    def test_hard_gate_cliff_caps_at_0_40(self) -> None:
        # Even with p=1.0 and O=1.0, G=0 caps score at 0.40
        s = score_item(soft_fraction=1.0, hard_gate=False, objective=1.0, objective_naive=0.2)
        self.assertAlmostEqual(s, 0.40, places=9)

    def test_full_pass_with_naive_baseline(self) -> None:
        # p=0.95, G=1, O=0.80, O_naive=0.60 -> O' = 0.50 -> 0.4*0.95 + 0.6*1*0.50 = 0.68
        s = score_item(soft_fraction=0.95, hard_gate=True, objective=0.80, objective_naive=0.60)
        self.assertAlmostEqual(s, 0.68, places=9)

    def test_pathological_nan_inf_negative(self) -> None:
        self.assertEqual(score_item(math.nan, True, 1.0), 0.0)
        self.assertEqual(score_item(1.0, math.nan, 1.0), 0.4)
        self.assertEqual(score_item(1.0, True, math.inf), 0.4)
        self.assertEqual(score_item(-0.5, True, -10.0), 0.0)
        self.assertEqual(compute_normalized_objective(0.5, 1.0), 0.0)
        self.assertEqual(compute_normalized_objective(1.0, 1.0), 1.0)

    def test_score_task_and_run_missing_tasks(self) -> None:
        task_s = score_task([
            {"p": 1.0, "G": True, "O": 1.0, "O_naive": 0.0},
            {"p": 1.0, "G": False, "O": 1.0, "O_naive": 0.0},
        ])
        self.assertAlmostEqual(task_s, 0.70, places=9)

        run_s = score_run({"t1": 0.8, "t2": math.nan}, ["t1", "t2", "t3"])
        self.assertAlmostEqual(run_s, 0.8 / 3.0, places=9)

    def test_correlation_cv_lb(self) -> None:
        corr = correlation_cv_lb([0.40, 0.55, 0.70, 0.85], [0.38, 0.52, 0.68, 0.84])
        self.assertGreater(corr["pearson"], 0.99)
        self.assertEqual(corr["spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
