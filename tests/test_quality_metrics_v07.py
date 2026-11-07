
"""Tests for OpenQuant labeled quality metrics."""

import pytest

pytest.importorskip("scipy")

from openquant.quality_metrics_v07 import evaluate_similarity


def test_identical_predictions_have_zero_quality_loss():
    values = [0, 1, 2, 3, 4]

    result = evaluate_similarity(
        values, values, values
    )

    assert result["fp32_spearman"] == pytest.approx(1)
    assert result["int4_spearman"] == pytest.approx(1)
    assert result["spearman_drop_points"] == pytest.approx(0)


def test_reversed_ranking_is_detected():
    gold = [0, 1, 2, 3, 4]

    result = evaluate_similarity(
        gold,
        gold,
        list(reversed(gold)),
    )

    assert result["fp32_spearman"] > 0.99
    assert result["int4_spearman"] < -0.99
    assert result["spearman_drop_points"] > 100


def test_constant_scores_are_rejected():
    with pytest.raises(ValueError, match="constant"):
        evaluate_similarity(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [1, 1, 1, 1],
        )


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="lengths"):
        evaluate_similarity(
            [0, 1, 2],
            [0, 1, 2],
            [0, 1],
        )
