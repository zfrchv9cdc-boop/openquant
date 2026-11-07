
"""OpenQuant: reproducible sentence-similarity quality metrics."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def evaluate_similarity(gold, reference, compressed):
    """Compare FP32 and INT4 predictions against labeled scores."""

    gold = np.asarray(gold, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    compressed = np.asarray(compressed, dtype=np.float64)

    if (
        gold.ndim != 1
        or reference.ndim != 1
        or compressed.ndim != 1
    ):
        raise ValueError("Scores must be one-dimensional")

    if not (
        len(gold)
        == len(reference)
        == len(compressed)
    ):
        raise ValueError("Score array lengths differ")

    if len(gold) < 3:
        raise ValueError("At least three sentence pairs required")

    for name, values in (
        ("gold", gold),
        ("reference", reference),
        ("compressed", compressed),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite scores")

        if np.unique(values).size < 2:
            raise ValueError(f"{name} scores are constant")

    ref_spearman = float(
        spearmanr(gold, reference).statistic
    )

    int4_spearman = float(
        spearmanr(gold, compressed).statistic
    )

    ref_pearson = float(
        pearsonr(gold, reference).statistic
    )

    int4_pearson = float(
        pearsonr(gold, compressed).statistic
    )

    return {
        "pairs": len(gold),
        "fp32_spearman": ref_spearman,
        "int4_spearman": int4_spearman,
        "spearman_drop_points": (
            ref_spearman - int4_spearman
        ) * 100,
        "fp32_pearson": ref_pearson,
        "int4_pearson": int4_pearson,
        "pearson_drop_points": (
            ref_pearson - int4_pearson
        ) * 100,
        "similarity_mean_absolute_difference": float(
            np.mean(np.abs(reference - compressed))
        ),
        "similarity_max_absolute_difference": float(
            np.max(np.abs(reference - compressed))
        ),
    }
