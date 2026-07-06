"""Tests for the calibration and decision-curve helpers in src.evaluation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.evaluation import (
    compute_calibration_slope_intercept,
    compute_net_benefit,
    plot_decision_curve,
)


def test_calibration_slope_intercept_well_calibrated() -> None:
    """A model whose probabilities match the true data-generating process
    should have calibration slope near 1 and intercept near 0."""
    rng = np.random.default_rng(0)
    n = 4000
    y_prob = rng.uniform(0.05, 0.95, size=n)
    y_true = (rng.uniform(size=n) < y_prob).astype(int)

    intercept, slope = compute_calibration_slope_intercept(y_true, y_prob)

    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_calibration_slope_intercept_overconfident_model() -> None:
    """Predictions pushed toward the extremes relative to the true rate
    should show a calibration slope well below 1 (overconfident)."""
    rng = np.random.default_rng(1)
    n = 4000
    true_prob = rng.uniform(0.2, 0.8, size=n)
    y_true = (rng.uniform(size=n) < true_prob).astype(int)
    # Push reported probabilities toward 0/1 relative to the true rate.
    y_prob = np.clip(0.5 + (true_prob - 0.5) * 3.0, 1e-3, 1 - 1e-3)

    _, slope = compute_calibration_slope_intercept(y_true, y_prob)

    assert slope < 0.7


def test_net_benefit_treat_none_is_always_zero() -> None:
    """The 'treat none' reference strategy has zero net benefit everywhere."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(size=200)
    thresholds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    result = compute_net_benefit(y_true, y_prob, thresholds)

    assert (result["net_benefit_none"] == 0.0).all()


def test_net_benefit_treat_all_matches_closed_form() -> None:
    """'Treat all' net benefit should match prevalence - (1-prevalence)*odds."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    prevalence = y_true.mean()
    thresholds = np.array([0.2, 0.5])

    result = compute_net_benefit(y_true, y_prob, thresholds)

    for _, row in result.iterrows():
        pt = row["threshold"]
        expected = prevalence - (1 - prevalence) * (pt / (1 - pt))
        assert row["net_benefit_all"] == pytest.approx(expected)


def test_net_benefit_perfect_classifier_dominates_reference_curves() -> None:
    """A perfectly separating classifier's net benefit should be at least as
    good as both the 'treat all' and 'treat none' curves below the
    prevalence-implied crossover threshold."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.05] * 50 + [0.95] * 50)
    thresholds = np.array([0.1, 0.2, 0.3, 0.4])

    result = compute_net_benefit(y_true, y_prob, thresholds)

    assert (result["net_benefit_model"] >= result["net_benefit_all"] - 1e-9).all()
    assert (result["net_benefit_model"] >= result["net_benefit_none"] - 1e-9).all()


def test_net_benefit_threshold_at_or_above_one_is_handled() -> None:
    """Thresholds of 1.0 (undefined odds) should return NaN, not raise."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    thresholds = np.array([0.5, 1.0])

    result = compute_net_benefit(y_true, y_prob, thresholds)

    last_row = result[result["threshold"] == 1.0].iloc[0]
    assert np.isnan(last_row["net_benefit_model"])
    assert np.isnan(last_row["net_benefit_all"])
    assert last_row["net_benefit_none"] == 0.0


def test_plot_decision_curve_returns_axes_with_three_lines() -> None:
    """Smoke test: the plot helper runs headlessly and draws all three curves."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(size=100)
    thresholds = np.linspace(0.05, 0.5, 10)

    ax = plot_decision_curve(y_true, y_prob, thresholds)

    assert ax is not None
    assert len(ax.lines) >= 3
