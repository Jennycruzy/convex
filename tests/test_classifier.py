"""The classifier's protocol: binary target, hard mapping, no look-ahead."""

from __future__ import annotations

import numpy as np
import pytest

from convex.classifier import (
    RegimeRule,
    brier_score,
    calibration_slope,
    fit_family,
    walk_forward,
)
from convex.config import load
from convex.errors import DataError
from convex.structures.base import Family


@pytest.fixture
def config():
    return load()


@pytest.fixture
def separable():
    """A signal a linear model should find: label follows the first predictor."""
    generator = np.random.default_rng(20260828)
    matrix = generator.normal(size=(200, 4))
    labels = (matrix[:, 0] + 0.25 * generator.normal(size=200) > 0).astype(int)
    return matrix, labels


def test_short_history_is_reported_rather_than_fitted(config):
    matrix = np.zeros((10, 3))
    labels = np.array([0, 1] * 5)
    model, report = fit_family(Family.PUT_BWB, matrix, labels, ["a", "b", "c"], config)
    assert model is None
    assert not report.trained
    assert "minimum" in report.note
    assert report.samples == 10


def test_single_class_history_is_reported_rather_than_fitted(config):
    matrix = np.random.default_rng(1).normal(size=(120, 3))
    labels = np.ones(120, dtype=int)
    model, report = fit_family(Family.STRADDLE, matrix, labels, ["a", "b", "c"], config)
    assert model is None
    assert "same label" in report.note


def test_a_fitted_family_reports_hit_rate_brier_and_calibration(config, separable):
    matrix, labels = separable
    model, report = fit_family(Family.PUT_BWB, matrix, labels, list("abcd"), config)
    assert model is not None and report.trained
    assert report.hit_rate > 0.8
    assert report.brier < 0.2
    assert 0.5 < report.calibration_slope < 1.5
    assert set(report.coefficients) == set("abcd")


def test_standardisation_survives_a_constant_predictor(config, separable):
    matrix, labels = separable
    matrix[:, 3] = 7.0  # a predictor that never moves
    model, report = fit_family(Family.PUT_BWB, matrix, labels, list("abcd"), config)
    assert model is not None
    probability = model.probability(matrix[0])
    assert 0.0 < probability < 1.0


def test_walk_forward_predicts_only_out_of_sample(config, separable):
    matrix, labels = separable
    probabilities, realised = walk_forward(Family.PUT_BWB, matrix, labels, list("abcd"), config)
    expected = matrix.shape[0] - config.int_("classifier.min_train_days")
    assert probabilities.size == expected == realised.size
    assert brier_score(probabilities, realised.astype(float)) < 0.25


def test_scoring_helpers_refuse_degenerate_input():
    with pytest.raises(DataError):
        brier_score(np.array([]), np.array([]))
    with pytest.raises(DataError, match="identical"):
        calibration_slope(np.full(10, 0.6), np.ones(10))


def test_the_regime_rule_favours_downside_when_variance_is_high():
    rule = RegimeRule()
    history = np.linspace(0.01, 0.05, 50)
    assert rule.regime(0.049, history) == "high_variance"
    assert rule.regime(0.011, history) == "low_variance"
    assert rule.regime(0.030, history) == "middle"

    assert rule.probability(Family.PUT_BWB, "high_variance") > 0.5
    assert rule.probability(Family.CALL_BWB, "high_variance") < 0.5
    assert rule.probability(Family.CALL_BWB, "low_variance") > 0.5
    assert rule.probability(Family.PUT_BWB, "middle") == 0.5


def test_the_regime_rule_refuses_to_guess_without_history():
    with pytest.raises(DataError, match="five prior"):
        RegimeRule().regime(0.02, [0.01, 0.02])
