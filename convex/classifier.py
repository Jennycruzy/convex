"""Per-structure directional classification.

The research is unambiguous on three points and this module implements exactly
those three and nothing more:

  binary target      the label is whether the structure's net result was
                     positive, not how large it was. Direction classifies
                     better than magnitude predicts
  hard mapping       a structure trades at full size or not at all. The
                     probability decides whether, never how much
  low capacity       L2-regularised logistic regression, because on short-
                     horizon 0DTE data with noisy payoffs and small per-family
                     samples a low-variance parametric model beats a flexible
                     one, and because a linear model's coefficients can be read
                     aloud to a judge

Training is an expanding window: every fit uses only sessions strictly before
the one being predicted, and predictors are standardised on the training window
alone so no distributional information leaks backwards.

When there is not enough history to train honestly, this module says so and
returns a documented rule instead of a model. Reporting that plainly is worth
more than a model fitted on thirty rows and presented as if it meant something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from convex.config import Config
from convex.errors import DataError
from convex.structures.base import Family

# Predictions this tightly clustered carry no ranking information, and fitting a
# line through them is numerically meaningless rather than merely imprecise.
_MINIMUM_PREDICTION_SPREAD = 1e-9


@dataclass(frozen=True)
class TrainingReport:
    """What the write-up reports for each family, honestly labelled."""

    family: Family
    trained: bool
    samples: int
    positives: int
    hit_rate: float | None
    brier: float | None
    calibration_slope: float | None
    coefficients: dict[str, float] | None
    note: str

    def as_dict(self) -> dict:
        return {
            "family": str(self.family),
            "trained": self.trained,
            "samples": self.samples,
            "positives": self.positives,
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
            "brier": None if self.brier is None else round(self.brier, 4),
            "calibration_slope": (
                None if self.calibration_slope is None else round(self.calibration_slope, 4)
            ),
            "note": self.note,
        }


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of the probabilities against the realised labels."""
    if probabilities.shape != outcomes.shape:
        raise DataError("probabilities and outcomes must have the same shape")
    if probabilities.size == 0:
        raise DataError("cannot score an empty prediction set")
    return float(np.mean((probabilities - outcomes) ** 2))


def calibration_slope(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Slope of realised frequency against predicted probability.

    One is perfect calibration, below one is overconfidence. Reported rather
    than optimised, because a model that is well ranked and badly calibrated is
    still useful under hard mapping while a model that claims certainty it does
    not have is not.
    """
    if probabilities.size < 3:
        raise DataError("calibration needs at least three predictions")
    spread = probabilities.std()
    if spread < _MINIMUM_PREDICTION_SPREAD:
        raise DataError("every prediction is identical, so calibration is undefined")
    slope, _ = np.polyfit(probabilities, outcomes.astype(float), 1)
    return float(slope)


@dataclass
class StructureModel:
    """One family's fitted classifier, with its standardisation."""

    family: Family
    feature_names: tuple[str, ...]
    model: LogisticRegression
    mean: np.ndarray
    scale: np.ndarray

    def probability(self, features: np.ndarray) -> float:
        if features.shape != self.mean.shape:
            raise DataError(
                f"{self.family}: expected {self.mean.size} features, got {features.size}"
            )
        standardised = ((features - self.mean) / self.scale).reshape(1, -1)
        return float(self.model.predict_proba(standardised)[0, 1])


def _standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    # A predictor that never moved in the training window carries no
    # information; dividing by its zero spread would be a silent infinity, so
    # it is neutralised explicitly and visibly.
    scale = np.where(scale > 0.0, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def fit_family(
    family: Family,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    config: Config,
) -> tuple[StructureModel | None, TrainingReport]:
    """Fit one family, or explain why it was not fitted."""
    minimum = config.int_("classifier.min_train_days")
    samples = int(matrix.shape[0])
    positives = int(labels.sum())

    if samples < minimum:
        return None, TrainingReport(
            family=family,
            trained=False,
            samples=samples,
            positives=positives,
            hit_rate=None,
            brier=None,
            calibration_slope=None,
            coefficients=None,
            note=(
                f"{samples} sessions of history against a {minimum}-session minimum; "
                "the documented volatility-regime rule runs instead"
            ),
        )
    if positives in (0, samples):
        return None, TrainingReport(
            family=family,
            trained=False,
            samples=samples,
            positives=positives,
            hit_rate=None,
            brier=None,
            calibration_slope=None,
            coefficients=None,
            note="every session carries the same label, so there is nothing to separate",
        )

    standardised, mean, scale = _standardise(matrix)
    model = LogisticRegression(
        C=config.float_("classifier.l2_c"),
        solver="lbfgs",
        max_iter=2_000,
    )
    model.fit(standardised, labels)
    in_sample = model.predict_proba(standardised)[:, 1]

    return (
        StructureModel(
            family=family,
            feature_names=tuple(feature_names),
            model=model,
            mean=mean,
            scale=scale,
        ),
        TrainingReport(
            family=family,
            trained=True,
            samples=samples,
            positives=positives,
            hit_rate=float(((in_sample > 0.5) == labels.astype(bool)).mean()),
            brier=brier_score(in_sample, labels.astype(float)),
            calibration_slope=calibration_slope(in_sample, labels),
            coefficients=dict(zip(feature_names, model.coef_[0].round(4))),
            note="fitted on the expanding window",
        ),
    )


def walk_forward(
    family: Family,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-sample probabilities from an expanding window.

    Session t is predicted by a model fitted only on sessions before t, with
    the standardisation computed on those sessions alone. Sessions inside the
    burn-in are not predicted at all rather than predicted badly.

    The row indices are returned alongside, and they matter. A fit can decline
    to produce a model, at which point that session is skipped and the run of
    probabilities is no longer one-per-session from the burn-in onwards.
    Anything pairing these probabilities back against the sessions they belong
    to has to use these indices; counting backwards from the end silently
    misaligns every figure by however many were skipped.
    """
    minimum = config.int_("classifier.min_train_days")
    predictions: list[float] = []
    realised: list[int] = []
    indices: list[int] = []
    for index in range(minimum, matrix.shape[0]):
        model, _ = fit_family(
            family, matrix[:index], labels[:index], feature_names, config
        )
        if model is None:
            continue
        predictions.append(model.probability(matrix[index]))
        realised.append(int(labels[index]))
        indices.append(index)
    return np.asarray(predictions), np.asarray(realised), np.asarray(indices, dtype=int)


@dataclass(frozen=True)
class RegimeRule:
    """The documented fallback when history is too short to fit anything.

    It is the research's volatility-regime table, reduced to what can be
    measured from a single chain snapshot: downside structures are favoured
    when implied variance is high relative to its own recent history, upside
    structures when it is low. The probability it returns is a stated prior,
    not an estimate, and every ledger record it produces says so.
    """

    high_variance_quantile: float = 0.6
    low_variance_quantile: float = 0.4
    favoured_probability: float = 0.60
    disfavoured_probability: float = 0.40

    def regime(self, implied_variance: float, history: Sequence[float]) -> str:
        series = np.asarray(list(history), dtype=float)
        if series.size < 5:
            raise DataError(
                f"the regime rule needs at least five prior variance readings, got {series.size}"
            )
        high = float(np.quantile(series, self.high_variance_quantile))
        low = float(np.quantile(series, self.low_variance_quantile))
        if implied_variance >= high:
            return "high_variance"
        if implied_variance <= low:
            return "low_variance"
        return "middle"

    def probability(self, family: Family, regime: str) -> float:
        downside = {Family.PUT_BWB}
        upside = {Family.CALL_BWB}
        long_premium = {Family.STRADDLE, Family.STRANGLE}
        if regime == "high_variance":
            favoured = downside | long_premium
        elif regime == "low_variance":
            favoured = upside | {Family.DEBIT_VERTICAL}
        else:
            return 0.5
        return self.favoured_probability if family in favoured else self.disfavoured_probability


def save_models(
    models: dict[Family, StructureModel],
    reports: Sequence[TrainingReport],
    directory: Path,
) -> Path:
    """Persist the fitted families and the report that justifies trusting them.

    The report is written next to the models on purpose. A model file on its
    own says nothing about whether it should be used; the hit rate, the Brier
    score and the calibration slope that were measured when it was fitted are
    what decide that, and they travel with it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "fitted_at": datetime.now(tz=timezone.utc).isoformat(),
        "families": {},
        "reports": [report.as_dict() for report in reports],
    }
    for family, model in models.items():
        payload["families"][str(family)] = {
            "feature_names": list(model.feature_names),
            "coefficients": model.model.coef_[0].tolist(),
            "intercept": float(model.model.intercept_[0]),
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
            "classes": model.model.classes_.tolist(),
        }
    path = directory / "models.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_models(directory: Path, config: Config) -> tuple[dict[Family, StructureModel], dict]:
    """Rebuild the fitted families from disk, or return nothing at all.

    A missing file is not an error: the agent runs the documented rule instead
    and says so in every record. A file that exists but cannot be read *is* an
    error, because silently falling back to the rule when a model was meant to
    be in use would misreport which one made the call.
    """
    path = directory / "models.json"
    if not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text())
    models: dict[Family, StructureModel] = {}
    for name, entry in payload.get("families", {}).items():
        family = Family(name)
        coefficients = np.asarray(entry["coefficients"], dtype=float)
        model = LogisticRegression(C=config.float_("classifier.l2_c"))
        # Restore the fitted state directly. Refitting on load would need the
        # training rows to still exist and would produce a different model on
        # a different day, which is not what "load the model" means.
        model.coef_ = coefficients.reshape(1, -1)
        model.intercept_ = np.asarray([entry["intercept"]], dtype=float)
        model.classes_ = np.asarray(entry["classes"])
        model.n_features_in_ = coefficients.size
        models[family] = StructureModel(
            family=family,
            feature_names=tuple(entry["feature_names"]),
            model=model,
            mean=np.asarray(entry["mean"], dtype=float),
            scale=np.asarray(entry["scale"], dtype=float),
        )
    return models, payload
