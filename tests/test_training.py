"""Recording chains, and learning from the recordings.

The archive exists because historical option quotes for a past 10:00 cannot be
fetched back: an expired contract's book is gone. If the snapshot a decision was
made on is not written down at the time, that day can never be labelled
honestly. These tests check the recording round-trips exactly, and that the
labels built from it are net of cost and free of look-ahead.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from convex import archive, training
from convex.agent import rank
from convex.config import load
from convex.errors import DataError
from convex.instruments import Right
from convex.scenarios import ScenarioSet
from convex.structures.base import Family
from tests.conftest import build_test_chain

TAKEN_AT = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)


def snapshot(day: date, spot: float = 650.0) -> archive.ChainSnapshot:
    return archive.ChainSnapshot(
        session_date=day,
        taken_at=TAKEN_AT.replace(year=day.year, month=day.month, day=day.day),
        spot=spot,
        expiry=day,
        entries=build_test_chain(spot=spot),
        cycle_id="c1",
    )


def scenarios() -> ScenarioSet:
    rng = np.random.default_rng(3)
    return ScenarioSet(
        log_returns=rng.normal(scale=0.006, size=400),
        source_days=tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(400)),
        entry_time=TAKEN_AT.time(),
        exit_time=datetime(2026, 8, 28, 16, 0).time(),
        volatility_scale=1.0,
        built_at=TAKEN_AT,
    )


# ---------------------------------------------------------------------- archive


def test_a_recorded_chain_round_trips_every_field_that_pricing_depends_on(tmp_path):
    original = snapshot(date(2026, 8, 28))
    archive.write(original, tmp_path)
    restored = archive.read(archive.path_for(tmp_path, date(2026, 8, 28)))

    assert restored.session_date == original.session_date
    assert restored.spot == original.spot
    assert len(restored.entries) == len(original.entries)
    for before, after in zip(original.entries, restored.entries):
        assert after.contract.symbol == before.contract.symbol
        assert after.contract.strike == before.contract.strike
        assert after.quote.bid == before.quote.bid
        assert after.quote.ask == before.quote.ask
        assert after.quote.mid == before.quote.mid
        assert after.greeks.implied_volatility == before.greeks.implied_volatility


def test_a_recorded_chain_is_evidence_and_is_never_overwritten(tmp_path):
    archive.write(snapshot(date(2026, 8, 28)), tmp_path)
    with pytest.raises(DataError, match="not rewritten"):
        archive.write(snapshot(date(2026, 8, 28)), tmp_path)


def test_an_empty_snapshot_is_refused_at_construction():
    with pytest.raises(DataError, match="no contracts"):
        archive.ChainSnapshot(
            session_date=date(2026, 8, 28), taken_at=TAKEN_AT, spot=650.0,
            expiry=date(2026, 8, 28), entries=[],
        )


def test_sessions_are_listed_oldest_first(tmp_path):
    for day in (date(2026, 8, 28), date(2026, 8, 26), date(2026, 8, 27)):
        archive.write(snapshot(day), tmp_path)
    assert archive.sessions(tmp_path) == [
        date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28)
    ]
    assert archive.sessions(tmp_path / "nothing-here") == []


# --------------------------------------------------------------------- labelling


def test_the_label_is_net_of_cost_not_gross():
    config = load()
    days = [date(2026, 8, 24) + timedelta(days=i) for i in range(3)]
    samples = training.build_samples(
        [snapshot(day) for day in days],
        {day: 651.0 for day in days},
        scenarios(),
        config,
        rank,
    )
    assert samples
    for sample in samples:
        # net_pnl is what the label is taken from, and it already has the
        # execution cost subtracted.
        assert sample.label == (1 if sample.net_pnl > 0 else 0)


def test_a_session_without_enough_history_behind_it_is_dropped_not_padded():
    config = load()
    day = date(2026, 8, 24)
    thin = ScenarioSet(
        log_returns=np.array([0.001, -0.002]),
        source_days=(date(2026, 8, 20), date(2026, 8, 21)),
        entry_time=TAKEN_AT.time(),
        exit_time=datetime(2026, 8, 28, 16, 0).time(),
        volatility_scale=1.0,
        built_at=TAKEN_AT,
    )
    assert training.build_samples([snapshot(day)], {day: 651.0}, thin, config, rank) == []


def test_only_sessions_before_the_labelled_day_enter_its_features():
    config = load()
    day = date(2026, 8, 24)
    # Every scenario day falls after the session being labelled, so none of
    # them may be used and the row cannot be built at all.
    future = ScenarioSet(
        log_returns=np.full(60, 0.001),
        source_days=tuple(date(2026, 9, 1) + timedelta(days=i) for i in range(60)),
        entry_time=TAKEN_AT.time(),
        exit_time=datetime(2026, 8, 28, 16, 0).time(),
        volatility_scale=1.0,
        built_at=TAKEN_AT,
    )
    assert training.build_samples([snapshot(day)], {day: 651.0}, future, config, rank) == []


def test_a_session_with_no_known_close_is_skipped_rather_than_guessed():
    config = load()
    days = [date(2026, 8, 24), date(2026, 8, 25)]
    samples = training.build_samples(
        [snapshot(day) for day in days],
        {days[0]: 651.0},          # the second session has no settlement
        scenarios(),
        config,
        rank,
    )
    assert {sample.session_date for sample in samples} == {days[0]}


def test_every_family_has_its_lagged_columns_from_the_first_session_on():
    config = load()
    day = date(2026, 8, 24)
    samples = training.build_samples([snapshot(day)], {day: 651.0}, scenarios(), config, rank)
    for sample in samples:
        for name in training.feature_names_for(sample.family):
            assert name in sample.features, f"{sample.family} is missing {name}"


def test_a_family_sees_its_own_past_results_and_nobody_elses():
    names = training.feature_names_for(Family.PUT_BWB)
    assert "put_bwb_pnl_mean5" in names
    assert not any(name.startswith("call_bwb") for name in names)


def test_the_matrix_is_built_in_session_order_with_a_fixed_width():
    config = load()
    days = [date(2026, 8, 24) + timedelta(days=i) for i in range(4)]
    samples = training.build_samples(
        [snapshot(day) for day in days],
        {day: 651.0 for day in days},
        scenarios(), config, rank,
    )
    matrix, labels = training.to_matrix(samples, Family.PUT_BWB)
    assert matrix.shape[1] == len(training.feature_names_for(Family.PUT_BWB))
    assert matrix.shape[0] == labels.size


def test_a_family_with_no_rows_gives_an_empty_matrix_rather_than_raising():
    matrix, labels = training.to_matrix([], Family.PUT_BWB)
    assert matrix.shape[0] == 0 and labels.size == 0


# ------------------------------------------------------------------- settlement


def test_the_broken_wing_settlement_matches_the_closed_form(put_bwb_legs):
    from convex.structures.base import Candidate

    candidate = Candidate(
        family=Family.PUT_BWB, legs=tuple(put_bwb_legs), description="test bwb"
    )
    # +650P / -2x645P / +635P entered for a 0.20 credit.
    assert training.settlement_pnl_of(candidate, -0.20, 660.0) == pytest.approx(20.0)
    assert training.settlement_pnl_of(candidate, -0.20, 600.0) == pytest.approx(-480.0)


# ------------------------------------------------------- the whole path at once


def test_recorded_chains_become_fitted_models_the_agent_can_use(tmp_path):
    """Archive, label, fit out-of-sample, save, reload, predict.

    This is the loop the project depends on: the agent records what it saw, the
    trainer learns from the recordings, and the next cycle loads the result. If
    any link is broken the agent silently runs the fallback rule forever, which
    would still trade — just never with a model, while claiming it might.
    """
    from convex.classifier import fit_family, load_models, save_models, walk_forward

    config = load()
    start = date(2026, 3, 2)
    days, closes = [], {}
    # Enough calendar days to clear the 60-session burn-in on weekdays alone.
    for index in range(110):
        day = start + timedelta(days=index)
        if day.weekday() >= 5:
            continue
        days.append(day)
        # Spot drifts and the settlement varies around it, so the labels are
        # not all one class and the fit is a real fit.
        spot = 640.0 + (index % 11)
        # A put broken-wing butterfly keeps its credit when the underlying
        # holds up and loses when it falls through the wings, so the labels
        # only carry information if the settlements do both.
        closes[day] = spot + (2.5 if index % 3 else -14.0)
        archive.write(snapshot(day, spot=spot), tmp_path)

    assert archive.sessions(tmp_path) == days
    snapshots = list(archive.read_all(tmp_path))
    samples = training.build_samples(snapshots, closes, scenarios(), config, rank)
    assert {sample.session_date for sample in samples} == set(days)

    family = Family.PUT_BWB
    matrix, labels = training.to_matrix(samples, family)
    names = training.feature_names_for(family)
    assert matrix.shape == (len(days), len(names))
    assert set(labels.tolist()) == {0, 1}, "labels are degenerate; the fit would be meaningless"

    model, report = fit_family(family, matrix, labels, names, config)
    assert model is not None and report.trained
    assert report.samples == len(days)

    probabilities, realised = walk_forward(family, matrix, labels, names, config)
    assert probabilities.size > 0
    assert realised.size == probabilities.size
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    models_dir = tmp_path / "models"
    save_models({family: model}, [report], models_dir)
    restored, payload = load_models(models_dir, config)
    assert set(restored) == {family}
    assert payload["reports"][0]["family"] == str(family)

    probe = matrix[-1]
    assert restored[family].probability(probe) == model.probability(probe)
    # The reloaded model must accept exactly the row the live feature engine
    # produces, by name, or the agent would fall back without saying why.
    assert restored[family].feature_names == names


def test_a_family_below_the_burn_in_is_not_fitted_at_all(tmp_path):
    from convex.classifier import fit_family

    config = load()
    days = [date(2026, 3, 2) + timedelta(days=i) for i in range(6)]
    closes = {day: 651.0 for day in days}
    for day in days:
        archive.write(snapshot(day), tmp_path)
    samples = training.build_samples(
        list(archive.read_all(tmp_path)), closes, scenarios(), config, rank
    )
    matrix, labels = training.to_matrix(samples, Family.PUT_BWB)
    model, report = fit_family(
        Family.PUT_BWB, matrix, labels, training.feature_names_for(Family.PUT_BWB), config
    )
    assert model is None
    assert not report.trained
    assert str(config.int_("classifier.min_train_days")) in report.note
