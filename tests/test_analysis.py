from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainevo.abm.cli import run_one
from brainevo.abm.presets import PRESETS
from brainevo.analysis import (
    bimodality_table,
    establishment_cells,
    exponential_hazard,
    fit_two_component,
    hazard_vs_deficit,
    load_runs,
)
from brainevo.analysis.stats import across_levels, bimodal_fraction, naive_vs_seeded


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("results")
    base = PRESETS["exp1-overhead-sweep"]
    for P in (0.5, 2.0):
        for mode in ("naive", "above_switch"):
            preset = base.with_overrides(cost__P=P, init__arch_mode=mode, init__n_founders=120)
            for seed in range(3):
                run_one(preset, seed, root / "thesis" / preset.name / f"P{P:g}_arch_mode{mode}" / f"seed{seed:03d}",
                        snapshot_every=20, progress=False)
    return root


def test_load_runs_flattens_overrides(fixture_root: Path) -> None:
    runs = load_runs(fixture_root, sweep="thesis")
    assert len(runs) == 12
    assert set(runs["cost__P"]) == {0.5, 2.0}
    assert set(runs["init__arch_mode"]) == {"naive", "above_switch"}
    assert runs["has_snapshots"].all()


def test_establishment_cells_and_hazard(fixture_root: Path) -> None:
    runs = load_runs(fixture_root, sweep="thesis")
    cells = establishment_cells(runs, ["cost__P"])
    assert len(cells) == 2
    assert set(cells.columns) >= {"p_establish", "p_retained", "hazard", "hazard_lo", "hazard_hi"}
    seeded_retention = cells["p_retained"]
    assert ((seeded_retention >= 0) & (seeded_retention <= 1)).all()
    cells, fit = hazard_vs_deficit(cells)
    assert "valley_deficit" in cells and (cells["valley_deficit"] >= 0).all()
    assert fit.n_cells <= 2


def test_exponential_hazard_censoring() -> None:
    rate, lo, hi = exponential_hazard(np.array([10, -1, -1]), np.array([100, 100, 100]))
    assert rate == pytest.approx(1 / 210)
    assert lo < rate < hi
    rate0, lo0, hi0 = exponential_hazard(np.array([-1, -1]), np.array([50, 50]))
    assert rate0 == 0.0 and lo0 == 0.0 and hi0 == pytest.approx(0.03)


def test_two_component_fit_separates_bimodal_from_unimodal() -> None:
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.lognormal(np.log(5), 0.1, 300), rng.lognormal(np.log(40), 0.1, 300)])
    unimodal = rng.lognormal(np.log(15), 0.3, 600)
    assert fit_two_component(bimodal).is_bimodal
    assert not fit_two_component(unimodal).is_bimodal


def test_bimodality_table_and_fraction(fixture_root: Path) -> None:
    runs = load_runs(fixture_root, sweep="thesis")
    fits = bimodality_table(runs.head(4))
    assert {"delta_bic", "ashman_d", "is_bimodal"} <= set(fits.columns)
    frac = bimodal_fraction(fits, ["tag"])
    assert ((frac["ci_low"] <= frac["fraction"]) & (frac["fraction"] <= frac["ci_high"])).all()


def test_rerandomstats_contrasts(fixture_root: Path) -> None:
    runs = load_runs(fixture_root, sweep="thesis")
    tests = naive_vs_seeded(runs, ["cost__P"])
    assert len(tests) == 2 and "p_Eb_fdr" in tests
    pairwise = across_levels(runs[runs["init__arch_mode"] == "naive"], "cost__P", "Eb")
    assert isinstance(pairwise, pd.DataFrame)
