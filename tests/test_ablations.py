from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm.ablations import ABLATION_NAMES, ablate, ablation_presets, seeded_from
from brainevo.abm.presets import all_presets
from brainevo.abm.taxa import TAXA
from brainevo.ode.predictions import (
    ablation_table,
    haller_curve,
    minimal_circuit_bounds,
    reduction_vs_demand,
    reversibility,
)


@pytest.fixture(scope="module")
def table():
    return ablation_table().set_index(["taxon", "ablation"])


def test_every_ablation_builds_an_architecture() -> None:
    for taxon in TAXA.values():
        for name in ABLATION_NAMES:
            arch = ablate(taxon, name).build_architecture()
            assert np.isfinite(arch.E_f_saturation)
    assert len(ablation_presets()) >= 26
    assert set(ablation_presets()) <= set(all_presets())


def test_saturation_is_necessary_for_the_frog_loss(table) -> None:
    assert table.loc[("frog", "intact"), "neurons"] < 0.8 * table.loc[("frog", "intact"), "cap"]
    unsaturated = table.loc[("frog", "no-saturation")]
    assert unsaturated["neurons"] == pytest.approx(unsaturated["cap"], rel=0.02)


def test_minimal_circuit_is_necessary_for_any_valley(table) -> None:
    assert (table.xs("no-min-circuit", level="ablation")["deficit"] <= 1e-6).all()
    assert (table.xs("intact", level="ablation")["deficit"] > 0).all()


def test_oxygen_cap_is_necessary_for_the_insect_ceiling(table) -> None:
    assert table.loc[("arthropod", "no-oxygen-cap"), "neurons"] > 2 * table.loc[("arthropod", "intact"), "neurons"]


def test_thesis_prices_destroy_the_brain_share_ordering(table) -> None:
    thesis = table.xs("thesis-prices", level="ablation")
    assert (thesis["brain_share"] > 0.3).all()
    intact = table.xs("intact", level="ablation")["a_index"]
    assert intact["arthropod"] < intact["mouse"] < intact["primate"]
    assert thesis["a_index"].max() - thesis["a_index"].min() < 0.1


def test_reversibility_has_no_loop_under_grounded_costs() -> None:
    loop = reversibility(TAXA["mouse"], np.linspace(0.3, 3.0, 6))
    down = loop[loop["direction"] == "down"].set_index("M")["neurons"]
    up = loop[loop["direction"] == "up"].set_index("M")["neurons"]
    assert np.allclose(down.sort_index(), up.sort_index(), atol=0.02)


def test_haller_exponent_in_range_for_interior_optimum() -> None:
    _, slope = haller_curve(TAXA["frog"], np.logspace(-1, 1, 7))
    assert 0.6 < slope < 0.85


def test_reduction_and_minimal_circuit_tables() -> None:
    red = reduction_vs_demand(TAXA["mouse"], np.array([0.05, 1.0, 3.0]))
    assert red["fraction_retained"].iloc[-1] == pytest.approx(1.0)
    assert not red["back_end_present"].iloc[0]
    bounds = minimal_circuit_bounds()
    assert bounds["n_min_upper_bound"].dropna().max() < 0.05


def test_seeded_from_uses_ancestor_optimum() -> None:
    p = seeded_from(TAXA["mouse"], TAXA["frog"])
    assert p.init.arch_mode == "fixed_Eb" and p.init.initial_Eb > 0
