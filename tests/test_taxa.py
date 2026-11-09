from __future__ import annotations

import pytest

from brainevo.abm.presets import all_presets
from brainevo.abm.taxa import TAXA, secondary_loss_preset, sources_table
from brainevo.ode.meanfield import MeanField

MINK_BRAIN_SHARE = {"mouse": (0.02, 0.08), "primate": (0.08, 0.25), "arthropod": (0.01, 0.08)}


def _optimum(name: str) -> tuple[float, float, float, float]:
    arch = TAXA[name].architecture()
    mixed = MeanField(arch).best_mixed()
    assert mixed is not None
    n = float(arch.neurons_for_cost(mixed.Eb))
    capsule = float(arch.optimal_capsule(mixed.Eb))
    return mixed.Ef, mixed.Eb, n, (mixed.Eb + capsule) / arch.basal


def test_back_end_share_ordering_matches_biology() -> None:
    a = {name: _optimum(name) for name in TAXA}
    share = {k: v[1] / (v[0] + v[1]) for k, v in a.items()}
    assert share["arthropod"] < share["mouse"] < share["primate"]
    assert share["frog"] < share["mouse"]


def test_realised_brain_shares_fall_in_measured_ranges() -> None:
    for name, (low, high) in MINK_BRAIN_SHARE.items():
        assert low <= _optimum(name)[3] <= high, name


def test_frog_loses_neurons_relative_to_ancestor() -> None:
    n_mouse, n_frog = _optimum("mouse")[2], _optimum("frog")[2]
    assert n_frog < 0.8 * n_mouse
    assert n_frog < TAXA["frog"].architecture().hard_cap  # a choice, not the cap


def test_secondary_loss_preset_seeds_at_ancestor_optimum() -> None:
    preset = secondary_loss_preset(TAXA["frog"])
    assert preset.init.arch_mode == "fixed_Eb"
    assert preset.init.initial_Eb == pytest.approx(_optimum("mouse")[1], rel=1e-6)
    assert preset.init.initial_Ec is not None and preset.init.initial_Ec > 0.0


def test_taxon_presets_are_registered() -> None:
    names = set(all_presets())
    assert {"taxon-arthropod", "taxon-mouse", "taxon-frog", "taxon-primate", "taxon-frog-from-mouse"} <= names


def test_every_decision_has_a_source() -> None:
    table = sources_table()
    for taxon in TAXA.values():
        assert all(len(src) > 20 for src in taxon.sources.values()), taxon.name
    assert table.count("|") > 60
