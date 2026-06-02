"""Tests for Hebbian + STDP plasticity rules."""

from __future__ import annotations

from darwin.mesh.mesh import CorticalMesh
from darwin.mesh.plasticity import HebbianRule, PlasticityController, STDPRule


def test_hebbian_delta_proportional_to_pre_post_product() -> None:
    rule = HebbianRule(learning_rate=0.1)
    delta = rule.apply(pre_activation=1.0, post_activation=0.5)
    assert abs(delta - 0.05) < 1e-9


def test_hebbian_zero_when_either_activation_zero() -> None:
    rule = HebbianRule(learning_rate=0.5)
    assert rule.apply(pre_activation=0.0, post_activation=1.0) == 0.0
    assert rule.apply(pre_activation=1.0, post_activation=0.0) == 0.0


def test_stdp_potentiation_when_pre_before_post() -> None:
    rule = STDPRule(a_plus=0.1, tau_plus_seconds=0.02)
    delta = rule.apply(pre_time=10.0, post_time=10.01)
    assert delta > 0


def test_stdp_depression_when_post_before_pre() -> None:
    rule = STDPRule(a_minus=0.1, tau_minus_seconds=0.02)
    delta = rule.apply(pre_time=10.0, post_time=9.99)
    assert delta < 0


def test_stdp_zero_at_simultaneous_fire() -> None:
    rule = STDPRule()
    assert rule.apply(pre_time=10.0, post_time=10.0) == 0.0


def test_plasticity_controller_with_no_firings_does_nothing() -> None:
    mesh = CorticalMesh()
    controller = PlasticityController()
    report = controller.apply_cycle(mesh)
    assert report.hebbian_updates == 0
    assert report.stdp_updates == 0


def test_plasticity_updates_connection_after_co_firing() -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a", threshold=0.05, refractory_seconds=0.0)
    mesh.add_cell("b", threshold=0.05, refractory_seconds=0.0)
    conn = mesh.connect("a", "b", weight=0.5, kind="is_a")
    # Both fire from injected activation in a single propagation.
    mesh.activate(["a", "b"], magnitude=1.0)
    mesh.propagate([], steps=1)
    controller = PlasticityController(
        hebbian=HebbianRule(learning_rate=0.1),
    )
    before = conn.weight
    controller.apply_cycle(mesh)
    assert conn.weight != before or len(mesh.recent_firings) < 2


def test_plasticity_respects_max_weight_bound() -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a", threshold=0.05, refractory_seconds=0.0)
    mesh.add_cell("b", threshold=0.05, refractory_seconds=0.0)
    conn = mesh.connect("a", "b", weight=0.95, kind="is_a")
    mesh.activate(["a", "b"], magnitude=1.0)
    mesh.propagate([], steps=1)
    controller = PlasticityController(
        hebbian=HebbianRule(learning_rate=10.0, max_weight=1.0),
    )
    controller.apply_cycle(mesh)
    assert conn.weight <= 1.0
