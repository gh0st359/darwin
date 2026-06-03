"""Physics domain seed — forces, particles, energy, fundamental relations."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Forces.
        ("force", "is_a", "physical_quantity", 1.0),
        ("gravity", "is_a", "force", 1.0),
        ("electromagnetism", "is_a", "force", 1.0),
        ("strong_force", "is_a", "force", 1.0),
        ("weak_force", "is_a", "force", 1.0),
        ("friction", "is_a", "force", 1.0),
        ("tension", "is_a", "force", 1.0),
        ("normal_force", "is_a", "force", 1.0),
        # Particles.
        ("photon", "is_a", "particle", 1.0),
        ("electron", "is_a", "particle", 1.0),
        ("proton", "is_a", "particle", 1.0),
        ("neutron", "is_a", "particle", 1.0),
        ("quark", "is_a", "particle", 1.0),
        ("lepton", "is_a", "particle", 1.0),
        ("boson", "is_a", "particle", 1.0),
        ("fermion", "is_a", "particle", 1.0),
        ("electron", "is_a", "lepton", 1.0),
        ("electron", "is_a", "fermion", 1.0),
        ("photon", "is_a", "boson", 1.0),
        # Energy.
        ("kinetic_energy", "is_a", "energy", 1.0),
        ("potential_energy", "is_a", "energy", 1.0),
        ("thermal_energy", "is_a", "energy", 1.0),
        ("electromagnetic_energy", "is_a", "energy", 1.0),
        ("nuclear_energy", "is_a", "energy", 1.0),
        ("mechanical_energy", "is_a", "energy", 1.0),
        # Waves + light.
        ("wave", "is_a", "physical_phenomenon", 1.0),
        ("light", "is_a", "wave", 1.0),
        ("sound", "is_a", "wave", 1.0),
        ("radio_wave", "is_a", "wave", 1.0),
        ("microwave", "is_a", "wave", 1.0),
        ("light", "is_a", "electromagnetic_radiation", 1.0),
        ("photon", "carries", "light", 1.0),
        # Causal relations.
        ("gravity", "causes", "weight", 1.0),
        ("gravity", "attracts", "mass", 1.0),
        ("friction", "causes", "heat", 1.0),
        ("friction", "opposes", "motion", 1.0),
        ("acceleration", "causes", "change_in_velocity", 1.0),
        # Conservation.
        ("energy", "conserved_in", "closed_system", 1.0),
        ("momentum", "conserved_in", "closed_system", 1.0),
        ("charge", "conserved_in", "interaction", 1.0),
        # Units.
        ("meter", "measures", "length", 1.0),
        ("kilogram", "measures", "mass", 1.0),
        ("second", "measures", "time", 1.0),
        ("ampere", "measures", "current", 1.0),
        ("kelvin", "measures", "temperature", 1.0),
        ("newton", "measures", "force", 1.0),
        ("joule", "measures", "energy", 1.0),
        ("watt", "measures", "power", 1.0),
        # Relativity.
        ("speed_of_light", "is_a", "constant", 1.0),
        ("speed_of_light", "limits", "information_propagation", 1.0),
        ("mass", "curves", "spacetime", 0.9),
        # Quantum.
        ("photon", "exhibits", "wave_particle_duality", 1.0),
        ("electron", "exhibits", "wave_particle_duality", 1.0),
        ("uncertainty_principle", "constrains", "measurement", 1.0),
    ]


__all__ = ["relations"]
