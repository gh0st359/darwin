"""Chemistry domain seed — elements, compounds, reactions."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Matter hierarchy.
        ("element", "is_a", "substance", 1.0),
        ("compound", "is_a", "substance", 1.0),
        ("mixture", "is_a", "substance", 1.0),
        ("atom", "is_a", "particle", 1.0),
        ("molecule", "is_a", "particle", 1.0),
        ("ion", "is_a", "particle", 1.0),
        # Atomic structure.
        ("proton", "part_of", "nucleus", 1.0),
        ("neutron", "part_of", "nucleus", 1.0),
        ("electron", "part_of", "atom", 1.0),
        ("nucleus", "part_of", "atom", 1.0),
        ("quark", "part_of", "proton", 1.0),
        ("quark", "part_of", "neutron", 1.0),
        # Elements.
        ("hydrogen", "is_a", "element", 1.0),
        ("helium", "is_a", "element", 1.0),
        ("carbon", "is_a", "element", 1.0),
        ("nitrogen", "is_a", "element", 1.0),
        ("oxygen", "is_a", "element", 1.0),
        ("sodium", "is_a", "element", 1.0),
        ("chlorine", "is_a", "element", 1.0),
        ("iron", "is_a", "element", 1.0),
        ("gold", "is_a", "element", 1.0),
        ("silicon", "is_a", "element", 1.0),
        # Element classes.
        ("hydrogen", "is_a", "nonmetal", 1.0),
        ("carbon", "is_a", "nonmetal", 1.0),
        ("oxygen", "is_a", "nonmetal", 1.0),
        ("nitrogen", "is_a", "nonmetal", 1.0),
        ("iron", "is_a", "metal", 1.0),
        ("gold", "is_a", "metal", 1.0),
        ("copper", "is_a", "metal", 1.0),
        ("sodium", "is_a", "metal", 1.0),
        # Compounds.
        ("water", "is_a", "compound", 1.0),
        ("water", "composed_of", "hydrogen", 1.0),
        ("water", "composed_of", "oxygen", 1.0),
        ("salt", "is_a", "compound", 1.0),
        ("salt", "composed_of", "sodium", 1.0),
        ("salt", "composed_of", "chlorine", 1.0),
        ("carbon_dioxide", "is_a", "compound", 1.0),
        ("carbon_dioxide", "composed_of", "carbon", 1.0),
        ("carbon_dioxide", "composed_of", "oxygen", 1.0),
        ("methane", "is_a", "compound", 1.0),
        ("methane", "composed_of", "carbon", 1.0),
        ("methane", "composed_of", "hydrogen", 1.0),
        ("glucose", "is_a", "compound", 1.0),
        ("glucose", "is_a", "carbohydrate", 1.0),
        ("protein", "is_a", "compound", 1.0),
        ("protein", "composed_of", "amino_acid", 1.0),
        # Reactions.
        ("combustion", "is_a", "reaction", 1.0),
        ("oxidation", "is_a", "reaction", 1.0),
        ("reduction", "is_a", "reaction", 1.0),
        ("acid_base", "is_a", "reaction", 1.0),
        ("combustion", "requires", "oxygen", 1.0),
        ("combustion", "produces", "carbon_dioxide", 0.9),
        ("combustion", "produces", "heat", 1.0),
        # States.
        ("solid", "is_a", "state_of_matter", 1.0),
        ("liquid", "is_a", "state_of_matter", 1.0),
        ("gas", "is_a", "state_of_matter", 1.0),
        ("plasma", "is_a", "state_of_matter", 1.0),
        # Properties.
        ("acid", "has_property", "low_ph", 1.0),
        ("base", "has_property", "high_ph", 1.0),
        ("metal", "conducts", "electricity", 0.95),
        ("metal", "conducts", "heat", 0.95),
        ("salt", "dissolves_in", "water", 0.95),
    ]


__all__ = ["relations"]
