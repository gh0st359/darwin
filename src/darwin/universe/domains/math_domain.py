"""Math domain seed — number types, operations, structures."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Number hierarchy.
        ("natural_number", "is_a", "number", 1.0),
        ("integer", "is_a", "number", 1.0),
        ("rational", "is_a", "number", 1.0),
        ("irrational", "is_a", "number", 1.0),
        ("real", "is_a", "number", 1.0),
        ("complex", "is_a", "number", 1.0),
        ("natural_number", "is_a", "integer", 1.0),
        ("integer", "is_a", "rational", 1.0),
        ("rational", "is_a", "real", 1.0),
        ("irrational", "is_a", "real", 1.0),
        ("real", "is_a", "complex", 1.0),
        # Special numbers.
        ("pi", "is_a", "irrational", 1.0),
        ("e", "is_a", "irrational", 1.0),
        ("zero", "is_a", "integer", 1.0),
        ("one", "is_a", "integer", 1.0),
        # Operations.
        ("addition", "is_a", "operation", 1.0),
        ("subtraction", "is_a", "operation", 1.0),
        ("multiplication", "is_a", "operation", 1.0),
        ("division", "is_a", "operation", 1.0),
        ("exponentiation", "is_a", "operation", 1.0),
        ("logarithm", "is_a", "operation", 1.0),
        ("addition", "inverse_of", "subtraction", 1.0),
        ("subtraction", "inverse_of", "addition", 1.0),
        ("multiplication", "inverse_of", "division", 1.0),
        ("exponentiation", "inverse_of", "logarithm", 1.0),
        # Properties.
        ("addition", "has_property", "commutative", 1.0),
        ("multiplication", "has_property", "commutative", 1.0),
        ("addition", "has_property", "associative", 1.0),
        ("multiplication", "has_property", "associative", 1.0),
        ("subtraction", "lacks_property", "commutative", 1.0),
        ("division", "lacks_property", "commutative", 1.0),
        # Structures.
        ("group", "is_a", "algebraic_structure", 1.0),
        ("ring", "is_a", "algebraic_structure", 1.0),
        ("field", "is_a", "algebraic_structure", 1.0),
        ("vector_space", "is_a", "algebraic_structure", 1.0),
        ("ring", "is_a", "group", 1.0),
        ("field", "is_a", "ring", 1.0),
        # Geometry.
        ("triangle", "is_a", "polygon", 1.0),
        ("quadrilateral", "is_a", "polygon", 1.0),
        ("pentagon", "is_a", "polygon", 1.0),
        ("hexagon", "is_a", "polygon", 1.0),
        ("square", "is_a", "quadrilateral", 1.0),
        ("rectangle", "is_a", "quadrilateral", 1.0),
        ("rhombus", "is_a", "quadrilateral", 1.0),
        ("square", "is_a", "rectangle", 1.0),
        ("square", "is_a", "rhombus", 1.0),
        ("circle", "is_a", "shape", 1.0),
        ("polygon", "is_a", "shape", 1.0),
        # Calculus.
        ("derivative", "is_a", "operation", 1.0),
        ("integral", "is_a", "operation", 1.0),
        ("derivative", "inverse_of", "integral", 1.0),
        ("limit", "is_a", "concept", 1.0),
        ("continuity", "is_a", "property", 1.0),
        ("differentiability", "implies", "continuity", 1.0),
        # Theorems.
        ("pythagorean_theorem", "is_a", "theorem", 1.0),
        ("fundamental_theorem_of_calculus", "is_a", "theorem", 1.0),
        ("godel_incompleteness", "is_a", "theorem", 1.0),
    ]


__all__ = ["relations"]
