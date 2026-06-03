"""Geography domain seed — continents, countries, landforms, climates."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Continents.
        ("africa", "is_a", "continent", 1.0),
        ("antarctica", "is_a", "continent", 1.0),
        ("asia", "is_a", "continent", 1.0),
        ("australia", "is_a", "continent", 1.0),
        ("europe", "is_a", "continent", 1.0),
        ("north_america", "is_a", "continent", 1.0),
        ("south_america", "is_a", "continent", 1.0),
        # Selected countries.
        ("united_states", "is_a", "country", 1.0),
        ("canada", "is_a", "country", 1.0),
        ("mexico", "is_a", "country", 1.0),
        ("brazil", "is_a", "country", 1.0),
        ("argentina", "is_a", "country", 1.0),
        ("united_kingdom", "is_a", "country", 1.0),
        ("france", "is_a", "country", 1.0),
        ("germany", "is_a", "country", 1.0),
        ("italy", "is_a", "country", 1.0),
        ("spain", "is_a", "country", 1.0),
        ("china", "is_a", "country", 1.0),
        ("japan", "is_a", "country", 1.0),
        ("india", "is_a", "country", 1.0),
        ("egypt", "is_a", "country", 1.0),
        ("south_africa", "is_a", "country", 1.0),
        ("nigeria", "is_a", "country", 1.0),
        # Country → continent.
        ("united_states", "part_of", "north_america", 1.0),
        ("canada", "part_of", "north_america", 1.0),
        ("mexico", "part_of", "north_america", 1.0),
        ("brazil", "part_of", "south_america", 1.0),
        ("argentina", "part_of", "south_america", 1.0),
        ("united_kingdom", "part_of", "europe", 1.0),
        ("france", "part_of", "europe", 1.0),
        ("germany", "part_of", "europe", 1.0),
        ("italy", "part_of", "europe", 1.0),
        ("spain", "part_of", "europe", 1.0),
        ("china", "part_of", "asia", 1.0),
        ("japan", "part_of", "asia", 1.0),
        ("india", "part_of", "asia", 1.0),
        ("egypt", "part_of", "africa", 1.0),
        ("south_africa", "part_of", "africa", 1.0),
        ("nigeria", "part_of", "africa", 1.0),
        # Capitals.
        ("washington_dc", "is_capital_of", "united_states", 1.0),
        ("ottawa", "is_capital_of", "canada", 1.0),
        ("mexico_city", "is_capital_of", "mexico", 1.0),
        ("brasilia", "is_capital_of", "brazil", 1.0),
        ("buenos_aires", "is_capital_of", "argentina", 1.0),
        ("london", "is_capital_of", "united_kingdom", 1.0),
        ("paris", "is_capital_of", "france", 1.0),
        ("berlin", "is_capital_of", "germany", 1.0),
        ("rome", "is_capital_of", "italy", 1.0),
        ("madrid", "is_capital_of", "spain", 1.0),
        ("beijing", "is_capital_of", "china", 1.0),
        ("tokyo", "is_capital_of", "japan", 1.0),
        ("new_delhi", "is_capital_of", "india", 1.0),
        ("cairo", "is_capital_of", "egypt", 1.0),
        # Oceans + landforms.
        ("pacific_ocean", "is_a", "ocean", 1.0),
        ("atlantic_ocean", "is_a", "ocean", 1.0),
        ("indian_ocean", "is_a", "ocean", 1.0),
        ("arctic_ocean", "is_a", "ocean", 1.0),
        ("southern_ocean", "is_a", "ocean", 1.0),
        ("river", "is_a", "waterway", 1.0),
        ("lake", "is_a", "waterway", 1.0),
        ("sea", "is_a", "waterway", 1.0),
        ("mountain", "is_a", "landform", 1.0),
        ("plateau", "is_a", "landform", 1.0),
        ("valley", "is_a", "landform", 1.0),
        ("desert", "is_a", "landform", 1.0),
        ("plain", "is_a", "landform", 1.0),
        # Climates.
        ("tropical", "is_a", "climate", 1.0),
        ("temperate", "is_a", "climate", 1.0),
        ("arid", "is_a", "climate", 1.0),
        ("polar", "is_a", "climate", 1.0),
        ("mediterranean", "is_a", "climate", 1.0),
    ]


__all__ = ["relations"]
