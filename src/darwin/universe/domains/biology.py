"""Biology domain seed — taxonomy + key processes + cellular structure."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Kingdoms.
        ("animal", "is_a", "organism", 1.0),
        ("plant", "is_a", "organism", 1.0),
        ("fungus", "is_a", "organism", 1.0),
        ("protist", "is_a", "organism", 1.0),
        ("bacterium", "is_a", "organism", 1.0),
        ("archaeon", "is_a", "organism", 1.0),
        # Animal taxonomy.
        ("mammal", "is_a", "animal", 1.0),
        ("bird", "is_a", "animal", 1.0),
        ("reptile", "is_a", "animal", 1.0),
        ("amphibian", "is_a", "animal", 1.0),
        ("fish", "is_a", "animal", 1.0),
        ("insect", "is_a", "animal", 1.0),
        ("arachnid", "is_a", "animal", 1.0),
        ("mollusk", "is_a", "animal", 1.0),
        # Mammals.
        ("dog", "is_a", "mammal", 1.0),
        ("cat", "is_a", "mammal", 1.0),
        ("horse", "is_a", "mammal", 1.0),
        ("whale", "is_a", "mammal", 1.0),
        ("human", "is_a", "mammal", 1.0),
        ("mouse", "is_a", "mammal", 1.0),
        ("bat", "is_a", "mammal", 1.0),
        ("primate", "is_a", "mammal", 1.0),
        ("human", "is_a", "primate", 1.0),
        ("chimpanzee", "is_a", "primate", 1.0),
        # Birds.
        ("eagle", "is_a", "bird", 1.0),
        ("sparrow", "is_a", "bird", 1.0),
        ("penguin", "is_a", "bird", 1.0),
        ("ostrich", "is_a", "bird", 1.0),
        # Plants.
        ("tree", "is_a", "plant", 1.0),
        ("flower", "is_a", "plant", 1.0),
        ("moss", "is_a", "plant", 1.0),
        ("fern", "is_a", "plant", 1.0),
        ("oak", "is_a", "tree", 1.0),
        ("pine", "is_a", "tree", 1.0),
        # Cellular structure.
        ("cell", "is_a", "biological_unit", 1.0),
        ("eukaryote", "is_a", "cell", 1.0),
        ("prokaryote", "is_a", "cell", 1.0),
        ("nucleus", "part_of", "eukaryote", 1.0),
        ("mitochondrion", "part_of", "eukaryote", 1.0),
        ("ribosome", "part_of", "cell", 1.0),
        ("membrane", "part_of", "cell", 1.0),
        ("cytoplasm", "part_of", "cell", 1.0),
        ("dna", "part_of", "nucleus", 1.0),
        ("chromosome", "part_of", "nucleus", 1.0),
        ("gene", "part_of", "chromosome", 1.0),
        # Tissues + organs.
        ("tissue", "part_of", "organ", 1.0),
        ("cell", "part_of", "tissue", 1.0),
        ("organ", "part_of", "organism", 1.0),
        ("heart", "is_a", "organ", 1.0),
        ("brain", "is_a", "organ", 1.0),
        ("lung", "is_a", "organ", 1.0),
        ("liver", "is_a", "organ", 1.0),
        ("kidney", "is_a", "organ", 1.0),
        ("skin", "is_a", "organ", 1.0),
        ("neuron", "is_a", "cell", 1.0),
        ("neuron", "part_of", "brain", 0.9),
        # Processes.
        ("photosynthesis", "is_a", "process", 1.0),
        ("respiration", "is_a", "process", 1.0),
        ("digestion", "is_a", "process", 1.0),
        ("reproduction", "is_a", "process", 1.0),
        ("mitosis", "is_a", "process", 1.0),
        ("meiosis", "is_a", "process", 1.0),
        ("evolution", "is_a", "process", 1.0),
        # Causal relations.
        ("photosynthesis", "requires", "sunlight", 1.0),
        ("photosynthesis", "requires", "water", 1.0),
        ("photosynthesis", "requires", "carbon_dioxide", 1.0),
        ("photosynthesis", "produces", "glucose", 1.0),
        ("photosynthesis", "produces", "oxygen", 1.0),
        ("respiration", "requires", "oxygen", 1.0),
        ("respiration", "produces", "carbon_dioxide", 1.0),
        ("respiration", "produces", "energy", 1.0),
        ("dna", "encodes", "gene", 1.0),
        ("gene", "expresses", "protein", 1.0),
        # Capabilities.
        ("bird", "can", "fly", 0.85),
        ("penguin", "cannot", "fly", 0.95),
        ("ostrich", "cannot", "fly", 0.95),
        ("fish", "can", "swim", 0.95),
        ("mammal", "has", "fur", 0.7),
        ("mammal", "has", "warm_blood", 0.95),
        ("reptile", "has", "cold_blood", 0.95),
    ]


__all__ = ["relations"]
