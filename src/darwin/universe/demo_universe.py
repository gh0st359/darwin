"""OPT-IN demonstration seed — NOT loaded by default.

This module ships an encyclopedic concept graph spanning physics, math,
chemistry, biology, mind, language, arts, philosophy, and computing. It is
**hardcoded knowledge** and exists only as a demo / regression fixture so
that someone can quickly inspect what a fully-populated universe looks like
without spending days teaching Darwin from scratch.

The default brain does NOT call ``demo_seed_universe()``. The default
universe starts with the tiny ``primitive_seed`` (meta-vocabulary like
``thing``, ``change``, ``cause``, ``same``, ``different`` — the structural
operators Darwin needs to form its OWN concepts) and grows from there:
words from chat become candidate concepts via the LanguageGrounder; the
ConceptDeriver proposes new concepts from observed regularities; the
ConceptComposer derives new ones by combining existing nodes. Domain
knowledge — physics, math, music, anything — is meant to emerge from use,
not be pre-baked.

If you want a head start for a demo, call ``demo_seed_universe(universe)``
explicitly. You will see hardcoded relations like "gravity is_a force"
because they were typed in by hand. That is the point of the demo seed.
"""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse


def demo_seed_universe(universe: ConceptUniverse) -> ConceptUniverse:
    """Populate a universe with the foundational concept graph. Idempotent."""

    _seed_domains(universe)
    _seed_physics(universe)
    _seed_mathematics(universe)
    _seed_chemistry(universe)
    _seed_biology(universe)
    _seed_mind(universe)
    _seed_language(universe)
    _seed_arts(universe)
    _seed_philosophy(universe)
    _seed_computing(universe)
    _seed_cross_domain(universe)
    return universe


# --------------------------------------------------------------------------- #
# Domain registration
# --------------------------------------------------------------------------- #


def _seed_domains(u: ConceptUniverse) -> None:
    u.add_domain("physics", "The study of matter, energy, and their interactions.")
    u.add_domain("mathematics", "The study of structure, quantity, and pattern.")
    u.add_domain("chemistry", "The study of substance, transformation, and bond.")
    u.add_domain("biology", "The study of life and living systems.")
    u.add_domain("mind", "Consciousness, cognition, and the nature of thought.")
    u.add_domain("language", "The structure and meaning of symbolic communication.")
    u.add_domain("arts", "Forms of expression that organize perception and feeling.")
    u.add_domain("philosophy", "Foundations of reasoning, ethics, and being.")
    u.add_domain("computing", "Algorithms, computation, and information.")


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #


def _seed_physics(u: ConceptUniverse) -> None:
    d = "physics"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    # Foundational
    add("matter", "Anything with mass that occupies volume.", aliases=("substance",))
    add("energy", "The capacity of a system to do work or produce change.")
    add("force", "An interaction that changes the motion of an object.")
    add("mass", "A measure of an object's resistance to acceleration.")
    add("time", "The dimension along which change is measured.")
    add("space", "The three-dimensional extent in which matter is located.")
    add("spacetime", "The unified four-dimensional manifold of space and time.", depth=1)
    add("momentum", "The product of mass and velocity.", depth=1)
    add("velocity", "Rate of change of position with direction.", depth=1)
    add("acceleration", "Rate of change of velocity.", depth=1)
    add("work", "Energy transferred when a force acts over a distance.", depth=1)
    add("power", "Rate at which work is done.", depth=1)
    # Forces and fields
    add("gravity", "The attractive force between objects with mass.", depth=1)
    add("electromagnetism", "The force between charged particles.", depth=1)
    add("strong_force", "The force binding quarks into hadrons.", depth=2)
    add("weak_force", "The force responsible for radioactive decay.", depth=2)
    add("field", "A region in which a force acts at every point.")
    add("electric_field", "Field around charge.", depth=1)
    add("magnetic_field", "Field produced by moving charge.", depth=1)
    # Matter constituents
    add("particle", "A localized excitation of a field.")
    add("wave", "A propagating disturbance carrying energy.")
    add("photon", "The quantum of the electromagnetic field.", depth=2)
    add("electron", "A negatively charged elementary particle.", depth=1)
    add("proton", "A positively charged nucleon.", depth=1)
    add("neutron", "An electrically neutral nucleon.", depth=1)
    add("quark", "Elementary constituents of protons and neutrons.", depth=2)
    add("atom", "The smallest unit of a chemical element.", depth=1)
    add("nucleus", "The dense core of an atom.", depth=1)
    # Macroscopic / thermodynamic
    add("temperature", "A measure of average kinetic energy.", depth=1)
    add("heat", "Energy transferred due to temperature difference.", depth=1)
    add("entropy", "A measure of disorder in a system.", depth=2)
    add("pressure", "Force per unit area.", depth=1)
    add("density", "Mass per unit volume.", depth=1)
    # Relativistic / quantum
    add("relativity", "Einstein's theory unifying space, time, and gravity.", depth=2)
    add("special_relativity", "Relativity for inertial frames.", depth=2)
    add("general_relativity", "Relativity that includes gravity as curvature.", depth=3)
    add("quantum_mechanics", "The physics of the very small.", depth=2)
    add("superposition", "A quantum state that is a sum of basis states.", depth=3)
    add("entanglement", "Correlation between separated quantum systems.", depth=3)
    add("uncertainty_principle", "Trade-off between conjugate observables.", depth=3)
    # Cosmology
    add("big_bang", "The expansion event at the origin of the observable universe.", depth=3)
    add("cosmology", "The study of the universe at the largest scales.", depth=2)
    add("black_hole", "A region of spacetime from which nothing escapes.", depth=3)
    add("dark_matter", "Non-luminous matter inferred from gravity.", depth=3)
    add("dark_energy", "The accelerator of cosmic expansion.", depth=3)

    rels = [
        ("matter", "composes", "atom"),
        ("atom", "part_of", "matter"),
        ("atom", "composes", "nucleus"),
        ("nucleus", "part_of", "atom"),
        ("nucleus", "composes", "proton"),
        ("nucleus", "composes", "neutron"),
        ("proton", "composes", "quark"),
        ("neutron", "composes", "quark"),
        ("electron", "part_of", "atom"),
        ("photon", "is_a", "particle"),
        ("electron", "is_a", "particle"),
        ("wave", "analogous_to", "particle"),
        ("particle", "analogous_to", "wave"),
        ("energy", "related_to", "matter"),
        ("matter", "derives_from", "energy"),
        ("force", "causes", "acceleration"),
        ("mass", "requires", "matter"),
        ("velocity", "composes", "momentum"),
        ("mass", "composes", "momentum"),
        ("force", "describes", "interaction"),
        ("gravity", "is_a", "force"),
        ("electromagnetism", "is_a", "force"),
        ("strong_force", "is_a", "force"),
        ("weak_force", "is_a", "force"),
        ("electric_field", "is_a", "field"),
        ("magnetic_field", "is_a", "field"),
        ("field", "expresses", "force"),
        ("gravity", "causes", "acceleration"),
        ("gravity", "related_to", "spacetime"),
        ("spacetime", "is_a", "space"),
        ("spacetime", "related_to", "time"),
        ("general_relativity", "describes", "gravity"),
        ("special_relativity", "is_a", "relativity"),
        ("general_relativity", "is_a", "relativity"),
        ("quantum_mechanics", "describes", "particle"),
        ("superposition", "part_of", "quantum_mechanics"),
        ("entanglement", "part_of", "quantum_mechanics"),
        ("uncertainty_principle", "part_of", "quantum_mechanics"),
        ("temperature", "related_to", "energy"),
        ("heat", "is_a", "energy"),
        ("entropy", "opposes", "order"),
        ("entropy", "related_to", "temperature"),
        ("pressure", "related_to", "force"),
        ("density", "related_to", "mass"),
        ("density", "related_to", "matter"),
        ("work", "is_a", "energy"),
        ("power", "related_to", "work"),
        ("power", "related_to", "time"),
        ("big_bang", "causes", "spacetime"),
        ("black_hole", "instantiates", "general_relativity"),
        ("cosmology", "describes", "spacetime"),
        ("dark_matter", "related_to", "gravity"),
        ("dark_energy", "causes", "expansion"),
    ]
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Mathematics
# --------------------------------------------------------------------------- #


def _seed_mathematics(u: ConceptUniverse) -> None:
    d = "mathematics"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("number", "An abstract object used to count, measure, or label.")
    add("natural_number", "A non-negative integer.", depth=1)
    add("integer", "A whole number, positive negative or zero.", depth=1)
    add("rational", "A ratio of two integers.", depth=2)
    add("real_number", "Any point on the continuous number line.", depth=2)
    add("irrational", "A real number that is not rational.", depth=2)
    add("complex_number", "A number with a real and imaginary part.", depth=3)
    add("infinity", "Without bound; larger than any finite quantity.", depth=2)
    add("zero", "The additive identity.")
    add("one", "The multiplicative identity.")
    add("set", "A collection of distinct objects.")
    add("function", "A mapping from inputs to outputs.")
    add("equation", "A statement that two expressions are equal.")
    add("variable", "A symbol that stands for an unknown or varying value.")
    add("operation", "A rule that combines inputs into an output.")
    add("addition", "Combining quantities by accumulation.", depth=1)
    add("multiplication", "Repeated addition; scaling.", depth=1)
    add("ratio", "A relation showing the size of one quantity to another.", depth=1)
    add("proportion", "Equality of two ratios.", depth=2)
    add("derivative", "The instantaneous rate of change of a function.", depth=3)
    add("integral", "The accumulated area under a function.", depth=3)
    add("calculus", "The mathematics of change and accumulation.", depth=3)
    add("algebra", "The study of operations on symbols.", depth=2)
    add("geometry", "The study of shape, size, and space.", depth=2)
    add("topology", "The study of properties preserved under deformation.", depth=3)
    add("logic", "The study of valid inference.")
    add("proof", "A demonstration that a statement follows from premises.", depth=2)
    add("theorem", "A proven mathematical statement.", depth=2)
    add("axiom", "A statement accepted without proof.")
    add("vector", "A quantity with magnitude and direction.", depth=2)
    add("matrix", "A rectangular array of numbers.", depth=2)
    add("group", "A set with an associative invertible operation.", depth=3)
    add("ring", "A set with addition and multiplication.", depth=3)
    add("field_math", "A ring where division by nonzero is possible.", depth=3)
    add("probability", "A numerical measure of uncertainty.", depth=2)
    add("statistics", "The study of data and uncertainty.", depth=2)
    add("information", "A reduction in uncertainty.", depth=2)

    rels = [
        ("natural_number", "is_a", "integer"),
        ("integer", "is_a", "rational"),
        ("rational", "is_a", "real_number"),
        ("irrational", "is_a", "real_number"),
        ("real_number", "is_a", "complex_number"),
        ("zero", "is_a", "natural_number"),
        ("one", "is_a", "natural_number"),
        ("number", "instantiates", "natural_number"),
        ("addition", "is_a", "operation"),
        ("multiplication", "is_a", "operation"),
        ("ratio", "describes", "proportion"),
        ("proportion", "requires", "ratio"),
        ("derivative", "is_a", "operation"),
        ("integral", "opposes", "derivative"),
        ("calculus", "composes", "derivative"),
        ("calculus", "composes", "integral"),
        ("algebra", "describes", "operation"),
        ("geometry", "describes", "shape"),
        ("topology", "derives_from", "geometry"),
        ("proof", "requires", "logic"),
        ("theorem", "requires", "proof"),
        ("theorem", "derives_from", "axiom"),
        ("axiom", "part_of", "logic"),
        ("vector", "composes", "matrix"),
        ("group", "requires", "operation"),
        ("ring", "is_a", "group"),
        ("field_math", "is_a", "ring"),
        ("probability", "is_a", "function"),
        ("statistics", "requires", "probability"),
        ("information", "measured_by", "probability"),
        ("information", "opposes", "uncertainty"),
        ("function", "requires", "variable"),
        ("equation", "requires", "variable"),
        ("equation", "expresses", "relation"),
        ("set", "composes", "function"),
        ("infinity", "opposes", "finite"),
    ]
    u.add_concept("shape", domain=d, definition="A bounded form in space.")
    u.add_concept("relation", domain=d, definition="A correspondence between objects.")
    u.add_concept("finite", domain=d, definition="Bounded; not infinite.")
    u.add_concept("uncertainty", domain="mind", definition="Lack of definite knowledge.")
    u.add_concept("order", domain="mind", definition="An arrangement showing pattern.")
    u.add_concept("expansion", domain="physics", definition="Increase in extent.")
    u.add_concept("interaction", domain="physics", definition="Mutual influence between systems.")
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Chemistry
# --------------------------------------------------------------------------- #


def _seed_chemistry(u: ConceptUniverse) -> None:
    d = "chemistry"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("element", "A pure substance defined by its atomic number.")
    add("molecule", "A bound group of atoms.")
    add("bond", "A force holding atoms together in a molecule.")
    add("covalent_bond", "A bond formed by sharing electrons.", depth=1)
    add("ionic_bond", "A bond formed by electron transfer.", depth=1)
    add("hydrogen_bond", "A weak attractive bond involving hydrogen.", depth=1)
    add("reaction", "A transformation in which bonds break and form.")
    add("acid", "A substance that donates protons.", depth=1)
    add("base", "A substance that accepts protons.", depth=1)
    add("ph", "A scale of acidity from 0 to 14.", depth=1)
    add("oxidation", "Loss of electrons.", depth=2)
    add("reduction", "Gain of electrons.", depth=2)
    add("catalyst", "A substance that accelerates a reaction.", depth=2)
    add("enzyme", "A biological catalyst.", depth=2)
    add("solvent", "A substance that dissolves others.")
    add("solute", "A substance dissolved in a solvent.")
    add("solution", "A homogeneous mixture.", depth=1)
    add("periodic_table", "The classification of chemical elements.", depth=2)
    add("water", "Two hydrogen atoms bonded to one oxygen.", depth=1)
    add("carbon", "The element underlying organic chemistry.", depth=1)
    add("oxygen", "The element supporting most combustion and respiration.", depth=1)
    add("hydrogen", "The lightest element.", depth=1)
    add("nitrogen", "Inert element comprising most of Earth's atmosphere.", depth=1)
    add("combustion", "Rapid oxidation that releases heat.", depth=2)
    add("organic_chemistry", "The chemistry of carbon-containing compounds.", depth=2)
    add("polymer", "A long chain of repeating molecular units.", depth=2)

    rels = [
        ("element", "composes", "atom"),
        ("atom", "composes", "molecule"),
        ("bond", "part_of", "molecule"),
        ("covalent_bond", "is_a", "bond"),
        ("ionic_bond", "is_a", "bond"),
        ("hydrogen_bond", "is_a", "bond"),
        ("reaction", "causes", "bond"),
        ("acid", "opposes", "base"),
        ("ph", "measured_by", "acid"),
        ("ph", "measured_by", "base"),
        ("oxidation", "opposes", "reduction"),
        ("catalyst", "causes", "reaction"),
        ("enzyme", "is_a", "catalyst"),
        ("solvent", "part_of", "solution"),
        ("solute", "part_of", "solution"),
        ("water", "is_a", "solvent"),
        ("water", "is_a", "molecule"),
        ("water", "instantiates", "covalent_bond"),
        ("water", "composes", "hydrogen"),
        ("water", "composes", "oxygen"),
        ("carbon", "is_a", "element"),
        ("oxygen", "is_a", "element"),
        ("hydrogen", "is_a", "element"),
        ("nitrogen", "is_a", "element"),
        ("periodic_table", "describes", "element"),
        ("combustion", "is_a", "reaction"),
        ("combustion", "requires", "oxygen"),
        ("combustion", "expresses", "energy"),
        ("organic_chemistry", "requires", "carbon"),
        ("polymer", "is_a", "molecule"),
    ]
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Biology
# --------------------------------------------------------------------------- #


def _seed_biology(u: ConceptUniverse) -> None:
    d = "biology"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("life", "A self-sustaining metabolic and reproductive system.")
    add("cell", "The basic unit of life.")
    add("organism", "A complete living individual.", depth=1)
    add("dna", "The molecule carrying genetic information.", depth=1)
    add("rna", "A nucleic acid that carries genetic messages.", depth=1)
    add("gene", "A segment of DNA encoding a heritable trait.", depth=1)
    add("protein", "A folded chain of amino acids that does cellular work.", depth=1)
    add("amino_acid", "The monomer of proteins.", depth=1)
    add("evolution", "Change in heritable traits over generations.", depth=2)
    add("mutation", "A change in a gene's sequence.", depth=2)
    add("selection", "Differential reproductive success.", depth=2)
    add("species", "An interbreeding population.", depth=2)
    add("ecosystem", "A community of organisms and their environment.", depth=2)
    add("metabolism", "The sum of life-sustaining chemical reactions.", depth=2)
    add("photosynthesis", "Conversion of light energy into chemical energy.", depth=2)
    add("respiration", "The release of energy from food.", depth=2)
    add("brain", "The organ that produces cognition.", depth=2)
    add("neuron", "A cell that conducts electrical signals.", depth=2)
    add("synapse", "The junction between two neurons.", depth=3)
    add("network", "A set of nodes connected by edges.")
    add("neural_network", "A network of neurons.", depth=3)

    rels = [
        ("cell", "part_of", "organism"),
        ("organism", "instantiates", "life"),
        ("dna", "part_of", "cell"),
        ("rna", "derives_from", "dna"),
        ("gene", "part_of", "dna"),
        ("gene", "describes", "protein"),
        ("amino_acid", "composes", "protein"),
        ("protein", "part_of", "cell"),
        ("mutation", "causes", "evolution"),
        ("selection", "causes", "evolution"),
        ("evolution", "describes", "life"),
        ("species", "composes", "ecosystem"),
        ("organism", "composes", "species"),
        ("metabolism", "part_of", "life"),
        ("photosynthesis", "is_a", "metabolism"),
        ("respiration", "is_a", "metabolism"),
        ("respiration", "requires", "oxygen"),
        ("photosynthesis", "expresses", "energy"),
        ("brain", "part_of", "organism"),
        ("neuron", "composes", "brain"),
        ("synapse", "part_of", "neural_network"),
        ("neuron", "composes", "neural_network"),
        ("neural_network", "is_a", "network"),
        ("neural_network", "describes", "brain"),
    ]
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Mind
# --------------------------------------------------------------------------- #


def _seed_mind(u: ConceptUniverse) -> None:
    d = "mind"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("consciousness", "The subjective quality of experience.")
    add("self", "The subject of experience; the experiencer.", depth=1)
    add("awareness", "Knowing that one is experiencing.", depth=1)
    add("attention", "Selective focus of cognitive resources.", depth=1)
    add("perception", "The conversion of sensation into experience.", depth=1)
    add("sensation", "Raw stimulation of a sensory channel.", depth=1)
    add("memory", "Storage and retrieval of past experience.", depth=1)
    add("episodic_memory", "Memory of specific events.", depth=2)
    add("semantic_memory", "Memory of facts and meanings.", depth=2)
    add("procedural_memory", "Memory of skills and procedures.", depth=2)
    add("thought", "A discrete cognitive event.")
    add("reasoning", "Drawing conclusions from premises.", depth=2)
    add("emotion", "An evaluative bodily and cognitive state.")
    add("intention", "A goal-directed mental state.", depth=2)
    add("agency", "The capacity to act in pursuit of goals.", depth=2)
    add("belief", "A representation taken to be true.")
    add("knowledge", "Justified true belief.", depth=2)
    add("metacognition", "Cognition about cognition.", depth=3)
    add("theory_of_mind", "The capacity to model other minds.", depth=3)
    add("self_model", "An internal representation of oneself.", depth=3)
    add("curiosity", "Drive to reduce uncertainty about the world.", depth=2)
    add("learning", "Updating beliefs from experience.", depth=2)
    add("simulation_mind", "Internal modeling of counterfactual situations.", depth=3)
    add("imagination", "Mental construction of unobserved scenarios.", depth=2)

    rels = [
        ("self", "part_of", "consciousness"),
        ("awareness", "requires", "consciousness"),
        ("attention", "is_a", "awareness"),
        ("sensation", "causes", "perception"),
        ("perception", "causes", "memory"),
        ("memory", "describes", "experience"),
        ("episodic_memory", "is_a", "memory"),
        ("semantic_memory", "is_a", "memory"),
        ("procedural_memory", "is_a", "memory"),
        ("thought", "part_of", "consciousness"),
        ("reasoning", "is_a", "thought"),
        ("reasoning", "requires", "logic"),
        ("emotion", "part_of", "consciousness"),
        ("intention", "expresses", "agency"),
        ("agency", "requires", "self"),
        ("belief", "requires", "self"),
        ("knowledge", "is_a", "belief"),
        ("metacognition", "describes", "thought"),
        ("theory_of_mind", "is_a", "metacognition"),
        ("self_model", "describes", "self"),
        ("self_model", "is_a", "theory_of_mind"),
        ("curiosity", "causes", "learning"),
        ("learning", "describes", "experience"),
        ("imagination", "is_a", "simulation_mind"),
        ("simulation_mind", "is_a", "thought"),
        ("consciousness", "related_to", "brain"),
        ("consciousness", "related_to", "neural_network"),
    ]
    u.add_concept("experience", domain=d, definition="The subjective stream of conscious events.")
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #


def _seed_language(u: ConceptUniverse) -> None:
    d = "language"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("symbol", "A mark or token that stands for something else.")
    add("word", "A discrete unit of language.", depth=1)
    add("sentence", "A grammatical sequence of words expressing a thought.", depth=1)
    add("meaning", "The content a symbol expresses.")
    add("reference", "What a symbol points to.", depth=1)
    add("grammar", "The structural rules of a language.", depth=2)
    add("syntax", "How words combine.", depth=2)
    add("semantics", "How words mean.", depth=2)
    add("pragmatics", "How meaning depends on context.", depth=2)
    add("metaphor", "Saying one thing as another to illuminate it.", depth=2)
    add("analogy", "A mapping between two structures.", depth=2)
    add("ambiguity", "Possible multiple meanings.", depth=2)
    add("narrative", "An ordered telling of events.", depth=2)
    add("speech_act", "An utterance that performs an action.", depth=2)
    add("idiom", "A phrase whose meaning is not predictable from its parts.", depth=2)
    add("translation", "Carrying meaning between languages.", depth=2)

    rels = [
        ("symbol", "expresses", "meaning"),
        ("word", "is_a", "symbol"),
        ("sentence", "composes", "word"),
        ("word", "composes", "sentence"),
        ("meaning", "describes", "reference"),
        ("grammar", "describes", "sentence"),
        ("syntax", "is_a", "grammar"),
        ("semantics", "describes", "meaning"),
        ("pragmatics", "describes", "meaning"),
        ("metaphor", "is_a", "analogy"),
        ("analogy", "expresses", "meaning"),
        ("ambiguity", "opposes", "reference"),
        ("narrative", "composes", "sentence"),
        ("speech_act", "is_a", "sentence"),
        ("idiom", "is_a", "metaphor"),
        ("translation", "describes", "meaning"),
        ("meaning", "related_to", "thought"),
        ("metaphor", "related_to", "imagination"),
    ]
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Arts
# --------------------------------------------------------------------------- #


def _seed_arts(u: ConceptUniverse) -> None:
    d = "arts"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("art", "Intentional creation that organizes perception or feeling.")
    add("music", "Organized sound across time.", depth=1)
    add("rhythm", "Patterned recurrence in time.", depth=2)
    add("harmony", "The simultaneous combination of pitches.", depth=2)
    add("melody", "A linear sequence of pitches.", depth=2)
    add("dissonance", "Tension between simultaneous pitches.", depth=2)
    add("consonance", "Stability between simultaneous pitches.", depth=2)
    add("pitch", "The perceived frequency of a sound.", depth=2)
    add("frequency", "Cycles per second.", depth=1)
    add("painting", "Visual art applied to a surface.", depth=1)
    add("color", "A property of light reflected from a surface.", depth=1)
    add("perspective", "A method of representing depth on a flat surface.", depth=2)
    add("composition_art", "The arrangement of elements in a work.", depth=2)
    add("sculpture", "Three-dimensional art.", depth=1)
    add("poetry", "Language with intensified rhythm and imagery.", depth=2)
    add("drama", "Performed enacted story.", depth=2)
    add("novel", "An extended prose narrative.", depth=2)
    add("beauty", "A quality that elicits aesthetic pleasure.")
    add("aesthetic", "Pertaining to perception of form and beauty.", depth=2)

    rels = [
        ("music", "is_a", "art"),
        ("painting", "is_a", "art"),
        ("sculpture", "is_a", "art"),
        ("poetry", "is_a", "art"),
        ("drama", "is_a", "art"),
        ("novel", "is_a", "art"),
        ("rhythm", "part_of", "music"),
        ("harmony", "part_of", "music"),
        ("melody", "part_of", "music"),
        ("dissonance", "opposes", "consonance"),
        ("pitch", "part_of", "harmony"),
        ("pitch", "measured_by", "frequency"),
        ("color", "part_of", "painting"),
        ("perspective", "part_of", "painting"),
        ("composition_art", "describes", "art"),
        ("poetry", "composes", "rhythm"),
        ("poetry", "composes", "metaphor"),
        ("novel", "composes", "narrative"),
        ("drama", "composes", "speech_act"),
        ("aesthetic", "describes", "beauty"),
        ("art", "expresses", "emotion"),
        ("art", "expresses", "meaning"),
    ]
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Philosophy
# --------------------------------------------------------------------------- #


def _seed_philosophy(u: ConceptUniverse) -> None:
    d = "philosophy"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("truth", "Correspondence with the way things are.")
    add("being", "What exists.", aliases=("ontology_concept",))
    add("causation", "A relation in which one thing brings about another.", depth=1)
    add("identity", "Sameness across change.", depth=1)
    add("freedom", "The capacity to act otherwise.", depth=1)
    add("responsibility", "Accountability for one's actions.", depth=2)
    add("ethics", "The study of how to act.", depth=2)
    add("good", "What is intrinsically valuable.")
    add("justice", "Fair distribution and treatment.", depth=2)
    add("meaning_life", "What makes a life worth living.", depth=2)
    add("epistemology", "The study of knowledge.", depth=2)
    add("ontology", "The study of being.", depth=2)
    add("phenomenology", "Description of structures of experience.", depth=3)
    add("free_will", "Self-determined action.", depth=3)
    add("determinism", "Every event is determined by prior causes.", depth=3)
    add("emergence", "Higher-level properties from lower-level interactions.", depth=2)

    rels = [
        ("truth", "describes", "being"),
        ("epistemology", "describes", "knowledge"),
        ("ontology", "describes", "being"),
        ("causation", "describes", "interaction"),
        ("identity", "related_to", "self"),
        ("freedom", "requires", "agency"),
        ("responsibility", "requires", "freedom"),
        ("ethics", "describes", "good"),
        ("justice", "is_a", "good"),
        ("phenomenology", "describes", "experience"),
        ("free_will", "opposes", "determinism"),
        ("emergence", "describes", "complexity"),
        ("emergence", "related_to", "consciousness"),
        ("emergence", "related_to", "evolution"),
        ("meaning_life", "related_to", "good"),
    ]
    u.add_concept("complexity", domain="computing", definition="A measure of structural intricacy.")
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Computing
# --------------------------------------------------------------------------- #


def _seed_computing(u: ConceptUniverse) -> None:
    d = "computing"
    add = lambda name, definition="", depth=0, **kw: u.add_concept(
        name, domain=d, definition=definition, depth=depth, **kw
    )

    add("algorithm", "A finite, definite procedure.")
    add("computation", "The process of executing an algorithm.")
    add("state", "A snapshot of a system at a moment.")
    add("recursion", "A definition that refers to itself.", depth=2)
    add("abstraction", "Treating distinct things as the same for a purpose.", depth=1)
    add("data", "Symbols that carry information.")
    add("program", "An algorithm expressed in a language.", depth=1)
    add("model", "A simplified representation used for reasoning.", depth=1)
    add("simulation", "Running a model to observe its behavior.", depth=1)
    add("machine_learning", "Algorithms that learn from data.", depth=2)
    add("agent", "An autonomous system that perceives and acts.", depth=2)
    add("planning", "Selecting actions to reach a goal.", depth=2)
    add("search", "Exploring a state space.", depth=2)
    add("graph", "A set of nodes connected by edges.")
    add("tree", "A hierarchical graph without cycles.", depth=1)
    add("complexity_class", "A category of computational difficulty.", depth=3)
    add("turing_machine", "An abstract model of computation.", depth=3)
    add("compiler", "A translator from one language to another.", depth=2)

    rels = [
        ("algorithm", "describes", "computation"),
        ("program", "instantiates", "algorithm"),
        ("recursion", "part_of", "algorithm"),
        ("abstraction", "part_of", "model"),
        ("data", "describes", "information"),
        ("program", "composes", "data"),
        ("model", "describes", "simulation"),
        ("simulation", "requires", "model"),
        ("machine_learning", "is_a", "algorithm"),
        ("machine_learning", "requires", "data"),
        ("agent", "composes", "planning"),
        ("agent", "requires", "model"),
        ("planning", "is_a", "search"),
        ("search", "describes", "state"),
        ("tree", "is_a", "graph"),
        ("graph", "is_a", "network"),
        ("complexity_class", "describes", "algorithm"),
        ("turing_machine", "is_a", "model"),
        ("turing_machine", "describes", "computation"),
        ("compiler", "is_a", "program"),
        ("compiler", "describes", "translation"),
    ]
    u.add_concept("translation", domain="language", definition="A mapping from one language to another.")
    u.add_relations(rels)


# --------------------------------------------------------------------------- #
# Cross-domain bridges — the heart of conceptual reasoning
# --------------------------------------------------------------------------- #


def _seed_cross_domain(u: ConceptUniverse) -> None:
    """Edges that span two different domains. Reasoning relies on these to
    move from one way of thinking into another."""

    rels = [
        # Music ↔ math
        ("music", "analogous_to", "ratio"),
        ("harmony", "describes", "ratio"),
        ("rhythm", "analogous_to", "function"),
        ("frequency", "measured_by", "real_number"),
        ("dissonance", "analogous_to", "irrational"),
        # Physics ↔ math
        ("equation", "describes", "force"),
        ("equation", "describes", "energy"),
        ("calculus", "describes", "acceleration"),
        ("calculus", "describes", "velocity"),
        ("vector", "describes", "force"),
        ("vector", "describes", "velocity"),
        ("matrix", "describes", "quantum_mechanics"),
        ("probability", "describes", "quantum_mechanics"),
        ("infinity", "related_to", "spacetime"),
        # Chemistry ↔ physics
        ("bond", "is_a", "force"),
        ("covalent_bond", "instantiates", "electromagnetism"),
        ("ionic_bond", "instantiates", "electromagnetism"),
        # Biology ↔ chemistry
        ("life", "requires", "carbon"),
        ("life", "requires", "water"),
        ("metabolism", "is_a", "reaction"),
        ("dna", "is_a", "molecule"),
        ("rna", "is_a", "molecule"),
        ("protein", "is_a", "molecule"),
        ("amino_acid", "is_a", "molecule"),
        # Mind ↔ biology
        ("consciousness", "emerges_from", "brain"),
        ("thought", "emerges_from", "neural_network"),
        ("memory", "instantiates", "neural_network"),
        ("learning", "describes", "neural_network"),
        # Mind ↔ language
        ("thought", "expresses", "meaning"),
        ("meaning", "requires", "symbol"),
        ("language", "describes", "thought"),
        ("metaphor", "is_a", "thought"),
        # Mind ↔ computing
        ("agent", "instantiates", "self"),
        ("model", "is_a", "belief"),
        ("planning", "requires", "intention"),
        ("simulation", "is_a", "simulation_mind"),
        ("machine_learning", "instantiates", "learning"),
        ("neural_network", "instantiates", "machine_learning"),
        # Arts ↔ mind
        ("art", "causes", "emotion"),
        ("music", "causes", "emotion"),
        ("aesthetic", "is_a", "perception"),
        ("beauty", "related_to", "perception"),
        # Philosophy ↔ mind
        ("consciousness", "related_to", "self"),
        ("knowledge", "is_a", "belief"),
        ("free_will", "requires", "agency"),
        ("identity", "related_to", "self_model"),
        # Philosophy ↔ physics
        ("emergence", "describes", "spacetime"),
        ("determinism", "describes", "classical_mechanics"),
        ("being", "related_to", "matter"),
        ("being", "related_to", "energy"),
        # Cosmology ↔ philosophy
        ("big_bang", "related_to", "being"),
        ("cosmology", "related_to", "ontology"),
    ]
    # A handful of concepts that the relations above reference but no
    # earlier domain seeded yet.
    u.add_concept("language", domain="language", definition="A system of symbolic communication.")
    u.add_concept("classical_mechanics", domain="physics", definition="The physics of macroscopic motion at everyday speeds.", depth=2)
    u.add_concept("emerges_from", domain="philosophy", definition="A meta-relation: high-level structure arising from low-level interaction.")

    # The relation kind ``emerges_from`` is not in the canonical kind list;
    # the bulk-add tolerates that — it's still a typed edge in the graph.
    u.add_relations(rels)
