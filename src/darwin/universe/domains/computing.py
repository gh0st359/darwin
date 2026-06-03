"""Computing domain seed — data structures, algorithms, languages, systems."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Data structures.
        ("array", "is_a", "data_structure", 1.0),
        ("list", "is_a", "data_structure", 1.0),
        ("linked_list", "is_a", "list", 1.0),
        ("stack", "is_a", "data_structure", 1.0),
        ("queue", "is_a", "data_structure", 1.0),
        ("tree", "is_a", "data_structure", 1.0),
        ("graph", "is_a", "data_structure", 1.0),
        ("hashmap", "is_a", "data_structure", 1.0),
        ("set", "is_a", "data_structure", 1.0),
        ("binary_tree", "is_a", "tree", 1.0),
        ("bst", "is_a", "binary_tree", 1.0),
        ("heap", "is_a", "binary_tree", 1.0),
        ("trie", "is_a", "tree", 1.0),
        # Algorithms.
        ("bubble_sort", "is_a", "algorithm", 1.0),
        ("quicksort", "is_a", "algorithm", 1.0),
        ("mergesort", "is_a", "algorithm", 1.0),
        ("heapsort", "is_a", "algorithm", 1.0),
        ("bfs", "is_a", "algorithm", 1.0),
        ("dfs", "is_a", "algorithm", 1.0),
        ("dijkstra", "is_a", "algorithm", 1.0),
        ("a_star", "is_a", "algorithm", 1.0),
        ("binary_search", "is_a", "algorithm", 1.0),
        # Complexity.
        ("o_1", "is_a", "complexity", 1.0),
        ("o_log_n", "is_a", "complexity", 1.0),
        ("o_n", "is_a", "complexity", 1.0),
        ("o_n_log_n", "is_a", "complexity", 1.0),
        ("o_n_squared", "is_a", "complexity", 1.0),
        ("o_two_n", "is_a", "complexity", 1.0),
        ("binary_search", "has_complexity", "o_log_n", 1.0),
        ("bubble_sort", "has_complexity", "o_n_squared", 1.0),
        ("quicksort", "has_complexity", "o_n_log_n", 1.0),
        ("mergesort", "has_complexity", "o_n_log_n", 1.0),
        # Languages.
        ("python", "is_a", "programming_language", 1.0),
        ("c", "is_a", "programming_language", 1.0),
        ("rust", "is_a", "programming_language", 1.0),
        ("javascript", "is_a", "programming_language", 1.0),
        ("haskell", "is_a", "programming_language", 1.0),
        ("lisp", "is_a", "programming_language", 1.0),
        ("python", "is_a", "interpreted_language", 1.0),
        ("c", "is_a", "compiled_language", 1.0),
        ("rust", "is_a", "compiled_language", 1.0),
        ("haskell", "is_a", "functional_language", 1.0),
        ("lisp", "is_a", "functional_language", 1.0),
        # Paradigms.
        ("imperative", "is_a", "paradigm", 1.0),
        ("functional", "is_a", "paradigm", 1.0),
        ("object_oriented", "is_a", "paradigm", 1.0),
        ("logic", "is_a", "paradigm", 1.0),
        ("concurrent", "is_a", "paradigm", 1.0),
        # Systems.
        ("kernel", "part_of", "operating_system", 1.0),
        ("filesystem", "part_of", "operating_system", 1.0),
        ("scheduler", "part_of", "kernel", 1.0),
        ("process", "is_a", "execution_unit", 1.0),
        ("thread", "is_a", "execution_unit", 1.0),
        ("thread", "part_of", "process", 1.0),
        # Networks.
        ("tcp", "is_a", "protocol", 1.0),
        ("udp", "is_a", "protocol", 1.0),
        ("http", "is_a", "protocol", 1.0),
        ("https", "is_a", "protocol", 1.0),
        ("dns", "is_a", "protocol", 1.0),
        ("tcp", "provides", "reliability", 1.0),
        ("udp", "provides", "low_latency", 1.0),
        # CS theory.
        ("turing_machine", "is_a", "computational_model", 1.0),
        ("lambda_calculus", "is_a", "computational_model", 1.0),
        ("automaton", "is_a", "computational_model", 1.0),
        ("regular_language", "is_a", "language_class", 1.0),
        ("context_free_language", "is_a", "language_class", 1.0),
        ("p", "is_a", "complexity_class", 1.0),
        ("np", "is_a", "complexity_class", 1.0),
        ("p", "subset_of", "np", 1.0),
    ]


__all__ = ["relations"]
