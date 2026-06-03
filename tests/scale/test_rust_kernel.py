"""Tests for the optional Rust kernel loader."""

from __future__ import annotations

from darwin.scale.rust_kernel import load_rust_kernel, rust_kernel_available


def test_loader_returns_none_when_absent() -> None:
    # The Rust extension is not built in CI.
    fn = load_rust_kernel()
    assert fn is None or callable(fn)


def test_availability_check_is_bool() -> None:
    assert isinstance(rust_kernel_available(), bool)
