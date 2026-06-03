"""Optional Rust/PyO3 activation kernel loader.

When ``darwin_rust_kernel`` (a future PyO3 extension) is importable, this
module returns a handle to the kernel's ``propagate`` callable. Otherwise
it returns ``None`` silently so the runtime can fall through to the pure-
Python path.
"""

from __future__ import annotations

from typing import Any, Callable


def load_rust_kernel() -> Callable[..., Any] | None:
    try:
        import darwin_rust_kernel  # type: ignore[import]
    except Exception:
        return None
    fn = getattr(darwin_rust_kernel, "propagate", None)
    return fn if callable(fn) else None


def rust_kernel_available() -> bool:
    return load_rust_kernel() is not None


__all__ = ["load_rust_kernel", "rust_kernel_available"]
