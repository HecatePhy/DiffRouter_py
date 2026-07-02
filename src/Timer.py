from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar, Any, cast


T = TypeVar("T")


@dataclass
class _TimeRecord:
    label: str
    seconds: float


class Timer:
    """A simple timer utility.

    Supports:
    - Manual start/stop
    - Context manager (with Timer(...): ...)
    - Accumulating elapsed time across multiple start/stop cycles
    """

    def __init__(self, name: Optional[str] = None, logger: Optional[Callable[[str], None]] = None) -> None:
        self.name: Optional[str] = name
        self.logger: Optional[Callable[[str], None]] = logger
        self._start_monotonic: Optional[float] = None
        self._elapsed_seconds: float = 0.0

    # --- Core API ---
    def start(self) -> None:
        if self._start_monotonic is not None:
            return  # already running
        self._start_monotonic = time.monotonic()

    def stop(self) -> float:
        if self._start_monotonic is None:
            return self._elapsed_seconds
        end = time.monotonic()
        self._elapsed_seconds += end - self._start_monotonic
        self._start_monotonic = None
        return self._elapsed_seconds

    def reset(self) -> None:
        self._start_monotonic = None
        self._elapsed_seconds = 0.0

    @property
    def running(self) -> bool:
        return self._start_monotonic is not None

    @property
    def elapsed(self) -> float:
        if self._start_monotonic is None:
            return self._elapsed_seconds
        return self._elapsed_seconds + (time.monotonic() - self._start_monotonic)

    # --- Context manager ---
    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        if self.logger is not None:
            label = f"[{self.name}] " if self.name else ""
            self.logger(f"{label}{self.elapsed:.6f}s")


def timeit(
    func: Optional[Callable[..., T]] = None,
    *,
    name: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]] | Callable[..., T]:
    """Decorator to time a function.

    Usage:
        @timeit
        def work(): ...

        @timeit(name="step", logger=print)
        def work(): ...
    """

    def _decorate(f: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            t = Timer(name=name or f.__name__, logger=logger)
            t.start()
            try:
                return f(*args, **kwargs)
            finally:
                t.stop()
                if t.logger is not None:
                    label = f"[{t.name}] " if t.name else ""
                    t.logger(f"{label}{t.elapsed:.6f}s")

        return cast(Callable[..., T], wrapped)

    if func is not None:
        return _decorate(func)
    return _decorate


__all__ = ["Timer", "timeit"]
