"""
Circuit registry — named collection of parameterized circuit specifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple


@dataclass
class CircuitSpec:
    """Specification for a parameterized quantum circuit."""
    name: str
    fn: Callable
    n_params: int
    n_qubits: int
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"CircuitSpec(name={self.name!r}, "
            f"n_qubits={self.n_qubits}, n_params={self.n_params})"
        )


class CircuitRegistry:
    """
    Named registry of CircuitSpec objects.

    Supports decorator and explicit registration patterns.

    Usage
    -----
        registry = CircuitRegistry()

        # Decorator style
        @registry.register("ansatz_A", n_params=8, n_qubits=4)
        def ansatz_A(params):
            for i in range(4):
                qml.RX(params[i] * np.pi, wires=i)
            ...

        # Manual style
        registry.add(CircuitSpec("ansatz_B", fn, n_params=11, n_qubits=4))

        # Lookup
        spec = registry["ansatz_A"]   # or registry.get("ansatz_A")
        for name, spec in registry.items():
            ...
    """

    def __init__(self) -> None:
        self._circuits: Dict[str, CircuitSpec] = {}

    # ── registration ────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        n_params: int,
        n_qubits: int,
        description: str = "",
    ) -> Callable:
        """Decorator that registers a circuit function."""
        def decorator(fn: Callable) -> Callable:
            self.add(CircuitSpec(name, fn, n_params, n_qubits, description))
            return fn
        return decorator

    def add(self, spec: CircuitSpec) -> None:
        """Register a CircuitSpec directly."""
        self._circuits[spec.name] = spec

    # ── lookup ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[CircuitSpec]:
        return self._circuits.get(name)

    def __getitem__(self, name: str) -> CircuitSpec:
        try:
            return self._circuits[name]
        except KeyError:
            available = list(self._circuits)
            raise KeyError(
                f"Circuit {name!r} not found. Available: {available}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._circuits

    def __len__(self) -> int:
        return len(self._circuits)

    def __iter__(self) -> Iterator[str]:
        return iter(self._circuits)

    def names(self) -> List[str]:
        return list(self._circuits)

    def items(self) -> Iterator[Tuple[str, CircuitSpec]]:
        return self._circuits.items()

    def __repr__(self) -> str:
        names = list(self._circuits)
        return f"CircuitRegistry({names})"
