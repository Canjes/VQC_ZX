"""
Sentinel-based phase identification for symbolic ZX optimization.

Strategy: assign each parameter a unique non-Clifford float 1/prime_i.
After building the ZX graph with these sentinel values, scan all vertices
and replace matching phases with SymPy symbols before running full_reduce.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import sympy as sym

# Enough primes to support circuits with up to ~40 parameters.
# 1/p must be far from Clifford angles (0, 0.5, 1, 1.5) and from each other.
_PRIMES = [
    7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
    139, 149, 151, 157, 163, 167, 173, 179, 181,
]
_CLIFFORD = frozenset({0.0, 0.5, 1.0, 1.5})
_PARAM_NAMES = "abcdefghijklmnopqrstuvwxyz"


class SentinelMatcher:
    """
    Generates unique non-Clifford sentinel phases (1/prime_i) — one per
    parameter — and maps ZX graph vertices back to SymPy expressions.

    Phase conventions (all in units of pi):
      RX / RZ        : phase = theta/pi = s
      RY             : phase = 1 + theta/pi = 1 + s   (ZX Y-rotation)
      CRX / CRZ      : half-angles  +s/2  and  -(s/2) mod 2
      CRY            : 1 + s/2  and  (1 - s/2) mod 2

    Nearest-distance matching avoids sentinel collisions (e.g. 1/22 vs 1/23).
    """

    def __init__(self, n_params: int, match_tol: float = 0.003):
        if n_params > len(_PRIMES):
            raise ValueError(
                f"n_params={n_params} exceeds available primes ({len(_PRIMES)}). "
                "Extend _PRIMES in _sentinel.py."
            )
        self.n_params = n_params
        self.match_tol = match_tol
        self.sentinels: List[float] = [1.0 / _PRIMES[i] for i in range(n_params)]
        self.symbols: List[sym.Symbol] = [
            sym.Symbol(_PARAM_NAMES[i] if i < 26 else f"p{i}")
            for i in range(n_params)
        ]
        self._phase_map: Dict[float, sym.Expr] = self._build_phase_map()

    def _build_phase_map(self) -> Dict[float, sym.Expr]:
        m: Dict[float, sym.Expr] = {}
        for s, sym_i in zip(self.sentinels, self.symbols):
            m[s]                = sym_i            # RX / RZ  full angle
            m[s / 2]            = sym_i / 2        # CRX / CRZ  half (+)
            m[(2.0 - s / 2) % 2] = -sym_i / 2      # CRX / CRZ  half (-)
            m[1.0 + s]          = sym_i            # RY         full angle
            # CRY half-angles carry a genuine +1 (mod 2) relative to the
            # CRX/CRZ half-angle keys. The +1 must be recorded in the
            # expression itself, not just the lookup key, or it is lost:
            # both would otherwise map to the bare expression sym_i / 2,
            # making the two cases indistinguishable downstream.
            m[1.0 + s / 2]      = 1 + sym_i / 2    # CRY        half (+)
            m[(1.0 - s / 2) % 2] = 1 - sym_i / 2   # CRY        half (-)
        return m

    def assign_symbolic_phases(self, graph) -> List[Tuple[int, sym.Expr]]:
        """
        Scan *graph* and replace sentinel float phases with SymPy expressions.
        Returns [(vertex_id, expr), ...] for every matched vertex.
        """
        from pyzx.utils import VertexType

        found: List[Tuple[int, sym.Expr]] = []
        for v in graph.vertices():
            if graph.type(v) not in (VertexType.Z, VertexType.X):
                continue
            try:
                ph = float(graph.phase(v)) % 2.0
            except (TypeError, ValueError):
                continue
            if any(abs(ph - c) < self.match_tol for c in _CLIFFORD):
                continue
            best_dist, best_expr = self.match_tol, None
            for s_val, expr in self._phase_map.items():
                d = abs(ph - s_val % 2.0)
                if d < best_dist:
                    best_dist, best_expr = d, expr
            if best_expr is not None:
                graph.set_phase(v, best_expr)
                found.append((v, best_expr))
        return found
