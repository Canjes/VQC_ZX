"""
Step-by-step ZX reduction tracer.

Mirrors full_reduce(paramSafe=True) exactly but records a TraceStep after
every rule application, so you can see which rule fired, how many times,
and the resulting graph statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import sympy as sym


@dataclass
class TraceStep:
    """Snapshot of the ZX graph after one rule application."""
    rule: str
    n_applied: int
    n_vertices: int
    n_edges: int
    n_symbolic: int
    n_clifford: int


class ZXTracer:
    """
    Runs full_reduce(paramSafe=True) step by step, recording each rule.

    Usage
    -----
        tracer = ZXTracer()
        # graph must already have SymPy phases assigned
        steps = tracer.trace(graph)
        tracer.print_table(steps)
    """

    # frozenset used in _stats — defined here so it's computed once
    _CLIFFORD = frozenset({0.0, 0.5, 1.0, 1.5})

    def _stats(self, graph) -> dict:
        from pyzx.utils import VertexType

        n_sym = n_clif = 0
        for v in graph.vertices():
            t = graph.type(v)
            if t not in (VertexType.Z, VertexType.X):
                continue
            ph = graph.phase(v)
            if isinstance(ph, sym.Basic):
                n_sym += 1
            else:
                try:
                    if float(ph) % 2.0 in self._CLIFFORD:
                        n_clif += 1
                except (TypeError, ValueError):
                    pass
        return {
            "n_vertices": graph.num_vertices(),
            "n_edges": graph.num_edges(),
            "n_symbolic": n_sym,
            "n_clifford": n_clif,
        }

    def _record(self, steps: List[TraceStep], graph, rule: str, n: int) -> None:
        s = self._stats(graph)
        steps.append(TraceStep(rule=rule, n_applied=n, **s))

    def _interior_clifford(self, graph, steps: List[TraceStep]) -> None:
        """Mirrors interior_clifford_simp with per-rule logging."""
        import pyzx as zx
        from pyzx.simplify import to_gh, id_simp

        n = zx.spider_simp(graph, quiet=True)
        if n:
            self._record(steps, graph, "spider_simp", n)

        # to_gh converts X-spiders to Z-spiders (H-edge form).
        # This is structural — not a simplification — so n_applied=0.
        to_gh(graph)
        self._record(steps, graph, "to_gh [X→Z+H]", 0)

        for _ in range(200):
            n1 = id_simp(graph, quiet=True)       # remove degree-2 identity spiders
            n2 = zx.spider_simp(graph, quiet=True)
            n3 = zx.pivot_simp(graph, quiet=True)
            n4 = zx.lcomp_simp(graph, quiet=True)
            if n1: self._record(steps, graph, "id_simp", n1)
            if n2: self._record(steps, graph, "spider_simp", n2)
            if n3: self._record(steps, graph, "pivot_simp", n3)
            if n4: self._record(steps, graph, "lcomp_simp", n4)
            if n1 + n2 + n3 + n4 == 0:
                break

    def trace(self, graph) -> List[TraceStep]:
        """
        Run full_reduce(paramSafe=True) on *graph* with per-step logging.

        The graph is modified in place (same as full_reduce).
        Returns a list of TraceStep objects.
        """
        import pyzx as zx
        from pyzx.simplify import pivot_boundary_simp

        steps: List[TraceStep] = []
        self._record(steps, graph, "inicial", 0)

        # Phase 1: initial interior_clifford_simp
        self._interior_clifford(graph, steps)

        # Phase 2: initial pivot_gadget_simp
        n = zx.pivot_gadget_simp(graph, quiet=True)
        if n:
            self._record(steps, graph, "pivot_gadget_simp", n)

        # Main loop (mirrors the while True in full_reduce)
        for _ in range(50):
            self._interior_clifford(graph, steps)
            n_pb = pivot_boundary_simp(graph, quiet=True)
            if n_pb:
                self._record(steps, graph, "pivot_boundary_simp", n_pb)
            j = zx.pivot_gadget_simp(graph, quiet=True)
            if j:
                self._record(steps, graph, "pivot_gadget_simp", j)
            if j == 0:
                break

        self._record(steps, graph, "final", 0)
        return steps

    def print_table(self, steps: List[TraceStep]) -> None:
        """Pretty-print a trace table."""
        hdr = (
            f"{'#':>3}  {'Regra':<26}  {'Aplic':>5}  "
            f"{'Vert':>5}  {'Arest':>5}  {'Simbol':>6}  {'Clif':>5}"
        )
        print(hdr)
        print("-" * len(hdr))
        for i, s in enumerate(steps):
            print(
                f"{i:>3}  {s.rule:<26}  {s.n_applied:>5}  "
                f"{s.n_vertices:>5}  {s.n_edges:>5}  "
                f"{s.n_symbolic:>6}  {s.n_clifford:>5}"
            )
        if len(steps) >= 2:
            s0, sf = steps[0], steps[-1]
            pct = 100 * (1 - sf.n_vertices / s0.n_vertices) if s0.n_vertices else 0.0
            print(
                f"\nResult: {s0.n_vertices} → {sf.n_vertices} vertices "
                f"({pct:.1f}% reduction)  |  "
                f"{sf.n_symbolic} symbolic params surviving"
            )
