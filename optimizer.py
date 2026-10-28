"""
ZX symbolic optimizer.

Converts a PennyLane parameterized circuit into a ZX graph, labels every
parametric vertex with a SymPy symbol, runs the chosen ZX simplification
strategy (paramSafe=True), and extracts the reduced circuit.

The result is deterministic: same circuit topology + same strategy
→ same reduced structure, regardless of concrete parameter values.

Supported strategies
--------------------
    "full_reduce"        (default)
        full_reduce(paramSafe=True) — complete Clifford suite.
        Skips gadget_simp to preserve symbolic phases.

    "clifford_simp"
        interior_clifford_simp + pivot_boundary_simp loop.
        More conservative; no phase-gadget simplification.

    "interior_clifford"
        spider_simp → to_gh → id_simp + spider_simp + pivot_simp + lcomp_simp loop.
        Fastest; most conservative. Good baseline.

    "pivot_gadget"
        interior_clifford_simp + pivot_gadget_simp.
        Adds phase-gadget pivoting on top of interior Clifford.

All strategies skip gadget_simp and supplementarity_simp, which can
corrupt symbolic (SymPy) phase expressions.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

import sympy as sym

from ._sentinel import SentinelMatcher, _PRIMES

# ── Strategy registry ─────────────────────────────────────────────────────────

STRATEGIES: Dict[str, str] = {
    "full_reduce":       "Full Clifford simplification (paramSafe — skips gadget_simp)",
    "clifford_simp":     "Interior Clifford + boundary pivot loop",
    "interior_clifford": "spider_simp → to_gh → id/spider/pivot/lcomp loop",
    "pivot_gadget":      "Interior Clifford + pivot_gadget_simp",
}


def _apply_strategy(graph, strategy: str) -> None:
    """Apply the chosen ZX simplification strategy in-place."""
    import pyzx as zx
    from pyzx.simplify import (
        interior_clifford_simp,
        clifford_simp,
        pivot_gadget_simp,
        spider_simp,
        to_gh,
    )

    key = strategy.lower().replace("-", "_").replace(" ", "_")

    if key == "full_reduce":
        zx.full_reduce(graph, paramSafe=True)

    elif key == "clifford_simp":
        clifford_simp(graph, quiet=True)

    elif key == "interior_clifford":
        spider_simp(graph, quiet=True)
        to_gh(graph)
        from pyzx.simplify import id_simp, pivot_simp, lcomp_simp
        for _ in range(500):
            n1 = id_simp(graph, quiet=True)
            n2 = spider_simp(graph, quiet=True)
            n3 = pivot_simp(graph, quiet=True)
            n4 = lcomp_simp(graph, quiet=True)
            if n1 + n2 + n3 + n4 == 0:
                break

    elif key == "pivot_gadget":
        interior_clifford_simp(graph, quiet=True)
        pivot_gadget_simp(graph, quiet=True)

    else:
        raise ValueError(
            f"Unknown ZX strategy '{strategy}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )


# ── Tied parameter reconstruction ────────────────────────────────────────────

def _build_tied_groups(M, C_or_b, tol: float = 1e-9):
    """
    Fold the n_surv raw surviving-vertex expressions down to the genuine
    number of trainable degrees of freedom, rank(M).

    A CRX/CRY gate contributes two half-angle expressions, +-theta/2, to
    symbolic_structure. Left untied, the extracted circuit would have more
    trainable capacity than the original (independent values standing in
    for what is, physically, a single original parameter, or a fixed
    linear combination of a few). The simple case is a row that is
    "single-symbol" (depends on exactly one original parameter): rows that
    are single-symbol and share the same symbol are clearly redundant with
    each other and are tied into one group. That alone is not sufficient
    in general, a half-angle row can instead fuse with a *different*
    gate's half-angle into one combined expression (e.g. a row depending on
    symbols {i, j}) while its own sibling half survives on its own (a row
    depending on {i} alone): the two rows are still only two independent
    equations in {i, j}, so they must still only cost two free values
    between them, not one each on top of whatever {j} needed elsewhere.
    Only an explicit rank check catches this, so every candidate row, not
    only the single-symbol ones, is tested for whether it actually adds a
    new independent direction before it is given its own free value.

    Rows are offered to the rank test in this preference order, tied
    single-symbol groups first (one representative row each, since every
    other member of a true tie is trivially a scalar multiple of it), then
    the remaining single-symbol rows, then fused/mixed rows last. This
    reproduces the exact grouping of the simple, already-common case, and
    only reaches for the general affine fallback below when the circuit's
    fusion pattern is tangled enough to need it.

    Returns
    -------
    G, C, groups : psi = G @ free_vec + C
        G : (n_surv, n_groups)   C : (n_surv,)   groups : list[list[int]]
        (groups[g][0] is the representative "basis" row for free value g,
        for rows not in any group's first position, G[row] expresses that
        row as the exact affine combination of the basis rows' psi values
        that reproduces it for every theta.)
    """
    import numpy as np

    b = C_or_b
    n_surv, n_params = M.shape
    symbol_of = {}
    for row in range(n_surv):
        nz = np.nonzero(np.abs(M[row]) > tol)[0]
        if len(nz) == 1:
            symbol_of[row] = (int(nz[0]), float(M[row, nz[0]]))

    from collections import defaultdict
    by_symbol = defaultdict(list)
    for row, (sym_idx, _coeff) in symbol_of.items():
        by_symbol[sym_idx].append(row)

    # Candidate order: tied-symbol groups (full membership kept together),
    # then lone single-symbol rows, then fused/mixed rows.
    tied_candidates: List[List[int]] = []
    lone_rows: List[int] = []
    for sym_idx, rows in by_symbol.items():
        if len(rows) >= 2:
            tied_candidates.append(rows)
        else:
            lone_rows.append(rows[0])
    fused_rows = [row for row in range(n_surv) if row not in symbol_of]
    candidates = tied_candidates + [[r] for r in lone_rows] + [[r] for r in fused_rows]

    basis_rows: List[int] = []
    basis_matrix = np.zeros((0, n_params))
    groups: List[List[int]] = []
    for rows in candidates:
        rep = rows[0]
        trial = np.vstack([basis_matrix, M[rep:rep + 1]])
        if np.linalg.matrix_rank(trial, tol=tol) > len(basis_rows):
            basis_rows.append(rep)
            basis_matrix = trial
            groups.append(rows)
        # else: redundant with the existing basis, contributes no new
        # free value (its own affine expression is filled in below).

    n_groups = len(groups)
    row_to_group = {}
    for g, rows in enumerate(groups):
        for row in rows:
            row_to_group[row] = g

    basis_pinv = np.linalg.pinv(basis_matrix)  # (n_params, n_groups), right-inverse of basis_matrix
    b_basis = np.array([b[r] for r in basis_rows])

    G = np.zeros((n_surv, n_groups))
    C = np.zeros(n_surv)
    for row in range(n_surv):
        if row in row_to_group and row in basis_rows:
            g = row_to_group[row]
            G[row, g] = 1.0
            C[row] = 0.0
        elif row in row_to_group:
            # Other member of a tied group whose representative is basis_rows[g].
            g = row_to_group[row]
            rep = groups[g][0]
            # tied rows are scalar multiples of the representative row's symbol:
            # psi_row = (M[row]/M[rep] ratio) * (psi_rep - b[rep]) + b[row]
            sym_idx, coeff_rep = symbol_of[rep]
            coeff_row = float(M[row, sym_idx])
            scale = coeff_row / coeff_rep
            G[row, g] = scale
            C[row] = b[row] - scale * b[rep]
        else:
            # Not independent of the basis: express as its exact affine
            # combination (holds for every theta, since both sides are
            # affine in theta and agree on a spanning set of directions).
            row_coeffs = M[row:row + 1] @ basis_pinv  # (1, n_groups)
            G[row, :] = row_coeffs
            C[row] = b[row] - float(row_coeffs @ b_basis)
    return G, C, groups


@dataclass
class ParameterMap:
    """
    Exact affine relationship between the N original circuit parameters
    (theta, in turns, i.e. the same units circuit_fn expects before the
    internal ``* pi``) and the angles of the gates that survive ZX
    extraction (phi, in turns): ``phi = M @ theta + b``.

    Tied grouping folds this further down to the true degrees of freedom:
    ``psi = G @ free + C``, where ``free`` has one entry per group instead
    of one per surviving gate (see :func:`_build_tied_groups`).

    Build one with :meth:`OptimizationResult.build_parameter_map`, used by
    :meth:`VQCEvaluator.train_both_matched` to start the extracted circuit
    from the same point, in function-space, as the original circuit.
    """
    M: Any
    b: Any
    G: Any
    C: Any
    groups: List[List[int]]

    @property
    def n_surviving(self) -> int:
        return self.M.shape[0]

    @property
    def n_groups(self) -> int:
        return self.G.shape[1]

    def phi(self, theta_turns):
        """theta (original params, turns) -> phi(theta) (surviving gate angles, turns)."""
        import numpy as np
        return self.M @ np.asarray(theta_turns, dtype=float) + self.b

    def free_vec(self, theta_turns):
        """theta (turns) -> one free scalar per tied group (the reduced trainable space)."""
        import numpy as np
        phi = self.phi(theta_turns)
        return np.array([
            (phi[self.groups[g][0]] - self.C[self.groups[g][0]]) / self.G[self.groups[g][0], g]
            for g in range(self.n_groups)
        ])

    def psi(self, free_vec_turns):
        """free vector (turns, one per group) -> psi (turns, one per surviving gate)."""
        import numpy as np
        return self.G @ np.asarray(free_vec_turns, dtype=float) + self.C


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """
    Everything produced by one ZX optimization run.

    Fields
    ------
    n_params_original   : parameters in the original circuit
    n_params_optimized  : trainable parameters remaining after reduction
    n_ops_original      : gate count before reduction
    n_ops_optimized     : gate count after reduction
    n_symbolic_before   : parametric vertices found before reduction
    n_symbolic_after    : parametric vertices surviving after reduction
    symbolic_structure  : SymPy expressions of surviving parameters (as strings)
    original_param_names: names of original SymPy symbols (a, b, c, ...)
    init_params_rad     : concrete initial angles (rad) for each optimized param
    optimized_ops       : pyzx Circuit object (use with GateApplicator)
    strategy            : ZX simplification strategy used
    """
    n_params_original: int
    n_params_optimized: int
    n_ops_original: int
    n_ops_optimized: int
    n_symbolic_before: int
    n_symbolic_after: int
    symbolic_structure: List[str]
    original_param_names: List[str]
    init_params_rad: List[float]
    optimized_ops: Any          # pyzx.Circuit
    strategy: str = "full_reduce"

    def summary(self) -> str:
        lines = [
            f"strategy: {self.strategy}",
            f"ops:      {self.n_ops_original} → {self.n_ops_optimized}  "
            f"({100*(1-self.n_ops_optimized/max(self.n_ops_original,1)):.1f}% change)",
            f"params:   {self.n_params_original} → {self.n_params_optimized}  "
            f"(symbolic before={self.n_symbolic_before}, after={self.n_symbolic_after})",
        ]
        if self.symbolic_structure:
            lines.append(f"structure: {self.symbolic_structure}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (gates included)."""
        from .cache import _serialize_gate
        return {
            "n_params_original":    self.n_params_original,
            "n_params_optimized":   self.n_params_optimized,
            "n_ops_original":       self.n_ops_original,
            "n_ops_optimized":      self.n_ops_optimized,
            "n_symbolic_before":    self.n_symbolic_before,
            "n_symbolic_after":     self.n_symbolic_after,
            "symbolic_structure":   self.symbolic_structure,
            "original_param_names": self.original_param_names,
            "init_params_rad":      self.init_params_rad,
            "strategy":             self.strategy,
            "gates": [_serialize_gate(g) for g in self.optimized_ops.gates],
        }

    def build_parameter_map(self, tol: float = 1e-9) -> "ParameterMap":
        """
        Reconstruct the exact affine map between the N original parameters
        and the angles of the gates surviving extraction, tying structurally
        linked expressions (CRX/CRY +-theta/2 half-angle pairs) into a
        single trainable degree of freedom each.

        ``optimized_ops.gates`` and ``symbolic_structure`` are in different
        orders (gate-emission order vs. graph-traversal order), so this
        recovers the correspondence using the same "ID-tag" trick the
        optimizer itself uses internally: each surviving gate's concrete
        angle (derived from a fresh set of canonical sentinels, 1/prime_i)
        is matched, by nearest distance, back to the symbolic_structure
        entry it was derived from.

        Raises
        ------
        RuntimeError
            If the matching is not a bijection (sentinel collision — should
            not happen for the default match_tol, but is checked rather
            than silently mismatched).
        """
        import numpy as np

        n_surv = self.n_symbolic_after
        n_params = self.n_params_original

        fixed_sentinels = [1.0 / _PRIMES[i] for i in range(n_surv)]
        gate_order_turns = [v / math.pi for v in self.init_params_rad]
        match = []
        for v in gate_order_turns:
            i_star = min(range(n_surv), key=lambda i: abs(v - fixed_sentinels[i]))
            match.append(i_star)
        if sorted(match) != list(range(n_surv)):
            raise RuntimeError(
                "ID-tag matching between init_params_rad and "
                "symbolic_structure is not a bijection (sentinel collision) "
                "-- cannot build a reliable parameter map for this circuit."
            )

        symbols = [sym.Symbol(name) for name in self.original_param_names]
        symbol_map = {s.name: s for s in symbols}
        exprs = [sym.sympify(e, locals=symbol_map) for e in self.symbolic_structure]

        M = np.zeros((n_surv, n_params))
        b = np.zeros(n_surv)
        for j in range(n_surv):
            e = exprs[match[j]]
            for k, s in enumerate(symbols):
                M[j, k] = float(e.coeff(s))
            b[j] = float(e.subs({s: 0 for s in symbols}))

        G, C, groups = _build_tied_groups(M, b, tol=tol)
        return ParameterMap(M=M, b=b, G=G, C=C, groups=groups)

    def verify_equivalence(
        self,
        circuit_fn: Callable,
        n_qubits: int,
        n_trials: int = 5,
        seed: int = 777,
        tol: float = 1e-3,
    ) -> float:
        """
        Check that the original circuit and the extracted circuit, run from
        matched parameters (theta and build_parameter_map().phi(theta)),
        produce the same output.

        Returns the maximum absolute difference observed over *n_trials*
        random (theta, input) draws. A genuinely equivalent reconstruction
        should give a value at the level of floating-point error (~1e-12).
        Values near or above *tol* mean the reconstruction for this circuit
        should not be trusted for training, which can happen for some
        CRX-heavy circuits.
        """
        import numpy as np
        import pennylane as qml

        pmap = self.build_parameter_map()
        opt_gates = self.optimized_ops.gates

        dev = qml.device("default.qubit", wires=n_qubits)
        obs = qml.sum(*[qml.PauliZ(w) for w in range(n_qubits)])

        @qml.qnode(dev)
        def qnode_orig(x, theta):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            circuit_fn(theta)
            return qml.expval(obs)

        @qml.qnode(dev)
        def qnode_opt(x, psi):
            from .evaluator import GateApplicator
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            idx = [0]
            for g in opt_gates:
                GateApplicator.apply(g, params=psi, param_idx=idx)
            return qml.expval(obs)

        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_trials):
            x = rng.uniform(0, math.pi, (5, n_qubits))
            # theta is sampled directly in the same units circuit_fn expects
            # (its entries ARE the symbolic parameters a, b, c, ... used to
            # build M/b -- see build_parameter_map), not divided by pi again.
            theta = rng.uniform(0.0, 2 * math.pi, self.n_params_original)
            free_vec = pmap.free_vec(theta)
            psi_turns = pmap.psi(free_vec)
            psi_rad = psi_turns * math.pi

            out_orig = np.array(qnode_orig(x, theta))
            out_opt = np.array(qnode_opt(x, psi_rad))
            worst = max(worst, float(np.max(np.abs(out_orig - out_opt))))

        return worst


# ── Optimizer ─────────────────────────────────────────────────────────────────

class ZXOptimizer:
    """
    Symbolic ZX optimizer for parameterized quantum circuits.

    Pipeline
    --------
    1. Build ZX graph with sentinel values (1/prime_i) as phase placeholders.
    2. Replace sentinel phases with SymPy symbols (a, b, c, …).
    3. Run the chosen ZX strategy (param-safe: symbolic vertices are protected).
    4. Collect surviving SymPy vertices (= effective free parameters).
    5. Assign concrete sentinel values and extract a pyzx Circuit.

    Parameters
    ----------
    strategy  : ZX simplification strategy.
                One of: 'full_reduce' (default), 'clifford_simp',
                'interior_clifford', 'pivot_gadget'.
    match_tol : sentinel matching tolerance (default 0.003).

    Usage
    -----
        opt = ZXOptimizer(strategy="clifford_simp")
        result = opt.run(my_circuit_fn, n_params=8)
        print(result.summary())
    """

    def __init__(
        self,
        strategy: str = "full_reduce",
        match_tol: float = 0.003,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(STRATEGIES.keys())}"
            )
        self.strategy = strategy
        self.match_tol = match_tol

    def run(self, circuit_fn: Callable, n_params: int) -> OptimizationResult:
        """
        Optimize *circuit_fn* and return an OptimizationResult.

        Parameters
        ----------
        circuit_fn : callable(params) that builds a PennyLane circuit.
                     Rotation angles must be params[i] * np.pi.
        n_params   : number of parameters circuit_fn expects.
        """
        import pyzx as zx
        import pennylane as qml
        from pyzx.utils import VertexType

        matcher = SentinelMatcher(n_params, self.match_tol)

        # 1. Build ZX graph
        with qml.queuing.AnnotatedQueue() as q:
            circuit_fn(matcher.sentinels)
        tape = qml.tape.QuantumScript.from_queue(q)
        graph = qml.transforms.to_zx(tape)
        n_ops_orig = len(tape.operations)

        # 2. Replace sentinels with SymPy symbols
        found = matcher.assign_symbolic_phases(graph)
        n_sym_before = len(found)

        if n_sym_before == 0:
            warnings.warn(
                "No parametric vertices identified. "
                "Check that circuit_fn uses params[i] * np.pi for angles.",
                stacklevel=2,
            )

        # 3. Apply chosen strategy (always param-safe)
        _apply_strategy(graph, self.strategy)

        # 4. Collect surviving symbolic vertices
        remaining = [
            (v, graph.phase(v))
            for v in graph.vertices()
            if isinstance(graph.phase(v), sym.Basic)
            and graph.type(v) in (VertexType.Z, VertexType.X)
        ]
        symbolic_structure = [str(e) for _, e in remaining]

        # 5. Assign concrete values and extract
        fixed = SentinelMatcher(len(remaining), self.match_tol)
        for (v, _), val in zip(remaining, fixed.sentinels):
            graph.set_phase(v, Fraction(val).limit_denominator(1000))

        optimized_ops = zx.extract_circuit(graph)
        init_params_rad = [
            float(g.phase) * math.pi
            for g in optimized_ops.gates
            if g.name in ("ZPhase", "XPhase")
        ]

        return OptimizationResult(
            n_params_original=n_params,
            n_params_optimized=len(init_params_rad),
            n_ops_original=n_ops_orig,
            n_ops_optimized=len(optimized_ops.gates),
            n_symbolic_before=n_sym_before,
            n_symbolic_after=len(remaining),
            symbolic_structure=symbolic_structure,
            original_param_names=[str(s) for s in matcher.symbols],
            init_params_rad=init_params_rad,
            optimized_ops=optimized_ops,
            strategy=self.strategy,
        )
