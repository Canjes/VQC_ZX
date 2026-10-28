"""
VQC training and evaluation.

Public helpers
--------------
    make_embedding(name, **kw)   -> callable(x, n_qubits)
    make_observable(name, **kw)  -> callable(n_qubits) -> PL observable

    GateApplicator               — applies pyzx gates to a PennyLane circuit
    VQCEvaluator                 — trains and compares original vs ZX-optimized

Supported embedding names
--------------------------
    "angle_Y"   (default)  AngleEmbedding with Y rotations
    "angle_X"              AngleEmbedding with X rotations
    "angle_Z"              AngleEmbedding with Z rotations
    "amplitude"            AmplitudeEmbedding (normalised; needs 2^n_q features)
    "iqp"                  IQP-style: H layer + ZZPhase interactions + H layer
    callable               any fn(x, n_qubits) -> None

Supported observable names
---------------------------
    "sum_all"   (default)  (Z0 + Z1 + ... + Zn) / n_q
    "sum_subset"           sum over a subset of qubits / len(subset)
    "single"               Z0 only  (historical — limited signal)
    "tensor"               Z0 ⊗ Z1 ⊗ ... (tensor product over subset)
    "parity"               alias for "tensor"
    callable               any fn(n_qubits, wires) -> PL observable
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as _np_plain


# ── Embedding factory ─────────────────────────────────────────────────────────

def make_embedding(
    name: Union[str, Callable] = "angle_Y",
) -> Callable:
    """
    Return an embedding function ``fn(x, n_qubits) -> None``.

    Parameters
    ----------
    name : str or callable
        One of ``"angle_Y"``, ``"angle_X"``, ``"angle_Z"``,
        ``"amplitude"``, ``"iqp"``, or a custom callable.
    """
    if callable(name):
        return name

    key = name.lower()

    if key in ("angle_y", "angle_x", "angle_z"):
        rot = key[-1].upper()
        def _angle(x, n_qubits: int, _rot=rot) -> None:
            import pennylane as qml
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation=_rot)
        _angle.__name__ = f"AngleEmbedding_{rot}"
        return _angle

    if key == "amplitude":
        def _amp(x, n_qubits: int) -> None:
            import pennylane as qml
            # AmplitudeEmbedding requires 2^n_q features; normalise automatically.
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0)
        _amp.__name__ = "AmplitudeEmbedding"
        return _amp

    if key == "iqp":
        def _iqp(x, n_qubits: int) -> None:
            """
            IQP-style data encoding (Havlíček et al., 2019).
            Layer 1: Hadamard on all qubits.
            Layer 2: ZZPhase on adjacent pairs — angle = x[i] * x[j].
            Layer 3: Hadamard again.
            Features x are assumed in [0, pi].
            """
            import pennylane as qml
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits - 1):
                qml.IsingZZ(x[i] * x[i + 1], wires=[i, i + 1])
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
        _iqp.__name__ = "IQPEmbedding"
        return _iqp

    raise ValueError(
        f"Unknown embedding '{name}'. "
        "Choose from: 'angle_Y', 'angle_X', 'angle_Z', 'amplitude', 'iqp', "
        "or pass a callable(x, n_qubits)."
    )


# ── Observable factory ────────────────────────────────────────────────────────

def make_observable(
    name: Union[str, Callable] = "sum_all",
    *,
    wires: Optional[List[int]] = None,
):
    """
    Return an observable factory ``fn(n_qubits) -> PL observable``.

    Parameters
    ----------
    name  : str or callable
        One of ``"sum_all"``, ``"sum_subset"``, ``"single"``,
        ``"tensor"`` / ``"parity"``, or a custom callable.
    wires : list[int] or None
        For ``"sum_subset"`` and ``"tensor"``: the qubit indices to include.
        If None, all qubits are used (same as ``"sum_all"`` / ``"tensor"``).
    """
    if callable(name):
        # Accept fn(n_qubits) -> observable  OR  fn(n_qubits, wires) -> observable
        import inspect
        sig = inspect.signature(name)
        if len(sig.parameters) >= 2:
            return lambda n_q: name(n_q, wires)
        return name

    key = name.lower()

    if key == "sum_all":
        def _obs(n_qubits: int):
            import pennylane as qml
            return qml.sum(*[qml.PauliZ(i) for i in range(n_qubits)])
        _obs.__name__ = "sum_all"
        return _obs

    if key in ("sum_subset",):
        def _obs(n_qubits: int, _wires=wires):
            import pennylane as qml
            ws = _wires if _wires is not None else list(range(n_qubits))
            return qml.sum(*[qml.PauliZ(i) for i in ws])
        _obs.__name__ = f"sum_subset({wires})"
        return _obs

    if key == "single":
        def _obs(n_qubits: int):
            import pennylane as qml
            return qml.PauliZ(0)
        _obs.__name__ = "single_Z0"
        return _obs

    if key in ("tensor", "parity"):
        def _obs(n_qubits: int, _wires=wires):
            import pennylane as qml
            ws = _wires if _wires is not None else list(range(n_qubits))
            if len(ws) == 1:
                return qml.PauliZ(ws[0])
            obs = qml.PauliZ(ws[0])
            for w in ws[1:]:
                obs = obs @ qml.PauliZ(w)
            return obs
        _obs.__name__ = f"tensor_Z({wires})"
        return _obs

    raise ValueError(
        f"Unknown observable '{name}'. "
        "Choose from: 'sum_all', 'sum_subset', 'single', 'tensor'/'parity', "
        "or pass a callable(n_qubits) -> PL observable."
    )


def _normalisation_factor(obs_name: str, n_qubits: int, wires: Optional[List[int]]) -> float:
    """
    Return the divisor used to keep the output of qml.expval in [-1, 1].

    - sum_all / sum_subset: divide by the number of summed qubits.
    - single / tensor / parity: already in [-1, 1] — divide by 1.
    """
    key = obs_name.lower() if isinstance(obs_name, str) else ""
    if key in ("sum_all",):
        return float(n_qubits)
    if key in ("sum_subset",):
        ws = wires if wires is not None else list(range(n_qubits))
        return float(len(ws))
    return 1.0


# ── GateApplicator ────────────────────────────────────────────────────────────

class GateApplicator:
    """
    Translates pyzx gate objects into PennyLane operations.

    Supports: ZPhase, XPhase, CNOT, CZ, SWAP, HAD, S, T, Z, NOT.
    If params/param_idx are supplied, ZPhase/XPhase consume the next
    trainable parameter instead of their stored phase.
    """

    @staticmethod
    def apply(gate, params=None, param_idx: Optional[List[int]] = None) -> None:
        import pennylane as qml

        name = gate.name

        if name in ("ZPhase", "XPhase"):
            if params is not None and param_idx is not None:
                angle = params[param_idx[0]]
                param_idx[0] += 1
            else:
                angle = float(gate.phase) * math.pi
            wire = int(gate.target) if hasattr(gate, "target") else int(gate.targets[0])
            if name == "ZPhase":
                qml.RZ(angle, wires=wire)
            else:
                qml.RX(angle, wires=wire)

        elif name == "CNOT":
            qml.CNOT(wires=[int(gate.control), int(gate.target)])

        elif name == "CZ":
            qml.CZ(wires=[int(gate.control), int(gate.target)])

        elif name == "SWAP":
            if hasattr(gate, "targets"):
                ws = [int(w) for w in gate.targets]
            elif hasattr(gate, "target1"):
                ws = [int(gate.target1), int(gate.target2)]
            elif hasattr(gate, "q1"):
                ws = [int(gate.q1), int(gate.q2)]
            else:
                ws = [int(gate.control), int(gate.target)]
            qml.SWAP(wires=ws)

        elif name == "HAD":
            qml.Hadamard(wires=int(gate.target))

        elif name == "S":
            if getattr(gate, "adjoint", False):
                qml.adjoint(qml.S)(wires=int(gate.target))
            else:
                qml.S(wires=int(gate.target))

        elif name == "T":
            if getattr(gate, "adjoint", False):
                qml.adjoint(qml.T)(wires=int(gate.target))
            else:
                qml.T(wires=int(gate.target))

        elif name == "Z":
            qml.PauliZ(wires=int(gate.target))

        elif name == "NOT":
            qml.PauliX(wires=int(gate.target))

        else:
            raise ValueError(
                f"GateApplicator: unsupported gate type '{name}'. "
                "Add a case for it rather than silently dropping it, "
                "a dropped gate corrupts the circuit's unitary."
            )


# ── VQCEvaluator ─────────────────────────────────────────────────────────────

class VQCEvaluator:
    """
    Trains and compares the original and ZX-optimized versions of a circuit.

    Parameters
    ----------
    circuit_spec : CircuitSpec
        The registered circuit (fn, n_qubits, n_params).
    cache_data   : dict or OptimizationResult
        ZX optimisation result (from CacheStore or ZXOptimizer).
    embedding    : str or callable, optional
        How input features are encoded. Default ``"angle_Y"``.
        Choices: ``"angle_Y"``, ``"angle_X"``, ``"angle_Z"``,
                 ``"amplitude"``, ``"iqp"``, or ``callable(x, n_qubits)``.
    observable   : str or callable, optional
        Observable measured at the end of each circuit. Default ``"sum_all"``.
        Choices: ``"sum_all"``, ``"sum_subset"``, ``"single"``,
                 ``"tensor"``/``"parity"``, or ``callable(n_qubits)``.
    observable_wires : list[int] or None
        Qubit subset for ``"sum_subset"`` and ``"tensor"``.
        Ignored for ``"sum_all"`` and ``"single"``.
    lr : float
        Default learning rate for gradient descent.

    Example
    -------
        ev = VQCEvaluator(
            registry["circuito_3"],
            cache["circuito_3"],
            embedding="iqp",
            observable="sum_subset",
            observable_wires=[0, 1],
        )
        result = ev.train_both(X, y, steps=200)
    """

    EMBEDDING_CHOICES = ("angle_Y", "angle_X", "angle_Z", "amplitude", "iqp")
    OBSERVABLE_CHOICES = ("sum_all", "sum_subset", "single", "tensor", "parity")

    def __init__(
        self,
        circuit_spec: Any,
        cache_data: Any,
        embedding: Union[str, Callable] = "angle_Y",
        observable: Union[str, Callable] = "sum_all",
        observable_wires: Optional[List[int]] = None,
        lr: float = 0.05,
    ) -> None:
        self.spec = circuit_spec
        self.cache = cache_data
        self.lr = lr
        self._embedding_fn = make_embedding(embedding)
        self._obs_factory = make_observable(observable, wires=observable_wires)
        self._obs_name = observable if isinstance(observable, str) else "custom"
        self._obs_wires = observable_wires
        self._qnode_orig = None
        self._qnode_opt = None

    # ── internals ─────────────────────────────────────────────────────────────

    def _get_n_qubits(self) -> int:
        return self.spec.n_qubits

    def _get_n_params_orig(self) -> int:
        return self.spec.n_params

    def _get_n_params_opt(self) -> int:
        c = self.cache
        return c["n_params_optimized"] if isinstance(c, dict) else c.n_params_optimized

    def _get_opt_gates(self):
        c = self.cache
        if isinstance(c, dict):
            from .cache import _Gate, _Ops
            return _Ops(c["gates"]).gates
        return c.optimized_ops.gates

    def _build_qnodes(self) -> None:
        import pennylane as qml

        n_q = self._get_n_qubits()
        circuit_fn = self.spec.fn
        opt_gates = self._get_opt_gates()
        embedding_fn = self._embedding_fn
        observable = self._obs_factory(n_q)
        norm = _normalisation_factor(self._obs_name, n_q, self._obs_wires)
        dev = qml.device("default.qubit", wires=n_q)

        @qml.qnode(dev)
        def _orig(x, params):
            embedding_fn(x, n_q)
            circuit_fn(params)
            return qml.expval(observable)

        @qml.qnode(dev)
        def _opt(x, params):
            embedding_fn(x, n_q)
            idx = [0]
            for g in opt_gates:
                GateApplicator.apply(g, params=params, param_idx=idx)
            return qml.expval(observable)

        self._qnode_orig = lambda x, p: _orig(x, p) / norm
        self._qnode_opt  = lambda x, p: _opt(x, p)  / norm

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, X, params, use_optimized: bool = False):
        """Return normalised predictions in [-1, 1] for each sample in X."""
        if self._qnode_orig is None:
            self._build_qnodes()
        from pennylane import numpy as np
        fn = self._qnode_opt if use_optimized else self._qnode_orig
        return np.array([fn(x, params) for x in X])

    def predict_proba(self, X, params, use_optimized: bool = False):
        """Return probabilities in [0, 1] (linear rescale from [-1, 1])."""
        return (self.predict(X, params, use_optimized) + 1.0) / 2.0

    def _bce_loss(self, params, X, y, use_optimized: bool):
        from pennylane import numpy as np
        eps = 1e-7
        p = np.clip(self.predict_proba(X, params, use_optimized), eps, 1 - eps)
        return -np.mean(np.array(y) * np.log(p) + (1 - np.array(y)) * np.log(1 - p))

    def train(
        self,
        X,
        y,
        steps: int = 200,
        lr: Optional[float] = None,
        use_optimized: bool = False,
        init_params=None,
        log_every: int = 20,
        threshold: float = 0.5,
        seed: int = 0,
    ) -> Dict:
        """
        Train one circuit (original or optimized) with gradient descent.

        Returns a dict with keys:
            loss, accuracy_final, params_final,
            n_ops, n_params, steps_log, loss_log
        """
        import pennylane as qml
        from pennylane import numpy as np

        if self._qnode_orig is None:
            self._build_qnodes()

        lr = lr or self.lr
        n_params = self._get_n_params_opt() if use_optimized else self._get_n_params_orig()

        if init_params is None:
            rng = _np_plain.random.default_rng(seed)
            params = np.array(rng.uniform(0.0, 2 * math.pi, n_params), requires_grad=True)
        else:
            params = np.array(init_params, requires_grad=True)

        opt = qml.GradientDescentOptimizer(stepsize=lr)
        steps_log, loss_log = [], []

        # opt.step() + a separate loss evaluation, not opt.step_and_cost():
        # for circuits built via GateApplicator (use_optimized=True),
        # step_and_cost() silently returns a stale gradient update.
        for step in range(1, steps + 1):
            params = opt.step(lambda p: self._bce_loss(p, X, y, use_optimized), params)
            if step % log_every == 0:
                steps_log.append(step)
                loss_log.append(float(self._bce_loss(params, X, y, use_optimized)))

        proba = self.predict_proba(X, params, use_optimized)
        preds = (proba >= threshold).astype(int)
        accuracy = float(np.mean(preds == np.array(y)))

        c = self.cache
        n_ops = (
            c["n_ops_optimized"] if use_optimized else c["n_ops_original"]
        ) if isinstance(c, dict) else (
            c.n_ops_optimized if use_optimized else c.n_ops_original
        )

        return {
            "loss": loss_log[-1] if loss_log else float("nan"),
            "accuracy_final": accuracy,
            "params_final": params,
            "n_ops": n_ops,
            "n_params": n_params,
            "steps_log": steps_log,
            "loss_log": loss_log,
        }

    def train_both(
        self,
        X,
        y,
        steps: int = 200,
        lr: Optional[float] = None,
        log_every: int = 10,
        threshold: float = 0.5,
        data_seed: int = 0,
    ) -> Dict:
        """
        Train the original circuit from random parameters, and the
        optimized circuit from the cache's sentinel-derived init_params_rad.

        These two starting points are NOT equivalent, in function space:
        the original circuit starts from a random point on its own
        landscape, while the optimized circuit starts from whatever point
        the sentinel placeholder values happen to land on, which has no
        particular relationship to the original's starting point. This
        method is kept for the simple/historical comparison. For a
        starting point that is provably the same function, in both
        circuits, derived from a single random theta, use
        :meth:`train_both_matched` instead.

        Returns a merged dict suitable for saving to results JSON.
        """
        r_orig = self.train(
            X, y, steps=steps, lr=lr,
            use_optimized=False,
            log_every=log_every, threshold=threshold,
            seed=data_seed + 99,
        )

        c = self.cache
        init_opt = c["init_params_rad"] if isinstance(c, dict) else c.init_params_rad
        r_opt = self.train(
            X, y, steps=steps, lr=lr,
            use_optimized=True,
            init_params=init_opt,
            log_every=log_every, threshold=threshold,
        )

        sym_struct = (
            c.get("symbolic_structure", []) if isinstance(c, dict)
            else c.symbolic_structure
        )

        return {
            "accuracy_original":  r_orig["accuracy_final"],
            "accuracy_optimized": r_opt["accuracy_final"],
            "loss_original":      r_orig["loss_log"],
            "loss_optimized":     r_opt["loss_log"],
            "n_ops_original":     r_orig["n_ops"],
            "n_ops_optimized":    r_opt["n_ops"],
            "n_params_original":  r_orig["n_params"],
            "n_params_optimized": r_opt["n_params"],
            "symbolic_structure": sym_struct,
            "embedding":          getattr(self._embedding_fn, "__name__", "custom"),
            "observable":         self._obs_name,
            "observable_wires":   self._obs_wires,
        }

    def train_both_matched(
        self,
        X,
        y,
        steps: int = 200,
        lr: Optional[float] = None,
        log_every: int = 10,
        threshold: float = 0.5,
        theta_seed: int = 0,
        test_split: Optional[float] = None,
    ) -> Dict:
        """
        Train the original circuit from a random theta, and the optimized
        circuit from phi(theta), the SAME point, carried through the exact
        ZX parameter map (OptimizationResult.build_parameter_map()), so
        both circuits start the run computing the identical function of X.
        Any accuracy difference after training is then attributable to the
        extracted circuit's topology and trainability, not to an arbitrary
        difference in starting point (contrast with train_both()).

        The optimized circuit is trained in its reduced, tied parameter
        space (one free value per ParameterMap group), capping its
        trainable capacity at exactly the number of groups — never more
        than the number of surviving gates, so a CRX/CRY gate's split
        +-theta/2 pair acts as one trainable parameter, not two.

        Parameters
        ----------
        theta_seed : seed for the random original theta (radians, uniform
            [0, 2*pi), one per original parameter).
        test_split : if given (e.g. 0.2), the LAST that fraction of X/y is
            held out as a test set, and the returned dict reports accuracy
            separately for "_train" and "_test". If None (default), X/y
            are used for training only, reported as "_train".

        Raises
        ------
        TypeError
            If cache_data is a plain dict (e.g. loaded from CacheStore)
            instead of a live OptimizationResult, since build_parameter_map()
            needs the SymPy symbolic_structure directly.
        RuntimeError
            If OptimizationResult.verify_equivalence() finds the two
            circuits are not actually equivalent at the matched starting
            point. Known to happen for some CRX-heavy circuits. Call
            verify_equivalence() yourself first if you want to check this
            without risking the exception.
        """
        import pennylane as qml
        from pennylane import numpy as pnp

        if self._qnode_orig is None:
            self._build_qnodes()

        lr = lr or self.lr
        n_q = self._get_n_qubits()
        n_params = self._get_n_params_orig()

        c = self.cache
        if isinstance(c, dict):
            raise TypeError(
                "train_both_matched needs the live OptimizationResult "
                "(as returned by ZXOptimizer.run()), not a dict loaded "
                "from CacheStore -- build_parameter_map() needs the SymPy "
                "symbolic_structure directly."
            )
        eq_diff = c.verify_equivalence(self.spec.fn, n_q)
        if eq_diff > 1e-3:
            raise RuntimeError(
                f"Matched initialisation is not trustworthy for "
                f"'{self.spec.name}': verify_equivalence() found "
                f"max|diff|={eq_diff:.3e} (tol=1e-3). This is a known "
                f"issue for some CRX-heavy circuits -- do not train from "
                f"this starting point."
            )
        pmap = c.build_parameter_map()

        if test_split:
            n_train = int(round((1 - test_split) * len(X)))
            X_tr, y_tr = X[:n_train], y[:n_train]
            X_te, y_te = X[n_train:], y[n_train:]
        else:
            X_tr, y_tr = X, y
            X_te, y_te = None, None

        rng = _np_plain.random.default_rng(theta_seed)
        # theta is sampled directly in the same units circuit_fn expects
        # (its entries ARE the symbolic parameters a, b, c, ... used to
        # build the ParameterMap), not divided by pi again.
        theta = rng.uniform(0.0, 2 * math.pi, n_params)
        free_vec_0 = pmap.free_vec(theta)

        # original circuit: trained freely from theta
        theta_p = pnp.array(theta, requires_grad=True)
        opt1 = qml.GradientDescentOptimizer(stepsize=lr)
        loss_o_log = []
        for step in range(1, steps + 1):
            theta_p = opt1.step(lambda p: self._bce_loss(p, X_tr, y_tr, False), theta_p)
            if step % log_every == 0:
                loss_o_log.append(float(self._bce_loss(theta_p, X_tr, y_tr, False)))

        # optimized circuit: trained in the reduced, tied free-vector space.
        # The pi-scaling is baked into the initial value and the additive
        # constant ONCE, outside the loop -- multiplying the trainable
        # variable by a non-unit constant inside the loss, every step,
        # would scale the effective learning rate by that constant squared
        # (here pi^2, silently wrecking the optimisation).
        G = pnp.array(pmap.G, requires_grad=False)
        C_rad = pnp.array(pmap.C * math.pi, requires_grad=False)
        free_vec = pnp.array(free_vec_0 * math.pi, requires_grad=True)
        opt2 = qml.GradientDescentOptimizer(stepsize=lr)

        def _loss_opt(fv):
            psi = G @ fv + C_rad
            return self._bce_loss(psi, X_tr, y_tr, True)

        loss_p_log = []
        for step in range(1, steps + 1):
            free_vec = opt2.step(_loss_opt, free_vec)
            if step % log_every == 0:
                loss_p_log.append(float(_loss_opt(free_vec)))
        psi_final = pmap.G @ _np_plain.array(free_vec) + pmap.C * math.pi

        def _accuracy(use_optimized, params, Xs, ys):
            proba = self.predict_proba(Xs, params, use_optimized)
            preds = (proba >= threshold).astype(int)
            return float(pnp.mean(preds == _np_plain.array(ys)))

        out = {
            "theta_seed": theta_seed,
            "equivalence_max_abs_diff": eq_diff,
            "n_groups": pmap.n_groups,
            "n_surviving": pmap.n_surviving,
            "accuracy_original_train": _accuracy(False, theta_p, X_tr, y_tr),
            "accuracy_optimized_train": _accuracy(True, psi_final, X_tr, y_tr),
            "loss_original_train": loss_o_log,
            "loss_optimized_train": loss_p_log,
        }
        out["delta_train"] = out["accuracy_optimized_train"] - out["accuracy_original_train"]

        if test_split:
            out["accuracy_original_test"] = _accuracy(False, theta_p, X_te, y_te)
            out["accuracy_optimized_test"] = _accuracy(True, psi_final, X_te, y_te)
            out["delta_test"] = out["accuracy_optimized_test"] - out["accuracy_original_test"]

        return out


# ── Dataset helper ────────────────────────────────────────────────────────────

def create_dataset(n_samples: int, n_qubits: int, seed: int = 0):
    """
    Simple classification dataset: X ~ Uniform[0, pi]^n_qubits.
    Label y=1 if sum(X) > n_qubits * pi/2, else y=0.
    """
    from pennylane import numpy as np

    rng = _np_plain.random.default_rng(seed)
    X = rng.uniform(0, math.pi, (n_samples, n_qubits)).astype(float)
    y = (X.sum(axis=1) > n_qubits * math.pi / 2).astype(int)
    return np.array(X), np.array(y)
