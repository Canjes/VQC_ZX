"""
ZXPipeline — high-level orchestrator.

Ties together the registry, optimizer, cache and evaluator into a single
object so a notebook cell can drive the whole workflow.

Typical usage
-------------
    from vqc_zx import ZXPipeline, CircuitRegistry
    import pennylane as qml
    from pennylane import numpy as np

    registry = CircuitRegistry()

    @registry.register("my_circuit", n_params=8, n_qubits=4)
    def my_circuit(params):
        for i in range(4):
            qml.RX(params[i] * np.pi, wires=i)
        ...

    pipeline = ZXPipeline(registry, strategy="full_reduce")
    pipeline.optimize_all()              # runs ZX, caches per (circuit, strategy)
    pipeline.print_summary()

    X, y = pipeline.create_dataset(n_samples=120, seed=0)
    results = pipeline.run_experiment(X, y, steps=200, embedding="angle_Y")
    pipeline.save_results(results)       # → results/angle_Y__sum_all__full_reduce.json
    pipeline.print_results(results)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from .cache import CacheStore
from .evaluator import VQCEvaluator, create_dataset as _create_dataset
from .optimizer import ZXOptimizer, OptimizationResult, STRATEGIES
from .registry import CircuitRegistry, CircuitSpec
from .tracer import ZXTracer


class ZXPipeline:
    """
    Orchestrates the full VQC-ZX evaluation workflow.

    Parameters
    ----------
    registry    : CircuitRegistry with the circuits to evaluate.
    cache_dir   : directory for ZX cache files (one JSON per circuit+strategy).
    results_dir : directory where experiment JSONs are saved.
    strategy    : ZX simplification strategy. One of:
                  'full_reduce' (default), 'clifford_simp',
                  'interior_clifford', 'pivot_gadget'.
    match_tol   : sentinel matching tolerance (default 0.003).
    """

    def __init__(
        self,
        registry: CircuitRegistry,
        cache_dir: str = "./zx_cache",
        results_dir: str = "./results",
        strategy: str = "full_reduce",
        match_tol: float = 0.003,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(STRATEGIES.keys())}"
            )
        self.registry = registry
        self.strategy = strategy
        self.cache = CacheStore(cache_dir)
        self.results_dir = results_dir
        self._optimizer = ZXOptimizer(strategy=strategy, match_tol=match_tol)
        self._tracer = ZXTracer()
        self._live_results: Dict[str, OptimizationResult] = {}
        os.makedirs(results_dir, exist_ok=True)

    # ── Phase A: ZX optimization ──────────────────────────────────────────────

    def optimize(
        self, circuit_name: str, force: bool = False
    ) -> Dict:
        """
        Run ZX optimization for *circuit_name* and cache the result.

        Cache is keyed by (circuit_name, strategy).
        If already cached and force=False, returns the cached dict directly.
        Returns the cache dict (not the OptimizationResult object).
        """
        if not force and self.cache.has(circuit_name, self.strategy):
            data = self.cache.get(circuit_name, self.strategy)
            print(
                f"[cache/{self.strategy}] {circuit_name}: "
                f"ops {data['n_ops_original']} → {data['n_ops_optimized']}  "
                f"params {data['n_params_original']} → {data['n_params_optimized']}"
            )
            return data

        spec = self.registry[circuit_name]
        result = self._optimizer.run(spec.fn, spec.n_params)
        self.cache.put(circuit_name, result)
        self._live_results[circuit_name] = result

        print(f"[ZX/{self.strategy}] {circuit_name}")
        print(f"  {result.summary()}")
        return self.cache.get(circuit_name, self.strategy)

    def get_result(self, circuit_name: str) -> OptimizationResult:
        """
        Return the live OptimizationResult for *circuit_name* (not the
        serialized cache dict). Needed for build_parameter_map() and
        verify_equivalence(), which operate on the SymPy symbolic_structure
        directly and cannot be reconstructed from the on-disk JSON cache.

        Re-runs the optimizer if this circuit was loaded from an on-disk
        cache in this session rather than freshly optimized (e.g. after
        optimize_all(force=False) found an existing cache file), since the
        live SymPy structure is never persisted to disk.
        """
        if circuit_name not in self._live_results:
            spec = self.registry[circuit_name]
            self._live_results[circuit_name] = self._optimizer.run(spec.fn, spec.n_params)
        return self._live_results[circuit_name]

    def optimize_all(self, force: bool = False) -> Dict[str, Dict]:
        """Optimize every circuit in the registry. Returns {name: cache_dict}."""
        caches = {}
        for name in self.registry:
            caches[name] = self.optimize(name, force=force)
        return caches

    # ── ZX trace ──────────────────────────────────────────────────────────────

    def trace(self, circuit_name: str) -> None:
        """
        Print a step-by-step trace of the ZX reduction for *circuit_name*.
        Does NOT modify the cache.
        """
        import pennylane as qml
        from ._sentinel import SentinelMatcher

        spec = self.registry[circuit_name]
        matcher = SentinelMatcher(spec.n_params)

        with qml.queuing.AnnotatedQueue() as q:
            spec.fn(matcher.sentinels)
        tape = qml.tape.QuantumScript.from_queue(q)
        graph = qml.transforms.to_zx(tape)
        found = matcher.assign_symbolic_phases(graph)

        print(f"=== ZX trace: {circuit_name} (strategy: {self.strategy}) ===")
        print(f"Symbolic phases assigned: {len(found)}")
        steps = self._tracer.trace(graph)
        self._tracer.print_table(steps)

    # ── Phase B: training and evaluation ─────────────────────────────────────

    def create_dataset(self, n_samples: int = 120, seed: int = 0, circuit_name: Optional[str] = None):
        """
        Create a classification dataset.
        If circuit_name is given, uses that circuit's n_qubits.
        Otherwise defaults to the first circuit in the registry.
        """
        if circuit_name is None:
            circuit_name = next(iter(self.registry))
        n_qubits = self.registry[circuit_name].n_qubits
        return _create_dataset(n_samples, n_qubits, seed)

    def run_experiment(
        self,
        X,
        y,
        circuit_names: Optional[List[str]] = None,
        steps: int = 200,
        lr: float = 0.05,
        log_every: int = 10,
        threshold: float = 0.5,
        data_seed: int = 0,
        embedding: str = "angle_Y",
        observable: str = "sum_all",
        observable_wires=None,
    ) -> Dict:
        """
        Train original and optimized versions of each circuit.

        Returns a results dict matching the format used by save_results.
        The strategy used is taken from self.strategy (set at pipeline init).
        """
        circuit_names = circuit_names or list(self.registry)
        run = {
            "created_at": datetime.now().isoformat(),
            "config": {
                "n_samples":        int(len(X)),
                "steps":            int(steps),
                "lr":               float(lr),
                "data_seed":        int(data_seed),
                "log_every":        int(log_every),
                "threshold":        float(threshold),
                "optimization":     "paramzx_symbolic_sympy",
                "strategy":         self.strategy,
                "embedding":        embedding,
                "observable":       observable,
                "observable_wires": observable_wires,
            },
            "circuits": {},
        }

        for name in circuit_names:
            cache_data = self.cache.get(name, self.strategy)
            if cache_data is None:
                print(
                    f"[WARN] {name}: not cached for strategy '{self.strategy}' — "
                    "run optimize_all() first."
                )
                run["circuits"][name] = {"status": "not_cached"}
                continue
            try:
                ev = VQCEvaluator(
                    self.registry[name],
                    cache_data,
                    embedding=embedding,
                    observable=observable,
                    observable_wires=observable_wires,
                    lr=lr,
                )
                res = ev.train_both(
                    X, y,
                    steps=steps,
                    log_every=log_every,
                    threshold=threshold,
                    data_seed=data_seed,
                )
                res["status"] = "ok"
                run["circuits"][name] = res
            except Exception as exc:
                run["circuits"][name] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                print(f"[WARN] {name}: {type(exc).__name__}: {exc}")

        return run

    def save_results(
        self,
        run: Dict,
        filename: Optional[str] = None,
        append: bool = True,
    ) -> str:
        """
        Append *run* to the results JSON file and return its path.

        If *filename* is omitted, it is auto-generated from the run config as:
            {embedding}__{observable}__{strategy}.json

        For example: angle_Y__sum_all__full_reduce.json

        Also computes and updates an aggregate summary across all saved runs.
        """
        if filename is None:
            cfg = run.get("config", {})
            embedding  = cfg.get("embedding",  "angle_Y")
            observable = cfg.get("observable", "sum_all")
            strategy   = cfg.get("strategy",   "full_reduce")
            filename = f"{embedding}__{observable}__{strategy}.json"

        path = os.path.join(self.results_dir, filename)

        all_runs: List[Dict] = []
        if append and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                all_runs = existing
            elif "runs" in existing:
                all_runs = existing["runs"]

        all_runs.append(run)
        summary = self._build_summary(all_runs)

        output = {"summary": summary, "runs": all_runs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return path

    @staticmethod
    def _build_summary(runs: List[Dict]) -> Dict:
        import numpy as np

        all_names = sorted({
            name for r in runs for name in r.get("circuits", {})
        })
        summary: Dict = {"n_runs": len(runs), "circuits": {}}

        for name in all_names:
            ok, errors = [], 0
            for r in runs:
                c = r.get("circuits", {}).get(name)
                if c is None:
                    continue
                if c.get("status") == "ok":
                    ok.append(c)
                else:
                    errors += 1

            if not ok:
                summary["circuits"][name] = {"status": "error_only", "n_errors": errors}
                continue

            def _mean_logs(key: str):
                lists = [c[key] for c in ok if key in c and c[key]]
                if not lists:
                    return []
                n = min(len(l) for l in lists)
                return np.mean([l[:n] for l in lists], axis=0).tolist()

            summary["circuits"][name] = {
                "status": "ok",
                "n_valid_runs": len(ok),
                "n_error_runs": errors,
                "accuracy_original_mean":  float(np.mean([c["accuracy_original"]  for c in ok])),
                "accuracy_optimized_mean": float(np.mean([c["accuracy_optimized"] for c in ok])),
                "loss_original_mean":  _mean_logs("loss_original"),
                "loss_optimized_mean": _mean_logs("loss_optimized"),
            }

        return summary

    # ── display helpers ───────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a compact table of cached optimization results for the current strategy."""
        print(f"Strategy: {self.strategy}")
        print(
            f"{'circuit':<18}  {'ops orig':>8}  {'ops opt':>7}  "
            f"{'params orig':>11}  {'params opt':>10}  symbolic structure"
        )
        print("-" * 80)
        for name in self.registry:
            data = self.cache.get(name, self.strategy)
            if data is None:
                print(f"{name:<18}  (not cached)")
                continue
            syms = data.get("symbolic_structure", [])
            syms_str = str(syms) if syms else "—"
            print(
                f"{name:<18}  {data['n_ops_original']:8}  {data['n_ops_optimized']:7}  "
                f"{data['n_params_original']:11}  {data['n_params_optimized']:10}  {syms_str}"
            )

    def print_results(self, run: Dict) -> None:
        """Print a compact table of a single experiment run."""
        cfg = run.get("config", {})
        print(
            f"strategy={cfg.get('strategy','?')}  "
            f"embedding={cfg.get('embedding','?')}  "
            f"observable={cfg.get('observable','?')}"
        )
        print(f"{'circuit':<18}  {'acc orig':>8}  {'acc opt':>8}  {'delta':>7}")
        print("-" * 50)
        for name, c in run.get("circuits", {}).items():
            if c.get("status") == "ok":
                delta = c["accuracy_optimized"] - c["accuracy_original"]
                print(
                    f"{name:<18}  {c['accuracy_original']:8.3f}  "
                    f"{c['accuracy_optimized']:8.3f}  {delta:+7.3f}"
                )
            else:
                print(f"{name:<18}  {c.get('status', 'error').upper()}")
