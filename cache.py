"""
JSON-based cache for ZX optimization results.

Cache files are named  {circuit_name}__{strategy}.json
so different strategies produce independent cache entries.
The expensive ZX reduction only runs once per (circuit, strategy) pair.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .optimizer import OptimizationResult


def _serialize_gate(g) -> dict:
    d: dict = {"name": g.name}
    for attr in ("target", "control", "target1", "target2", "q1", "q2"):
        if hasattr(g, attr) and getattr(g, attr) is not None:
            d[attr] = int(getattr(g, attr))
    if hasattr(g, "targets") and g.targets is not None:
        d["targets"] = [int(w) for w in g.targets]
    if hasattr(g, "phase") and g.phase is not None:
        d["phase_pi"] = float(g.phase)
    return d


class _Gate:
    """Lightweight gate object reconstructed from a serialized dict."""
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)
        if "phase_pi" in d:
            self.phase = d["phase_pi"]


class _Ops:
    """Reconstructed circuit from a list of serialized gate dicts."""
    def __init__(self, gates: List[dict]) -> None:
        self.gates = [_Gate(g) for g in gates]


class CacheStore:
    """
    Persists and retrieves ZX optimization results as JSON files.

    File naming convention:
        {circuit_name}__{strategy}.json

    This ensures that different ZX strategies produce independent cache
    entries — you can compare 'full_reduce' vs 'clifford_simp' for the
    same circuit without one overwriting the other.

    Usage
    -----
        store = CacheStore("./zx_cache")
        store.put("circuito_2", result)           # strategy from result.strategy

        data = store.get("circuito_2", "full_reduce")   # returns dict or None
        ops  = store.load_ops("circuito_2", "full_reduce")  # returns _Ops
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, name: str, strategy: str = "full_reduce") -> str:
        return os.path.join(self.directory, f"{name}__{strategy}.json")

    # ── write ────────────────────────────────────────────────────────────────

    def put(self, name: str, result: OptimizationResult) -> None:
        """Serialize and write *result* to cache (filename uses result.strategy)."""
        data = {
            "circuit_name":          name,
            "strategy":              result.strategy,
            "optimization":          "paramzx_symbolic_sympy",
            "n_params_original":     result.n_params_original,
            "n_params_optimized":    result.n_params_optimized,
            "n_ops_original":        result.n_ops_original,
            "n_ops_optimized":       result.n_ops_optimized,
            "n_symbolic_before":     result.n_symbolic_before,
            "n_symbolic_after":      result.n_symbolic_after,
            "symbolic_structure":    result.symbolic_structure,
            "original_param_names":  result.original_param_names,
            "init_params_rad":       result.init_params_rad,
            "gates":                 [_serialize_gate(g) for g in result.optimized_ops.gates],
        }
        path = self._path(name, result.strategy)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── read ─────────────────────────────────────────────────────────────────

    def has(self, name: str, strategy: str = "full_reduce") -> bool:
        return os.path.exists(self._path(name, strategy))

    def get(self, name: str, strategy: str = "full_reduce") -> Optional[Dict]:
        """Return cached dict or None if not found."""
        path = self._path(name, strategy)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_ops(self, name: str, strategy: str = "full_reduce") -> Optional[_Ops]:
        """Return a reconstructed _Ops object or None."""
        data = self.get(name, strategy)
        if data is None:
            return None
        return _Ops(data["gates"])

    # ── management ───────────────────────────────────────────────────────────

    def delete(self, name: str, strategy: str = "full_reduce") -> None:
        path = self._path(name, strategy)
        if os.path.exists(path):
            os.remove(path)

    def list_cached(self) -> List[Tuple[str, str]]:
        """Return list of (circuit_name, strategy) tuples for all cache files."""
        results = []
        for fname in os.listdir(self.directory):
            if fname.endswith(".json") and "__" in fname:
                stem = fname[:-5]
                parts = stem.rsplit("__", 1)
                if len(parts) == 2:
                    results.append((parts[0], parts[1]))
        return sorted(results)

    def __repr__(self) -> str:
        cached = self.list_cached()
        return f"CacheStore(directory={self.directory!r}, cached={cached})"
