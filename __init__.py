"""
vqc_zx — VQC evaluation framework using ZX-calculus symbolic optimization.

Public API
----------
    from vqc_zx import (
        ZXPipeline,
        CircuitRegistry, CircuitSpec,
        ZXOptimizer, OptimizationResult, ParameterMap,
        ZXTracer, TraceStep,
        CacheStore,
        VQCEvaluator, GateApplicator,
        create_dataset,
        setup_pyzx,
    )
"""

from .cache import CacheStore
from .evaluator import (
    GateApplicator,
    VQCEvaluator,
    create_dataset,
    make_embedding,
    make_observable,
)
from .optimizer import OptimizationResult, ParameterMap, STRATEGIES, ZXOptimizer
from .pipeline import ZXPipeline
from .registry import CircuitRegistry, CircuitSpec
from .tracer import TraceStep, ZXTracer


def setup_pyzx(paramzx_path: str | None = None) -> str:
    """
    Add ParamZX to sys.path so `import pyzx` resolves to the local fork.

    Searches for ParamZX/ relative to this file's location if no path given.
    Returns the resolved path.
    """
    import os
    import sys

    if paramzx_path is None:
        # vqc_zx/ is typically inside the project root, alongside ParamZX/
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(pkg_dir, "..", "ParamZX"),
            os.path.join(pkg_dir, "ParamZX"),
        ]:
            c = os.path.normpath(candidate)
            if os.path.isdir(os.path.join(c, "pyzx")):
                paramzx_path = c
                break

    if paramzx_path is None:
        raise FileNotFoundError(
            "ParamZX/pyzx not found. Pass the path explicitly: "
            "setup_pyzx('/path/to/ParamZX')"
        )

    paramzx_path = os.path.normpath(paramzx_path)
    if paramzx_path not in sys.path:
        sys.path.insert(0, paramzx_path)

    return paramzx_path


__all__ = [
    "ZXPipeline",
    "CircuitRegistry",
    "CircuitSpec",
    "ZXOptimizer",
    "OptimizationResult",
    "ParameterMap",
    "STRATEGIES",
    "ZXTracer",
    "TraceStep",
    "CacheStore",
    "VQCEvaluator",
    "GateApplicator",
    "create_dataset",
    "make_embedding",
    "make_observable",
    "setup_pyzx",
]
