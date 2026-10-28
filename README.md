# vqc_zx

`vqc_zx` applies [ZX-calculus](https://en.wikipedia.org/wiki/ZX-calculus) simplification to variational (parameterised) quantum circuits, and compares the original circuit against its ZX-optimised counterpart on a machine learning task. It covers the whole workflow: registering a circuit, running the ZX simplification, caching the result, and training/evaluating both versions under a shared, verified starting point.

## Why ParamZX

Standard [PyZX](https://github.com/Quantomatic/pyzx) assumes every gate's phase is a fixed, known number. Its simplification rules test phases against specific values (0, π/2, π, 3π/2) to decide how to rewrite the graph, which breaks the moment a phase is actually a trainable parameter instead of a constant.

`vqc_zx` works around this with a **sentinel** approach: each trainable parameter is temporarily given a unique, non-Clifford placeholder value (the reciprocal of a distinct prime number) before the circuit is converted to a ZX graph. This only works because of [**ParamZX**](https://github.com/mjsutcliffe99/ParamZX), a fork of PyZX by Matthew Sutcliffe and Aleks Kissinger that extends the simplification engine to treat any symbolic (SymPy) phase as non-Clifford, regardless of its numeric value. Without this fork, `full_reduce` has no way to tell a placeholder phase apart from an ordinary Clifford one, and would silently rewrite it away.

If you use ParamZX, please cite the paper it is based on:

> M. Sutcliffe and A. Kissinger, *"Fast classical simulation of quantum circuits via parametric rewriting in the ZX-calculus"*, 2024. [arXiv:2403.06777](https://arxiv.org/abs/2403.06777)

## Installation

`vqc_zx` is not a standalone package: it needs the ParamZX fork of PyZX on the Python path, not the regular `pyzx` from PyPI. Clone both repositories as siblings:

```
your-project/
├── vqc_zx/      <- this repository
└── ParamZX/     <- https://github.com/mjsutcliffe99/ParamZX
```

```bash
git clone https://github.com/mjsutcliffe99/ParamZX.git
git clone <this-repo-url> vqc_zx
pip install -r vqc_zx/requirements.txt
```

`vqc_zx.setup_pyzx()` finds `ParamZX/` automatically if it is a sibling of (or inside) the `vqc_zx/` directory, so no manual `sys.path` editing is needed. Pass a path explicitly (`setup_pyzx("/path/to/ParamZX")`) if your layout is different.

## Quickstart

```python
from vqc_zx import setup_pyzx, CircuitRegistry, ZXOptimizer, VQCEvaluator, create_dataset

setup_pyzx()  # adds ParamZX/ to sys.path so "import pyzx" resolves to the fork

import pennylane as qml
from pennylane import numpy as np

registry = CircuitRegistry()

@registry.register("demo_crz", n_qubits=4, n_params=11)
def demo_crz(params):
    for i in range(4):
        qml.RX(params[i] * np.pi, wires=i)
    for i in range(4):
        qml.RZ(params[i + 4] * np.pi, wires=i)
    qml.CRZ(params[8] * np.pi, wires=[3, 2])
    qml.CRZ(params[9] * np.pi, wires=[2, 1])
    qml.CRZ(params[10] * np.pi, wires=[1, 0])

# Phase A: simplify the circuit with ZX-calculus
result = ZXOptimizer(strategy="full_reduce").run(demo_crz, n_params=11)
print(result.summary())

# Confirm the extracted circuit really is the same function as the original
diff = result.verify_equivalence(demo_crz, n_qubits=4)
print("max diff:", diff)  # should be near machine precision

# Phase B: train both versions from a matched starting point
X, y = create_dataset(n_samples=120, n_qubits=4, seed=0)
evaluator = VQCEvaluator(registry["demo_crz"], result, lr=0.05)
res = evaluator.train_both_matched(X, y, steps=200, theta_seed=0, test_split=0.2)
print(res["accuracy_original_test"], res["accuracy_optimized_test"])
```

For the full workflow, including the `ZXPipeline` orchestrator, caching, embeddings, observables, gate noise, and the reduction tracer, see `quickstart_vqc_zx.ipynb` and `tutorial_vqc_zx.ipynb`.

## Package structure

| Module | What it does |
|---|---|
| `_sentinel.py` | Assigns and recovers the non-Clifford sentinel phases that keep trainable parameters identifiable through ZX simplification. |
| `optimizer.py` | `ZXOptimizer` runs the ZX pipeline (convert, simplify, extract) and reconstructs the exact affine map between original and extracted parameters. |
| `tracer.py` | `ZXTracer` replays the simplification step by step, recording which rule fired and the resulting graph statistics. |
| `registry.py` | `CircuitRegistry` keeps named circuit definitions together with their qubit and parameter counts. |
| `cache.py` | `CacheStore` persists ZX optimisation results to disk, keyed by circuit name and strategy. |
| `evaluator.py` | `VQCEvaluator` trains and compares the original and ZX-optimised circuits; `GateApplicator` turns a `pyzx` gate sequence into PennyLane operations. |
| `pipeline.py` | `ZXPipeline` ties all of the above into one object for running a full experiment. |

## Status

Built as part of an MSc dissertation in Physics Engineering (University of Minho). The core finding, across the circuits tested, is that ZX simplification does not increase accuracy or reduce hardware cost, even though it does reveal genuine structural redundancy in some circuit families. See the dissertation for the full experimental results.

## License

[MIT](LICENSE).
