"""CMA-ES training with checkpoint/resume and restart-until-memorized.

Faithful to the authors' main_code.py:
  * optimizer: CMA-ES (`cma`), sigma0 = 0.7, tolfun = 5e-4 (labels) / 1e-4 (random states)
  * parameters initialized ~ Uniform[0, 2π]
  * OUTER LOOP: restart CMA-ES until training accuracy reaches `target_accuracy` (=1.0).
    This memorization step is the mechanism behind the generalization-gap result.

Added for robustness (Kaggle 12 h sessions / interruptions; not in the authors' code):
  * checkpoint of the full CMA-ES state + RNG every `checkpoint_every` generations, with
    automatic resume; results appended incrementally to CSV so partial sweeps are never lost.

The generic `train()` works on any (objective, accuracy_fn); `train_qcnn()` wires it to the
QCNN loss/accuracy for a labelled dataset.
"""
from __future__ import annotations

import csv
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cma
import numpy as np

from . import loss as loss_mod
from . import qcnn as qcnn_mod


@dataclass
class TrainConfig:
    dim: int                               # number of parameters
    sigma0: float = 0.7
    tolfun: float = 5e-4
    max_generations: int = 1500            # per CMA-ES run (before a restart)
    target_accuracy: float = 1.0           # restart until train accuracy >= this
    max_restarts: int = 30
    popsize: int | None = None             # base popsize; None -> CMA default 4 + 3 ln(dim)
    popsize_growth: float = 2.0            # IPOP: multiply popsize each restart (1.0 = off)
    time_budget: float | None = None       # wall-clock cap in seconds (stop restarting after)
    seed: int = 0
    checkpoint_every: int = 10             # generations between checkpoints
    accuracy_check_every: int = 20         # generations between in-run accuracy checks (early stop)
    reinit_on_restart: bool = True         # fresh U[0,2π] init each restart
    init_low: float = 0.0
    init_high: float = 2 * np.pi


@dataclass
class TrainResult:
    best_params: np.ndarray
    best_loss: float
    train_accuracy: float
    generations: int                       # total across restarts
    restarts: int
    success: bool
    seconds: float
    history: list = field(default_factory=list)  # (restart, gen, best_loss, acc)


# --------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------
def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)                      # atomic


def _load_checkpoint(path: Path) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _append_csv(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _base_popsize(cfg: TrainConfig) -> int:
    return cfg.popsize or (4 + int(3 * np.log(cfg.dim)))


def _popsize_for_restart(cfg: TrainConfig, restart: int) -> int:
    """IPOP: grow the population on each restart to escape local minima."""
    g = cfg.popsize_growth if cfg.popsize_growth and cfg.popsize_growth > 1 else 1.0
    return int(round(_base_popsize(cfg) * (g**restart)))


def _new_es(x0: np.ndarray, cfg: TrainConfig, cma_seed: int, popsize: int) -> cma.CMAEvolutionStrategy:
    opts = {"tolfun": cfg.tolfun, "maxiter": cfg.max_generations,
            "seed": cma_seed, "verbose": -9, "popsize": popsize}
    return cma.CMAEvolutionStrategy(list(x0), cfg.sigma0, opts)


# --------------------------------------------------------------------------------------
# generic training core
# --------------------------------------------------------------------------------------
def train(
    objective: Callable[[np.ndarray], float],
    accuracy_fn: Callable[[np.ndarray], float],
    cfg: TrainConfig,
    checkpoint_path: str | Path | None = None,
    result_path: str | Path | None = None,
    tag: str = "",
) -> TrainResult:
    """Minimize `objective` with CMA-ES, restarting until `accuracy_fn` >= target."""
    t0 = time.time()
    ckpt = Path(checkpoint_path) if checkpoint_path else None

    state = _load_checkpoint(ckpt)
    if state is not None and state.get("tag") != tag:
        state = None  # checkpoint belongs to a different job (stale/colliding) -> ignore
    if state is not None:
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        es = pickle.loads(state["es"])
        restart, total_gen = state["restart"], state["total_gen"]
        best_params, best_loss = state["best_params"], state["best_loss"]
        history = state["history"]
    else:
        rng = np.random.default_rng(cfg.seed)
        x0 = rng.uniform(cfg.init_low, cfg.init_high, cfg.dim)
        es = _new_es(x0, cfg, cma_seed=cfg.seed * 1000 + 1, popsize=_popsize_for_restart(cfg, 0))
        restart, total_gen = 0, 0
        best_params, best_loss = x0, float("inf")
        history = []

    def checkpoint():
        if ckpt is None:
            return
        _save_checkpoint(ckpt, {
            "tag": tag, "es": pickle.dumps(es), "rng_state": rng.bit_generator.state,
            "restart": restart, "total_gen": total_gen,
            "best_params": best_params, "best_loss": best_loss, "history": history,
        })

    def _time_up() -> bool:
        return cfg.time_budget is not None and (time.time() - t0) >= cfg.time_budget

    while restart <= cfg.max_restarts:
        early_acc = None
        while not es.stop() and not _time_up():
            sols = es.ask()
            es.tell(sols, [objective(np.asarray(s)) for s in sols])
            total_gen += 1
            if es.result.fbest < best_loss:
                best_loss, best_params = float(es.result.fbest), np.asarray(es.result.xbest)
            if total_gen % cfg.checkpoint_every == 0:
                checkpoint()
            # in-run early stop: don't keep grinding after the labels are already memorized
            if total_gen % cfg.accuracy_check_every == 0:
                early_acc = float(accuracy_fn(best_params))
                if early_acc >= cfg.target_accuracy:
                    break

        acc = early_acc if early_acc is not None and early_acc >= cfg.target_accuracy \
            else float(accuracy_fn(best_params))
        history.append((restart, total_gen, best_loss, acc))
        if result_path:
            _append_csv(result_path, {"tag": tag, "restart": restart, "generations": total_gen,
                                      "best_loss": best_loss, "train_accuracy": acc,
                                      "popsize": _popsize_for_restart(cfg, restart)})
        checkpoint()
        if acc >= cfg.target_accuracy:
            return TrainResult(best_params, best_loss, acc, total_gen, restart, True,
                               time.time() - t0, history)
        if _time_up():
            break  # wall-clock budget exhausted -> return best-so-far (success=False)

        # IPOP restart: fresh CMA-ES with a grown population (and optionally fresh init)
        restart += 1
        x0 = rng.uniform(cfg.init_low, cfg.init_high, cfg.dim) if cfg.reinit_on_restart else best_params
        es = _new_es(x0, cfg, cma_seed=cfg.seed * 1000 + restart + 1,
                     popsize=_popsize_for_restart(cfg, restart))

    acc = float(accuracy_fn(best_params))
    return TrainResult(best_params, best_loss, acc, total_gen, restart, False,
                       time.time() - t0, history)


# --------------------------------------------------------------------------------------
# QCNN convenience wrapper
# --------------------------------------------------------------------------------------
def train_qcnn(
    states: np.ndarray,
    labels: np.ndarray,
    n: int,
    cfg: TrainConfig | None = None,
    qcnn=None,
    checkpoint_path=None,
    result_path=None,
    tag: str = "",
    equalize: bool = True,
    **cfg_overrides,
) -> TrainResult:
    """Train the n-qubit QCNN to (memorize) the given (states, labels) via CMA-ES.

    `equalize` adds the authors' incorrect-class-equalizing term to the objective (default
    True — required to memorize the harder configs; see src/loss.py).
    """
    fn = qcnn or qcnn_mod.build_qcnn(n)
    if cfg is None:
        cfg = TrainConfig(dim=qcnn_mod.num_params(n), **cfg_overrides)

    def objective(p):
        return loss_mod.dataset_loss(fn, p, states, labels, equalize=equalize)

    def accuracy_fn(p):
        return loss_mod.accuracy(fn, p, states, labels)

    return train(objective, accuracy_fn, cfg, checkpoint_path, result_path, tag)
