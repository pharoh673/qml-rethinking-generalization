"""Single-job runner: one (experiment, n, N, seed[, ratio]) -> one Fig-3 data point.

Pipeline: load the authors' canonical couplings -> regenerate ED ground states (cached) ->
apply the experiment's randomization -> train the QCNN (CMA-ES, checkpointed) -> evaluate
train/test error and the empirical generalization gap |train_err - test_err| -> append a
summary row to the results CSV.

Test set: the shared 1000-point set, drawn from the SAME distribution as training per the
paper (random labels: i.i.d. uniform; random states: same resample protocol; corruption:
clean TRUE labels — corruption is applied to TRAINING only).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import data_generation as dg
from . import loss as loss_mod
from . import phases
from . import qcnn as qcnn_mod
from . import randomization as rnd
from .config import ExperimentConfig
from .training import TrainConfig, train_qcnn

# experiment name -> authors' top-level directory
_AUTHORS_SUBDIR = {
    "real_labels": "real_labels",
    "random_labels": "random_labels",
    "random_states": "random_states",
    "partial_corruption": "partially_corrupted_labels",
}
_TEST_SEED_OFFSET = 10_000_000  # decorrelate test-set randomization from training


def _load_standard(experiment: str, n: int, N: int | None, test_size: int, periodic: bool):
    """Couplings+labels+states for standard layouts ({exp}/{n}_qubits/...)."""
    j1p, j2p, labp = dg.authors_paths(experiment, n, N=N)
    pd = phases.load_dataset(j1p, j2p, labp)
    sl = slice(0, test_size) if N is None else slice(None)
    states = dg.ground_states_cached(pd.j1[sl], pd.j2[sl], n, periodic=periodic)
    return pd.j1[sl], pd.j2[sl], pd.labels[sl], states


def _load_partial_corruption(n: int, N: int | None, test_size: int, periodic: bool):
    """partially_corrupted_labels uses a flat layout (n=8 only): truth = LABELS_{N}_0."""
    if n != 8:
        raise NotImplementedError("partial_corruption is provided for n=8 only (as in the paper)")
    root = dg.AUTHORS_ROOT / "partially_corrupted_labels"
    if N is None:
        j1p, j2p = root / "J1coef_j1j2_1000_Test_Set", root / "J2coef_j1j2_1000_Test_Set"
        labp = root / "LABELS_1000_Test_Set"
    else:
        d = root / f"{N}_training_data"
        j1p, j2p, labp = d / f"J1coef_j1j2_{N}", d / f"J2coef_j1j2_{N}", d / f"LABELS_{N}_0"
    pd = phases.load_dataset(j1p, j2p, labp)
    sl = slice(0, test_size) if N is None else slice(None)
    states = dg.ground_states_cached(pd.j1[sl], pd.j2[sl], n, periodic=periodic)
    return pd.j1[sl], pd.j2[sl], pd.labels[sl], states


def prepare_datasets(experiment, n, N, seed, ratio, periodic, test_size):
    """Return (train_states, train_labels, test_states, test_labels) for one job."""
    loader = _load_partial_corruption if experiment == "partial_corruption" else (
        lambda *a: _load_standard(experiment, *a)
    )
    _, _, tr_lab, tr_states = loader(n, N, test_size, periodic)
    _, _, te_lab, te_states = loader(n, None, test_size, periodic)
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + _TEST_SEED_OFFSET)

    if experiment == "random_labels":
        tr_lab = rnd.random_labels(len(tr_states), rng_tr)
        te_lab = rnd.random_labels(len(te_states), rng_te)            # same (random) distribution
    elif experiment == "random_states":
        tr_states = rnd.gaussian_resample_states(tr_states, rng_tr)
        te_states = rnd.gaussian_resample_states(te_states, rng_te)   # labels stay true
    elif experiment == "partial_corruption":
        c = int(round(ratio * len(tr_lab)))
        tr_lab = rnd.corrupt_labels(tr_lab, c, rng_tr)                # corrupt TRAINING only
    # real_labels: clean truth, untouched
    return tr_states, tr_lab, te_states, te_lab


def _append_summary(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run_job(cfg: ExperimentConfig, job: dict, write_csv: bool = True, checkpoint: bool = True) -> dict:
    """Run one job dict (from cfg.jobs()) and return its summary row."""
    n, N, seed = cfg.n, job["N"], job["seed"]
    ratio = job.get("ratio")
    tr_states, tr_lab, te_states, te_lab = prepare_datasets(
        cfg.experiment, n, N, seed, ratio, cfg.periodic, cfg.test_size
    )

    fn = qcnn_mod.build_qcnn(n)
    tcfg = TrainConfig(
        dim=qcnn_mod.num_params(n), sigma0=cfg.sigma0, tolfun=cfg.tolfun,
        max_generations=cfg.max_generations, max_restarts=cfg.max_restarts,
        target_accuracy=cfg.target_accuracy, seed=seed,
        popsize=cfg.popsize, popsize_growth=cfg.popsize_growth, time_budget=cfg.time_budget,
    )
    rtag = "" if ratio is None else f"_r{ratio:.3f}"
    # tag uniquely identifies the job (incl. experiment) so a stale/colliding checkpoint
    # is never wrongly resumed (see training.train tag check).
    tag = f"{cfg.name}_{cfg.experiment}_n{n}_N{N}_s{seed}{rtag}"
    ckpt = Path(cfg.checkpoint_dir) / f"{tag}.pkl" if checkpoint else None
    res = train_qcnn(tr_states, tr_lab, n, tcfg, qcnn=fn, checkpoint_path=ckpt,
                     tag=tag, equalize=cfg.equalize_loss)

    train_err = loss_mod.error_probability(fn, res.best_params, tr_states, tr_lab)
    test_err = loss_mod.error_probability(fn, res.best_params, te_states, te_lab)
    row = {
        "name": cfg.name, "experiment": cfg.experiment, "n": n, "N": N, "seed": seed,
        "ratio": ratio, "train_error": round(train_err, 5), "test_error": round(test_err, 5),
        "gen_gap": round(abs(train_err - test_err), 5),
        "train_accuracy": round(res.train_accuracy, 5), "memorized": res.success,
        "generations": res.generations, "restarts": res.restarts,
        "seconds": round(res.seconds, 1),
    }
    if write_csv:
        _append_summary(cfg.results_csv, row)
    return row
