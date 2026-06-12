"""Config schema (dataclass + YAML) for config-driven experiments.

    python run_experiment.py --config configs/random_labels_n8.yaml
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

VALID_EXPERIMENTS = {"real_labels", "random_labels", "random_states", "partial_corruption"}


@dataclass
class ExperimentConfig:
    experiment: str                          # one of VALID_EXPERIMENTS
    n: int                                   # qubits
    N_values: list[int]                      # training-set sizes to sweep
    seeds: list[int] = field(default_factory=lambda: [0])
    corruption_ratios: list[float] | None = None  # partial_corruption only
    periodic: bool = False                   # OBC by default (matches authors' DMRG)
    test_size: int = 1000

    # CMA-ES (authors' defaults + memorization aids)
    sigma0: float = 0.7
    tolfun: float = 5e-4
    max_generations: int = 1500
    max_restarts: int = 30
    target_accuracy: float = 1.0
    equalize_loss: bool = True   # authors' incorrect-class-equalizing term (needed to memorize)
    popsize: int | None = None   # base CMA population (None -> 4 + 3 ln(dim))
    popsize_growth: float = 2.0  # IPOP: grow population each restart (1.0 = off)
    time_budget: float | None = None  # per-job wall-clock cap in seconds (None = uncapped)

    # IO
    name: str = "experiment"
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "logs"

    def __post_init__(self):
        if self.experiment not in VALID_EXPERIMENTS:
            raise ValueError(f"experiment={self.experiment!r} not in {sorted(VALID_EXPERIMENTS)}")
        if self.experiment == "partial_corruption" and not self.corruption_ratios:
            raise ValueError("partial_corruption requires `corruption_ratios`")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    def jobs(self) -> list[dict]:
        """Expand the config into the cartesian product of (N, seed[, ratio]) jobs."""
        out = []
        ratios = self.corruption_ratios if self.experiment == "partial_corruption" else [None]
        for N in self.N_values:
            for seed in self.seeds:
                for ratio in ratios:
                    job = {"experiment": self.experiment, "n": self.n, "N": N, "seed": seed}
                    if ratio is not None:
                        job["ratio"] = ratio
                    out.append(job)
        return out

    @property
    def results_csv(self) -> Path:
        return Path(self.results_dir) / f"{self.name}.csv"
