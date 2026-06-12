#!/usr/bin/env python3
"""Fetch the authors' released artifacts into data/authors/.

We treat the authors' coupling/label files as the CANONICAL dataset:
    J1coef_*  J2coef_*   -> the (j1, j2) coupling points (Uniform[-4,4]^2 samples)
    LABELS_*             -> the 4-class phase labels (0=SPT, 1=FM, 2=AFM, 3=trivial)

The phase-labeling FUNCTION is not in the released code and the paper defers it to
Verresen et al. (2017); our free-fermion reconstruction only matches ~62% of their
labels (curved boundaries). Hence we adopt their published labels verbatim rather
than regenerate them from a guessed map.  See README -> "Open items / ZENODO-CHECK".

Their ground-state files (train_groundstates.npy) are TensorCircuit-MPS tensors /
Qibo statevectors and are downloaded only for VALIDATION (observable cross-checks);
the pipeline itself regenerates ground states from the couplings in native
PennyLane format (see src/data_generation.py).

Source: https://github.com/bpcarlos/understanding_QML_rethinking_gen  (v1.0.0,
Zenodo DOI 10.5281/zenodo.10277124).

Platform-agnostic: tries `git clone --depth 1`, falls back to the GitHub zip.
Runs unchanged on Windows, WSL2, Docker, and Kaggle.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/bpcarlos/understanding_QML_rethinking_gen.git"
ZIP_URL = "https://github.com/bpcarlos/understanding_QML_rethinking_gen/archive/refs/heads/main.zip"

# data/authors/ relative to repo root (this file lives in scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "data" / "authors"


def _have_git() -> bool:
    return shutil.which("git") is not None


def _clone(dest: Path) -> bool:
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError) as exc:  # pragma: no cover
        print(f"[fetch] git clone failed ({exc}); falling back to zip download.")
        return False


def _download_zip(dest: Path) -> bool:
    print(f"[fetch] downloading {ZIP_URL} ...")
    with urllib.request.urlopen(ZIP_URL) as resp:  # noqa: S310 (trusted URL)
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        tmp = dest.parent / "_authors_zip_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        zf.extractall(tmp)
        # the zip extracts to <repo>-main/ ; move its contents up into dest
        roots = [p for p in tmp.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"unexpected zip layout: {roots}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(roots[0]), str(dest))
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"[fetch] {DEST} already populated — skipping. (delete it to re-fetch.)")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    ok = (_have_git() and _clone(DEST)) or _download_zip(DEST)
    if not ok:
        print("[fetch] ERROR: could not obtain the authors' data.", file=sys.stderr)
        return 1

    # quick inventory so the user can see what landed
    labels = sorted(DEST.rglob("LABELS_*"))
    j1 = sorted(DEST.rglob("J1coef_*"))
    gs = sorted(DEST.rglob("train_groundstates.npy"))
    print(f"[fetch] done -> {DEST}")
    print(f"[fetch]   LABELS_* files : {len(labels)}")
    print(f"[fetch]   J1coef_* files : {len(j1)}")
    print(f"[fetch]   ground-state .npy files (validation only): {len(gs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
