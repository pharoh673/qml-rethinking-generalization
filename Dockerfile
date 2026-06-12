# One-command reproducible environment.
# Build/run via docker-compose (preferred):
#   docker compose build
#   docker compose run experiment --config configs/random_labels_n8.yaml
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default BLAS/threading to single-threaded INSIDE each container/worker so that
# the job-runner's process-level parallelism doesn't oversubscribe the 16 threads.
# (The job-runner sets these per worker too; this is the safe default.)
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# build-essential: some scientific wheels / quimb extras may compile.
# git: used by scripts/fetch_authors_data.py to pull the authors' released data.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install -e .

# run_experiment.py is the single entry point; compose passes --config ... through.
ENTRYPOINT ["python", "run_experiment.py"]
