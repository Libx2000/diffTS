# Physics-Constrained Residual Diffusion for Global Ocean T/S Forecasting

Minimal research implementation of a global probabilistic ocean temperature and salinity forecasting pipeline. The workflow combines a deterministic 3-D U-Net, conditional residual diffusion, validation-based ensemble calibration, and a bounded physics-constrained correction module.

This public package contains source code only. GLORYS data, preprocessing artifacts, checkpoints, evaluation tables, figures, logs, and manuscript files are intentionally excluded.

## Pipeline

1. Prepare and quality-control daily GLORYS temperature and salinity fields.
2. Train and evaluate the deterministic 3-D U-Net backbone.
3. Estimate deterministic residual statistics and train the conditional residual diffusion model.
4. Fit the ensemble calibration on validation data only.
5. Train and evaluate the bounded joint temperature-salinity physics correction.
6. Run the final test/OOD evaluation and correction-strength ablation.

## Repository layout

```text
config.yaml              Portable experiment configuration
scripts/                 Training, inference, calibration and evaluation entry points
src/diffts/              Models, datasets, losses, metrics and physical diagnostics
tests/                   Lightweight model and metric smoke tests
environment.yml          Conda environment without a platform-specific PyTorch build
requirements.txt         Python dependencies other than PyTorch
upload_to_github.bat     Interactive one-click GitHub upload helper for Windows
```

## Data layout

The code expects one NetCDF file per variable and year:

```text
reanalysis_31year/
├── thetao_glor_1.5deg/
│   └── thetao_glor_YYYY_0_1.5deg.nc
└── so_glor_1.5deg/
    └── so_glor_YYYY_0_1.5deg.nc
```

The expected variables are `thetao` and `so`, sampled on a 1.5° global grid with 15 retained depth levels. Edit `data.root` in `config.yaml` before running. Generated files are written to `artifacts/` and `runs/`, both of which are ignored by Git.

## Installation

```bash
conda env create -f environment.yml
conda activate diffts
```

Install a CUDA-enabled PyTorch build compatible with the GPU and driver by following the official PyTorch installation selector. This is intentionally not pinned in `environment.yml`, because the correct CUDA wheel depends on the target workstation or cluster.

Alternatively, after installing PyTorch:

```bash
python -m pip install -r requirements.txt
```

## Quick validation

These tests do not require GLORYS data or trained checkpoints:

```bash
python tests/smoke_deterministic.py
python tests/smoke_diffusion.py
python tests/smoke_physics_correction.py
python tests/smoke_evaluation.py
```

## Reproduce the workflow

Run all commands from the repository root.

### 1. Data preparation

```bash
python scripts/stage1_prepare.py --config config.yaml --mode inspect
python scripts/stage1_prepare.py --config config.yaml --mode prepare
```

### 2. Deterministic backbone

```bash
python scripts/train_deterministic.py --config config.yaml
python scripts/evaluate_deterministic.py --config config.yaml --split test
python scripts/evaluate_deterministic.py --config config.yaml --split ood
```

### 3. Residual diffusion

```bash
python scripts/stage3_prepare_residual_stats.py --config config.yaml
python scripts/train_residual_diffusion.py --config config.yaml
python scripts/evaluate_residual_diffusion.py --config config.yaml --split test
python scripts/evaluate_residual_diffusion.py --config config.yaml --split ood
```

### 4. Validation-only calibration

Fit calibration parameters on the validation split, then optionally verify them on later periods:

```bash
python scripts/fit_baseline_calibration.py --config config.yaml --mode fit --split val
python scripts/fit_baseline_calibration.py --config config.yaml --mode evaluate --split test
python scripts/fit_baseline_calibration.py --config config.yaml --mode evaluate --split ood
```

### 5. Physics-constrained correction

```bash
python scripts/train_stage4.py --config config.yaml
python scripts/evaluate_stage4.py --config config.yaml --split test
python scripts/evaluate_stage4.py --config config.yaml --split ood
```

### 6. Final evaluation

```bash
python scripts/evaluate_stage5.py --config config.yaml --split test
python scripts/evaluate_stage5.py --config config.yaml --split ood
python scripts/analyze_stage5.py --runs-dir ./runs --bootstrap 2000 --seed 42 --noninferiority-margin-percent 0.5
```

Use `--max-samples N` on ensemble evaluation commands for a short end-to-end check before a full run. Training and ensemble inference are intended for a CUDA GPU.

## Upload to GitHub

1. Create an empty repository on GitHub without adding a README, `.gitignore`, or license.
2. Double-click `upload_to_github.bat`.
3. Paste the repository URL, for example `https://github.com/USERNAME/REPOSITORY.git`.
4. Complete GitHub authentication if prompted.

The helper initializes Git, creates the first commit, sets the branch to `main`, adds or updates `origin`, and pushes without using force.

## Reproducibility notes

- Calibration is fitted on validation data only; test and OOD periods are not used for fitting.
- Random seeds, lead times, temporal splits, correction bounds, and architecture settings are recorded in `config.yaml`.
- Model weights and data-derived artifacts are not distributed. They can be regenerated by following the commands above.
- Before publication, add the paper DOI and repository URL to `CITATION.cff`.

## License

No software license has been selected in this package. Add the license approved by all authors and your institution before making the GitHub repository public.
