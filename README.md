# ECO-Waste-PCCM

## What Is Included

ECO-Waste-PCCM treats workbook completion as a split-respecting decision loop:

1. Build typed city, country, and Defra workbook matrices from public waste data.
2. Fit cross-fitted completion experts under shared masks and validation splits.
3. Select risk-specific L1/L2 completions rather than a single generic imputation target.
4. Project predictions into feasible accounting states with shared constraint operators.
5. Attach guarded residual-risk evidence as an audit signal.
6. Rank cells for reveal-refit human verification without automatically replacing official data.

The implementation is designed for reproducible evaluation, not for automatic public-sector data correction.

## Repository Layout

```text
.
├── data/
│   └── raw/                 # Place public raw input files here
├── outputs/                 # Generated tables, protocol files, and summaries
├── scripts/
│   ├── ecowaste_core.py
│   ├── run_experiment.py
│   ├── run_longterm_experiment.py
│   ├── run_aaai_experiments.py
│   ├── run_dual_head_analysis.py
│   ├── run_technical_depth_experiments.py
│   ├── run_defra_external_validation.py
│   ├── run_country_generalization.py
│   ├── run_evidence_enhancement.py
│   ├── build_cost_aware_verification.py
│   └── run_algorithm_experiments.ps1
└── requirements.txt
```

## Data

The scripts expect public raw files under `data/raw/`. See `data/README.md` for the expected filenames. Raw third-party datasets are not committed in this release directory, because redistribution depends on the original data providers.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The scripts use `ECOWASTE_PROJECT_ROOT` to find `data/` and `outputs/`. The provided PowerShell runner sets this variable automatically.

## Run

Run the main algorithmic experiment suite:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_algorithm_experiments.ps1
```

Run individual experiments when you only need one section:

```powershell
python .\scripts\run_longterm_experiment.py
python .\scripts\run_defra_external_validation.py
python .\scripts\run_country_generalization.py
python .\scripts\run_evidence_enhancement.py
```

Generated results are written under `outputs/`, mainly as CSV and JSON files.

## Main Scripts

- `ecowaste_core.py`: shared data loading, feature typing, constraints, risk construction, and utility functions.
- `run_experiment.py`: core PCCM pipeline, baselines, cross-fitted experts, projection, evidence, acquisition, and protocol outputs.
- `run_longterm_experiment.py`: city random-mask and structured missingness experiments.
- `run_defra_external_validation.py`: Defra external workbook validation and latest-year audit simulation.
- `run_country_generalization.py`: country-level transfer and stress-boundary experiments.
- `run_evidence_enhancement.py`: guarded residual-risk evidence analysis.
- `run_dual_head_analysis.py`: L1/L2 risk-head comparison.
- `run_technical_depth_experiments.py`: component and decision-operator ablations.
- `build_cost_aware_verification.py`: cost-aware verification summary tables.
