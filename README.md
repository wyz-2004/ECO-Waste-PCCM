# ECO-Waste-PCCM

ECO-Waste-PCCM is a compact main-experiment project for auditable constrained
completion of public-sector waste workbooks. It reads the World Bank What a
Waste 3.0 city workbook, completes missing numeric cells, projects outputs into
bounded and closure-consistent states, and returns a ranked audit queue for
human review.

This release contains only the main experiment. Extra evaluation suites,
manuscript-formatting scripts, and paper-production assets are not part of this
project directory.

## Project Layout

```text
ECO-Waste-PCCM/
  config/
    default.json
  data/
    raw/
      What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx
  outputs/
    main/
  src/
    ecowaste_pccm/
      data.py
      model.py
      constraints.py
      metrics.py
      pipeline.py
  run.py
  requirements.txt
```

## Method Scope

The main pipeline performs one workflow:

1. Read the What a Waste 3.0 city workbook.
2. Build a typed city-feature matrix from numeric workbook fields.
3. Standardize percentage fields to proportions and fit split-respecting completion experts.
4. Learn simplex weights on held-out validation cells.
5. Project delivered values into bounded and closure-consistent states.
6. Evaluate the held-out main task.
7. Export the completed workbook and ranked audit queue.

The output is an auditable workbook state, not an automatic replacement for
official records.

## Setup

```powershell
cd E:\waste\ECO-Waste-PCCM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Run the complete main experiment with one command:

```powershell
python run.py
```

Optional overrides:

```powershell
python run.py --seed 7 --output-dir outputs/main_seed7
```

## Outputs

The default run writes to `outputs/main/`:

- `completed_workbook.csv`: original reported cells plus completed missing cells.
- `audit_queue.csv`: ranked missing cells for human review.
- `holdout_predictions.csv`: split-safe held-out predictions for the main task.
- `feature_dictionary.csv`: feature type, bounds, and closure metadata.
- `main_metrics.json`: machine-readable metric summary.
- `run_summary.txt`: short human-readable summary.
