# Verification record

Verified on 13 November 2026 with Python 3.12.

- 5 pytest regression tests passed.
- Python source compilation passed.
- DOM integration checks passed for rendered metrics, primary form submission, result rendering, error states, text escaping and button reset.
- All notebook code cells executed in an isolated IPython process and saved with outputs. Socket-based Jupyter kernels were unavailable in the build environment; the notebooks retain a normal Python 3 kernelspec for local Jupyter use.
- Browser connection timed out; pixel-level browser rendering and mobile interactions have not been visually verified. Responsive CSS breakpoints are implemented.
- HTTP integration tests use FastAPI TestClient. No public deployment is included.

## Measured results

Dataset: 6,000 synthetic transactions, seed 42. Fraud prevalence: 2.82%. Holdout: 1,200 transactions. Selected by validation AP: XGBoost.

| Model | Test precision | Test recall | Test F1 | Test AP | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.162 | 0.176 | 0.169 | 0.104 | 0.027 |
| Random Forest | 0.174 | 0.235 | 0.200 | 0.150 | 0.033 |
| XGBoost | 0.140 | 0.176 | 0.156 | 0.152 | 0.032 |

Test AP bootstrap 95% interval for the selected model: [0.073, 0.268]. This interval concerns one synthetic sample only. Notebook includes SHAP beeswarm and waterfall figures. CSV import is checked for valid rows, missing columns and invalid feature values.

## Reproduce

Install `requirements.lock`, then run `python -m pytest -q`. Install Node 24+, run `npm ci && npm run test:ui`. Open the notebook in Jupyter and run all cells. See README for ports and configuration.
