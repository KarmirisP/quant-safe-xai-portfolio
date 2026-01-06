# Quant-Safe XAI Pipeline for Dynamic Portfolio Management

This repository contains code and a reproducible research paper describing a **Quant-Safe** machine-learning pipeline for equity return prediction and portfolio construction, with an emphasis on:

- **Data-leakage prevention** (point-in-time features, walk-forward evaluation)
- **Explainability** via SHAP (out-of-sample explanations only)
- **Actionable portfolio layer** (top-N selection, volatility scaling, caps, turnover control)
- **Execution + accounting** (IBKR fractional shares, reconciled fills, daily mark-to-market performance logs)

## Contents

- `code/`
  - `quant_pipeline_final.py` — research backtest pipeline (walk-forward, SHAP, outputs)
  - `quant_pipeline_final_live_ibkr_accounting.py` — live trading + accounting-quality logs (IBKR)
  - `live_forecast.py` — train-on-history / score-latest inference script (no walk-forward)
- `paper/`
  - `main.tex`, `references.bib` — LaTeX paper source
  - `equity_curve.png`, `shap_summary.png` — figures generated from results
- `results/`
  - example outputs (metrics, equity curve, SHAP global importance)

## Methodology (high level)

1. **Signal layer**
   - Train on all *labeled* history (where forward horizon return is known)
   - Score latest *unlabeled* rows for live inference

2. **Evaluation layer (research)**
   - Walk-forward validation to avoid look-ahead bias
   - Out-of-sample SHAP computed only on test folds

3. **Portfolio layer**
   - Top-N by predicted 6-month return
   - Volatility scaling (inverse 3-month volatility)
   - Position caps + cash buffer + turnover control

4. **Execution + accounting layer (live)**
   - IBKR fractional shares via `ib_insync`
   - Fill reconciliation into trade blotter
   - Daily mark-to-market equity curve
   - Realized/unrealized P&L by ticker
   - Slippage estimates (implementation shortfall vs reference price)

## Setup

Create a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

##Running the research backtest
bash

python code/quant_pipeline_final.py
Outputs (example):

results_DJI_predictions.csv

results_DJI_equity_curve.csv

results_DJI_metrics.csv

results_DJI_shap_oos_global.csv

##Running live inference (no execution)
bash

python code/live_forecast.py
Running IBKR execution + accounting (paper trading recommended)
Prerequisites:

IBKR TWS or IB Gateway running

API enabled: Enable ActiveX and Socket Clients

##Use paper account first

bash

python code/quant_pipeline_final_live_ibkr_accounting.py --dry-run
Then (paper trading):

bash

python code/quant_pipeline_final_live_ibkr_accounting.py --paper

##Notes and limitations
Historical backtests can be inflated by frictions (slippage, fees), survivorship bias, and data availability.

The primary contribution here is the Quant-Safe architecture and explainable regime detection, not a guarantee of future performance.

Use paper trading and small size; validate stability out-of-sample before risking capital.

Citation
If you use this repository in academic work, please cite the Zenodo DOI (see badge/DOI in the GitHub release and Zenodo record).

License
See LICENSE.
