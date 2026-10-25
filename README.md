# Context-Dependent Model Suitability in Epidemiological Forecasting: Evidence from Bangladesh Measles Surveillance

This repository contains the replication codebase and datasets for the dual-resolution empirical forecasting study benchmarking classical structural time-series models against machine learning architectures across national and subnational measles surveillance in Bangladesh.

---

## 1. Experimental Architecture

The framework is partitioned into two independent empirical phases:
* **Phase 1 (National Monthly, 2012–2025):** Evaluates 7 forecasting architectures (ETS, SARIMAX, Prophet, XGBoost, LightGBM, SVR, MLP) across 14 years of WHO surveillance under Base and Google Trends augmented feature sets.
* **Phase 2 (Subnational Daily, 2026):** Evaluates models under high volatility and severe sparsity across all 8 administrative divisions of Bangladesh during the 2026 epidemic.

Validation follows an expanding-window rolling-origin protocol (64 forecast instances in Phase 1; 55 instances per division in Phase 2). Multiple testing is controlled via the Benjamini–Hochberg False Discovery Rate (FDR) procedure.

---

## 2. Environment Setup

Experiments were executed under Python 3.14.4. To guarantee deterministic numerical replication, install the locked dependencies:

```bash
# Clone the repository
git clone https://github.com/[YOUR-USERNAME]/[YOUR-REPO-NAME].git
cd [YOUR-REPO-NAME]

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt