# People Analytics - Turnover Prediction

[Português](README.md) | **English**

![version](https://img.shields.io/badge/version-v1.0.0-blueviolet?style=flat-square)
![status](https://img.shields.io/badge/status-Completed-2ECC71?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![DAX](https://img.shields.io/badge/DAX-Analytical--Calculations-4F9DE0?style=flat-square)
![Power BI](https://img.shields.io/badge/Power--BI-Premium--Interface-F2C811?style=flat-square&logo=powerbi&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-006400?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8E44AD?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[![Live Dashboard](https://img.shields.io/badge/Power_BI-View_Live_Dashboard-F2C811?style=flat-square&logo=powerbi&logoColor=black)](https://app.powerbi.com/view?r=eyJrIjoiNzg1MmZhNjctMTI0Ny00ZDIxLWFlMjItZTZiMDRhZjFlZWUwIiwidCI6IjI4NDVhN2ExLWQ3ZTMtNDBjNC1hMGYwLWY4NWI5OWY2Mjc2YyJ9)

End-to-end People Analytics project combining data engineering, machine learning and Power BI visualization to predict and analyze employee turnover.

## Table of Contents

- [About the Project](#about-the-project)
- [Structure](#structure)
- [Data Pipeline](#data-pipeline)
- [Machine Learning Model](#machine-learning-model)
  - [Model Comparison](#model-comparison-5-fold-cross-validation)
  - [Validation Methodology](#validation-methodology)
  - [Known Limitations](#known-limitations)
  - [Top 10 Exit Factors](#top-10-exit-factors)
  - [Risk Distribution](#risk-distribution)
- [Power BI Dashboard](#power-bi-dashboard)
  - [DAX Measures](#dax-measures)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Key Results](#key-results)
- [Data Dictionary](docs/dicionario_dados.md)
- [License](#license)

## About the Project

- Dataset: IBM HR Analytics Employee Attrition (1,470 employees, 35 original variables)
- After cleaning and feature engineering: 52 columns (see `docs/dicionario_dados.md`)
- Goal: identify employees at risk of leaving and the factors that contribute most
- Stack: Python (pandas, scikit-learn, XGBoost, SHAP) + Power BI

## Structure

```
dados/
    raw/                          Original CSV (source of truth)
    processed/                    Clean data, ML scoring and support tables
scripts/
    pipeline_limpeza_hr.py        ETL: cleaning, downcasting, PT translation
    pipeline_ml_optimizado.py     ML: 5 models + SMOTE + Optuna + scoring
    gerar_graficos.py             Visualizations for documentation
docs/
    dicionario_dados.md           Description of each column
    imagens/                      Model charts
    imagens/dashboard/            Dashboard page screenshots
models/
    modelo_attrition_optimizado.pkl
templates/                        SVG background generators for the pages (HTML)
dashboards/
    People_Analytics_HR_Attrition.pbip
    *.Report/                     Report with 4 pages
    *.SemanticModel/              Model: processed table + ML tables
requirements.txt                  Project dependencies
requirements-lock.txt             Pinned versions (reproducibility)
```

## Data Pipeline

The cleaning pipeline turns the raw CSV into an optimized dataset:

- Removes 4 useless columns (zero variance + ID)
- Type downcasting (511 KB -> 94 KB, 82% reduction)
- Translates columns and categorical values to Portuguese
- Creates pre-computed bands (age, salary, distance, tenure)
- Creates ordered ordinal labels for charts
- Applies log1p to MonthlyIncome (skewness 1.37 -> 0.29)
- Zero rows removed (outliers kept for business relevance)

## Machine Learning Model

### Model Comparison (5-fold Cross-Validation)

![Comparison](docs/imagens/comparacao_modelos.png)

| Model | AUC-ROC | Recall | F1 | Precision | Accuracy |
|--------|---------|--------|-----|-----------|----------|
| Logistic Regression | 0.812 | 73.8% | 0.479 | 35.4% | 74.1% |
| Random Forest | 0.813 | 18.6% | 0.301 | 80.0% | 86.1% |
| LightGBM | 0.803 | 38.0% | 0.468 | 60.8% | 86.1% |
| XGBoost | 0.814 | 43.0% | 0.511 | 63.0% | 86.7% |
| XGBoost + SMOTE + Optuna | 0.819 | 60.8% | 0.519 | 45.3% | 81.8% |

Selected model: **XGBoost + SMOTE + Optuna** (best AUC-ROC and best F1, with high recall).
In People Analytics, detecting who will leave matters more than avoiding false alarms,
so we prioritize recall without giving up a good AUC.

### Validation Methodology

- **Stratified 5-fold cross-validation**: all reported metrics are out-of-fold
  (each row is predicted while outside the training set), not in-sample.
- **SMOTE inside the CV pipeline** (`ImbPipeline`): oversampling only sees the
  training data of each fold, preventing synthetic-data leakage into the test set.
- **Baselines with `class_weight="balanced"`** for a fair comparison against the final model.
- **Optuna**: 80 trials maximizing AUC-ROC to tune the XGBoost hyperparameters.

### Known Limitations

- **Modest AUC gain**: the optimized model (0.819) beats the XGBoost baseline (0.814)
  by a margin within noise. The real gain is in recall (43% -> 61%), an effect of
  SMOTE + `scale_pos_weight`, not of a fundamentally superior model.
- **45% precision**: fewer than half of those flagged as high risk actually leave.
  Acceptable by business logic (a false alarm is cheap), but it must be communicated.
- **Per-employee scoring is in-sample**: the `RiscoSaida_Prob` feeding the dashboard
  comes from the model trained on the full dataset, so it is optimistic. The honest
  metrics (CV) are reported separately in the table above.
- **Small sample**: 237 exits out of 1,470 records; metrics have relevant variance.
- **Encoding**: nominal categorical variables use LabelEncoder (artificial ordering);
  trees tolerate it, but dedicated encoding would be cleaner.

### Confusion Matrix

![Matrix](docs/imagens/matriz_confusao.png)

### ROC Curve

![ROC](docs/imagens/curva_roc.png)

### Top 10 Exit Factors

![Features](docs/imagens/feature_importance.png)

Main turnover drivers (model importance):
1. Stock option level (those without tend to leave more) - 9.7%
2. Work pressure (overtime + poor balance) - 5.7%
3. Job involvement - 5.7%
4. Performance rating - 5.0%
5. Environment satisfaction - 4.4%

### SHAP - Explainability

![SHAP](docs/imagens/shap_summary.png)

### Risk Distribution

![Risk](docs/imagens/distribuicao_risco.png)

The scoring classifies each employee into 3 levels, using fixed business
thresholds over the exit probability:

| Class | Rule | Employees |
|--------|-------|---------------|
| High | probability >= 40% | 312 |
| Medium | probability 20-40% | 155 |
| Low | probability < 20% | 1,003 |

### Turnover by Department

![Department](docs/imagens/turnover_departamento.png)

## Power BI Dashboard

> **[Open the live dashboard](https://app.powerbi.com/view?r=eyJrIjoiNzg1MmZhNjctMTI0Ny00ZDIxLWFlMjItZTZiMDRhZjFlZWUwIiwidCI6IjI4NDVhN2ExLWQ3ZTMtNDBjNC1hMGYwLWY4NWI5OWY2Mjc2YyJ9)** (Power BI Service)

The report is organized into 4 pages, each with its own SVG background
(generated by the templates in `templates/`) and slicers for Department,
Gender and Overtime:

### Home Page

![Home Page](docs/imagens/dashboard/pagina_inicial.png)

### 1. Risk Overview

Turnover rate, exits and high-risk employees, broken down by job role,
department and age band.

![Risk Overview](docs/imagens/dashboard/visao_geral_de_risco.png)

### 2. Exit Factors

Average salary, salary gap, salary band and overtime versus turnover,
with a risk table by job role and department.

![Exit Factors](docs/imagens/dashboard/fatores_de_saida.png)

### 3. Career and Growth

Turnover by tenure, stock options, job level and travel frequency
as retention factors.

![Career and Growth](docs/imagens/dashboard/carreira_e_crescimento.png)

### 4. Predictive Model

XGBoost + SMOTE + Optuna: metrics, feature importance and distribution
by risk level.

![Predictive Model](docs/imagens/dashboard/modelo_preditivo.png)

### DAX Measures

The model contains measures organized by display folder:

- **General KPIs**: Total Employees, Exits, Turnover Rate, Stayed
- **Compensation**: Average Salary (overall/exits/stayed), Salary Gap, % No Stock Options
- **Satisfaction**: Averages of environment, job, involvement, balance, relationship
- **Demographics**: Average age, distance, years at company, years since promotion, % overtime
- **ML Risk**: High/Medium/Low risk, % high risk, average risk, real turnover among high risk, estimated cost
- **Model KPIs**: AUC-ROC, Recall, F1 and Precision of the selected model (ml_comparacao_modelos table)

### ML Support Tables

- **ml_comparacao_modelos**: metrics of the 5 tested models (feeds the Predictive Model page)
- **ml_feature_importance**: importance of each feature (%)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

cd scripts

# 1. Cleaning and preparation
python pipeline_limpeza_hr.py

# 2. ML training and scoring (~2 min with Optuna)
python pipeline_ml_optimizado.py

# 3. Generate charts (optional)
python gerar_graficos.py
```

## Requirements

```
python >= 3.9
pandas
numpy
pyarrow
scikit-learn
xgboost
lightgbm
shap
optuna
imbalanced-learn
matplotlib
seaborn
joblib
```

Exact versions in `requirements.txt` (and `requirements-lock.txt` for reproducibility).

## Key Results

- Turnover rate: **16.1%** (237 of 1,470)
- The model detects **61%** of employees who leave (recall)
- Selected model AUC-ROC: **0.819**
- 312 employees classified as high risk (>= 40% probability)
- Estimated turnover cost: average salary x 1.5 x exits
- Top factor: absence of stock options

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
