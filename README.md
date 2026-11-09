# Optiver Realized Volatility Prediction

An econometric and machine learning pipeline built for the Optiver Realized Volatility Prediction Kaggle competition. Utilizing high-frequency limit order book and trade data provided by the quantitative trading firm Optiver, this project predicts 10-minute realized volatility by combining market microstructure features, volatility modeling, macro-timeline reconstruction, and dual-tree gradient boosting ensembles. Models are evaluated using Root Mean Square Percentage Error (RMSPE).

## Experiment & Results

The pipeline was developed incrementally, with each experiment introducing additional features or modeling components. The following table summarizes the Out-of-Fold (OOF) cross-validation results.

| Stage | Architecture / Feature Addition | OOF RMSPE | Improvement |
| --- | --- | ---: | ---: |
| **Baseline** | Base WAP, spreads, and volume aggregations | 0.23296 | - |
| **Trial 1** | **+ HAR-RV & Volatility Slope:** Added multi-scale realized volatility features (\(RV_{150}\), \(RV_{300}\)) and OLS-based volatility trend estimation. | 0.23262 | -0.00034 |
| **Trial 2** | **+ Order Flow Imbalance (OFI):** Added tick-level bid/ask order flow features to capture changes in liquidity and trading pressure. | 0.23221 | -0.00041 |
| **Trial 3** | **+ t-SNE Time Reconstruction:** Used cross-sectional volatility patterns and market beta correlations to infer the ordering of anonymized `time_id` values. | 0.23156 | -0.00065 |
| **Trial 4** | **+ LightGBM & CatBoost Ensemble:** Combined leaf-wise and symmetric tree boosting models with complementary categorical feature handling. | **0.23099** | -0.00057 |

**Total improvement: 0.00197 RMSPE** compared with the baseline.

---

## Methodology

### 1. Market Microstructure Features

- **Order Flow Imbalance (OFI):** Captures changes in bid/ask pressure from order book updates, including price movements and volume changes. OFI variance over the 10-minute window is used to measure fluctuations in liquidity and order flow.

- **Heterogeneous Autoregressive Model (HAR-RV):** Introduces realized volatility features at different time scales, including \(RV_{150}\), \(RV_{300}\), and \(RV_{600}\), along with their ratios and interactions. These features help capture volatility persistence and changes in volatility intensity.

- **Volatility Slope:** Applies vectorized OLS regression to estimate the volatility trend across four quarters of the trading window. The resulting slope helps distinguish increasing, decreasing, and relatively stable volatility patterns.

### 2. Time Reconstruction

The original chronological ordering of `time_id` values is anonymized. To recover temporal relationships, the pipeline uses cross-sectional volatility patterns across stocks.

- Applies **t-SNE (perplexity = 400)** to the cross-sectional volatility matrix to obtain a low-dimensional representation of market states.
- Exploits correlations in stock volatility and market beta to identify similarities between different time periods.
- Uses the inferred temporal relationships to construct additional market-level volatility and regime features.

This approach allows the model to incorporate cross-stock information that is not available from individual order book observations alone.

### 3. Ensemble Architecture

The final model combines LightGBM and CatBoost to leverage their different tree-growing strategies and categorical feature handling.

- **LightGBM:** Uses leaf-wise tree growth. The `stock_id` feature is handled through out-of-fold target encoding.
- **CatBoost:** Uses symmetric trees and Ordered Target Statistics to handle categorical features directly.
- **Loss Function:** Both models use weighted RMSE with sample weights proportional to \(1/y^2\), aligning the training objective with RMSPE.

---

## Repository Structure

```text
realized-volatility-prediction/
├── .gitignore
├── requirements.txt
├── README.md
├── data/                       # Add to .gitignore
│   ├── book_train.parquet/
│   ├── trade_train.parquet/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── kaggle_submission.ipynb
└── src/
    ├── __init__.py
    ├── config.py               # Hyperparameters and path variables
    ├── metrics.py              # Define metric RMSPE
    ├── data_loader.py          # Parallel processing wrappers (joblib)
    ├── features/
    │   ├── __init__.py
    │   ├── har_rv.py           # Multi-scale realized volatility and volatility slope
    │   ├── microstructure.py   # Feature Engineering for Limit Order Book and Trading data
    │   └── macro_tsne.py      # Macro-timeline reconstruction
    ├── models/
    │   ├── __init__.py
    │   ├── lightgbm_model.py   # LightGBM with target encoding
    │   └── catboost_model.py   # CatBoost with categorical features
    └── train.py                # GroupKFold loop and ensembling logic

```

## Execution & Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Placement

Place the Optiver `.parquet` and `.csv` files inside the `data/` directory. Add this directory to `.gitignore` to avoid committing large dataset files.

### Running the Pipeline

Run the complete training pipeline:

```bash
python src/train.py
```

The pipeline executes feature engineering, t-SNE-based time reconstruction, cross-validation, model training, and ensemble blending.

For Kaggle submissions, see the [Kaggle Notebook](https://www.kaggle.com/code/quanglapnguyen/optiver-trading), which contains the corresponding inference pipeline.

