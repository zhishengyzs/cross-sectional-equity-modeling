# Rolling Cross-Sectional Strategy - 50 NASDAQ Stocks

A quantitative trading strategy that uses rolling cross-sectional Ridge regression to predict stock returns and select top-performing stocks from a universe of 50 NASDAQ stocks.

## Quickstart

From the repository root:

```bash
pip install -r requirements.txt
python -m src.strategy_50 --config config.yaml
python -m src.generate_plots --config config.yaml
```

- **Strategy** reads data from `data/` and writes all CSV outputs to `outputs/tables/`.
- **Plots** read those CSVs from `outputs/tables/` and write PNGs to `outputs/figures/`.

The `outputs/` directory is **not tracked** by git. After a fresh clone you must run the above commands to regenerate tables and figures.

## Project Overview

This project implements a machine learning-based trading strategy that:
- Uses a rolling 36-month window to train Ridge regression models
- Predicts 20-day forward returns for 50 NASDAQ stocks
- Selects the top 5 stocks each month based on predictions
- Backtests the strategy against SPY benchmark
- Analyzes performance across different VIX regimes

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository and go to the project root.
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Strategy

From the **repository root**:

```bash
python -m src.strategy_50 --config config.yaml
```

Default config path is `config.yaml`. This will:
- Load stock data from `data/` (see `config.yaml`: `data_dir`)
- Compute technical features (returns, momentum, volatility, RSI)
- Train rolling cross-sectional models
- Generate predictions and select top 5 stocks per month
- Calculate performance metrics
- **Save all CSV outputs to `outputs/tables/`:**
  - `top5_selection.csv` – monthly stock selections
  - `backtest_results.csv` – portfolio returns vs SPY
  - `vix_regime.csv` – performance by VIX regime
  - `performance.csv` – summary statistics
  - `predictions.csv` – all model predictions

### Generating Visualizations

After running the strategy:

```bash
python -m src.generate_plots --config config.yaml
```

This reads from `outputs/tables/` and writes to `outputs/figures/`:
- `plot_cumulative_returns.png` – cumulative returns comparison
- `plot_monthly_returns.png` – monthly returns bar chart
- `plot_vix_regime.png` – strategy returns vs VIX
- `plot_drawdown.png` – drawdown analysis
- `plot_performance_table.png` – performance summary table

### Configuration

Edit `config.yaml` at the repo root to change:
- `data_dir` – input data directory (default: `data`)
- `output_tables` – where strategy CSVs are written (default: `outputs/tables`)
- `output_figures` – where plots are written (default: `outputs/figures`)
- `train_window_months`, `horizon_days`, `top_k`, `model`, `alpha` – strategy parameters

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── src/
│   ├── strategy_50.py      # Main strategy implementation
│   └── generate_plots.py   # Visualization generation
├── data/                    # TRACKED – required to run (stock CSVs, VIX, SPY)
├── outputs/                 # GENERATED – not tracked; regenerate with scripts
│   ├── tables/              # backtest_results.csv, predictions.csv, etc.
│   └── figures/             # plot_*.png
└── reports/
    └── report.docx
```

## Strategy Details

### Features
- **Returns**: 1-day, 5-day, 20-day returns
- **Momentum**: 5-day, 20-day, 63-day (3-month) momentum
- **Volatility**: 20-day and 60-day annualized volatility
- **RSI**: 14-day Relative Strength Index
- **VIX**: Volatility index as market regime indicator

### Model
- **Algorithm**: Ridge Regression (L2 regularization, alpha from config)
- **Training Window**: 36 months rolling (config: `train_window_months`)
- **Prediction Horizon**: 20-day forward returns
- **Selection**: Top 5 stocks per month, equal-weighted (config: `top_k`)

### Performance Metrics
- Annualized return and volatility
- Sharpe ratio
- Maximum drawdown
- Hit rate vs SPY benchmark
- VIX regime analysis

## Data Requirements

The `data/` directory is **tracked** and must be present after cloning. It should contain CSV files with:
- `Date` (or `DATE` for VIX)
- `Close` or `Adj Close` (price data)
- `Volume` (trading volume)

Required files: one CSV per stock ticker, plus `VIX.csv` and `SPY.csv`.

## Notes

- The strategy uses only the provided data in `data/`; no external fetching.
- A 36-month rolling window is used, so the first predictions start after 36 months of data.
- All predictions are made without look-ahead bias (training on [t-36, t-1], predicting for t).
- Outputs in `outputs/` are regenerated on each run and are not committed to the repo.
