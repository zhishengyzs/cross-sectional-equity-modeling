#!/usr/bin/env python3
"""
Rolling Cross-Sectional Strategy - 50 NASDAQ Stocks
Final Version with Complete Data
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import yaml

warnings.filterwarnings('ignore')

RF_ANNUAL = 0.02


def load_config(config_path: str) -> dict:
    """Load config YAML; paths resolved relative to cwd (repo root)."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_all_data(data_dir: Path):
    """Load all 50 stock CSVs + VIX + SPY"""
    print("Loading data from", data_dir)

    all_stocks = []
    tickers = []

    for f in sorted(data_dir.glob('*.csv')):
        ticker = f.stem

        # Skip VIX and SPY for stock universe
        if ticker in ['VIX', 'SPY']:
            continue

        try:
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'])
            df['ticker'] = ticker

            # Use Adj Close if available
            if 'Adj Close' in df.columns:
                df['price'] = df['Adj Close']
            else:
                df['price'] = df['Close']

            df['volume'] = df['Volume']

            if len(df) > 500:
                all_stocks.append(df[['Date', 'price', 'volume', 'ticker']])
                tickers.append(ticker)
        except Exception as e:
            print(f"  Error loading {ticker}: {e}")

    stock_df = pd.concat(all_stocks, ignore_index=True)
    print(f"Loaded {len(tickers)} stocks: {sorted(tickers)}")

    # Load VIX
    vix_path = data_dir / 'VIX.csv'
    vix_df = pd.read_csv(vix_path)
    if 'DATE' in vix_df.columns:
        vix_df = vix_df.rename(columns={'DATE': 'Date'})
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    vix_df['VIX'] = vix_df['Close']
    vix_df = vix_df[['Date', 'VIX']].dropna()
    print(f"VIX data: {len(vix_df)} days")

    # Load SPY
    spy_path = data_dir / 'SPY.csv'
    spy_df = pd.read_csv(spy_path)
    spy_df['Date'] = pd.to_datetime(spy_df['Date'])
    if 'Adj Close' in spy_df.columns:
        spy_df['SPY'] = spy_df['Adj Close']
    else:
        spy_df['SPY'] = spy_df['Close']
    spy_df = spy_df[['Date', 'SPY']]
    print(f"SPY data: {len(spy_df)} days")

    return stock_df, vix_df, spy_df, tickers


def compute_features(df):
    """Compute all features per ticker"""
    print("Computing features...")
    results = []

    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker].sort_values('Date').copy()

        # Returns
        sub['ret_1d'] = sub['price'].pct_change(1)
        sub['ret_5d'] = sub['price'].pct_change(5)
        sub['ret_20d'] = sub['price'].pct_change(20)
        sub['log_ret'] = np.log(sub['price'] / sub['price'].shift(1))

        # Momentum (including 3-month = 63 trading days)
        sub['mom_5d'] = np.log(sub['price'] / sub['price'].shift(5))
        sub['mom_20d'] = np.log(sub['price'] / sub['price'].shift(20))
        sub['mom_63d'] = np.log(sub['price'] / sub['price'].shift(63))

        # Volatility
        sub['vol_20d'] = sub['log_ret'].rolling(20).std() * np.sqrt(252)
        sub['vol_60d'] = sub['log_ret'].rolling(60).std() * np.sqrt(252)

        # RSI (14-day)
        delta = sub['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        sub['rsi_14'] = 100 - (100 / (1 + gain / loss))

        # Forward return (target)
        sub['fwd_ret_20d'] = sub['price'].shift(-20) / sub['price'] - 1

        results.append(sub)

    return pd.concat(results, ignore_index=True)


def create_monthly_panel(df, vix_df):
    """Create month-end panel with VIX"""
    print("Creating monthly panel...")

    df['month_end'] = df['Date'] + pd.offsets.MonthEnd(0)
    monthly = df.groupby(['ticker', 'month_end']).tail(1).copy()

    # Merge VIX
    vix_df['month_end'] = vix_df['Date'] + pd.offsets.MonthEnd(0)
    vix_monthly = vix_df.groupby('month_end')['VIX'].last().reset_index()
    monthly = monthly.merge(vix_monthly, on='month_end', how='left')
    monthly['VIX'] = monthly['VIX'].ffill()

    return monthly


def rolling_cross_sectional_ols(panel, rolling_window: int, alpha: float):
    """
    Rolling Cross-Sectional OLS
    Train on [t-36, t-1], predict for t
    NO LOOK-AHEAD BIAS
    """
    from sklearn.linear_model import Ridge

    print(f"Running rolling OLS (window={rolling_window} months)...")

    features = ['ret_1d', 'ret_5d', 'ret_20d',
                'mom_5d', 'mom_20d', 'mom_63d',
                'vol_20d', 'vol_60d', 'rsi_14', 'VIX']

    months = sorted(panel['month_end'].unique())
    print(f"Total months: {len(months)}")

    all_preds = []

    for i, current_month in enumerate(months):
        if i < rolling_window:
            continue

        # Training: past ROLLING_WINDOW months (EXCLUDING current)
        train_months = months[i - rolling_window : i]
        train_data = panel[panel['month_end'].isin(train_months)].copy()
        train_data = train_data.dropna(subset=features + ['fwd_ret_20d'])

        # Test: current month only
        test_data = panel[panel['month_end'] == current_month].copy()
        test_data = test_data.dropna(subset=features)

        if len(train_data) < 100 or len(test_data) < 10:
            continue

        X_train = train_data[features].values
        y_train = train_data['fwd_ret_20d'].values
        X_test = test_data[features].values

        # Ridge regression
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        test_data['pred'] = model.predict(X_test)
        all_preds.append(test_data[['month_end', 'ticker', 'pred', 'fwd_ret_20d', 'VIX']])

    result = pd.concat(all_preds, ignore_index=True)
    print(f"Generated predictions for {result['month_end'].nunique()} months")
    return result


def select_top_n(predictions, top_n: int):
    """Select top N stocks each month"""
    print(f"Selecting Top-{top_n} stocks per month...")

    selections = []
    for month in predictions['month_end'].unique():
        month_data = predictions[predictions['month_end'] == month].copy()
        month_data = month_data.dropna(subset=['pred', 'fwd_ret_20d'])

        if len(month_data) < top_n:
            continue

        top = month_data.nlargest(top_n, 'pred').copy()
        top['weight'] = 1.0 / top_n
        top['rank'] = range(1, top_n + 1)
        selections.append(top)

    return pd.concat(selections, ignore_index=True)


def backtest(selections, spy_df):
    """Compute portfolio returns and merge with SPY"""
    print("Computing backtest results...")

    # Portfolio returns
    port_ret = selections.groupby('month_end').apply(
        lambda x: np.average(x['fwd_ret_20d'], weights=x['weight'])
    ).reset_index(name='port_ret')

    # Add VIX
    vix_monthly = selections.groupby('month_end')['VIX'].first().reset_index()
    port_ret = port_ret.merge(vix_monthly, on='month_end')

    # SPY returns
    spy_df['month_end'] = spy_df['Date'] + pd.offsets.MonthEnd(0)
    spy_monthly = spy_df.groupby('month_end')['SPY'].last().reset_index()
    spy_monthly['spx_ret'] = spy_monthly['SPY'].pct_change()

    port_ret = port_ret.merge(spy_monthly[['month_end', 'spx_ret']], on='month_end', how='inner')
    port_ret = port_ret.dropna()

    return port_ret


def calculate_stats(port_ret):
    """Calculate performance statistics"""
    n = len(port_ret)

    # Strategy stats
    ann_ret = port_ret['port_ret'].mean() * 12
    ann_vol = port_ret['port_ret'].std() * np.sqrt(12)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum_wealth = (1 + port_ret['port_ret']).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = (port_ret['port_ret'] > port_ret['spx_ret']).mean()

    # SPY stats
    spx_ann_ret = port_ret['spx_ret'].mean() * 12
    spx_ann_vol = port_ret['spx_ret'].std() * np.sqrt(12)
    spx_sharpe = (spx_ann_ret - RF_ANNUAL) / spx_ann_vol if spx_ann_vol > 0 else 0

    return {
        'N_Months': n,
        'Strategy_Ann_Return': ann_ret,
        'Strategy_Ann_Vol': ann_vol,
        'Strategy_Sharpe': sharpe,
        'Strategy_MaxDD': max_dd,
        'Hit_Rate_vs_SPX': hit_rate,
        'SPX_Ann_Return': spx_ann_ret,
        'SPX_Ann_Vol': spx_ann_vol,
        'SPX_Sharpe': spx_sharpe
    }


def vix_regime_analysis(port_ret):
    """Analyze performance by VIX regime"""
    vix_median = port_ret['VIX'].median()
    print(f"VIX Median: {vix_median:.2f}")

    def regime_stats(df, name):
        if len(df) < 3:
            return None
        ann_ret = df['port_ret'].mean() * 12
        ann_vol = df['port_ret'].std() * np.sqrt(12)
        sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else 0
        hit_rate = (df['port_ret'] > df['spx_ret']).mean()
        return {
            'Regime': name,
            'N_Months': len(df),
            'Ann_Return': ann_ret,
            'Ann_Vol': ann_vol,
            'Sharpe': sharpe,
            'Hit_Rate': hit_rate
        }

    high_vix = port_ret[port_ret['VIX'] > vix_median]
    low_vix = port_ret[port_ret['VIX'] <= vix_median]

    high_stats = regime_stats(high_vix, f'High_VIX_>{vix_median:.1f}')
    low_stats = regime_stats(low_vix, f'Low_VIX_<={vix_median:.1f}')

    return pd.DataFrame([high_stats, low_stats]), vix_median


def main():
    parser = argparse.ArgumentParser(description='Rolling cross-sectional strategy')
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path.cwd()
    data_dir = root / config['data_dir']
    output_tables = root / config['output_tables']
    output_tables.mkdir(parents=True, exist_ok=True)

    rolling_window = config['train_window_months']
    top_n = config['top_k']
    alpha = config.get('alpha', 1.0)

    print("=" * 70)
    print("ROLLING CROSS-SECTIONAL STRATEGY - 50 NASDAQ STOCKS")
    print("=" * 70)

    # Load data
    stock_df, vix_df, spy_df, tickers = load_all_data(data_dir)
    print(f"\nStock universe: {len(tickers)} stocks")
    print(f"Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")

    # Feature engineering
    features_df = compute_features(stock_df)

    # Create monthly panel
    monthly_panel = create_monthly_panel(features_df, vix_df)
    print(f"Monthly panel: {monthly_panel.shape[0]} rows, {monthly_panel['month_end'].nunique()} months")

    # Rolling prediction
    predictions = rolling_cross_sectional_ols(monthly_panel, rolling_window, alpha)

    # Select top N
    selections = select_top_n(predictions, top_n)
    selections.to_csv(output_tables / 'top5_selection.csv', index=False)
    print("Saved:", output_tables / 'top5_selection.csv')

    # Backtest
    port_ret = backtest(selections, spy_df)

    # Calculate stats
    stats = calculate_stats(port_ret)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY (50 STOCKS)")
    print("=" * 70)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # VIX regime analysis
    regime_df, vix_med = vix_regime_analysis(port_ret)

    print("\n" + "=" * 70)
    print("VIX REGIME ANALYSIS")
    print("=" * 70)
    print(regime_df.to_string(index=False))

    # Save results
    port_ret.to_csv(output_tables / 'backtest_results.csv', index=False)
    regime_df.to_csv(output_tables / 'vix_regime.csv', index=False)
    pd.DataFrame([stats]).to_csv(output_tables / 'performance.csv', index=False)
    predictions.to_csv(output_tables / 'predictions.csv', index=False)

    print("\n" + "=" * 70)
    print("FILES SAVED:")
    print(f"  - {output_tables / 'top5_selection.csv'}")
    print(f"  - {output_tables / 'backtest_results.csv'}")
    print(f"  - {output_tables / 'vix_regime.csv'}")
    print(f"  - {output_tables / 'performance.csv'}")
    print(f"  - {output_tables / 'predictions.csv'}")
    print("=" * 70)

    return stats, regime_df


if __name__ == "__main__":
    main()
