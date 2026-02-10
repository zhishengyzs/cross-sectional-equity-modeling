#!/usr/bin/env python3
"""Generate visualizations for the report"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load config YAML; paths resolved relative to cwd (repo root)."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Generate strategy plots')
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path.cwd()
    output_tables = root / config['output_tables']
    output_figures = root / config['output_figures']
    output_figures.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11

    # Load data
    backtest_path = output_tables / 'backtest_results.csv'
    backtest = pd.read_csv(backtest_path)
    backtest['month_end'] = pd.to_datetime(backtest['month_end'])

    # 1. Cumulative Returns
    fig, ax = plt.subplots(figsize=(12, 6))

    cum_strategy = (1 + backtest['port_ret']).cumprod()
    cum_spx = (1 + backtest['spx_ret']).cumprod()

    ax.plot(backtest['month_end'], cum_strategy, 'b-', linewidth=2, label='Strategy (Top-5)')
    ax.plot(backtest['month_end'], cum_spx, 'r--', linewidth=2, label='SPY Benchmark')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Starting = 1.0)')
    ax.set_title('Strategy vs SPY: Cumulative Returns (Feb 2024 - Oct 2025)')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_figures / 'plot_cumulative_returns.png', dpi=150)
    plt.close()
    print("Saved:", output_figures / 'plot_cumulative_returns.png')

    # 2. Monthly Returns Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(backtest))
    width = 0.35

    bars1 = ax.bar(x - width/2, backtest['port_ret']*100, width, label='Strategy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, backtest['spx_ret']*100, width, label='SPY', color='coral', alpha=0.8)

    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Return (%)')
    ax.set_title('Monthly Returns: Strategy vs SPY')
    ax.set_xticks(x[::1])
    ax.set_xticklabels([d.strftime('%Y-%m') for d in backtest['month_end']], rotation=45)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_figures / 'plot_monthly_returns.png', dpi=150)
    plt.close()
    print("Saved:", output_figures / 'plot_monthly_returns.png')

    # 3. VIX and Strategy Returns
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(backtest['month_end'], backtest['port_ret']*100, width=20,
            color=['green' if r > 0 else 'red' for r in backtest['port_ret']], alpha=0.6, label='Strategy Return')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Strategy Return (%)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(backtest['month_end'], backtest['VIX'], 'b-', linewidth=2, label='VIX')
    ax2.set_ylabel('VIX', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    vix_median = backtest['VIX'].median()
    ax2.axhline(y=vix_median, color='blue', linestyle=':', alpha=0.7, label=f'VIX Median ({vix_median:.1f})')

    fig.suptitle('Strategy Returns and VIX Regime')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_figures / 'plot_vix_regime.png', dpi=150)
    plt.close()
    print("Saved:", output_figures / 'plot_vix_regime.png')

    # 4. Drawdown Chart
    fig, ax = plt.subplots(figsize=(12, 5))

    cum_wealth = (1 + backtest['port_ret']).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max * 100

    ax.fill_between(backtest['month_end'], drawdown, 0, color='red', alpha=0.3)
    ax.plot(backtest['month_end'], drawdown, 'r-', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Strategy Drawdown')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_figures / 'plot_drawdown.png', dpi=150)
    plt.close()
    print("Saved:", output_figures / 'plot_drawdown.png')

    # 5. Performance Summary Table as Image
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    data = [
        ['Metric', 'Strategy', 'SPY'],
        ['Ann. Return', '35.75%', '21.05%'],
        ['Ann. Volatility', '32.20%', '11.46%'],
        ['Sharpe Ratio', '1.05', '1.66'],
        ['Max Drawdown', '-16.90%', '-'],
        ['Hit Rate vs SPY', '50.00%', '-'],
        ['N Months', '20', '20']
    ]

    table = ax.table(cellText=data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('Performance Summary (Feb 2024 - Oct 2025)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_figures / 'plot_performance_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved:", output_figures / 'plot_performance_table.png')

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
