#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== Headless plotting for CI =====
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

# ===== Imports =====
import os
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ===== Parameters =====
WINDOW = 250  # rolling window size (e.g., 250 trading days ≈ 1 year)

PAIRS = [
    ('BRK-B', 'SPY'),     # growth vs quality value
    ('GLD', 'SPY'),       # growth vs value
    ('BTC-USD', 'SPY'),   # digital liquidity vs traditional growth
    ('TQQQ', 'SPY'),      # risk-on vs safe-haven (leveraged growth sentiment)
    ('NVDA', 'SMH'),       # nvidia v.s. semiconductors
    ('NVDA', 'QQQ'),      # AI & innovation v.s. broad market
    ('GOOG', 'QQQ'),       # real-economy vs bonds
    ('META', 'QQQ'),
    ('SPY', '^HSI')   # dollar vs gold (global risk flow)
]

START_DATE = "1990-01-01"
#START_DATE = "1993-01-01"


# ===== Helper Functions =====
def get_rolling_return(df: pd.DataFrame, sym: str, window: int = 100) -> pd.DataFrame:
    """Compute rolling cumulative log-returns over `window` days and return a single-column DataFrame."""
    df = df.copy()
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame(columns=[f'RollingCumReturn_{sym}_{window}'])
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    col = f'RollingCumReturn_{sym}_{window}'
    df[col] = np.exp(df['LogReturn'].rolling(window=window).sum()) - 1
    out = df[[col]].dropna()
    out.index = pd.to_datetime(out.index)
    return out

def percentile_rank(series: np.ndarray, value: float) -> float:
    """Percentile rank in [0,100]."""
    arr = np.asarray(series)
    n = arr.size
    if n == 0:
        return float('nan')
    # rank using <= (right side), consistent with np.searchsorted on sorted array, side='right'
    return (np.sum(arr <= value) / n) * 100.0

# ===== Main plotting routine =====
def main():
    # Collect unique tickers from PAIRS
    all_tickers = sorted(list(set([t for pair in PAIRS for t in pair])))

    # Download all tickers once
    raw_data = {}
    for sym in all_tickers:
        try:
            df = yf.download(sym, start=START_DATE, auto_adjust=False, progress=False)
        except Exception as e:
            print(f"[WARN] Failed to download {sym}: {e}")
            df = pd.DataFrame()
        raw_data[sym] = df

    # Prepare figure (3x3 grid; hide extras if <9 pairs)
    n_rows, n_cols = 3, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2))
    axs = axs.flatten()

    for idx, (TARGET, BASE) in enumerate(PAIRS):
        ax = axs[idx]
        target_raw = raw_data.get(TARGET, pd.DataFrame()).copy()
        base_raw = raw_data.get(BASE, pd.DataFrame()).copy()

        # Compute rolling returns
        q1 = get_rolling_return(target_raw, TARGET, window=WINDOW)
        q2 = get_rolling_return(base_raw, BASE, window=WINDOW)

        # Align on dates
        df = pd.merge(q1, q2, left_index=True, right_index=True, how='inner')
        tcol = f'RollingCumReturn_{TARGET}_{WINDOW}'
        bcol = f'RollingCumReturn_{BASE}_{WINDOW}'

        if df.empty or tcol not in df.columns or bcol not in df.columns:
            ax.set_title(f'{TARGET} − {BASE} ({WINDOW}-Day) | no data', fontsize=11)
            ax.axis('off')
            continue

        diff = df[tcol] - df[bcol]

        # BTC-only clip (no default clip for others)
        if 'BTC' in TARGET or 'NVDA' in TARGET or 'TQQQ' in TARGET:
            diff = diff.clip(-1.5, 1.5)

        latest_diff = float(diff.iloc[-1])
        latest_pct = percentile_rank(diff.values, latest_diff)

        # === quantile coloring logic ===
        q10, q90 = np.nanquantile(diff, [0.1, 0.9])
        ax.set_title(
           f'{TARGET} − {BASE} | pct {latest_pct:.1f}%',
            fontsize=11,
            color='red' if (latest_diff < q10 or latest_diff > q90) else 'black'
        )

        q10, q25, q75, q90 = np.nanquantile(diff, [0.1, 0.25, 0.75, 0.9])
        color = (
            'red' if (latest_diff < q10 or latest_diff > q90)
            else 'orange' if (q10 <= latest_diff < q25 or q75 < latest_diff <= q90)
            else 'black'
        )

        ax.set_title(f'{TARGET} − {BASE} | pct {latest_pct:.1f}%', fontsize=11, color=color)

        # Plot
        ax.plot(df.index, diff.values, label=f'{TARGET} − {BASE}', color=f'C{idx % 10}')
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_title(f'{TARGET} − {BASE} | pct {latest_pct:.1f}%', fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        for label in ax.get_xticklabels():
            label.set_rotation(0) 
        ax.set_xlabel("")  
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel(f"{WINDOW}-CumReturn Diff")
        #ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any extra subplots
    for j in range(len(PAIRS), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()

    # ===== Save outputs for GitHub Pages =====
    os.makedirs("docs", exist_ok=True)
    out_png = "docs/rolling_pairs.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")

    # Simple landing page with timestamp + image
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rolling Pair Diffs</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
  .meta {{ color: #555; margin-bottom: 12px; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  .note {{ margin-top: 16px; color: #666; font-size: 0.95rem; }}
  code {{ background:#f6f8fa; padding:2px 4px; border-radius:4px; }}
</style>
</head>
<body>
  <h1>Rolling Pair Diffs ({WINDOW}-day)</h1>
  <div class="meta">Last updated: {ts}</div>
  <p><img src="rolling_pairs.png" alt="rolling pairs figure"></p>
  <div class="note">
    Data via <code>yfinance</code>. Figure regenerates daily via GitHub Actions.
  </div>
</body></html>
"""
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Wrote {out_png} and docs/index.html")

if __name__ == "__main__":
    main()







