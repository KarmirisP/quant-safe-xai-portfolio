#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo Appendix Generator (from results_DJI_predictions.csv)

Inputs:
  - results_DJI_predictions.csv with columns:
      date,ticker,target,price,prediction,window_end

Outputs (default to ./paper/):
  - mc_equity_fan.png
  - mc_summary_hist.png
  - mc_summary_table.csv  (LaTeX-friendly)
  - daily_returns.csv     (audit trail)

Method:
  1) Build daily portfolio returns from cross-sectional predictions
     - selection: top N by prediction
     - weights: equal-weight or inverse-vol (63d rolling vol)
     - rebalance: daily OR monthly (config)
     - costs: per turnover (bps)

  2) Block-bootstrap Monte Carlo of daily returns (stationary bootstrap)
     - preserves short-range dependence
     - produces distributions for CAGR / MaxDD

This is intended for a paper appendix robustness check.
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min())

def cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return float("nan")
    total_return = equity[-1] / equity[0]
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return float(total_return ** (1.0 / years) - 1.0)

def ann_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))

def sharpe(returns: np.ndarray, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    ex = returns - rf_daily
    sd = np.std(ex, ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.mean(ex) / sd * math.sqrt(periods_per_year))


# -----------------------------
# Portfolio construction from predictions
# -----------------------------
@dataclass
class PortfolioSpec:
    top_n: int = 5
    weight_mode: str = "inv_vol"   # "equal" or "inv_vol"
    vol_lookback: int = 63         # trading days
    rebalance: str = "D"           # "D" daily, "M" month-end
    cost_bps: float = 15.0         # per $ turnover (round-trip simplification)
    max_weight: float = 0.25       # cap (optional)


def compute_inv_vol_weights(vol: pd.Series, max_weight: float) -> pd.Series:
    vol = vol.replace([np.inf, -np.inf], np.nan)
    vol = vol.clip(lower=1e-8)
    w = 1.0 / vol
    w = w / w.sum() if w.sum() > 0 else w * 0.0
    if max_weight is not None and max_weight > 0:
        # iterative cap redistribution
        w = w.copy()
        for _ in range(10):
            over = w[w > max_weight]
            if over.empty:
                break
            excess = float((over - max_weight).sum())
            w.loc[over.index] = max_weight
            under = w[w < max_weight]
            if under.sum() <= 0 or excess <= 0:
                break
            w.loc[under.index] += excess * (under / under.sum())
        w = w / w.sum() if w.sum() > 0 else w
    return w


def build_daily_returns_from_predictions(df: pd.DataFrame, spec: PortfolioSpec) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      - positions_df: rows=date, columns=ticker weights (post-cost turnover computed separately)
      - port_ret: pd.Series daily returns aligned to dates (t -> t+1)
    """
    df = df.copy()
    df["date"] = to_datetime(df["date"])
    df = df.dropna(subset=["date", "ticker", "price", "prediction"])
    df = df.sort_values(["date", "ticker"])

    # Wide prices for realized next-day returns
    px = df.pivot(index="date", columns="ticker", values="price").sort_index()
    ret1 = px.pct_change(1).shift(-1)  # return from t to t+1, aligned at t

    # Rolling vol (for inv_vol weights)
    vol = px.pct_change(1).rolling(spec.vol_lookback).std()

    # Rebalance dates
    if spec.rebalance.upper() == "D":
        rebal_dates = px.index
    elif spec.rebalance.upper() == "M":
        # month-end based on available trading days
        rebal_dates = px.resample("M").last().index
        rebal_dates = rebal_dates.intersection(px.index)
    else:
        raise ValueError("rebalance must be 'D' or 'M'")

    # Build weights on each rebalance date
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    prev_w = pd.Series(0.0, index=px.columns)

    for d in px.index:
        if d not in rebal_dates:
            # carry forward weights (no rebalance)
            weights.loc[d] = prev_w.values
            continue

        day_panel = df[df["date"] == d].set_index("ticker")
        day_panel = day_panel.loc[day_panel.index.intersection(px.columns)]

        if day_panel.empty:
            weights.loc[d] = prev_w.values
            continue

        # Select top N by prediction
        top = day_panel["prediction"].sort_values(ascending=False).head(spec.top_n)
        sel = top.index.tolist()

        if spec.weight_mode == "equal":
            w = pd.Series(0.0, index=px.columns)
            w.loc[sel] = 1.0 / len(sel)
        else:
            # inv vol on selected names
            vv = vol.loc[d, sel]
            w_sel = compute_inv_vol_weights(vv, spec.max_weight)
            w = pd.Series(0.0, index=px.columns)
            w.loc[sel] = w_sel

        w = w.fillna(0.0)
        weights.loc[d] = w.values
        prev_w = w

    # Turn weights into daily portfolio returns with transaction costs on rebalance days
    port_ret = (weights * ret1).sum(axis=1).fillna(0.0)

    # Turnover and costs
    w_shift = weights.shift(1).fillna(0.0)
    turnover = (weights - w_shift).abs().sum(axis=1)  # 1.0 = 100% turnover
    # Apply cost only on rebalance days (but turnover is 0 on non-rebalance due to carry)
    cost = (turnover * (spec.cost_bps / 10000.0)).fillna(0.0)
    port_ret_net = port_ret - cost

    audit = pd.DataFrame({
        "gross_return": port_ret,
        "turnover": turnover,
        "cost": cost,
        "net_return": port_ret_net
    }, index=px.index)

    return audit, port_ret_net


# -----------------------------
# Stationary bootstrap Monte Carlo
# -----------------------------
def stationary_bootstrap_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap indices.
    p: probability of starting a new block (expected block length = 1/p)
    """
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx


def monte_carlo_paths(returns: np.ndarray, n_paths: int, block_len: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
      - equity_paths: (n_paths, T) equity indexed to 1.0 start
      - cagr: (n_paths,)
      - maxdd: (n_paths,)
      - vol: (n_paths,)
      - sharpe: (n_paths,)
    """
    rng = np.random.default_rng(seed)
    T = len(returns)
    p = 1.0 / max(1, block_len)

    equity_paths = np.empty((n_paths, T), dtype=float)
    cagr_v = np.empty(n_paths, dtype=float)
    maxdd_v = np.empty(n_paths, dtype=float)
    vol_v = np.empty(n_paths, dtype=float)
    shrp_v = np.empty(n_paths, dtype=float)

    for k in range(n_paths):
        ii = stationary_bootstrap_indices(T, p, rng)
        r = returns[ii]
        eq = np.cumprod(1.0 + r)
        equity_paths[k, :] = eq
        cagr_v[k] = cagr(eq)
        maxdd_v[k] = max_drawdown(eq)
        vol_v[k] = ann_vol(r)
        shrp_v[k] = sharpe(r)

    return {
        "equity_paths": equity_paths,
        "cagr": cagr_v,
        "maxdd": maxdd_v,
        "vol": vol_v,
        "sharpe": shrp_v
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_fan_chart(dates: pd.DatetimeIndex, equity_paths: np.ndarray, out_png: str) -> None:
    qs = [5, 25, 50, 75, 95]
    bands = np.percentile(equity_paths, qs, axis=0)

    plt.figure(figsize=(11, 6))
    plt.plot(dates, bands[2], linewidth=2)  # median
    plt.fill_between(dates, bands[1], bands[3], alpha=0.25)
    plt.fill_between(dates, bands[0], bands[4], alpha=0.15)
    plt.yscale("log")
    plt.title("Monte Carlo Equity Fan (Stationary Block Bootstrap)")
    plt.xlabel("Date")
    plt.ylabel("Equity (log scale, start=1.0)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_histograms(stats: Dict[str, np.ndarray], out_png: str) -> None:
    plt.figure(figsize=(11, 6))
    # Two panels on one figure is okay for an appendix summary image
    # but we keep it simple and readable with two hist overlays.
    plt.hist(stats["cagr"], bins=50, alpha=0.6, label="CAGR")
    plt.hist(stats["maxdd"], bins=50, alpha=0.6, label="Max Drawdown")
    plt.title("Monte Carlo Distributions (CAGR and Max Drawdown)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def summarize_stats(stats: Dict[str, np.ndarray]) -> pd.DataFrame:
    def pct(x, p):
        return float(np.percentile(x, p))

    rows = []
    for name, arr in [("CAGR", stats["cagr"]),
                      ("MaxDD", stats["maxdd"]),
                      ("AnnVol", stats["vol"]),
                      ("Sharpe", stats["sharpe"])]:
        rows.append({
            "metric": name,
            "p05": pct(arr, 5),
            "p25": pct(arr, 25),
            "p50": pct(arr, 50),
            "p75": pct(arr, 75),
            "p95": pct(arr, 95),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
        })
    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True, help="Path to results_DJI_predictions.csv")
    ap.add_argument("--out_dir", type=str, default="paper", help="Output dir (e.g., paper/)")
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--rebalance", type=str, default="D", choices=["D", "M"])
    ap.add_argument("--weight_mode", type=str, default="inv_vol", choices=["equal", "inv_vol"])
    ap.add_argument("--vol_lookback", type=int, default=63)
    ap.add_argument("--cost_bps", type=float, default=15.0)
    ap.add_argument("--max_weight", type=float, default=0.25)
    ap.add_argument("--n_paths", type=int, default=5000)
    ap.add_argument("--block_len", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.pred_csv)
    spec = PortfolioSpec(
        top_n=args.top_n,
        weight_mode=args.weight_mode,
        vol_lookback=args.vol_lookback,
        rebalance=args.rebalance,
        cost_bps=args.cost_bps,
        max_weight=args.max_weight
    )

    audit, rets = build_daily_returns_from_predictions(df, spec)
    audit_path = os.path.join(args.out_dir, "daily_returns.csv")
    audit.to_csv(audit_path, index_label="date")

    returns = audit["net_return"].values.astype(float)
    dates = audit.index

    stats = monte_carlo_paths(returns, n_paths=args.n_paths, block_len=args.block_len, seed=args.seed)

    fan_png = os.path.join(args.out_dir, "mc_equity_fan.png")
    hist_png = os.path.join(args.out_dir, "mc_summary_hist.png")
    plot_fan_chart(dates, stats["equity_paths"], fan_png)
    plot_histograms(stats, hist_png)

    summ = summarize_stats(stats)
    summ_path = os.path.join(args.out_dir, "mc_summary_table.csv")
    summ.to_csv(summ_path, index=False)

    print("Wrote:")
    print(" ", audit_path)
    print(" ", fan_png)
    print(" ", hist_png)
    print(" ", summ_path)
    print("\nPreview summary:")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
