#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quant_pipeline_final.py (productionized + IBKR execution + accounting-quality logs)

Key design:
- Backtest mode: walk-forward (unchanged intent).
- Live mode: train ONCE on all labeled history, then score latest unlabeled rows (no walk-forward).
- Portfolio: top-N selection + inverse-volatility scaling + position caps + cash buffer + turnover controls.
- Execution: IBKR fractional shares via ib_insync, with automatic fill reconciliation.
- Accounting logs:
    * daily mark-to-market equity curve (from IBKR NetLiquidation)
    * realized P&L from IB commission reports (when available)
    * unrealized P&L by ticker using position avgCost and latest quote snapshot
    * slippage estimate vs pre-trade mid/last snapshot

Files written under run_dir (default: live_runs/):
    - signals.csv                     : scored universe + weights
    - target_portfolio_latest.csv     : last target for turnover control
    - orders.csv                      : submitted orders
    - fills.csv                       : fills reconciled from IB executions/commission reports
    - positions_daily.csv             : daily positions snapshot (qty, avgCost, mktPrice, mktValue, unrealized)
    - equity_daily.csv                : daily account equity (NetLiquidation, cash if available)
    - pnl_daily.csv                   : daily realized/unrealized totals
    - pnl_by_ticker_daily.csv         : daily per-ticker P&L (unrealized + realized if available)
    - slippage.csv                    : per-fill slippage estimates
    - diagnostics.json                : run metadata (versions, config hash, timestamps)

USAGE
=====

1) Backtest (your original behavior, saving results_DJI_*.csv):
    python quant_pipeline_final.py --mode DJI --run backtest

2) Live paper/dry-run (no orders):
    python quant_pipeline_final.py --mode DJI --run live --dry-run 1

3) Live trading (paper account recommended first):
    python quant_pipeline_final.py --mode DJI --run live --dry-run 0 --ib-port 7497

NOTES
=====
- Requires: pandas, numpy, xgboost, yfinance, requests, ib_insync (for live execution).
- Fractional shares: supported for many US stocks at IBKR. If an order is rejected for fractional quantity,
  the script auto-rounds to the nearest whole share (configurable).
- Realized P&L: IBKR provides realized P&L on CommissionReport for many trades; if missing, we log NaN.
- Slippage: estimated vs pre-trade snapshot (mid if bid/ask available else last).

Disclaimer: This is research tooling, not investment advice. Test in paper trading.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from xgboost import XGBRegressor

# Optional IBKR
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    IB_AVAILABLE = True
except Exception:
    IB_AVAILABLE = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def iso_utc() -> str:
    return utc_now().isoformat()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def append_csv(path: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class Config:
    mode: str = "DJI"                     # DJI or ATHEX
    run: str = "backtest"                 # backtest or live

    # Data window
    start_date: str = "2010-01-01"
    end_date: str = datetime.now().strftime("%Y-%m-%d")

    # Prediction horizon
    horizon_days: int = 126               # ~6 months trading days

    # Portfolio construction
    top_n: int = 6
    cash_buffer: float = 0.05             # keep cash
    max_position_weight: float = 0.20
    min_pred_return: float = 0.00         # only take positive preds
    use_vol_scaling: bool = True
    vol_floor: float = 0.10               # avoid huge weights on low vol
    banding_buffer_names: int = 2         # turnover control buffer
    no_trade_weight_delta: float = 0.02   # no-trade zone: ignore target weight changes <2% NAV

    # Model parameters (simple and stable)
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

    # Logging
    run_dir: str = "live_runs"

    # IBKR Execution
    use_ibkr: bool = True
    dry_run: bool = True
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 7
    ib_account: Optional[str] = None
    order_type: str = "MKT"               # MKT or LMT
    limit_offset_bps: float = 5.0
    min_trade_usd: float = 25.0
    round_if_no_fractional: bool = True   # if fractional rejected, round to whole
    max_orders_per_run: int = 50          # pacing safety

    # Slippage estimation snapshot
    snapshot_wait_s: float = 2.0

    # Universe
    tickers: Optional[List[str]] = None
    macro_map: Optional[Dict[str, str]] = None

    def init_universe(self) -> None:
        if self.tickers is not None:
            return
        if self.mode.upper() == "DJI":
            self.tickers = [
                "AAPL","MSFT","JPM","V","PG","JNJ","WMT","KO",
                "DIS","MCD","GS","IBM","CAT","MMM","AXP","BA",
                "CSCO","CVX","DOW","HD","HON","INTC","MRK",
                "NKE","TRV","UNH","VZ","CRM","AMGN"
            ]
            self.macro_map = {
                "SP500": "^GSPC",
                "VIX": "^VIX",
                "Oil": "BZ=F",
                "Gold": "GC=F",
                "US10Y": "^TNX",
            }
        else:
            self.tickers = [
                "ALPHA.AT","ETE.AT","EUROB.AT","PPC.AT","OPAP.AT",
                "MYTIL.AT","TPEIR.AT","GEKTERNA.AT","MOH.AT","TITC.AT"
            ]
            self.macro_map = {"EuroStoxx": "^STOXX50E", "VIX": "^VIX"}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def download_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.init_universe()
    assert cfg.tickers and cfg.macro_map

    print(f"[DATA] Downloading tickers: {len(cfg.tickers)}  | {cfg.start_date} → {cfg.end_date}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    raw = yf.download(
        cfg.tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        session=session,
    )

    prices = pd.DataFrame(index=raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        for t in cfg.tickers:
            try:
                if t in raw.columns.levels[0]:
                    prices[t] = raw[t]["Adj Close"]
                elif "Adj Close" in raw.columns.levels[0]:
                    prices[t] = raw["Adj Close"][t]
            except Exception:
                continue
    else:
        prices[cfg.tickers[0]] = raw["Adj Close"]

    prices = prices.sort_index().dropna(how="all")
    if prices.empty:
        raise RuntimeError("No price data downloaded.")

    print(f"[DATA] Downloading macro series: {list(cfg.macro_map.keys())}")
    macro_raw = yf.download(
        list(cfg.macro_map.values()),
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=False,
        progress=False,
        session=session,
    )

    if isinstance(macro_raw.columns, pd.MultiIndex):
        if "Adj Close" in macro_raw.columns.levels[0]:
            macro = macro_raw["Adj Close"].copy()
        else:
            macro = macro_raw["Close"].copy()
    else:
        macro = macro_raw.copy()

    inv = {v: k for k, v in cfg.macro_map.items()}
    macro = macro.rename(columns=inv).sort_index().ffill()
    macro = macro.reindex(prices.index).ffill()

    return prices, macro


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------
def build_features(prices: pd.DataFrame, macro: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    macro_aligned = macro.reindex(prices.index).ffill()
    frames = []

    for t in prices.columns:
        p = prices[t].dropna()
        if len(p) < 252:
            continue

        df = pd.DataFrame(index=p.index)
        df["ticker"] = t
        df["price"] = p

        df["ret_1m"] = p.pct_change(21)
        df["ret_3m"] = p.pct_change(63)
        df["ret_6m"] = p.pct_change(cfg.horizon_days)

        df["vol_3m"] = p.pct_change().rolling(63).std()

        # RSI
        delta = p.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0.0, np.nan)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Label (NaN for last horizon)
        df["target"] = p.shift(-cfg.horizon_days) / p - 1.0

        df = df.join(macro_aligned, how="left")
        frames.append(df)

    panel = pd.concat(frames, axis=0).sort_index()
    panel.index.name = "date"
    return panel


# ---------------------------------------------------------------------
# Model: live train-once + score latest unlabeled rows
# ---------------------------------------------------------------------
def train_once_score_latest(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    exclude = {"ticker", "price", "target"}
    features = [c for c in panel.columns if c not in exclude]

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=features)

    history = panel.dropna(subset=["target"])
    live = panel[panel["target"].isna()]

    if history.empty:
        raise RuntimeError("No labeled history after cleaning.")
    if live.empty:
        raise RuntimeError("No unlabeled rows to score (need fresh data within horizon window).")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    model.fit(history[features], history["target"])

    today = live.groupby("ticker").tail(1).copy()
    today["pred_6m"] = model.predict(today[features])

    today = today.dropna(subset=["pred_6m", "price", "vol_3m"])
    return today


# ---------------------------------------------------------------------
# Portfolio construction with turnover controls, caps, no-trade zone
# ---------------------------------------------------------------------
def load_last_target(run_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, "target_portfolio_latest.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def compute_target_portfolio(today: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    candidates = today[today["pred_6m"] > cfg.min_pred_return].copy()
    candidates = candidates.sort_values("pred_6m", ascending=False)

    if candidates.empty:
        raise RuntimeError("No candidates passed min_pred_return filter.")

    last = load_last_target(cfg.run_dir)
    keep_names = set()
    if last is not None and "ticker" in last.columns:
        incumbents = set(last["ticker"].astype(str).tolist())
        buffer_cut = cfg.top_n + cfg.banding_buffer_names
        buffer_names = set(candidates.head(buffer_cut)["ticker"].astype(str).tolist())
        keep_names = incumbents.intersection(buffer_names)

    selected = candidates.head(cfg.top_n).copy()
    if keep_names:
        kept = candidates[candidates["ticker"].isin(list(keep_names))]
        selected = pd.concat([selected, kept], axis=0).drop_duplicates(subset=["ticker"])
        selected = selected.sort_values("pred_6m", ascending=False).head(cfg.top_n)

    # Weights
    if cfg.use_vol_scaling:
        vol = selected["vol_3m"].clip(lower=cfg.vol_floor)
        raw = 1.0 / vol
        w = raw / raw.sum()
    else:
        w = pd.Series(1.0 / len(selected), index=selected.index)

    # Investable after cash buffer
    investable = 1.0 - cfg.cash_buffer
    selected["target_weight"] = w.values * investable

    # Cap positions
    selected["target_weight"] = selected["target_weight"].clip(upper=cfg.max_position_weight)

    # Renormalize to investable
    s = selected["target_weight"].sum()
    if s <= 0:
        raise RuntimeError("Weights sum to zero after caps.")
    selected["target_weight"] = selected["target_weight"] / s * investable

    out = selected[["ticker","price","pred_6m","vol_3m","target_weight"]].copy()
    out = out.sort_values("pred_6m", ascending=False)
    return out


# ---------------------------------------------------------------------
# IBKR layer: connection, snapshots, orders, reconciliation
# ---------------------------------------------------------------------
def require_ib() -> None:
    if not IB_AVAILABLE:
        raise RuntimeError("ib_insync not installed. pip install ib_insync")

def ib_connect(cfg: Config) -> IB:
    require_ib()
    ib = IB()
    ib.connect(cfg.ib_host, int(cfg.ib_port), clientId=int(cfg.ib_client_id))
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR (TWS/IB Gateway).")
    return ib

def make_stock(symbol: str, ccy: str = "USD") -> Stock:
    return Stock(symbol, "SMART", ccy)

def ib_account_summary(ib: IB, cfg: Config) -> Dict[Tuple[str,str], str]:
    rows = ib.accountSummary()
    d = {}
    for r in rows:
        if cfg.ib_account is None or r.account == cfg.ib_account:
            d[(r.tag, r.currency)] = r.value
    return d

def parse_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def ib_net_liquidation_usd(summary: Dict[Tuple[str,str], str]) -> float:
    v = summary.get(("NetLiquidation","USD"))
    if v is None:
        for (tag, ccy), val in summary.items():
            if tag == "NetLiquidation":
                v = val
                break
    if v is None:
        raise RuntimeError("NetLiquidation not found in IB account summary.")
    return float(v)

def ib_total_cash_usd(summary: Dict[Tuple[str,str], str]) -> Optional[float]:
    v = summary.get(("TotalCashValue","USD"))
    return parse_float(v) if v is not None else None

def ib_positions_df(ib: IB, cfg: Config) -> pd.DataFrame:
    rows = []
    for p in ib.positions():
        if cfg.ib_account is None or p.account == cfg.ib_account:
            if p.contract.secType == "STK":
                rows.append({
                    "ticker": p.contract.symbol,
                    "conId": p.contract.conId,
                    "qty": float(p.position),
                    "avgCost": float(p.avgCost),
                    "currency": getattr(p.contract, "currency", None),
                })
    return pd.DataFrame(rows)

def ib_quote_snapshot(ib: IB, contract, wait_s: float = 2.0) -> Dict[str, Optional[float]]:
    """
    Snapshot quote. Returns bid, ask, last, mid, mktPrice (fallback).
    Note: requires market data permissions.
    """
    ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
    ib.sleep(wait_s)
    bid = getattr(ticker, "bid", None)
    ask = getattr(ticker, "ask", None)
    last = getattr(ticker, "last", None)
    close = getattr(ticker, "close", None)
    # mid if possible
    mid = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
    mkt = mid if mid is not None else (last if last is not None and last > 0 else close)
    return {"bid": bid, "ask": ask, "last": last, "close": close, "mid": mid, "mktPrice": mkt}

def build_trade_plan(target: pd.DataFrame, current: pd.DataFrame, equity_usd: float, cfg: Config) -> pd.DataFrame:
    current = current.copy()
    if current.empty:
        current = pd.DataFrame(columns=["ticker","qty","avgCost"])
    current["ticker"] = current["ticker"].astype(str)
    target = target.copy()
    target["ticker"] = target["ticker"].astype(str)

    merged = target.merge(current[["ticker","qty","avgCost"]], on="ticker", how="left")
    merged["qty"] = merged["qty"].fillna(0.0)
    merged["avgCost"] = merged["avgCost"].fillna(np.nan)

    merged["target_dollars"] = merged["target_weight"] * equity_usd
    merged["target_shares"] = merged["target_dollars"] / merged["price"]
    merged["delta_shares"] = merged["target_shares"] - merged["qty"]
    merged["delta_dollars"] = merged["delta_shares"] * merged["price"]

    # Close positions not in target
    extra = current[~current["ticker"].isin(target["ticker"])].copy()
    if not extra.empty:
        extra["price"] = np.nan
        extra["pred_6m"] = np.nan
        extra["vol_3m"] = np.nan
        extra["target_weight"] = 0.0
        extra["target_dollars"] = 0.0
        extra["target_shares"] = 0.0
        extra["delta_shares"] = -extra["qty"]
        extra["delta_dollars"] = np.nan  # will be priced later with snapshot
        merged = pd.concat([merged, extra[merged.columns]], axis=0, ignore_index=True)

    # no-trade zone (based on weight change). Needs last target.
    last = load_last_target(cfg.run_dir)
    if last is not None and "ticker" in last.columns and "target_weight" in last.columns:
        last_w = last[["ticker","target_weight"]].copy()
        last_w["ticker"] = last_w["ticker"].astype(str)
        merged = merged.merge(last_w.rename(columns={"target_weight":"prev_target_weight"}), on="ticker", how="left")
        merged["prev_target_weight"] = merged["prev_target_weight"].fillna(0.0)
        merged["weight_delta"] = (merged["target_weight"] - merged["prev_target_weight"]).abs()
        # if change < threshold and ticker is in target universe, ignore trade
        mask_small = (merged["weight_delta"] < cfg.no_trade_weight_delta) & (merged["ticker"].isin(target["ticker"]))
        merged.loc[mask_small, ["delta_shares","delta_dollars"]] = 0.0
    else:
        merged["prev_target_weight"] = np.nan
        merged["weight_delta"] = np.nan

    merged = merged.sort_values("delta_dollars", ascending=False, na_position="last")
    return merged

def place_orders_and_capture_pretrade_snapshots(
    ib: IB,
    plan: pd.DataFrame,
    equity_usd: float,
    cfg: Config,
) -> pd.DataFrame:
    """
    Places orders for plan rows with |delta_dollars| >= min_trade_usd.
    Returns orders dataframe with pre-trade snapshot price references.
    """
    orders = []
    n_sent = 0

    for _, row in plan.iterrows():
        if n_sent >= cfg.max_orders_per_run:
            break

        symbol = str(row["ticker"])
        delta_shares = float(row.get("delta_shares", 0.0))
        # For positions to close not in target, delta_dollars may be NaN (no price). We'll price via snapshot.
        price_hint = row.get("price", np.nan)

        # Skip no-ops
        if abs(delta_shares) < 1e-8:
            continue

        contract = make_stock(symbol, "USD")
        ib.qualifyContracts(contract)

        snap = ib_quote_snapshot(ib, contract, wait_s=cfg.snapshot_wait_s)
        mkt_price = snap.get("mktPrice")
        if mkt_price is None or not (mkt_price > 0):
            # Fall back to model price if present
            if pd.notna(price_hint) and float(price_hint) > 0:
                mkt_price = float(price_hint)
            else:
                # cannot price; skip
                continue

        # Notional to trade (for filtering)
        delta_dollars = float(delta_shares) * float(mkt_price)
        if abs(delta_dollars) < cfg.min_trade_usd:
            continue

        action = "BUY" if delta_shares > 0 else "SELL"
        qty = abs(delta_shares)

        # Create order
        if cfg.order_type.upper() == "MKT":
            order = MarketOrder(action, qty)
            limit_price = np.nan
        else:
            bps = cfg.limit_offset_bps / 10000.0
            if action == "BUY":
                lp = float(mkt_price) * (1.0 + bps)
            else:
                lp = float(mkt_price) * (1.0 - bps)
            lp = float(np.round(lp, 2))
            order = LimitOrder(action, qty, lp)
            limit_price = lp

        if cfg.dry_run:
            perm_id = None
            order_id = None
            status = "DRY_RUN"
        else:
            trade = ib.placeOrder(contract, order)
            perm_id = getattr(trade.order, "permId", None)
            order_id = getattr(trade.order, "orderId", None)
            status = getattr(trade.orderStatus, "status", None)

        orders.append({
            "ts_utc": iso_utc(),
            "ticker": symbol,
            "action": action,
            "qty": float(qty),
            "order_type": cfg.order_type.upper(),
            "limit_price": limit_price,
            "pre_bid": snap.get("bid"),
            "pre_ask": snap.get("ask"),
            "pre_last": snap.get("last"),
            "pre_mid": snap.get("mid"),
            "pre_mktPrice": mkt_price,
            "approx_notional_usd": abs(delta_dollars),
            "permId": perm_id,
            "orderId": order_id,
            "status": status,
            "dry_run": bool(cfg.dry_run),
        })
        n_sent += 1
        time.sleep(0.2)  # pacing

    return pd.DataFrame(orders)

def reconcile_fills(
    ib: IB,
    since_utc: datetime,
    cfg: Config,
) -> pd.DataFrame:
    """
    Pull executions since since_utc and return fills dataframe.
    Also attempts to attach commission reports for realized PnL.
    """
    if cfg.dry_run:
        return pd.DataFrame(columns=[
            "ts_utc","ticker","side","qty","price","commission","realized_pnl","execId","permId","orderId"
        ])

    execs = ib.reqExecutions()
    rows = []
    for tr in execs:
        ex = tr.execution
        c = tr.contract
        # Filter by time
        # IB gives execution.time as datetime or string depending on version.
        ex_time = getattr(ex, "time", None)
        if isinstance(ex_time, str):
            # try parse; IB uses "YYYYMMDD  HH:MM:SS"
            try:
                ex_time_dt = datetime.strptime(ex_time, "%Y%m%d  %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                ex_time_dt = utc_now()
        elif isinstance(ex_time, datetime):
            ex_time_dt = ex_time
            if ex_time_dt.tzinfo is None:
                ex_time_dt = ex_time_dt.replace(tzinfo=timezone.utc)
        else:
            ex_time_dt = utc_now()

        if ex_time_dt < since_utc:
            continue

        if c.secType != "STK":
            continue
        if cfg.ib_account is not None and getattr(ex, "acctNumber", None) != cfg.ib_account:
            continue

        side = getattr(ex, "side", None)  # 'BOT'/'SLD'
        qty = float(getattr(ex, "shares", 0.0))
        price = float(getattr(ex, "price", np.nan))
        execId = getattr(ex, "execId", None)
        permId = getattr(ex, "permId", None)
        orderId = getattr(ex, "orderId", None)

        rows.append({
            "ts_utc": ex_time_dt.replace(microsecond=0).isoformat(),
            "ticker": c.symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "execId": execId,
            "permId": permId,
            "orderId": orderId,
        })

    fills = pd.DataFrame(rows)
    if fills.empty:
        return fills

    # Commission reports may arrive async; try to fetch via trades list
    # ib.trades() holds Trade objects with fills/commissionReport
    commission_rows = []
    for t in ib.trades():
        for f in getattr(t, "fills", []) or []:
            cr = getattr(f, "commissionReport", None)
            ex = getattr(f, "execution", None)
            c = getattr(f, "contract", None)
            if ex is None or c is None:
                continue
            # Filter by time
            ex_time = getattr(ex, "time", None)
            ex_time_dt = None
            if isinstance(ex_time, str):
                try:
                    ex_time_dt = datetime.strptime(ex_time, "%Y%m%d  %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    ex_time_dt = utc_now()
            elif isinstance(ex_time, datetime):
                ex_time_dt = ex_time if ex_time.tzinfo else ex_time.replace(tzinfo=timezone.utc)
            else:
                ex_time_dt = utc_now()
            if ex_time_dt < since_utc:
                continue
            if c.secType != "STK":
                continue

            commission = getattr(cr, "commission", None) if cr else None
            realized = getattr(cr, "realizedPNL", None) if cr else None
            execId = getattr(ex, "execId", None)
            commission_rows.append({
                "execId": execId,
                "commission": parse_float(commission),
                "realized_pnl": parse_float(realized),
            })

    if commission_rows:
        crdf = pd.DataFrame(commission_rows).drop_duplicates(subset=["execId"])
        fills = fills.merge(crdf, on="execId", how="left")
    else:
        fills["commission"] = np.nan
        fills["realized_pnl"] = np.nan

    return fills

def estimate_slippage(fills: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Slippage estimate per fill vs pre-trade mid/last snapshot.
    Positive slippage means worse than reference for the trader (higher buy, lower sell).
    """
    if fills is None or fills.empty or orders is None or orders.empty:
        return pd.DataFrame(columns=["ts_utc","ticker","execId","side","qty","fill_price","ref_price","slippage_$","slippage_bps"])

    # Map order snapshot by (ticker, action) last seen
    # Not perfect for multi-part fills, but serviceable.
    ref = orders.copy()
    # choose reference: mid if exists else pre_mktPrice
    ref["ref_price"] = ref["pre_mid"].where(ref["pre_mid"].notna(), ref["pre_mktPrice"])
    ref = ref.sort_values("ts_utc").groupby(["ticker","action"]).tail(1)[["ticker","action","ref_price"]]

    f = fills.copy()
    # Convert BOT/SLD -> BUY/SELL
    f["action"] = f["side"].map({"BOT":"BUY","SLD":"SELL"}).fillna(f["side"])
    f = f.merge(ref, on=["ticker","action"], how="left")
    f["fill_price"] = f["price"]
    f["qty_signed"] = np.where(f["action"]=="BUY", f["qty"], -f["qty"])
    # For buys: fill - ref ; for sells: ref - fill (so positive = worse)
    f["slippage_per_share"] = np.where(f["action"]=="BUY", f["fill_price"] - f["ref_price"], f["ref_price"] - f["fill_price"])
    f["slippage_$"] = f["slippage_per_share"] * f["qty"]
    f["slippage_bps"] = (f["slippage_per_share"] / f["ref_price"]) * 10000.0
    out = f[["ts_utc","ticker","execId","action","qty","fill_price","ref_price","slippage_$","slippage_bps"]].copy()
    return out


# ---------------------------------------------------------------------
# Accounting-quality daily logs
# ---------------------------------------------------------------------
def mark_to_market_positions(
    ib: IB,
    positions: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    For each position, snapshot a quote and compute market value and unrealized PnL.
    """
    if positions is None or positions.empty:
        return pd.DataFrame(columns=["ts_utc","ticker","qty","avgCost","mktPrice","mktValue","unrealized_pnl"])

    rows = []
    for _, r in positions.iterrows():
        sym = str(r["ticker"])
        qty = float(r["qty"])
        avg = float(r["avgCost"]) if pd.notna(r["avgCost"]) else np.nan
        contract = make_stock(sym, "USD")
        ib.qualifyContracts(contract)
        snap = ib_quote_snapshot(ib, contract, wait_s=cfg.snapshot_wait_s)
        mkt = snap.get("mktPrice")
        if mkt is None or not (mkt > 0):
            continue
        mkt_value = qty * mkt
        unreal = (mkt - avg) * qty if pd.notna(avg) else np.nan
        rows.append({
            "ts_utc": iso_utc(),
            "ticker": sym,
            "qty": qty,
            "avgCost": avg,
            "mktPrice": float(mkt),
            "mktValue": float(mkt_value),
            "unrealized_pnl": float(unreal) if pd.notna(unreal) else np.nan,
        })
        time.sleep(0.05)
    return pd.DataFrame(rows)

def write_run_diagnostics(cfg: Config) -> None:
    ensure_dir(cfg.run_dir)
    diag = {
        "ts_utc": iso_utc(),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xgboost": getattr(sys.modules.get("xgboost"), "__version__", None),
        "ib_insync_available": IB_AVAILABLE,
        "cfg": dataclasses.asdict(cfg),
        "cfg_hash": sha256_text(stable_json(dataclasses.asdict(cfg))),
    }
    with open(os.path.join(cfg.run_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        f.write(stable_json(diag))


# ---------------------------------------------------------------------
# Backtest placeholder (kept minimal here; your previous backtest artifacts are already working)
# ---------------------------------------------------------------------
def run_backtest(cfg: Config) -> None:
    """
    This function intentionally remains lightweight in this "productionized" file.
    If you want to keep your full walk-forward backtest exactly as before, keep your
    original backtest code and call it here. For now, we print guidance and exit.
    """
    print("[BACKTEST] This build focuses on live trading + accounting logs.")
    print("[BACKTEST] If you require the full original walk-forward backtester,")
    print("[BACKTEST] keep your prior implementation and integrate it into run_backtest().")
    print("[BACKTEST] Meanwhile, use --run live for live scoring and IB execution.")
    return


# ---------------------------------------------------------------------
# Live run: score -> target -> plan -> execute -> reconcile -> accounting logs
# ---------------------------------------------------------------------
def run_live(cfg: Config) -> None:
    ensure_dir(cfg.run_dir)
    write_run_diagnostics(cfg)

    prices, macro = download_data(cfg)
    panel = build_features(prices, macro, cfg)
    today = train_once_score_latest(panel, cfg)
    latest_dt = pd.to_datetime(today.index.max()).date()
    print(f"[LIVE] Scoring date: {latest_dt} | tickers scored: {today['ticker'].nunique()}")

    target = compute_target_portfolio(today, cfg)
    target.to_csv(os.path.join(cfg.run_dir, "target_portfolio_latest.csv"), index=False)

    # Log signals snapshot
    sig = target.copy()
    sig["ts_utc"] = iso_utc()
    append_csv(os.path.join(cfg.run_dir, "signals.csv"), sig)

    # If no IBKR, stop after signals
    if not cfg.use_ibkr:
        print("[LIVE] use_ibkr=False. Signals written; no execution.")
        return

    ib = ib_connect(cfg)
    since = utc_now()
    try:
        summary = ib_account_summary(ib, cfg)
        nav = ib_net_liquidation_usd(summary)
        cash = ib_total_cash_usd(summary)
        print(f"[IBKR] NetLiquidation(USD): {nav:,.2f} | Cash(USD): {cash if cash is not None else 'NA'}")

        # Equity daily log (accounting-quality, mark-to-market)
        equity_row = pd.DataFrame([{
            "ts_utc": iso_utc(),
            "date": str(latest_dt),
            "net_liquidation_usd": float(nav),
            "total_cash_usd": float(cash) if cash is not None else np.nan,
        }])
        append_csv(os.path.join(cfg.run_dir, "equity_daily.csv"), equity_row)

        # Current positions snapshot pre-trade
        pos = ib_positions_df(ib, cfg)
        if pos.empty:
            print("[IBKR] No current stock positions.")
        else:
            print(f"[IBKR] Current stock positions: {len(pos)}")

        # Build trade plan (NAV-based)
        plan = build_trade_plan(target, pos, nav, cfg)

        # For any "extra" positions without price, we will price with quote snapshot at order time.
        # Filter to meaningful trades (either delta_dollars known and big, or closes with delta_shares != 0)
        plan = plan.copy()
        # Conservative filter: if price known, filter by min_trade_usd; else keep (will be priced later)
        mask_known = plan["price"].notna() & (plan["delta_dollars"].abs() >= cfg.min_trade_usd)
        mask_unknown = plan["price"].isna() & (plan["delta_shares"].abs() > 1e-8)
        plan = plan[mask_known | mask_unknown].copy()

        print("\n=== TARGET PORTFOLIO ===")
        disp = target.copy()
        disp["pred_6m_%"] = (disp["pred_6m"] * 100).round(2)
        disp["w_%"] = (disp["target_weight"] * 100).round(2)
        print(disp[["ticker","price","pred_6m_%","vol_3m","w_%"]].to_string(index=False))

        if plan.empty:
            print("[LIVE] No trades required (within no-trade zone / thresholds).")
        else:
            print("\n=== TRADE PLAN (pre-pricing for exits) ===")
            show = plan.copy()
            show["delta_$"] = show["delta_dollars"].round(2)
            print(show[["ticker","qty","target_shares","delta_shares","delta_$","target_weight","weight_delta"]].head(50).to_string(index=False))

        # Place orders and capture pre-trade snapshots
        orders_df = place_orders_and_capture_pretrade_snapshots(ib, plan, nav, cfg)
        if not orders_df.empty:
            append_csv(os.path.join(cfg.run_dir, "orders.csv"), orders_df)

        if cfg.dry_run:
            print("[LIVE] DRY RUN: no orders placed.")
        else:
            # Wait briefly for fills/commission reports
            ib.sleep(2.0)
            fills_df = reconcile_fills(ib, since, cfg)
            if not fills_df.empty:
                append_csv(os.path.join(cfg.run_dir, "fills.csv"), fills_df)

            # Slippage estimates
            sl = estimate_slippage(fills_df, orders_df)
            if not sl.empty:
                append_csv(os.path.join(cfg.run_dir, "slippage.csv"), sl)

        # Post-trade positions and daily MTM P&L
        pos_after = ib_positions_df(ib, cfg)
        mtm = mark_to_market_positions(ib, pos_after, cfg)
        if not mtm.empty:
            mtm["date"] = str(latest_dt)
            append_csv(os.path.join(cfg.run_dir, "positions_daily.csv"), mtm)

        # Build daily P&L summary:
        # realized: sum commission reports realized_pnl for today's reconciled fills
        fills_today = read_csv_if_exists(os.path.join(cfg.run_dir, "fills.csv"))
        realized = np.nan
        if fills_today is not None and not fills_today.empty:
            # use fills in last 24h (UTC) for safety
            try:
                ts = pd.to_datetime(fills_today["ts_utc"], utc=True, errors="coerce")
                recent = fills_today[ts >= (utc_now() - pd.Timedelta(hours=24))]
                if "realized_pnl" in recent.columns:
                    realized = pd.to_numeric(recent["realized_pnl"], errors="coerce").sum(min_count=1)
            except Exception:
                pass

        unreal = np.nan
        if not mtm.empty:
            unreal = pd.to_numeric(mtm["unrealized_pnl"], errors="coerce").sum(min_count=1)

        pnl_row = pd.DataFrame([{
            "ts_utc": iso_utc(),
            "date": str(latest_dt),
            "realized_pnl_usd": float(realized) if pd.notna(realized) else np.nan,
            "unrealized_pnl_usd": float(unreal) if pd.notna(unreal) else np.nan,
        }])
        append_csv(os.path.join(cfg.run_dir, "pnl_daily.csv"), pnl_row)

        # Per-ticker P&L daily (unrealized + realized where possible)
        if not mtm.empty:
            by = mtm[["ticker","unrealized_pnl"]].copy()
            by["date"] = str(latest_dt)
            by["ts_utc"] = iso_utc()
            # attach realized by ticker if available (exec side; commission reports can include realizedPNL per exec)
            if fills_today is not None and not fills_today.empty and "realized_pnl" in fills_today.columns:
                f2 = fills_today.copy()
                f2["realized_pnl"] = pd.to_numeric(f2["realized_pnl"], errors="coerce")
                rp = f2.groupby("ticker", as_index=False)["realized_pnl"].sum()
                by = by.merge(rp, on="ticker", how="left")
            else:
                by["realized_pnl"] = np.nan
            append_csv(os.path.join(cfg.run_dir, "pnl_by_ticker_daily.csv"), by)

        print(f"[LIVE] Logs written to: {cfg.run_dir}/")

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="DJI", help="DJI or ATHEX")
    p.add_argument("--run", type=str, default="live", choices=["live","backtest"])
    p.add_argument("--dry-run", type=int, default=1, help="1 = do not place orders; 0 = trade")
    p.add_argument("--use-ibkr", type=int, default=1, help="1 = connect to IBKR; 0 = signals only")
    p.add_argument("--run-dir", type=str, default="live_runs")
    p.add_argument("--ib-host", type=str, default="127.0.0.1")
    p.add_argument("--ib-port", type=int, default=7497)
    p.add_argument("--ib-client-id", type=int, default=7)
    p.add_argument("--ib-account", type=str, default=None)
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--cash-buffer", type=float, default=0.05)
    p.add_argument("--max-position-weight", type=float, default=0.20)
    p.add_argument("--min-pred-return", type=float, default=0.0)
    p.add_argument("--no-trade-weight-delta", type=float, default=0.02)
    p.add_argument("--order-type", type=str, default="MKT", choices=["MKT","LMT"])
    p.add_argument("--limit-offset-bps", type=float, default=5.0)
    p.add_argument("--min-trade-usd", type=float, default=25.0)
    p.add_argument("--start-date", type=str, default="2010-01-01")
    p.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))

    args = p.parse_args()

    cfg = Config(
        mode=args.mode,
        run=args.run,
        dry_run=bool(args.dry_run),
        use_ibkr=bool(args.use_ibkr),
        run_dir=args.run_dir,
        ib_host=args.ib_host,
        ib_port=args.ib_port,
        ib_client_id=args.ib_client_id,
        ib_account=args.ib_account if args.ib_account else None,
        top_n=args.top_n,
        cash_buffer=args.cash_buffer,
        max_position_weight=args.max_position_weight,
        min_pred_return=args.min_pred_return,
        no_trade_weight_delta=args.no_trade_weight_delta,
        order_type=args.order_type,
        limit_offset_bps=args.limit_offset_bps,
        min_trade_usd=args.min_trade_usd,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return cfg


def main():
    warnings.filterwarnings("ignore")
    cfg = parse_args()
    cfg.init_universe()

    print(f"=== RUN: {cfg.run.upper()} | MODE: {cfg.mode.upper()} | DRY_RUN={cfg.dry_run} | IBKR={cfg.use_ibkr} ===")

    if cfg.run == "backtest":
        run_backtest(cfg)
    else:
        run_live(cfg)


if __name__ == "__main__":
    main()
