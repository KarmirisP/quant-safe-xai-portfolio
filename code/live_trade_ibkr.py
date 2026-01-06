import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---- Reuse your pipeline components ----
from quant_pipeline_final import Config, DataEngine, FeatureEngineer, GreekMacroEngine

# ---- Model ----
from xgboost import XGBRegressor

# ---- Optional IBKR execution ----
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
    IB_AVAILABLE = True
except Exception:
    IB_AVAILABLE = False

warnings.filterwarnings("ignore")


# =========================
# Live trading configuration
# =========================
@dataclass
class LiveTradeConfig:
    mode: str = "DJI"
    start_date: str = "2010-01-01"
    end_date: str = datetime.now().strftime("%Y-%m-%d")

    # Portfolio rules
    top_n: int = 6
    min_pred_return: float = 0.0         # require pred > 0
    cash_buffer: float = 0.05            # keep 5% cash
    max_position_weight: float = 0.20    # cap 20% per name
    use_vol_scaling: bool = True         # weight ∝ 1/vol_3m
    vol_floor: float = 0.10              # avoid extreme weights on tiny vol
    no_trade_weight_delta: float = 0.02  # don't trade if |Δweight| < 2% NAV (turnover control)

    # Banding turnover control: keep incumbent if still within top (N + buffer)
    band_buffer: int = 2

    # IBKR
    USE_IBKR: bool = True
    DRY_RUN: bool = True                 # set False to actually place orders
    IB_HOST: str = "127.0.0.1"
    IB_PORT: int = 7497                  # commonly paper TWS; adjust for your setup
    IB_CLIENT_ID: int = 7
    IB_ACCOUNT: str | None = None

    ORDER_TYPE: str = "MKT"              # "MKT" or "LMT"
    LIMIT_OFFSET_BPS: float = 5.0        # for LMT
    MIN_TRADE_USD: float = 25.0          # ignore tiny trades

    RUN_DIR: str = "live_runs"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_csv(path: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# =========================
# Build the same macro features as your backtest
# =========================
def build_macro_features(macro_adjclose: pd.DataFrame) -> pd.DataFrame:
    macro_feat = pd.DataFrame(index=macro_adjclose.index)
    for c in macro_adjclose.columns:
        s = macro_adjclose[c]
        macro_feat[f"{c}_ret_1m"] = s.pct_change(21)
        macro_feat[f"{c}_ret_3m"] = s.pct_change(63)
        macro_feat[f"{c}_z_1y"] = (s - s.rolling(252).mean()) / (s.rolling(252).std() + 1e-9)
    return macro_feat.dropna(how="all")


# =========================
# Signal layer: train once on all labeled history, score latest unlabeled rows
# =========================
def train_once_score_latest(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Select numeric features, exclude target/price
    drop_cols = {"target", "price"}
    feature_cols = [c for c in panel.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(panel[c])]

    # Impute feature NaNs using medians (fit-time convenience; you may want stricter handling later)
    data = panel.copy()
    for c in feature_cols:
        if data[c].isna().any():
            med = data[c].median(skipna=True)
            data[c] = data[c].fillna(med if pd.notna(med) else 0.0)

    # Labeled history = where target exists
    history = data.dropna(subset=["target", "price"])
    if history.empty:
        raise RuntimeError("No labeled history after cleaning (target missing everywhere).")

    # Live rows = target is NaN but features exist (future not realized yet)
    live = data[data["target"].isna()].dropna(subset=["price"])
    if live.empty:
        raise RuntimeError("No live rows found to score (check end_date / feature construction).")

    # Train once
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(history[feature_cols], history["target"])

    # Score latest row per ticker
    latest_per_ticker = live.groupby(level="ticker").tail(1).copy()
    latest_per_ticker["pred_6m"] = model.predict(latest_per_ticker[feature_cols])

    # Keep essentials
    out = latest_per_ticker.copy()
    if "vol_3m" not in out.columns:
        raise RuntimeError("vol_3m missing; required for vol scaling. Check feature engineering.")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_6m", "price", "vol_3m"])
    return out


# =========================
# Portfolio layer: selection, vol scaling, caps, cash, turnover
# =========================
def load_last_target(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "target_portfolio_latest.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def build_target_portfolio(scored: pd.DataFrame, lcfg: LiveTradeConfig) -> pd.DataFrame:
    df = scored.reset_index().rename(columns={"date": "asof_date"} if "date" in scored.index.names else {})
    df = df.rename(columns={"level_0": "asof_date"}).copy() if "level_0" in df.columns else df
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    df["ticker"] = df["ticker"].astype(str)

    # Signal filter and sort
    df = df[df["pred_6m"] > lcfg.min_pred_return].sort_values("pred_6m", ascending=False)
    if df.empty:
        raise RuntimeError("No names pass min_pred_return filter.")

    # Turnover banding: keep incumbents if still within top (N + buffer)
    last = load_last_target(lcfg.RUN_DIR)
    keep = set()
    if last is not None and "ticker" in last.columns:
        buffer_cut = lcfg.top_n + lcfg.band_buffer
        buffer_names = set(df.head(buffer_cut)["ticker"].tolist())
        keep = set(last["ticker"]).intersection(buffer_names)

    selected = df.head(lcfg.top_n).copy()
    if keep:
        selected = pd.concat([selected, df[df["ticker"].isin(list(keep))]]).drop_duplicates("ticker")
        selected = selected.sort_values("pred_6m", ascending=False).head(lcfg.top_n)

    # Weighting
    investable = 1.0 - lcfg.cash_buffer
    if lcfg.use_vol_scaling:
        vol = selected["vol_3m"].clip(lower=lcfg.vol_floor)
        raw = 1.0 / vol
        w = raw / raw.sum()
    else:
        w = np.repeat(1.0 / len(selected), len(selected))

    selected["target_weight"] = w * investable

    # Cap and renormalize
    selected["target_weight"] = selected["target_weight"].clip(upper=lcfg.max_position_weight)
    s = selected["target_weight"].sum()
    selected["target_weight"] = selected["target_weight"] / s * investable

    cols = ["asof_date", "ticker", "price", "pred_6m", "vol_3m", "target_weight"]
    return selected[cols].copy()


# =========================
# Execution/accounting layer (IBKR)
# =========================
def ib_connect(lcfg: LiveTradeConfig) -> IB:
    if not IB_AVAILABLE:
        raise RuntimeError("ib_insync not installed. Run: pip install ib_insync")
    ib = IB()
    ib.connect(lcfg.IB_HOST, lcfg.IB_PORT, clientId=lcfg.IB_CLIENT_ID)
    if not ib.isConnected():
        raise RuntimeError("IBKR connection failed. Is TWS/IB Gateway running with API enabled?")
    return ib


def ib_net_liquidation_usd(ib: IB, account: str | None) -> float:
    rows = ib.accountSummary()
    # Prefer USD NetLiquidation
    for r in rows:
        if (account is None or r.account == account) and r.tag == "NetLiquidation" and r.currency == "USD":
            return float(r.value)
    # Fallback any currency
    for r in rows:
        if (account is None or r.account == account) and r.tag == "NetLiquidation":
            return float(r.value)
    raise RuntimeError("Could not read NetLiquidation from accountSummary.")


def ib_positions_stk(ib: IB, account: str | None) -> pd.DataFrame:
    rows = []
    for p in ib.positions():
        if (account is None or p.account == account) and p.contract.secType == "STK":
            rows.append({"ticker": p.contract.symbol, "position": float(p.position), "avgCost": float(p.avgCost)})
    return pd.DataFrame(rows)


def stock_contract(symbol: str) -> Stock:
    return Stock(symbol, "SMART", "USD")


def build_trade_plan(target: pd.DataFrame, positions: pd.DataFrame, equity: float, lcfg: LiveTradeConfig) -> pd.DataFrame:
    tgt = target.copy()
    pos = positions.copy() if positions is not None else pd.DataFrame(columns=["ticker", "position"])
    if pos.empty:
        pos = pd.DataFrame({"ticker": [], "position": []})

    m = tgt.merge(pos[["ticker", "position"]], on="ticker", how="left").fillna({"position": 0.0})
    m["target_dollars"] = m["target_weight"] * equity
    m["target_shares"] = m["target_dollars"] / m["price"]
    m["delta_shares"] = m["target_shares"] - m["position"]
    m["delta_dollars"] = m["delta_shares"] * m["price"]

    # No-trade zone (turnover control): skip if implied weight change is small
    m["delta_weight"] = (m["delta_dollars"] / equity).replace([np.inf, -np.inf], 0.0)
    m = m[m["delta_weight"].abs() >= lcfg.no_trade_weight_delta]

    # Ignore small notionals
    m = m[m["delta_dollars"].abs() >= lcfg.MIN_TRADE_USD]

    return m.sort_values("delta_dollars", ascending=False)


def place_fractional_orders(ib: IB, plan: pd.DataFrame, lcfg: LiveTradeConfig) -> list[dict]:
    logs = []
    for _, r in plan.iterrows():
        symbol = r["ticker"]
        px = float(r["price"])
        delta_dollars = float(r["delta_dollars"])
        qty = float(np.round(delta_dollars / px, 4))
        if abs(delta_dollars) < lcfg.MIN_TRADE_USD or abs(qty) < 1e-6:
            continue

        contract = stock_contract(symbol)
        ib.qualifyContracts(contract)

        action = "BUY" if qty > 0 else "SELL"
        q = abs(qty)

        if lcfg.ORDER_TYPE.upper() == "MKT":
            order = MarketOrder(action, q)
        else:
            bps = lcfg.LIMIT_OFFSET_BPS / 10000.0
            limit_px = px * (1.0 + bps) if action == "BUY" else px * (1.0 - bps)
            order = LimitOrder(action, q, float(np.round(limit_px, 2)))

        if lcfg.DRY_RUN:
            logs.append({
                "ts_utc": now_utc_iso(),
                "ticker": symbol,
                "action": action,
                "qty": q,
                "approx_price": px,
                "approx_notional_usd": q * px,
                "order_type": lcfg.ORDER_TYPE.upper(),
                "dry_run": True,
            })
            continue

        trade = ib.placeOrder(contract, order)
        logs.append({
            "ts_utc": now_utc_iso(),
            "ticker": symbol,
            "action": action,
            "qty": q,
            "approx_price": px,
            "approx_notional_usd": q * px,
            "order_type": lcfg.ORDER_TYPE.upper(),
            "dry_run": False,
            "ib_orderId": getattr(trade.order, "orderId", None),
        })

        # pacing
        time.sleep(0.25)

    return logs


# =========================
# Main runner
# =========================
def main():
    lcfg = LiveTradeConfig()
    ensure_dir(lcfg.RUN_DIR)

    # Use your existing Config/DataEngine so macro + fundamentals align with research
    cfg = Config(mode=lcfg.mode)
    cfg.start_date = lcfg.start_date
    cfg.end_date = lcfg.end_date

    engine = DataEngine(cfg)

    # Download all needed tickers
    all_tickers = cfg.tickers + [cfg.benchmark] + list(cfg.macro_map.values())
    all_tickers = list(dict.fromkeys(all_tickers))
    ohlcv = engine.download_ohlcv(all_tickers)

    # Build macro_adjclose
    def get_adjclose(t: str) -> pd.Series:
        if isinstance(ohlcv.columns, pd.MultiIndex) and t in ohlcv.columns.get_level_values(0):
            return ohlcv[t]["Adj Close"].rename(t)
        if "Adj Close" in ohlcv.columns:
            return ohlcv["Adj Close"].rename(t)
        raise KeyError(f"Adj Close not found for {t}")

    macro_prices = {}
    for name, tkr in cfg.macro_map.items():
        try:
            macro_prices[name] = get_adjclose(tkr)
        except Exception:
            continue
    macro_adjclose = pd.DataFrame(macro_prices).dropna(how="all")
    macro_feat = build_macro_features(macro_adjclose)

    # Add Greek macro if ATHEX
    if cfg.mode.upper() == "ATHEX":
        try:
            gr = GreekMacroEngine.get_daily_features(cfg.start_date)
            macro_feat = macro_feat.join(gr, how="outer").ffill()
        except Exception as e:
            print(f"[WARN] Greek macro fetch failed: {e}")

    fundamentals_q = engine.download_quarterly_fundamentals(cfg.tickers)

    # IMPORTANT: keep unlabeled rows for live scoring
    panel = FeatureEngineer.make_panel_keep_unlabeled(ohlcv, cfg.tickers, cfg)
    panel = FeatureEngineer.add_macro(panel, macro_feat)
    try:
        panel = FeatureEngineer.add_fundamentals(panel, fundamentals_q, cfg)
    except Exception as e:
        print(f"[WARN] Fundamentals join skipped: {e}")

    scored = train_once_score_latest(panel, cfg)
    asof = scored.index.get_level_values(0).max()
    print(f"\nLIVE SCORE ASOF: {asof.date()} | names scored: {scored.index.get_level_values(1).nunique()}")

    target = build_target_portfolio(scored, lcfg)
    target = target.sort_values("pred_6m", ascending=False)

    # Persist “latest target” for turnover banding
    target.to_csv(os.path.join(lcfg.RUN_DIR, "target_portfolio_latest.csv"), index=False)

    # Log signals
    target_log = target.copy()
    target_log["ts_utc"] = now_utc_iso()
    append_csv(os.path.join(lcfg.RUN_DIR, "signals.csv"), target_log)

    print("\n=== TARGET PORTFOLIO ===")
    disp = target.copy()
    disp["pred_6m_%"] = (disp["pred_6m"] * 100).round(1)
    disp["w_%"] = (disp["target_weight"] * 100).round(2)
    print(disp[["ticker", "price", "pred_6m_%", "vol_3m", "w_%"]].to_string(index=False))

    if not lcfg.USE_IBKR:
        print("\nIBKR execution disabled. Done.")
        return

    ib = ib_connect(lcfg)
    try:
        equity = ib_net_liquidation_usd(ib, lcfg.IB_ACCOUNT)
        append_csv(os.path.join(lcfg.RUN_DIR, "equity_curve.csv"),
                   pd.DataFrame([{"ts_utc": now_utc_iso(), "equity_usd": equity}]))

        positions = ib_positions_stk(ib, lcfg.IB_ACCOUNT)
        if positions is None or positions.empty:
            positions = pd.DataFrame(columns=["ticker", "position", "avgCost"])
        pos_log = positions.copy()
        pos_log["ts_utc"] = now_utc_iso()
        append_csv(os.path.join(lcfg.RUN_DIR, "positions.csv"), pos_log)

        plan = build_trade_plan(target, positions, equity, lcfg)

        print("\n=== TRADE PLAN ===")
        show = plan.copy()
        show["delta_$"] = show["delta_dollars"].round(2)
        show["delta_sh"] = show["delta_shares"].round(4)
        show["tgt_sh"] = show["target_shares"].round(4)
        print(show[["ticker", "price", "position", "tgt_sh", "delta_sh", "delta_$"]].to_string(index=False))

        trade_logs = place_fractional_orders(ib, plan, lcfg)
        if trade_logs:
            append_csv(os.path.join(lcfg.RUN_DIR, "trades.csv"), pd.DataFrame(trade_logs))

        if lcfg.DRY_RUN:
            print("\nDRY_RUN=True: no orders sent. Trades logged as dry-run entries.")
        else:
            print(f"\nOrders sent: {len(trade_logs)}")

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
