import os
import time
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import yfinance as yf
import requests
import requests_cache

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import shap

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# =========================
# Configuration
# =========================

@dataclass
class Config:
    mode: str = "DJI"  # "DJI" or "ATHEX"

    start_date: str = "2010-01-01"
    end_date: str = "2026-01-01"

    # Prediction target: forward return over horizon_days (trading days)
    horizon_days: int = 126  # ~6 months

    # Walk-forward
    train_years: int = 5
    step_months: int = 1  # retrain each month

    # Portfolio
    top_n: int = 5
    tx_cost_bps: float = 15.0  # 0.15%

    # Data robustness
    cache_dir: str = "market_data_cache"
    min_history_days: int = 600
    max_retries: int = 3
    retry_sleep_sec: float = 2.0
    batch_size: int = 10  # keep requests smaller to reduce blocks

    # Fundamentals (best-effort from Yahoo)
    fundamentals_lag_days: int = 45
    
    @property
    def fundamentals_lag(self) -> int:
        return self.fundamentals_lag_days

    # SHAP
    shap_max_rows_per_window: int = 1000

    # Random seed
    seed: int = 42

    # Universe / benchmark / macro tickers
    tickers: List[str] = field(default_factory=list)
    benchmark: str = ""
    macro_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode.upper() == "DJI":
            # Dow 30-ish, liquid, good for validation
            self.tickers = [
                "AAPL", "MSFT", "JPM", "V", "PG", "JNJ", "WMT", "KO",
                "DIS", "MCD", "GS", "IBM", "CAT", "MMM", "AXP", "BA",
                "CSCO", "CVX", "HD", "HON", "INTC", "MRK", "NKE", "TRV",
                "UNH", "VZ", "CRM", "AMGN", "DOW", "RTX"
            ]
            self.benchmark = "^DJI"
            self.macro_map = {
                "SP500": "^GSPC",
                "VIX": "^VIX",
                "OIL": "BZ=F",
                "GOLD": "GC=F",
                "DXY": "DX-Y.NYB",
                "US10Y": "^TNX",
            }

        elif self.mode.upper() == "ATHEX":
            # Example Greek large caps (you can expand)
            self.tickers = [
                "ALPHA.AT", "ETE.AT", "EUROB.AT", "PPC.AT", "OPAP.AT",
                "MYTIL.AT", "TPEIR.AT", "GEKTERNA.AT", "MOH.AT",
                "TITC.AT", "BELA.AT", "TENERGY.AT", "ELPE.AT", "LAMDA.AT"
            ]
            self.benchmark = "^FTASE"
            self.macro_map = {
                "EUROSTOXX50": "^STOXX50E",
                "VIX": "^VIX",
                "OIL": "BZ=F",
                "GOLD": "GC=F",
                "EURUSD": "EURUSD=X",
            }
        else:
            raise ValueError("mode must be 'DJI' or 'ATHEX'")


# =========================
# Utilities
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_name(s: str) -> str:
    return s.replace("^", "IDX_").replace("=", "_").replace(".", "_").replace("/", "_")

def month_start(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=d.year, month=d.month, day=1)

def to_bps(x: float) -> float:
    return x / 10000.0


# =========================
# Data Engine (Yahoo + caching + batching + retries)
# =========================

class DataEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ensure_dir(cfg.cache_dir)

        # Cache HTTP responses to reduce repeated calls (helps rate limiting)
        requests_cache.install_cache(
            cache_name=os.path.join(cfg.cache_dir, "http_cache"),
            backend="sqlite",
            expire_after=3600  # 1 hour
        )

        # yfinance uses requests under the hood; a warmed requests stack can help.
        yf.set_tz_cache_location(os.path.join(cfg.cache_dir, "yf_tz_cache"))

    def _parquet_path(self, key: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"{safe_name(key)}.parquet")

    def download_ohlcv(self, tickers: List[str]) -> pd.DataFrame:
        """
        Returns a MultiIndex columns dataframe like:
        columns = (ticker, field) where field in [Open, High, Low, Close, Adj Close, Volume]
        """
        all_frames = []
        tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order

        # Batch to reduce triggering blocks
        for i in range(0, len(tickers), self.cfg.batch_size):
            batch = tickers[i:i + self.cfg.batch_size]
            key = "ohlcv_" + "_".join([safe_name(t) for t in batch]) + f"_{self.cfg.start_date}_{self.cfg.end_date}"
            path = self._parquet_path(key)

            if os.path.exists(path):
                df = pd.read_parquet(path)
                all_frames.append(df)
                continue

            df = self._download_with_retries(batch)
            if df is None or df.empty:
                continue

            df.to_parquet(path)
            all_frames.append(df)

            # Small pause between batches
            time.sleep(0.5)

        if not all_frames:
            raise RuntimeError("No data returned from yfinance for OHLCV. Your network/Yahoo access is blocked or rate-limited.")

        # Merge batches (columns are MultiIndex)
        out = pd.concat(all_frames, axis=1)
        out = out.loc[~out.index.duplicated(keep="first")].sort_index()
        return out

    def _download_with_retries(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        last_err = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                df = yf.download(
                    tickers=tickers,
                    start=self.cfg.start_date,
                    end=self.cfg.end_date,
                    progress=False,
                    group_by="ticker",
                    threads=False,          # threads sometimes increases blocks
                    auto_adjust=False
                )
                if df is None or df.empty:
                    raise RuntimeError("Empty response from yf.download")
                return df
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_sleep_sec * attempt)
        print(f"[WARN] batch failed {tickers}: {last_err}")
        return None

    def download_quarterly_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """
        Best-effort quarterly fundamentals from Yahoo via yfinance.
        Returns a panel indexed by (report_date, ticker) with columns:
          NetMargin, ROE, DebtToEquity
        """
        frames = []
        for t in tickers:
            path = self._parquet_path(f"fund_{t}_{self.cfg.start_date}_{self.cfg.end_date}")
            if os.path.exists(path):
                frames.append(pd.read_parquet(path))
                continue

            try:
                tk = yf.Ticker(t)
                inc = tk.quarterly_income_stmt.T
                bs = tk.quarterly_balance_sheet.T

                if inc is None or bs is None or inc.empty or bs.empty:
                    continue

                inc.index = pd.to_datetime(inc.index)
                bs.index = pd.to_datetime(bs.index)
                df = inc.join(bs, how="outer", lsuffix="_inc", rsuffix="_bs").sort_index()

                def find_col(cands: List[str]) -> Optional[str]:
                    for c in cands:
                        hits = [col for col in df.columns if c.lower() == str(col).lower()]
                        if hits:
                            return hits[0]
                    for c in cands:
                        hits = [col for col in df.columns if c.lower() in str(col).lower()]
                        if hits:
                            return hits[0]
                    return None

                rev = find_col(["Total Revenue", "TotalRevenue"])
                net = find_col(["Net Income", "NetIncome"])
                eq = find_col(["Stockholders Equity", "Total Equity", "TotalStockholderEquity"])
                debt = find_col(["Total Debt", "TotalDebt"])

                clean = pd.DataFrame(index=df.index)
                if rev and net:
                    clean["NetMargin"] = df[net] / df[rev].replace(0, np.nan)
                if net and eq:
                    clean["ROE"] = df[net] / df[eq].replace(0, np.nan)
                if debt and eq:
                    clean["DebtToEquity"] = df[debt] / df[eq].replace(0, np.nan)

                clean["ticker"] = t
                clean = clean.reset_index().rename(columns={"index": "report_date"})
                clean = clean.set_index(["report_date", "ticker"])

                clean.to_parquet(path)
                frames.append(clean)

                time.sleep(0.2)

            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames).sort_index()
        return out


# =========================
# Greek Macro (FRED, free CSV)
# =========================

class GreekMacroEngine:
    @staticmethod
    def fetch_fred_series(series_id: str, col_name: str, start_date: str) -> pd.Series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, parse_dates=["DATE"])
        df = df.rename(columns={series_id: col_name}).set_index("DATE")
        df = df[df.index >= pd.to_datetime(start_date)]
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        return df[col_name]

    @staticmethod
    def get_daily_features(start_date: str) -> pd.DataFrame:
        # Greece 10Y yield + CPI (common free series)
        # IRLTLT01GRM156N and CPALTT01GRM659N are widely used FRED series for Greece (yield, CPI)
        gr10y = GreekMacroEngine.fetch_fred_series("IRLTLT01GRM156N", "GR_10Y", start_date)
        grcpi = GreekMacroEngine.fetch_fred_series("CPALTT01GRM659N", "GR_CPI", start_date)

        macro = pd.concat([gr10y, grcpi], axis=1).sort_index()

        # Convert to daily (forward fill)
        macro = macro.resample("D").ffill()

        out = pd.DataFrame(index=macro.index)
        out["GR_10Y_Level"] = macro["GR_10Y"]
        out["GR_10Y_Change_3M"] = macro["GR_10Y"].diff(90)  # approx 3 months (calendar)
        out["GR_CPI_YoY"] = macro["GR_CPI"].pct_change(365)

        return out.ffill()


# =========================
# Feature Engineering
# =========================

class FeatureEngineer:
    @staticmethod
    def rsi(price: pd.Series, period: int = 14) -> pd.Series:
        delta = price.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss.replace(0, np.nan))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def make_panel(ohlcv: pd.DataFrame, tickers: List[str], cfg: Config) -> pd.DataFrame:
        """
        Returns a panel indexed by (date, ticker) with:
          tech features + target + price
        """
        frames = []
        for t in tickers:
            try:
                if isinstance(ohlcv.columns, pd.MultiIndex) and t in ohlcv.columns.get_level_values(0):
                    px = ohlcv[t]["Adj Close"].dropna()
                else:
                    # single ticker fallback
                    px = ohlcv["Adj Close"].dropna()
            except Exception:
                continue

            if len(px) < cfg.min_history_days:
                continue

            df = pd.DataFrame(index=px.index)
            df["ticker"] = t

            # Momentum / returns
            df["ret_1m"] = px.pct_change(21)
            df["ret_3m"] = px.pct_change(63)
            df["ret_6m"] = px.pct_change(126)

            # Volatility
            df["vol_3m"] = px.pct_change().rolling(63).std()

            # RSI / drawdown
            df["rsi_14"] = FeatureEngineer.rsi(px, 14)
            roll_max = px.rolling(252).max()
            df["drawdown_52w"] = (px / roll_max) - 1.0

            # price for backtest execution
            df["price"] = px

            # Target (forward return over horizon)
            df["target"] = px.shift(-cfg.horizon_days) / px - 1.0

            frames.append(df)

        panel = pd.concat(frames).dropna(subset=["target"]).sort_index()
        panel.index.name = "date"
        panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()
        return panel

    @staticmethod
    def add_macro(panel: pd.DataFrame, macro_feat: pd.DataFrame) -> pd.DataFrame:
        """
        Join macro features to a panel that is indexed by date (and may include ticker as a column
        or as part of a MultiIndex). This function is robust to either structure.

        Expected:
        - macro_feat: DatetimeIndex, columns = macro features
        - panel: either
            A) index = DatetimeIndex, column 'ticker' exists, or
            B) MultiIndex (date, ticker)
        """
        if macro_feat is None or macro_feat.empty:
            return panel

        print("panel index:", type(panel.index), panel.index[:3])
        print("macro index:", type(macro_feat.index), macro_feat.index[:3])
        print("macro columns:", list(macro_feat.columns))

        # Ensure macro_feat index is DatetimeIndex and sorted
        macro_feat = macro_feat.copy()
        macro_feat.index = pd.to_datetime(macro_feat.index)
        macro_feat = macro_feat.sort_index().ffill()

        out = panel.copy()

        # Case B: MultiIndex (date, ticker)
        if isinstance(out.index, pd.MultiIndex):
            # assume first level is date
            date_level = out.index.names[0] or 0
            dates = out.index.get_level_values(0)
            # reindex macro to all dates then join by date level
            macro_aligned = macro_feat.reindex(pd.Index(pd.to_datetime(dates))).ffill()
            macro_aligned.index = out.index  # broadcast down to (date,ticker)
            out = pd.concat([out, macro_aligned], axis=1)
            return out

        # Case A: DatetimeIndex
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        macro_aligned = macro_feat.reindex(out.index).ffill()
        out = pd.concat([out, macro_aligned], axis=1)
        return out

    @staticmethod
    def add_fundamentals(panel: pd.DataFrame, fundamentals_q: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        """
        Robust fundamentals join with reporting lag.
        - panel: either indexed by date with 'ticker' column OR MultiIndex (date, ticker)
        - fundamentals_q: expected to have columns including ['ticker'] and a DatetimeIndex of report dates
                        OR MultiIndex (date, ticker). We will normalize it.
        """
        if fundamentals_q is None or fundamentals_q.empty:
            return panel

        out = panel.copy()
        
        print("FUND panel index:", type(panel.index), panel.index[:3])
        print("FUND panel cols:", list(panel.columns)[:20])
        print("FUND fundamentals index:", type(fundamentals_q.index), fundamentals_q.index[:3])
        print("FUND fundamentals cols:", list(fundamentals_q.columns)[:20])


        # --- Normalize panel to MultiIndex (date, ticker) without losing data
        if isinstance(out.index, pd.MultiIndex):
            # Assume first level is date and second is ticker (common pattern)
            out_mi = out.copy()
            out_mi.index = out_mi.index.set_names(["date", "ticker"])
        else:
            # Single DatetimeIndex: must have a ticker column to become a panel
            if "ticker" not in out.columns:
                raise KeyError(
                    "Panel has a single-level date index but no 'ticker' column. "
                    "Your feature builder must add a 'ticker' column before calling add_fundamentals(), "
                    "or you must keep the panel as a MultiIndex (date, ticker)."
                )
            out_mi = out.copy()
            out_mi = out_mi.reset_index().rename(columns={out.index.name or "index": "date"})
            out_mi["date"] = pd.to_datetime(out_mi["date"])
            out_mi = out_mi.set_index(["date", "ticker"]).sort_index()

        # --- Normalize fundamentals to MultiIndex (date, ticker)
        f = fundamentals_q.copy()

        # If fundamentals comes as DatetimeIndex + 'ticker' column
        if not isinstance(f.index, pd.MultiIndex):
            if "ticker" not in f.columns:
                # If fundamentals was concatenated with a ticker column elsewhere, this should exist.
                # If not, we cannot safely join.
                return out  # fail-soft
            f = f.copy()
            f = f.reset_index().rename(columns={f.index.name or "index": "date"})
            f["date"] = pd.to_datetime(f["date"]) + pd.Timedelta(days=int(cfg.fundamentals_lag))
            f = f.set_index(["date", "ticker"]).sort_index()
        else:
            # MultiIndex: apply lag to date level
            f = f.copy()
            f.index = f.index.set_names(["date", "ticker"])
            # shift the date level by lag days
            dates = pd.to_datetime(f.index.get_level_values("date")) + pd.Timedelta(days=int(cfg.fundamentals_lag))
            tickers = f.index.get_level_values("ticker")
            f.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
            f = f.sort_index()

        # --- Join fundamentals and forward-fill within ticker (quarterly -> daily)
        joined = out_mi.join(f, how="left")
        joined = joined.groupby(level="ticker").ffill()
        
        # Track whether fundamentals are actually available (after ffill)
        fund_cols = [c for c in f.columns if c in joined.columns]
        if fund_cols:
            joined["has_fundamentals"] = joined[fund_cols].notna().any(axis=1).astype(int)

            # For early history where Yahoo has no fundamentals at all, fill with cross-sectional medians (or 0.0)
            # Median is usually safer than 0 for ratios like ROE.
            for c in fund_cols:
                med = joined[c].median(skipna=True)
                joined[c] = joined[c].fillna(med if pd.notna(med) else 0.0)
        else:
            joined["has_fundamentals"] = 0

        # Return in the same shape as input
        if isinstance(out.index, pd.MultiIndex):
            return joined
        else:
            # back to single date index + ticker column if you want; safest is keep MultiIndex
            # but to preserve your prior behavior:
            joined_reset = joined.reset_index().set_index("date").sort_index()
            return joined_reset


# =========================
# Walk-forward training + out-of-sample SHAP
# =========================

class WalkForwardXGB:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.param_dist = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.5, 1.0, 2.0],
        }

    def run(self, panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          preds_df: rows indexed by (date, ticker) with prediction, target, price
          shap_global_df: per-window mean(|SHAP|) indexed by window_end
          metrics_df: per-window MSE, counts
        """
        # Select features
        drop_cols = {"target", "price"}
        feature_cols = [c for c in panel.columns if c not in drop_cols]
        # Remove non-numeric
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(panel[c])]

        # Clean NAs
        # Impute remaining missing values in features (fundamentals/macro gaps)
        data = panel.copy()
        data = data.dropna(subset=["target", "price"])

        # Keep only numeric features already selected
        for c in feature_cols:
            if data[c].isna().any():
                med = data[c].median(skipna=True)
                data[c] = data[c].fillna(med if pd.notna(med) else 0.0)

        print("[DEBUG] panel rows:", len(panel))
        print("[DEBUG] after target/price drop:", len(data))
        print("[DEBUG] unique dates:", data.index.get_level_values(0).nunique())
        print("[DEBUG] date min/max:", data.index.get_level_values(0).min(), data.index.get_level_values(0).max())

        # Walk-forward boundaries in calendar months
        dates = pd.Index(sorted(data.index.get_level_values(0).unique()))
        if len(dates) < 252 * (self.cfg.train_years + 1):
            raise RuntimeError("Not enough history after cleaning to run walk-forward.")

        start = dates.min()
        end = dates.max()

        # Build list of month starts
        month_starts = pd.date_range(month_start(start), month_start(end), freq="MS")
        # We start after train_years of data
        train_start_min = month_start(start) + pd.DateOffset(years=self.cfg.train_years)

        preds_chunks = []
        shap_chunks = []
        metrics = []

        for w_end in month_starts:
            if w_end < train_start_min:
                continue

            w_test_start = w_end
            w_test_end = w_end + pd.DateOffset(months=self.cfg.step_months)

            if w_test_start > end:
                break

            w_train_start = w_end - pd.DateOffset(years=self.cfg.train_years)

            train_mask = (data.index.get_level_values(0) >= w_train_start) & (data.index.get_level_values(0) < w_test_start)
            test_mask = (data.index.get_level_values(0) >= w_test_start) & (data.index.get_level_values(0) < w_test_end)

            train_df = data.loc[train_mask]
            test_df = data.loc[test_mask]

            if len(train_df) < 500 or len(test_df) < 50:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df["target"]
            X_test = test_df[feature_cols]
            y_test = test_df["target"]

            model = self._fit_xgb(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = float(mean_squared_error(y_test, y_pred))
            metrics.append({
                "window_end": w_end,
                "train_start": w_train_start,
                "train_end": w_test_start,
                "test_start": w_test_start,
                "test_end": w_test_end,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "mse": mse
            })

            # Predictions out-of-sample for this window
            pc = test_df[["target", "price"]].copy()
            pc["prediction"] = y_pred
            pc["window_end"] = w_end
            preds_chunks.append(pc)

            # OOS SHAP on *test only* (sampled)
            shap_rows = X_test
            if len(shap_rows) > self.cfg.shap_max_rows_per_window:
                shap_rows = shap_rows.sample(self.cfg.shap_max_rows_per_window, random_state=self.cfg.seed)

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(shap_rows)

            mean_abs = np.abs(shap_vals).mean(axis=0)
            s = pd.Series(mean_abs, index=feature_cols, name=w_end)
            shap_chunks.append(s)

            print(f"[WF] {w_end.date()} test={len(test_df)} mse={mse:.6f}")

        preds_df = pd.concat(preds_chunks).sort_index()
        shap_global_df = pd.DataFrame(shap_chunks).sort_index()
        metrics_df = pd.DataFrame(metrics).sort_values("window_end")

        return preds_df, shap_global_df, metrics_df

    def _fit_xgb(self, X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.cfg.seed,
            n_jobs=-1,
            tree_method="hist"
        )
        cv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=self.param_dist,
            n_iter=15,
            scoring="neg_mean_squared_error",
            cv=cv,
            random_state=self.cfg.seed,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X, y)
        return search.best_estimator_


# =========================
# Backtest (monthly rebalance, equal weight, costs)
# =========================

class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, preds: pd.DataFrame, benchmark_adjclose: pd.Series) -> pd.DataFrame:
        """
        preds index: (date,ticker) with columns [target, price, prediction, window_end]
        backtest: monthly rebalance using prediction ranking
        """
        df = preds.reset_index().rename(columns={"level_0": "date"} if "level_0" in preds.columns else {})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"])

        # Identify rebalance dates as month starts present in df
        df["month"] = df["date"].dt.to_period("M").astype(str)
        rebalance_dates = df.groupby("month")["date"].min().values
        rebalance_dates = pd.to_datetime(rebalance_dates)

        # Build a price lookup
        price_tbl = df.pivot(index="date", columns="ticker", values="price").sort_index()

        portfolio_value = 10000.0
        holdings = {}  # ticker -> shares
        curve = []

        all_dates = price_tbl.index

        for current_date in all_dates:
            # Mark-to-market
            if holdings:
                pv = 0.0
                for t, sh in holdings.items():
                    if t in price_tbl.columns:
                        px = price_tbl.at[current_date, t]
                        if pd.notna(px):
                            pv += sh * px
                portfolio_value = pv if pv > 0 else portfolio_value
            curve.append((current_date, portfolio_value))

            # Rebalance at rebalance dates
            if current_date in rebalance_dates:
                todays = df[df["date"] == current_date].dropna(subset=["prediction", "price"])
                if len(todays) < self.cfg.top_n:
                    continue

                top = todays.sort_values("prediction", ascending=False).head(self.cfg.top_n)

                # Apply transaction cost on full turnover (simple approximation)
                portfolio_value *= (1.0 - to_bps(self.cfg.tx_cost_bps))

                # Build new holdings equal-weight
                holdings = {}
                w = portfolio_value / len(top)
                for _, r in top.iterrows():
                    px = float(r["price"])
                    if px > 0:
                        holdings[r["ticker"]] = w / px

        curve_df = pd.DataFrame(curve, columns=["date", "portfolio"]).set_index("date")

        # Benchmark (normalize to 10,000)
        bench = benchmark_adjclose.reindex(curve_df.index).ffill()
        bench = 10000.0 * (bench / bench.iloc[0])
        out = curve_df.join(bench.rename("benchmark"), how="inner")
        return out


# =========================
# Pipeline
# =========================

def run_pipeline(cfg: Config) -> None:
    print(f"\n=== RUN MODE: {cfg.mode} ===")
    ensure_dir(cfg.cache_dir)

    engine = DataEngine(cfg)

    all_tickers = cfg.tickers + [cfg.benchmark] + list(cfg.macro_map.values())
    all_tickers = list(dict.fromkeys(all_tickers))

    ohlcv = engine.download_ohlcv(all_tickers)

    # Split OHLCV into (universe, benchmark, macro)
    def get_adjclose(t: str) -> pd.Series:
        if isinstance(ohlcv.columns, pd.MultiIndex) and t in ohlcv.columns.get_level_values(0):
            return ohlcv[t]["Adj Close"].rename(t)
        # If single-ticker download happened (rare here), try direct
        if "Adj Close" in ohlcv.columns:
            return ohlcv["Adj Close"].rename(t)
        raise KeyError(f"Adj Close not found for {t}")

    # Universe panel
    universe_prices = {}
    for t in cfg.tickers:
        try:
            universe_prices[t] = get_adjclose(t)
        except Exception:
            continue
    universe_adjclose = pd.DataFrame(universe_prices).dropna(how="all")

    # Benchmark
    bench_adjclose = get_adjclose(cfg.benchmark).dropna()

    # Macro (Yahoo-based)
    macro_prices = {}
    for name, tkr in cfg.macro_map.items():
        try:
            macro_prices[name] = get_adjclose(tkr)
        except Exception:
            continue
    macro_adjclose = pd.DataFrame(macro_prices).dropna(how="all")

    # Convert macro to daily features
    macro_feat = pd.DataFrame(index=macro_adjclose.index)
    for c in macro_adjclose.columns:
        s = macro_adjclose[c]
        macro_feat[f"{c}_ret_1m"] = s.pct_change(21)
        macro_feat[f"{c}_ret_3m"] = s.pct_change(63)
        macro_feat[f"{c}_z_1y"] = (s - s.rolling(252).mean()) / (s.rolling(252).std() + 1e-9)
    macro_feat = macro_feat.dropna(how="all")

    # Add Greek macro from FRED if ATHEX
    if cfg.mode.upper() == "ATHEX":
        try:
            gr = GreekMacroEngine.get_daily_features(cfg.start_date)
            macro_feat = macro_feat.join(gr, how="outer").ffill()
        except Exception as e:
            print(f"[WARN] Greek macro fetch failed: {e}")

    # Fundamentals (best-effort)
    fundamentals_q = engine.download_quarterly_fundamentals(cfg.tickers)

    # Build panel
    panel = FeatureEngineer.make_panel(ohlcv, cfg.tickers, cfg)
    panel = FeatureEngineer.add_macro(panel, macro_feat)
    try:
        panel = FeatureEngineer.add_fundamentals(panel, fundamentals_q, cfg)
    except Exception as e:
        print(f"[WARN] Fundamentals join skipped due to: {e}")


    # Drop any remaining NA rows in features/target
    panel = panel.dropna(subset=["target", "price"])

    # Train + OOS SHAP
    wf = WalkForwardXGB(cfg)
    preds, shap_global, metrics = wf.run(panel)

    # Backtest
    bt = Backtester(cfg)
    equity = bt.run(preds, bench_adjclose)

    # Outputs
    out_prefix = f"results_{cfg.mode}"
    preds.to_csv(f"{out_prefix}_predictions.csv")
    shap_global.to_csv(f"{out_prefix}_shap_oos_global.csv")
    metrics.to_csv(f"{out_prefix}_metrics.csv", index=False)
    equity.to_csv(f"{out_prefix}_equity_curve.csv")

    # Summary
    total_ret = equity["portfolio"].iloc[-1] / equity["portfolio"].iloc[0] - 1.0
    bench_ret = equity["benchmark"].iloc[-1] / equity["benchmark"].iloc[0] - 1.0
    print("\n=== SUMMARY ===")
    print(f"Portfolio total return: {total_ret*100:.2f}%")
    print(f"Benchmark total return: {bench_ret*100:.2f}%")
    print(f"Outputs saved with prefix: {out_prefix}_*.csv")


if __name__ == "__main__":
    # Start with DJI until your data access is confirmed stable
    cfg = Config(mode="DJI")
    # When ready:
    # cfg = Config(mode="ATHEX")

    run_pipeline(cfg)
