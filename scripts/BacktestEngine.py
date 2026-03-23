"""
BacktestEngine.py
-----------------
Realistic portfolio backtest that addresses the 4 main problems that
prevent a quant model from beating the S&P 500:

  1. Transaction costs   — applied per actual turnover
  2. Benchmark comparison — SPY used as reference
  3. Portfolio turnover  — tracked explicitly each period
  4. Signal-weighted sizing — position size ∝ predicted return magnitude

Usage
-----
    from BacktestEngine import BacktestEngine

    # After running model.test_model() so test_df has 'pred_return':
    engine = BacktestEngine(
        test_df         = model.test_df,
        target_col      = model.target_key,
        holding_period  = 5,        # N days
        n_positions     = 20,       # stocks per side (long / short)
        cost_bps        = 5.0,      # one-way cost in basis points
        sizing          = "signal", # "equal" or "signal"
        spy_csv         = "../Data/SPY.csv",  # optional benchmark
    )
    results = engine.run()
    engine.plot(results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class BacktestEngine:
    """
    Simulates a long-short equity strategy using model predictions.

    Parameters
    ----------
    test_df : pd.DataFrame
        Output of Model.test_df after calling test_model(). Must contain
        columns: 'date', 'act_symbol', 'pred_return', and the target column.
    target_col : str
        Name of the realised-return column (e.g. 'FWD_LOG_RET_5_T').
    holding_period : int
        How many trading days each portfolio is held (same as the target
        horizon N).  Rebalancing happens every N days.
    n_positions : int
        Number of stocks in the long book (and optionally the short book).
    cost_bps : float
        One-way transaction cost in basis points (5 bps = 0.05%).
        Round-trip = 2 × cost_bps × turnover_rate.
    sizing : str
        'equal'  — equal dollar weight for every position.
        'signal' — weight ∝ |predicted return| (signal-proportional sizing).
    long_only : bool
        If True run a long-only strategy; if False run long-short.
    spy_csv : str or None
        Path to a CSV with at least columns ['date','close'] for SPY.
        If None, benchmark comparison is skipped.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        target_col: str,
        holding_period: int = 5,
        n_positions: int = 20,
        cost_bps: float = 5.0,
        sizing: str = "signal",
        long_only: bool = False,
        spy_csv: str | None = None,
    ):
        if "pred_return" not in test_df.columns:
            raise ValueError("test_df must contain 'pred_return'. Run model.test_model() first.")
        if target_col not in test_df.columns:
            raise ValueError(f"'{target_col}' not found in test_df.")

        self.df = test_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date")

        self.target_col = target_col
        self.N = holding_period
        self.n_positions = n_positions
        self.cost_bps = cost_bps / 10_000        # convert to decimal
        self.sizing = sizing
        self.long_only = long_only
        self.spy_csv = spy_csv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Runs the full backtest simulation.

        Returns
        -------
        dict with keys:
            equity_curve   — pd.Series, daily cumulative log-return
            turnover_series— pd.Series, turnover rate per rebalance date
            metrics        — dict of summary statistics
            rebal_log      — pd.DataFrame, period-by-period detail
        """
        sorted_dates = sorted(self.df["date"].unique())
        rebal_dates  = sorted_dates[::self.N]

        period_returns   = []
        period_costs     = []
        period_turnovers = []
        period_dates     = []

        prev_long, prev_short = set(), set()

        for i, date in enumerate(rebal_dates):
            day_df = self.df[self.df["date"] == date].copy()
            if len(day_df) < self.n_positions * 2:
                continue

            n = min(self.n_positions, len(day_df) // 2)

            long_stocks  = day_df.nlargest(n, "pred_return")
            short_stocks = day_df.nsmallest(n, "pred_return")

            # --- Position sizing ---
            long_weights  = self._compute_weights(long_stocks)
            short_weights = self._compute_weights(short_stocks)

            # --- Realised returns (use target_col as realised return proxy) ---
            long_ret  = (long_stocks[self.target_col].values * long_weights).sum()
            short_ret = (short_stocks[self.target_col].values * short_weights).sum()

            if self.long_only:
                gross_return = long_ret
            else:
                gross_return = long_ret - short_ret

            # --- Turnover ---
            curr_long  = set(long_stocks["act_symbol"])
            curr_short = set(short_stocks["act_symbol"])

            if prev_long or prev_short:
                long_to  = self._turnover(prev_long,  curr_long)
                short_to = self._turnover(prev_short, curr_short) if not self.long_only else 0.0
                avg_to   = (long_to + short_to) / (1 if self.long_only else 2)
            else:
                avg_to = 1.0   # first period — full cost

            # Round-trip cost (2× one-way) scaled by turnover
            cost = 2 * self.cost_bps * avg_to

            period_returns.append(gross_return)
            period_costs.append(cost)
            period_turnovers.append(avg_to)
            period_dates.append(date)

            prev_long, prev_short = curr_long, curr_short

        if not period_dates:
            raise RuntimeError("No rebalancing periods found. Check your test_df dates.")

        results_df = pd.DataFrame({
            "date":     period_dates,
            "gross_ret": period_returns,
            "cost":      period_costs,
            "net_ret":   np.array(period_returns) - np.array(period_costs),
            "turnover":  period_turnovers,
        }).set_index("date")

        # --- Build daily equity curve (approximate: spread return over N days) ---
        daily_net = results_df["net_ret"] / self.N
        equity_curve = daily_net.cumsum()

        # --- SPY benchmark ---
        spy_curve = self._load_spy(equity_curve.index)

        # --- Summary metrics ---
        metrics = self._compute_metrics(results_df, equity_curve, spy_curve)

        return {
            "equity_curve":    equity_curve,
            "spy_curve":       spy_curve,
            "turnover_series": results_df["turnover"],
            "metrics":         metrics,
            "rebal_log":       results_df,
        }

    def plot(self, results: dict):
        """Produces a 3-panel dashboard from run() output."""
        eq   = results["equity_curve"]
        spy  = results["spy_curve"]
        to   = results["turnover_series"]
        m    = results["metrics"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("BacktestEngine — Strategy vs S&P 500", fontsize=13, fontweight="bold")

        # Panel 1: Equity curves
        axes[0].plot(eq.index, eq.values, color="purple", linewidth=2, label="Strategy (Net)")
        if spy is not None:
            spy_aligned = spy.reindex(eq.index, method="ffill").dropna()
            axes[0].plot(spy_aligned.index, spy_aligned.values, color="steelblue",
                         linewidth=2, linestyle="--", label="SPY")
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_title("Cumulative Log Return")
        axes[0].legend()
        axes[0].set_xlabel("Date")

        # Panel 2: Rolling Sharpe (6-month window)
        window = max(10, 126 // self.N)
        rolling_ret  = results["rebal_log"]["net_ret"]
        rolling_sr   = (rolling_ret.rolling(window).mean() /
                        rolling_ret.rolling(window).std()) * np.sqrt(252 / self.N)
        axes[1].plot(rolling_sr.index, rolling_sr.values, color="green", linewidth=2)
        axes[1].axhline(0,   color="black", linewidth=1)
        axes[1].axhline(1.0, color="red",   linewidth=1, linestyle="--", label="Sharpe=1")
        axes[1].set_title(f"Rolling {window}-period Sharpe")
        axes[1].legend()

        # Panel 3: Turnover bar chart
        axes[2].bar(to.index, to.values * 100, color="coral", edgecolor="black", alpha=0.7)
        axes[2].set_title("Portfolio Turnover per Rebalance (%)")
        axes[2].set_ylabel("Turnover %")
        axes[2].set_xlabel("Date")

        plt.tight_layout()
        plt.show()

        # Print metrics table
        print("\nPERFORMANCE SUMMARY")
        print("=" * 45)
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:.4f}")
            else:
                print(f"  {k:<30} {v}")
        print("=" * 45)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weights(self, stock_df: pd.DataFrame) -> np.ndarray:
        if self.sizing == "equal" or len(stock_df) == 0:
            return np.ones(len(stock_df)) / len(stock_df)

        # Signal-proportional: weight ∝ |predicted return|
        raw = np.abs(stock_df["pred_return"].values)
        total = raw.sum()
        if total == 0:
            return np.ones(len(stock_df)) / len(stock_df)
        return raw / total

    @staticmethod
    def _turnover(prev: set, curr: set) -> float:
        """Fraction of portfolio that changed (symmetric difference / union)."""
        union = prev | curr
        if not union:
            return 0.0
        return len(prev.symmetric_difference(curr)) / len(union)

    def _load_spy(self, strategy_index: pd.DatetimeIndex) -> pd.Series | None:
        if self.spy_csv is None:
            return None
        try:
            spy = pd.read_csv(self.spy_csv)

            # Normalize column names: lowercase + strip spaces
            spy.columns = [c.strip().lower().replace("/", "_").replace(" ", "_") for c in spy.columns]

            # Accept several common date column names
            date_col = next((c for c in spy.columns if c in ("date", "time", "timestamp")), None)
            if date_col is None:
                raise ValueError(f"No date column found. Columns: {spy.columns.tolist()}")

            # Accept several common close price column names
            close_col = next(
                (c for c in spy.columns if c in ("close", "close_last", "adj_close", "adjclose", "price")),
                None,
            )
            if close_col is None:
                raise ValueError(f"No close column found. Columns: {spy.columns.tolist()}")

            spy[date_col]  = pd.to_datetime(spy[date_col])
            spy[close_col] = pd.to_numeric(spy[close_col].astype(str).str.replace(",", ""), errors="coerce")

            spy = spy.sort_values(date_col).set_index(date_col)
            spy_log = np.log(spy[close_col] / spy[close_col].shift(1)).dropna()
            spy_cum = spy_log.cumsum()

            # Align to strategy rebalancing dates
            spy_aligned = spy_cum.reindex(strategy_index, method="ffill")

            # Zero-base at strategy start
            first_valid = spy_aligned.dropna()
            offset = float(first_valid.iloc[0]) if len(first_valid) > 0 else 0.0
            return spy_aligned - offset
        except Exception as e:
            print(f"[BacktestEngine] Could not load SPY benchmark: {e}")
            return None

    def _compute_metrics(
        self,
        results_df: pd.DataFrame,
        equity_curve: pd.Series,
        spy_curve: pd.Series | None,
    ) -> dict:
        net = results_df["net_ret"]
        ann = np.sqrt(252 / self.N)

        mean_ret  = net.mean()
        std_ret   = net.std()
        sharpe    = (mean_ret / std_ret * ann) if std_ret != 0 else np.nan

        # Max drawdown on equity curve
        roll_max  = equity_curve.cummax()
        drawdown  = equity_curve - roll_max
        max_dd    = drawdown.min()
        calmar    = (mean_ret * 252 / self.N) / abs(max_dd) if max_dd != 0 else np.nan

        avg_to    = results_df["turnover"].mean()
        avg_cost  = results_df["cost"].mean()
        win_rate  = (net > 0).mean()

        metrics = {
            "Annualized Sharpe (Net)":  sharpe,
            "Mean Period Return (Net)":  mean_ret,
            "Ann. Return (approx)":      mean_ret * (252 / self.N),
            "Max Drawdown":              max_dd,
            "Calmar Ratio":              calmar,
            "Win Rate":                  win_rate,
            "Avg Turnover per Period":   avg_to,
            f"Avg Cost per Period ({self.cost_bps*10_000:.0f}bps/leg)": avg_cost,
            "N Rebalancing Periods":     len(results_df),
        }

        if spy_curve is not None:
            spy_aligned = spy_curve.reindex(equity_curve.index, method="ffill").dropna()
            if len(spy_aligned) > 1:
                spy_total  = spy_aligned.iloc[-1] - spy_aligned.iloc[0]
                strat_total = equity_curve.iloc[-1] if not equity_curve.empty else 0.0
                metrics["SPY Total Return (log)"]      = float(spy_total)
                metrics["Strategy Total Return (log)"] = float(strat_total)
                metrics["Alpha vs SPY (log)"]          = float(strat_total - spy_total)

        return metrics
