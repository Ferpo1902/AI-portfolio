"""
run_strategy.py
---------------
End-to-end script to train and backtest a stock-picking strategy
designed to beat the S&P 500.

Fixes applied vs the baseline model:
  1. 15 predictive features (up from 4) covering momentum, mean-reversion,
     microstructure, macro, and statistical regime
  2. Target: 5-day forward log return (FWD_LOG_RET) instead of same-day intraday
  3. Walk-forward analysis — one model per year, tested out-of-sample
  4. Transaction costs deducted (5 bps/leg, scaled by actual turnover)
  5. Long-only portfolio selection — compared directly to SPY
  6. Signal-weighted sizing — position size ∝ |predicted return|

HOW TO RUN
----------
From the scripts/ directory:

    python run_strategy.py

Prerequisites:
  - ../Data/all_ohlcv.feather  (run DataRetrieval/setup_data.py first)
  - ../Data/SPY.csv            (already present in Data/)
  - ../Data/ETFs.feather       (run DataRetrieval/setup_data.py first)

The script will:
  1. Build a liquid universe (or load from cache)
  2. Train one LightGBM model per year via walk-forward analysis
  3. Run a realistic backtest and compare the equity curve to SPY
  4. Print a clear verdict: does it beat the S&P 500?
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Make sure scripts/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_analysis import (
    Model, Universe, TopLiquidityFilter, PriceFilter, AdvancedStatsFilter
)
from FeatureEngine import FeatureRequest, RateFeatureRequest
from AdvancedModels import run_walk_forward_analysis
from BacktestEngine import BacktestEngine

# ==============================================================
# CONFIG — tweak these to experiment
# ==============================================================

DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
UNIVERSE_PATH  = os.path.join(DATA_DIR, "Universes", "strategy_v1.feather")
OHLCV_PATH     = os.path.join(DATA_DIR, "all_ohlcv.feather")
SPY_CSV        = os.path.join(DATA_DIR, "SPY.csv")
MODEL_FOLDER   = os.path.join(DATA_DIR, "models", "sp500_beater")

# Universe filters
UNIVERSE_SIZE  = 500        # top N stocks by monthly dollar volume
PRICE_MIN      = 10.0       # exclude penny stocks

# Walk-forward windows
TRAIN_YEARS    = 4          # years of data used to train each fold
VAL_YEARS      = 1          # years for hyperparameter validation
TEST_YEARS     = 1          # out-of-sample test window per fold
N_TRIALS       = 50         # Optuna hyperparameter trials per fold

# Backtest
HOLDING_PERIOD = 5          # calendar days between rebalances (weekly)
N_POSITIONS    = 25         # stocks held in the long portfolio
COST_BPS       = 5.0        # one-way cost in basis points (realistic for US large-cap)
SIZING         = "signal"   # "equal" or "signal" (weight ∝ |pred_return|)
LONG_ONLY      = True       # True = long-only (compares to SPY buy-and-hold)


# ==============================================================
# STEP 1 — FEATURES
# ==============================================================
# Academic literature on cross-sectional return predictability shows these
# factors have the strongest out-of-sample evidence:
#
# Momentum (12-1 month)  — Jegadeesh & Titman 1993, still works
# Short-term reversal    — Jegadeesh 1990 (1-week mean reversion)
# Low volatility anomaly — Baker, Bradley & Wurgler 2011
# Liquidity premium      — Amihud 2002
# Quality/Earnings       — Sloan 1996 (accruals)
# Macro regime (VIX,     — risk-on/off filters
#   yield curve)

def get_features():
    """
    Returns the optimized list of FeatureRequests.

    All features use shift=-1 so the model sees YESTERDAY's signal
    to predict tomorrow's return (no lookahead bias).
    """
    return [
        # --- Volatility & Microstructure ---
        # Low-vol anomaly: low-volatility stocks outperform on risk-adjusted basis
        FeatureRequest(
            name='VOL_ZSCORE',
            params={'timeperiod': 20},
            shift=-1,
            input_type='raw',
            alias='vol_z_20',
        ),
        # Illiquidity premium: less liquid stocks earn higher returns
        FeatureRequest(
            name='SPREAD_AR',
            params={'timeperiod': 20},
            shift=-1,
            alias='spread_20',
            transform='rank',       # rank cross-sectionally to remove skew
        ),
        # Distance from VWAP: stocks far below VWAP tend to mean-revert up
        FeatureRequest(
            name='VWAP_Z',
            params={'timeperiod': 20},
            shift=-1,
            alias='vwap_z_20',
        ),
        # Candle efficiency: strong directional days signal conviction
        FeatureRequest(
            name='RANGE_EFFICIENCY',
            shift=-1,
            alias='rng_eff',
        ),
        # Overnight gap relative to volatility: news-driven dislocations
        FeatureRequest(
            name='GAP_SIGMA',
            shift=-1,
            alias='gap_sig',
        ),

        # --- Momentum (strongest documented factor) ---
        # 12-month momentum: stocks up over last year keep outperforming
        FeatureRequest(
            name='MOM',
            params={'timeperiod': 252},
            shift=-1,
            input_type='log_ret',
            alias='mom_252',
        ),
        # 1-month short-term reversal: last month's losers bounce back
        FeatureRequest(
            name='MOM',
            params={'timeperiod': 21},
            shift=-1,
            input_type='log_ret',
            alias='mom_21',
        ),
        # RSI: overbought/oversold signal
        FeatureRequest(
            name='RSI',
            params={'timeperiod': 14},
            shift=-1,
            input_type='raw',
            alias='rsi_14',
        ),
        # Trend regime: ADX identifies trending vs choppy stocks
        FeatureRequest(
            name='ADX_REGIME',
            params={'timeperiod': 14},
            shift=-1,
            alias='adx_reg',
        ),
        # 50-day MA crossover: medium-term trend signal
        FeatureRequest(
            name='MA_crossover',
            params={'timeperiod': 50},
            shift=-1,
            alias='ma50_cross',
        ),

        # --- Statistical Properties ---
        # Hurst exponent: H > 0.5 = trending, H < 0.5 = mean-reverting
        FeatureRequest(
            name='HURST',
            params={'timeperiod': 100},
            shift=-1,
            input_type='raw',
            alias='hurst_100',
        ),
        # Autocorrelation: positive = momentum, negative = mean reversion
        FeatureRequest(
            name='AUTOCORR',
            params={'timeperiod': 20},
            shift=-1,
            input_type='log_ret',
            alias='autocorr_20',
        ),
        # Price z-score: how far the stock is from its rolling mean (mean reversion)
        FeatureRequest(
            name='ZSCORE',
            params={'timeperiod': 20},
            shift=-1,
            input_type='log_ret',
            alias='price_z_20',
        ),
        # Beta: low-beta stocks outperform (low-beta anomaly)
        FeatureRequest(
            name='pandas_beta',
            params={'timeperiod': 60},
            shift=-1,
            alias='beta_60',
        ),

        # --- Macro Regime ---
        # VIX: when VIX is high, risk premia are higher; adjust exposure
        FeatureRequest(
            name='VIX',
            shift=-1,
            alias='vix',
        ),
        # Yield curve spread (10y - 2y): positive = expansion, negative = recession risk
        RateFeatureRequest(
            name='RATE_SPREAD',
            term1='10_year',
            term2='2_year',
        ),
    ]


# ==============================================================
# STEP 2 — TARGET
# ==============================================================
def get_target():
    """
    5-day forward log return.

    Why 5 days (weekly)?
    - Long enough to capture real alpha (not microstructure noise)
    - Short enough to rebalance frequently and compound returns
    - Aligns with weekly options expiry cycles (extra liquidity)

    IMPORTANT: No alias — the column name must match "FWD_LOG_RET_5_..."
    so that internal regex in evaluate_quantile_spread extracts N=5 correctly.
    """
    return FeatureRequest(
        name='FWD_LOG_RET',
        params={'timeperiod': 5},
        shift=0,
        input_type='raw',
    )


# ==============================================================
# STEP 3 — BUILD UNIVERSE
# ==============================================================
def build_universe():
    """
    Creates a liquid, investable universe of US equities.
    Saves to feather for reuse (skip re-building if file exists).
    """
    if os.path.exists(UNIVERSE_PATH):
        print(f"[Universe] Loading cached universe: {UNIVERSE_PATH}")
        return pd.read_feather(UNIVERSE_PATH)

    if not os.path.exists(OHLCV_PATH):
        raise FileNotFoundError(
            f"OHLCV data not found at {OHLCV_PATH}.\n"
            "Run DataRetrieval/setup_data.py first to download and cache the data."
        )

    print("[Universe] Building universe from scratch (may take a few minutes)...")
    df = pd.read_feather(OHLCV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Three-stage filter pipeline:
    # 1. Keep the 500 most liquid stocks by monthly dollar volume
    # 2. Exclude penny stocks (price < $10)
    # 3. Exclude stocks that crashed > 60% in the past year (fallen knives)
    filters = [
        TopLiquidityFilter(N=UNIVERSE_SIZE),
        PriceFilter(min_price=PRICE_MIN, method="avg_price_over_last_year"),
        AdvancedStatsFilter(
            min_history_days=252,   # at least 1 year of history
            max_crash=-0.60,        # exclude > 60% annual crash
            require_uptrend=None,   # don't force uptrend (too restrictive)
            volatility_n=None,      # no volatility ranking at universe level
        ),
    ]

    universe = Universe(master_df=df)
    universe.add_filters(filters)

    os.makedirs(os.path.dirname(UNIVERSE_PATH), exist_ok=True)
    universe_data = universe.get_all_universe_data(save_name=UNIVERSE_PATH)

    print(f"[Universe] Saved to {UNIVERSE_PATH}")
    print(f"[Universe] {universe_data['act_symbol'].nunique()} unique stocks, "
          f"{universe_data['date'].nunique()} trading days")
    return universe_data


# ==============================================================
# STEP 4 — WALK-FORWARD TRAINING
# ==============================================================
def train_walk_forward():
    """
    Trains one LightGBM model per TEST_YEARS-year window, rolling forward.
    Returns the concatenated out-of-sample predictions across all folds.
    """
    print("\n[Training] Starting walk-forward analysis...")
    print(f"  Train: {TRAIN_YEARS}y | Val: {VAL_YEARS}y | Test: {TEST_YEARS}y per fold")
    print(f"  Features: {len(get_features())} | Optuna trials: {N_TRIALS}")

    features = get_features()
    target   = get_target()

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    oos_df, metrics_df = run_walk_forward_analysis(
        universe_path = UNIVERSE_PATH,
        target        = target,
        features      = features,
        folder_path   = MODEL_FOLDER,
        target_type   = "regression",
        train_years   = TRAIN_YEARS,
        val_years     = VAL_YEARS,
        test_years    = TEST_YEARS,
        n_trials      = N_TRIALS,
    )

    return oos_df, metrics_df


# ==============================================================
# STEP 5 — BACKTEST vs SPY
# ==============================================================
def run_backtest(oos_df):
    """
    Simulates a weekly-rebalanced long-only portfolio using model predictions.
    Compares equity curve to SPY (buy-and-hold benchmark).
    """
    # Identify the target column (contains "T" as a standalone token)
    target_candidates = [c for c in oos_df.columns if "T" in c.split("_")]
    if not target_candidates:
        raise ValueError("No target column found in OOS dataframe. Check column names.")
    target_col = target_candidates[0]
    print(f"\n[Backtest] Using realised return column: '{target_col}'")

    engine = BacktestEngine(
        test_df        = oos_df,
        target_col     = target_col,
        holding_period = HOLDING_PERIOD,
        n_positions    = N_POSITIONS,
        cost_bps       = COST_BPS,
        sizing         = SIZING,
        long_only      = LONG_ONLY,
        spy_csv        = SPY_CSV,
    )

    results = engine.run()
    engine.plot(results)
    return results


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 65)
    print("  SP500 BEATER — Quant Equity Strategy")
    print(f"  Universe: top {UNIVERSE_SIZE} liquid US stocks")
    print(f"  Target: {HOLDING_PERIOD}-day forward log return")
    print(f"  Positions: {N_POSITIONS} long | Costs: {COST_BPS}bps/leg")
    print("=" * 65)

    # 1. Universe
    build_universe()

    # 2. Walk-forward training
    oos_df, metrics_df = train_walk_forward()

    if oos_df.empty:
        print("\n[ERROR] No out-of-sample predictions were generated.")
        print("  Possible causes:")
        print("  - Universe date range too short for the train/val/test windows")
        print("  - OHLCV data missing or corrupt")
        return

    # 3. Print fold-by-fold metrics
    print("\n[Results] Walk-Forward Metrics by Fold:")
    print(metrics_df.to_string(index=False))

    # 4. Backtest
    results = run_backtest(oos_df)

    # 5. Final verdict
    m      = results["metrics"]
    sharpe = m.get("Annualized Sharpe (Net)", float("nan"))
    alpha  = m.get("Alpha vs SPY (log)",      float("nan"))
    max_dd = m.get("Max Drawdown",             float("nan"))
    calmar = m.get("Calmar Ratio",             float("nan"))
    to     = m.get("Avg Turnover per Period",  float("nan"))

    print("\n" + "=" * 65)
    print("  FINAL VERDICT")
    print("=" * 65)

    if not np.isnan(alpha) and not np.isnan(sharpe):
        beats_spy    = alpha > 0
        decent_risk  = sharpe > 1.0
        tradeable    = abs(max_dd) < 0.40   # drawdown under 40%

        status = "BEATS S&P 500" if (beats_spy and decent_risk) else "DOES NOT BEAT S&P 500 YET"
        print(f"  Status        : {status}")
        print(f"  Alpha vs SPY  : {alpha:+.2%}  (positive = beats benchmark)")
        print(f"  Sharpe (net)  : {sharpe:.2f}   (target > 1.0)")
        print(f"  Max Drawdown  : {max_dd:.2%}")
        print(f"  Calmar Ratio  : {calmar:.2f}   (return / max drawdown)")
        print(f"  Avg Turnover  : {to:.0%}  per {HOLDING_PERIOD}-day period")
        print()

        if beats_spy and decent_risk and tradeable:
            print("  ACTION: Strategy is backtest-validated.")
            print("  Next step: paper trade for 3-6 months before using real capital.")
        elif beats_spy and not decent_risk:
            print("  ACTION: Alpha exists but Sharpe is low.")
            print("  Try: increase N_POSITIONS, reduce COST_BPS, or improve features.")
        elif not beats_spy:
            print("  ACTION: No alpha detected yet.")
            print("  Try: extend training data, add fundamental features,")
            print("       or change HOLDING_PERIOD to 10 or 20 days.")
    else:
        print("  SPY benchmark not available — check SPY_CSV path.")
        print(f"  Sharpe (net): {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd:.2%}")

    print("=" * 65)


if __name__ == "__main__":
    main()
