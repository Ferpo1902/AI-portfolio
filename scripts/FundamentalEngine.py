import pandas as pd
import numpy as np

'''
Everything is written to work on quarterly data
'''

#0. Functions
def compute_eps_growth_qoq(df: pd.DataFrame):
    """
    EPS growth quarter-over-quarter.
    Computes the shifted series once and reuses it to avoid
    double-groupby misalignment.
    """
    shifted = df.groupby('act_symbol')['eps'].shift(1)
    return (df['eps'] - shifted) / shifted.abs().replace(0, np.nan)

def compute_eps_growth_yoy(df: pd.DataFrame):
    """
    EPS growth year-over-year (4 quarters back).
    Assumes quarterly data — raises a warning if annual rows are detected.
    """
    shifted = df.groupby('act_symbol')['eps'].shift(4)
    return (df['eps'] - shifted) / shifted.abs().replace(0, np.nan)

def compute_revenue_growth_yoy(df: pd.DataFrame):
    """
    Revenue growth year-over-year (4 quarters back).
    Assumes quarterly data — raises a warning if annual rows are detected.
    """
    shifted = df.groupby('act_symbol')['revenue'].shift(4)
    return (df['revenue'] - shifted) / shifted.abs().replace(0, np.nan)


def compute_asset_growth_yoy(df: pd.DataFrame):
    """
    Total asset growth year-over-year (4 quarters back).
    Assumes quarterly data — raises a warning if annual rows are detected.
    """
    shifted = df.groupby('act_symbol')['total_assets'].shift(4)
    return (df['total_assets'] - shifted) / shifted.abs().replace(0, np.nan)

def compute_quick_ratio(df: pd.DataFrame):
    """
    Quick ratio using the additive method:
        (cash + short_term_investments + net_receivables) / current_liabilities

    This is more conservative than subtracting inventory from current assets,
    as it excludes illiquid current assets like prepaid expenses, deferred costs,
    and other non-cash items that the inventory-subtraction method implicitly includes.

    Required columns:
        - cash_and_equivalents
        - short_term_investments
        - net_receivables
        - current_liabilities
    """
    liquid_assets = (
        df['cash_and_equivalents']
        + df['short_term_investments'].fillna(0)
        + df['net_receivables'].fillna(0)
    )
    return liquid_assets / df['current_liabilities'].replace(0, np.nan)

# ==========================================
# 1. THE REGISTRY
# ==========================================
FUNDAMENTAL_REGISTRY = {
    
    # --- PROBABILITY & EFFICIENCY ---
    'ROE': {
        'fn': lambda df: df['net_income'] / df['total_equity'].replace(0, np.nan),
        'inputs': ['net_income', 'total_equity']
    },
    'ROA': {
        'fn': lambda df: df['net_income'] / df['total_assets'].replace(0, np.nan),
        'inputs':['net_income', 'total_assets']
    },
    'GROSS_MARGIN': {
        'fn': lambda df: df['gross_profit'] / df['revenue'].replace(0, np.nan),
        'inputs': ['gross_profit', 'revenue']
    },
    'OPERATING_MARGIN': {
        'fn': lambda df: df['operating_income'] / df['revenue'].replace(0, np.nan),
        'inputs':['operating_income', 'revenue']
    },
    'ASSET_TURNOVER': {
        'fn': lambda df: df['revenue'] / df['total_assets'].replace(0, np.nan),
        'inputs': ['revenue', 'total_assets']
    },
    'CASH_FLOW_ROA': {
        'fn': lambda df: df['operating_cash_flow'] / df['total_assets'].replace(0, np.nan),
        'inputs':['operating_cash_flow', 'total_assets']
    },

    # --- LIQUIDITY / SOLVENCY (Balance Sheet) ---
    'CURRENT_RATIO': {
        'fn': lambda df: df['current_assets'] / df['current_liabilities'].replace(0, np.nan),
        'inputs':['current_assets', 'current_liabilities']
    },
    'QUICK_RATIO': {
        'fn': lambda df: (df['current_assets'] - df['inventory']) / df['current_liabilities'].replace(0, np.nan),
        'inputs':['current_assets', 'inventory', 'current_liabilities']
    },
    'DEBT_TO_EQUITY': {
        'fn': lambda df: df['total_debt'] / df['total_equity'].replace(0, np.nan),
        'inputs': ['total_debt', 'total_equity']
    },
    'ASSET_LEVERAGE': {
        'fn': lambda df: df['total_assets'] / df['total_equity'].replace(0, np.nan),
        'inputs':['total_assets', 'total_equity']
    },
    'CASH_RATIO': {
        'fn': lambda df: df['cash_and_equivalents'] / df['current_liabilities'].replace(0, np.nan),
        'inputs':['cash_and_equivalents', 'current_liabilities']
    },

    # --- ADVANCED QUANT FACTORS ---
    'SLOANS_ACCRUALS': {
        'fn': lambda df: (df['net_income'] - df['operating_cash_flow']) / df['total_assets'].replace(0, np.nan),
        'inputs':['net_income', 'operating_cash_flow', 'total_assets']
    },
    'CF_TO_NET_INCOME': {
        'fn': lambda df: df['operating_cash_flow'] / df['net_income'].replace(0, np.nan),
        'inputs':['operating_cash_flow', 'net_income']
    },

    # --- MOMENTUM (Assumes Sparse Quarterly Rows) ---
    'EPS_GROWTH_QOQ': {
        # Shift 1 = 1 Quarter back
        'fn': compute_eps_growth_qoq,
        'inputs': ['eps']
    },
    'EPS_GROWTH_YOY': {
        # Shift 4 = 4 Quarters back (1 Year)
        'fn': compute_eps_growth_yoy,
        'inputs': ['eps']
    },
    'REVENUE_GROWTH_YOY': {
        'fn': lambda df: (df['revenue'] - df.groupby('act_symbol')['revenue'].shift(4)) / df.groupby('act_symbol')['revenue'].shift(4).abs().replace(0, np.nan),
        'inputs': ['revenue']
    },
    'ASSET_GROWTH_YOY': {
        'fn': lambda df: (df['total_assets'] - df.groupby('act_symbol')['total_assets'].shift(4)) / df.groupby('act_symbol')['total_assets'].shift(4).abs().replace(0, np.nan),
        'inputs':['total_assets']
    },

    # --- EVENT-DRIVEN ---
    'SUE': {
        # Standardized Unexpected Earnings: (Actual - Estimate) / Standard Deviation of Estimates
        'fn': lambda df: (df['eps'] - df['eps_est']) / df['eps_est_std'].replace(0, np.nan),
        'inputs': ['eps', 'eps_est', 'eps_est_std']
    },
    'PEAD_SURPRISE_PROXY': {
        # Actual drift requires price data. Here we calculate Earnings Surprise %, 
        # which is the fundamental catalyst/proxy for PEAD.
        'fn': lambda df: (df['eps'] - df['eps_est']) / df['eps_est'].abs().replace(0, np.nan),
        'inputs':['eps', 'eps_est']
    },
    'DAYS_TO_NEXT_EARNINGS': {
        'fn': lambda df: (pd.to_datetime(df['next_earnings_date']) - pd.to_datetime(df['date'])).dt.days,
        'inputs':['date', 'next_earnings_date']
    },
    'DAYS_SINCE_LAST_EARNINGS': {
        'fn': lambda df: (pd.to_datetime(df['date']) - pd.to_datetime(df['last_earnings_date'])).dt.days,
        'inputs':['date', 'last_earnings_date']
    }
}


# ==========================================
# 2. THE REQUEST OBJECT
# ==========================================
class FundamentalRequest:
    def __init__(self, name, alias=None):
        if name not in FUNDAMENTAL_REGISTRY:
            raise ValueError(f"Fundamental Feature '{name}' not found in Registry")
        self.name = name
        self.alias = alias if alias else f"F_{name}"

# ==========================================
# 3. THE ENGINE
# ==========================================
class FundamentalEngine:
    def __init__(self, requests):
        self.requests = requests

    def compute(self, df):
        print(f"Fundamental Engine: Computing {len(self.requests)} sparse features...")
        
        # 1. Safety Sort (Critical for Growth/Shift calculations grouping by symbol)
        df = df.sort_values(['act_symbol', 'date']).copy()
        
        # 2. Compute
        for req in self.requests:
            print(f"  -> {req.name}")
            config = FUNDAMENTAL_REGISTRY[req.name]
            
            # Check if required raw columns exist
            missing =[col for col in config['inputs'] if col not in df.columns]
            
            if missing:
                print(f"[Warning] Missing raw columns {missing} for {req.name}. Yielding NaNs.")
                df[req.alias] = np.nan
            else:
                try:
                    # Execute the registry function
                    df[req.alias] = config['fn'](df)
                    
                    # Clean up infs caused by extremely small denominator values
                    df[req.alias] = df[req.alias].replace([np.inf, -np.inf], np.nan)
                    
                except Exception as e:
                    print(f"[Error] Failed to compute {req.name}: {e}. Yielding NaNs.")
                    df[req.alias] = np.nan
                
        return df