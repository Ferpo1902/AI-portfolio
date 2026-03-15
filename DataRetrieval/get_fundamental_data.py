'''
Script for setting up fundamental data from the earnings database
'''

import pandas as pd
from sqlalchemy import create_engine
from functools import reduce

# 1. SETUP SQLALCHEMY ENGINE (Fixes the Pandas Warning)
# Note: Ensure you have sqlalchemy installed (`pip install sqlalchemy`)
engine = create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/earnings")
print("Connected to database via SQLAlchemy.")

# ==========================================
# 2. PULL QUARTERLY SEC FINANCIALS
# ==========================================
print("\n--- Pulling Financial Statements ---")
financial_tables =[
    "income_statement",
    "balance_sheet_assets",
    "balance_sheet_liabilities",
    "balance_sheet_equity",
    "cash_flow_statement"
]

dataframes =[]
for table in financial_tables:
    print(f"  -> Pulling {table}...")
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    
    # Ensure date is a proper datetime object for safe merging
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    dataframes.append(df)

# ==========================================
# 3. DYNAMICALLY MERGE TABLES
# ==========================================
print("\nMerging financial statements...")

def merge_dfs(left, right):
    # Dynamically find overlapping keys so we NEVER trigger a KeyError
    # It will use['act_symbol', 'date', 'period'] if they exist in both
    potential_keys =['act_symbol', 'date', 'period']
    merge_keys =[col for col in potential_keys if col in left.columns and col in right.columns]
    
    # Merge and flag duplicate overlapping columns
    merged = pd.merge(left, right, on=merge_keys, how='outer', suffixes=('', '_dup'))
    
    # Drop the redundant duplicate columns cleanly
    dup_cols =[col for col in merged.columns if col.endswith('_dup')]
    merged.drop(columns=dup_cols, inplace=True)
    
    return merged

all_fundamentals = reduce(merge_dfs, dataframes)

# Sort safely by symbol and the Quarter-End Date
all_fundamentals.sort_values(['act_symbol', 'date'], inplace=True)
all_fundamentals.reset_index(drop=True, inplace=True)

# ==========================================
# 4. REMOVE ETFS
# ==========================================
print("\nRemoving ETFs...")
try:
    etf_data = pd.read_feather("../Data/ETFs.feather")
    ETFs = etf_data['act_symbol'].unique()
    all_fundamentals = all_fundamentals[~all_fundamentals['act_symbol'].isin(ETFs)]
    print("ETFs successfully removed.")
except FileNotFoundError:
    print("Warning: ETFs.feather not found. Skipping ETF filtering.")

# Save the dataset
all_fundamentals.to_feather("../Data/all_fundamentals.feather")
print(f"Saved Quarterly Fundamentals: {all_fundamentals.shape}")

# ==========================================
# 5. PULL THE EARNINGS CALENDAR (CRITICAL)
# ==========================================
# You MUST have this table to prevent look-ahead bias in your Feature Engine
print("\n--- Pulling Earnings Calendar ---")
print("  -> Pulling earnings_calendar...")

calendar_df = pd.read_sql("SELECT * FROM earnings_calendar", engine)

if 'date' in calendar_df.columns:
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
calendar_df.to_feather("../Data/earnings_calendar.feather")
print(f"Saved Earnings Calendar: {calendar_df.shape}")

# Dispose the engine connection safely
engine.dispose()
print("\nData extraction complete!")