'''
Script for cleaning up interest rate data so that it contains
all dates in the ohlcv data, filling missing values from the 
most recent previous date.
'''
import pandas as pd
import numpy as np

def clean_rates_data():
    # --- 1. Load Data ---
    ohlcv_df = pd.read_feather("../Data/all_ohlcv.feather")
    rates_df = pd.read_csv("../Data/treasury_rates.csv")

    # --- 2. The Core Logic ---

    # Step A: Ensure 'date' columns are proper datetime objects
    ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
    rates_df['date'] = pd.to_datetime(rates_df['date'])

    # Step B: Sort both by date (crucial for forward-filling time series)
    ohlcv_df = ohlcv_df.sort_values('date')
    rates_df = rates_df.sort_values('date')

    # Step C: Get UNIQUE dates from OHLCV data
    # (OHLCV datasets often have multiple tickers per day, we just need the unique days)
    target_dates = pd.Series(ohlcv_df['date'].unique()).sort_values()

    # Step D: Set the date column as the index for the rates dataframe
    rates_df.set_index('date', inplace=True)

    # Step E: Reindex using EXACTLY the unique OHLCV dates. 
    # method='ffill' pulls the value from the most recent prior date for all term columns.
    rates_df = rates_df.reindex(target_dates, method='ffill')

    # Optional Step: If the very first date in your OHLCV data is earlier than the 
    # first date in your interest rate data, ffill() won't catch it. bfill() catches leading NaNs.
    rates_df = rates_df.bfill()

    # Step F: Reset the index to turn 'date' back into a standard column
    rates_df.index.name = 'date'
    cleaned_rates = rates_df.reset_index()

    # --- 3. Save the Data ---
    cleaned_rates.to_csv("clean_rates.csv", index=False)
    
    return ohlcv_df, cleaned_rates

def test_dates_match(ohlcv_df, cleaned_rates):
    '''
    Tests that every single date present in the original OHLCV dataframe 
    is now successfully present in the cleaned rates dataframe.
    '''
    # Convert dates to sets for hyper-fast comparison
    ohlcv_dates = set(ohlcv_df['date'])
    rates_dates = set(cleaned_rates['date'])
    
    # Check if ohlcv_dates is a subset of rates_dates
    test_passed = ohlcv_dates.issubset(rates_dates)
    
    return test_passed

if __name__ == "__main__":
    print("Cleaning interest rate data...")
    ohlcv_data, cleaned_rates_data = clean_rates_data()
    cleaned_rates_data.to_csv("../Data/clean_interest_rates.csv")

    print("Running validation test...")
    is_valid = test_dates_match(ohlcv_data, cleaned_rates_data)
    
    if is_valid:
        print("✅ TEST PASSED: True. All OHLCV dates are present in the rate data.")
    else:
        print("❌ TEST FAILED: False. Some OHLCV dates are missing from the rate data.")
        # Optional: You can raise an error here to stop a larger pipeline from continuing
        # raise ValueError("Date mismatch between OHLCV and Interest Rate data.")
