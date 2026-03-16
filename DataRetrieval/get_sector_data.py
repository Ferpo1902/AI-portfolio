'''
Pulls sector data from Yahoo finance for each act symbol in the full ETF-free dataset
'''
import pandas as pd
import yfinance as yf
from pdb import set_trace as st

def fetch_sector_data(unique_tickers):
    """
    Fetches sector data for a list of tickers and returns a DataFrame.
    """
    print(f"Fetching sectors for {len(unique_tickers)} tickers...")
    sector_map =[]
    
    for ticker in unique_tickers:
        try:
            # yfinance returns a dictionary of info, we just want 'sector'
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            sector_map.append({'act_symbol': ticker, 'sector': sector})
        except Exception as e:
            sector_map.append({'act_symbol': ticker, 'sector': 'Unknown'})
            
    return pd.DataFrame(sector_map)

df = pd.read_feather("../Data/all_ohlcv.feather")
tickers = df.act_symbol.unique()
sectors = fetch_sector_data(tickers)
st()