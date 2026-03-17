import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
from pdb import set_trace as st
import numpy as np

def get_sector(ticker: str, user_agent: str):
    """
    Fetches the SIC description for a given ticker from the SEC EDGAR API.
    By using the browse-edgar endpoint with ATOM output, this dynamically 
    resolves both active and delisted tickers.
    
    :param ticker: The stock ticker symbol (e.g., 'AAPL' or 'TWTR')
    :param user_agent: Required by SEC. Format: 'Name/Company ContactEmail'
    :return: String with SIC description or "Unknown" if not found/error
    """
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate"
    }
    
    # The CGI endpoint accepts tickers directly (including delisted) and returns an ATOM XML feed
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&action=getcompany&output=atom"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # If a ticker is completely invalid, the SEC returns a standard HTML search page rather than XML.
        # We can catch this by checking the Content-Type header.
        if "xml" not in response.headers.get("Content-Type", ""):
            return "Unknown"
            
        # Parse the XML feed
        root = ET.fromstring(response.content)
        
        # Search the XML tree for the assigned-sic-desc tag
        # We use .endswith() to safely bypass hardcoding the exact XML namespace SEC uses
        for elem in root.iter():
            if elem.tag.endswith('assigned-sic-desc'):
                return elem.text.strip() if elem.text else "Unknown"
                
        return "Unknown"
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch metadata for {ticker}: {e}")
        return "Unknown"
    except ET.ParseError as e:
        print(f"Failed to parse XML for {ticker}: {e}")
        return "Unknown"


MY_USER_AGENT = "Pat LaChapelle patlachapelle3@gmail.com" 
df = pd.read_feather("../Data/all_ohlcv_no_ETFs.feather")
sectors = []
tickers = []
for c, ticker in enumerate(df.act_symbol.unique()):
    sector = get_sector(ticker, MY_USER_AGENT)
    tickers.append(ticker)
    sectors.append(sector)
    print(f"{np.round(100*c/12770, 2)}% complete", ticker, sector)

sector_df = pd.DataFrame({"act_symbol": tickers, "sector": sectors})
st()