from typing import List, Tuple, Any

import yfinance as yf  # type: ignore
import pandas as pd
from tqdm import tqdm
import marimo as mo  # to have marimo-compatible progress bars


def download_close_price(ticker: str) -> pd.Series:
    """
    Download the close price of a stock from Yahoo Finance.

    Args:
        ticker: str, the ticker (name) of the stock to download

    Returns:
        pd.Series, the close price of the stock
    """
    # cast ticker to upper case
    ticker = ticker.upper()
    data = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    return data["Close"][ticker]


def get_close_price_df(
    tickers: str, drop_missing_dates: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Download the close price of a list of stocks from Yahoo Finance.

    Args:
        tickers: str, a string of tickers separated by a comma
        drop_missing_dates: bool, if True, drop dates for which at least one ticker is missing

    Returns:
        pd.DataFrame, the close price of the stocks
        list, the tickers that were successfully downloaded
        list, the tickers that were not found
    """
    all_data = dict()
    symbols = [x.strip() for x in tickers.split(",")]

    # define progress bar
    # Use Any for progress bar to accommodate both tqdm and marimo progress bar
    symbols_iter: Any
    if mo.running_in_notebook():
        symbols_iter = mo.status.progress_bar(
            symbols, title="Downloading Yahoo finance data"
        )
    else:
        symbols_iter = tqdm(symbols, desc="Downloading Yahoo finance data")

    ticker_hits: List[str] = []
    ticker_misses: List[str] = []
    for ticker in symbols_iter:
        try:
            t_data = download_close_price(ticker)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            t_data = pd.Series()
        if len(t_data) > 0:
            all_data[ticker] = t_data
            ticker_hits.append(ticker)
        else:
            ticker_misses.append(ticker)
            print(f"Ticker {ticker} not found, skipping")

    if drop_missing_dates:
        # drop missing dates
        res_df = pd.DataFrame(all_data).dropna()
    else:
        # fill missing dates with NaN
        res_df = pd.DataFrame(all_data)
    return res_df, ticker_hits, ticker_misses
