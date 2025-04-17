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
    data = yf.download(ticker, period="max", progress=False)
    return data["Close"][ticker]


def get_close_price_df(tickers: str) -> pd.DataFrame:
    """
    Download the close price of a list of stocks from Yahoo Finance.

    Args:
        tickers: str, a string of tickers separated by a comma

    Returns:
        pd.DataFrame, the close price of the stocks
    """
    all_data = dict()
    symbols = [x.strip() for x in tickers.split(",")]

    # define progress bar
    if mo.running_in_notebook():
        symbols_iter = mo.status.progress_bar(
            symbols, title="Downloading Yahoo finance data"
        )
    else:
        symbols_iter = tqdm(symbols, desc="Downloading Yahoo finance data")

    for ticker in symbols_iter:
        t_data = download_close_price(ticker)
        if len(t_data) > 0:
            all_data[ticker] = download_close_price(ticker)
        else:
            print(f"Ticker {ticker} not found, skipping")
    return pd.DataFrame(all_data).dropna()
