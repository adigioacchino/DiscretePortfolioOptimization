import yfinance as yf
import pandas as pd

from tqdm import tqdm
import marimo as mo # to have marimo-compatible progress bars

def download_close_price(ticker: str):
    """
    Download the close price of a stock from Yahoo Finance.
    :param ticker: str, the ticker of the stock
    :return: pd.Series, the close price of the stock
    """
    data = yf.download(ticker, period="max", progress=False)
    return data['Close']

def get_close_price_df(tickers: str) -> pd.DataFrame:
    """
    Download the close price of a list of stocks from Yahoo Finance.
    :param tickers: str, the tickers of the stocks separated by comma
    :return: pd.DataFrame, the close price of the stocks
    """
    all_data = dict()
    symbols = [x.strip() for x in tickers.split(",")]

    # define progress bar
    if mo.running_in_notebook():
        symbols_iter = mo.status.progress_bar(symbols, title="Downloading Yahoo finance data")
    else:
        symbols_iter = tqdm(symbols, desc="Downloading Yahoo finance data")

    for ticker in symbols_iter:
        t_data = download_close_price(ticker)
        if len(t_data) > 0:
            all_data[ticker] = download_close_price(ticker)
        else:
            print(f"Ticker {ticker} not found, skipping")
    return pd.DataFrame(all_data).dropna()