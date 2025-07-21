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
    tickers: str, drop_missing_dates: bool = True, target_currency: str = "USD"
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Download the close price of a list of stocks from Yahoo Finance.

    Args:
        tickers: str, a string of tickers separated by a comma
        drop_missing_dates: bool, if True, drop dates for which at least one ticker is missing
        target_currency: str, the currency to convert the stock prices to (default is 'USD')

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
        # convert data to target currency
        res_df = currency_conversion(res_df, target_currency=target_currency).dropna()
    else:
        # fill missing dates with NaN
        res_df = pd.DataFrame(all_data)
        # convert data to target currency
        res_df = currency_conversion(res_df, target_currency=target_currency)

    return res_df, ticker_hits, ticker_misses


def currency_conversion(data: pd.DataFrame, target_currency: str) -> pd.DataFrame:
    """
    Convert the currency of the stock prices in the DataFrame.

    Args:
        data: pd.DataFrame, the stock prices
        currency: str, the currency to convert to (e.g., 'USD', 'EUR')

    Returns:
        pd.DataFrame, the stock prices converted to the specified currency
    """
    tickers = data.columns.tolist()
    currency_map = _get_ticker_currencies(tickers)

    all_currency = list(set(list(currency_map.values())))
    all_currency_not_target = [c for c in all_currency if c != target_currency]

    currency_conversion = {}
    forex_list = []
    for cur in all_currency_not_target:
        if target_currency == "USD":
            fx_ticker = f"{cur}=X"
        else:
            fx_ticker = f"{target_currency}{cur}=X"
        forex_list.append(fx_ticker)
        currency_conversion[cur] = fx_ticker

    if len(forex_list) == 0:
        return data
    else:
        fx_history = yf.Tickers(forex_list).history(
            period="max", auto_adjust=True, progress=False
        )["Close"]

        # Align prices and forex, using forward fill for missing values
        data_al, fx_al = data.align(fx_history, join="left", axis=0)
        fx_al = fx_al.ffill()

        for col in data_al.columns:
            reference_cur = currency_map[col]
            if reference_cur != target_currency:
                fx_col = currency_conversion[reference_cur]
                data_al.loc[:, col] /= fx_al.loc[:, fx_col]

        return data_al


def _get_ticker_currencies(ticker_list: List[str]) -> dict[str, str]:
    """
    Get the currency of each ticker in the list using Yahoo Finance.
    Args:
        ticker_list: List[str], a list of tickers
    Returns:
        dict[str, str], a dictionary mapping tickers to their currencies. In case of failure, it defaults to 'USD'.
    """
    tickers_data = yf.Tickers(ticker_list)
    currency_map = {}

    for ticker in ticker_list:
        try:
            info = tickers_data.tickers[ticker].info
            currency = info.get("currency", "USD")  # Default fallback
            currency_map[ticker] = currency.upper()
        except Exception as e:
            print(f"⚠️ Could not get currency for {ticker}: {e}")
            currency_map[ticker] = "USD"  # Fallback to USD

    return currency_map
