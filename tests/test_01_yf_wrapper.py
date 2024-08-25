from discrete_portfolio_optimization.yfinance_download import download_close_price, get_close_price_df

import pandas as pd
from pytest import mark

@mark.dependency()
def test_download_close_price():
    # Test the function download_close_price
    # The function should return a pd.Series
    ticker = "AAPL"
    close_price = download_close_price(ticker)
    assert isinstance(close_price, pd.Series)
    assert close_price.index.name == "Date"
    assert close_price.name == "Close"

@mark.dependency()
def test_get_close_price_df():
    # Test the function get_close_price_df
    # The function should return a pd.DataFrame
    tickers = "AAPL , MSFT "
    close_price_df = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]

    # again without spaces
    tickers = "AAPL,MSFT"
    close_price_df = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]

    # now with a non-existing ticker
    tickers = "AAPL,MSFT,XYZ"
    close_price_df = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]