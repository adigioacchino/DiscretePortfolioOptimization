from discrete_portfolio_optimization.yfinance_download import download_close_price, get_close_price_df

import pandas as pd
from pytest import mark, approx

@mark.dependency()
def test_download_close_price():
    # Test the function download_close_price
    # The function should return a pd.Series
    ticker = "AAPL"
    close_price = download_close_price(ticker)
    assert isinstance(close_price, pd.Series)
    assert close_price.index.name == "Date"
    assert close_price.name == ticker

    ticker = "aapl"
    close_price2 = download_close_price(ticker)
    # take first 100 elements of the series to compare
    close_price = close_price.head(100)
    close_price2 = close_price2.head(100)
    assert all(close_price.index == close_price2.index)
    assert close_price.values == approx(close_price2.values)

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
    tickers = "AAPL,MSFT,ABCDEFG"
    close_price_df = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]