from DiscretePortfolioOptimization.yfinance_download import (
    download_close_price,
    get_close_price_df,
)

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
    assert close_price.values == approx(close_price2.values, abs=1e-5)


@mark.dependency()
def test_get_close_price_df():
    # Test the function get_close_price_df
    # The function should return a pd.DataFrame
    tickers = "AAPL , MSFT "
    close_price_df, tickers_hits, tickers_miss = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]
    assert tickers_hits == ["AAPL", "MSFT"]
    assert tickers_miss == []

    # again without spaces
    tickers = "AAPL,MSFT"
    close_price_df, tickers_hits, tickers_miss = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]
    assert tickers_hits == ["AAPL", "MSFT"]
    assert tickers_miss == []

    # now with a non-existing ticker
    tickers = "AAPL,MSFT,ABCDEFG"
    close_price_df, tickers_hits, tickers_miss = get_close_price_df(tickers)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.columns.tolist() == ["AAPL", "MSFT"]
    assert tickers_hits == ["AAPL", "MSFT"]
    assert tickers_miss == ["ABCDEFG"]

    # test drop_missing_dates
    tickers = "AAPL, ABNB"
    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=True)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=False)
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2000-01-01")

    tickers = "AAPL, ABNB"
    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=True, target_currency='EUR')
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=False, target_currency='EUR')
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2012-01-01")

    tickers = "BPSO.MI, ENI.MI, ABNB"
    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=True, target_currency='USD')
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=False, target_currency='USD')
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2012-01-01")

    tickers = "BPSO.MI, ENI.MI, ABNB"
    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=True, target_currency='EUR')
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(tickers, drop_missing_dates=False, target_currency='EUR')
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2010-01-01")
