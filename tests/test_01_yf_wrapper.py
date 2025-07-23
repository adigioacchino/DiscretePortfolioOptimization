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


@mark.dependency()
def test_change_currency():
    # Consistency check when target currency is defined to USD or not (should be the same)
    tickers = "AAPL, ABNB"
    close_price_df_1, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="USD"
    )
    close_price_df_2, _, _ = get_close_price_df(tickers, drop_missing_dates=True)
    assert all(close_price_df_1.iloc[-1] == close_price_df_2.iloc[-1]), (
        "The last prices in USD and without target currency should be the same"
    )

    # Checking that the price in EUR was lower the in USD during the first 6 months of 2025
    # case when the tickers are in USD
    tickers = "AAPL, ABNB"
    close_price_df_eur, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="EUR"
    )
    close_price_df_usd, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="USD"
    )
    # align the indices (remove the dates that are not in both dataframes)
    common_dates = close_price_df_eur.index.intersection(close_price_df_usd.index)
    close_price_df_eur = close_price_df_eur.loc[common_dates]
    close_price_df_usd = close_price_df_usd.loc[common_dates]
    # check that the price in EUR is lower than in USD for the first 6 months of 2025
    assert all(
        close_price_df_eur.loc["2025-01-01":"2025-06-30"]
        < close_price_df_usd.loc["2025-01-01":"2025-06-30"]
    ), "The price in EUR should be lower than in USD for the first 6 months of 2025"
    # case when the tickers are in EUR
    tickers = "BPSO.MI, ENI.MI"
    close_price_df_eur, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="EUR"
    )
    close_price_df_usd, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="USD"
    )
    # align the indices (remove the dates that are not in both dataframes)
    common_dates = close_price_df_eur.index.intersection(close_price_df_usd.index)
    close_price_df_eur = close_price_df_eur.loc[common_dates]
    close_price_df_usd = close_price_df_usd.loc[common_dates]
    # check that the price in EUR is lower than in USD for the first 6 months of 2025
    assert all(
        close_price_df_eur.loc["2025-01-01":"2025-06-30"]
        < close_price_df_usd.loc["2025-01-01":"2025-06-30"]
    ), (
        "The price in EUR should be lower than in USD for the first 6 months of 2025 (BPSO.MI, ENI.MI tickers)"
    )

    # Asserting data coverage when currency is changed
    tickers = "AAPL, ABNB"
    close_price_df, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="EUR"
    )
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(
        tickers, drop_missing_dates=False, target_currency="EUR"
    )
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2012-01-01")

    tickers = "BPSO.MI, ENI.MI, ABNB"
    close_price_df, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="USD"
    )
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(
        tickers, drop_missing_dates=False, target_currency="USD"
    )
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2012-01-01")

    tickers = "BPSO.MI, ENI.MI, ABNB"
    close_price_df, _, _ = get_close_price_df(
        tickers, drop_missing_dates=True, target_currency="EUR"
    )
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() > pd.Timestamp("2020-01-01")

    close_price_df, _, _ = get_close_price_df(
        tickers, drop_missing_dates=False, target_currency="EUR"
    )
    assert isinstance(close_price_df, pd.DataFrame)
    assert close_price_df.index.name == "Date"
    assert close_price_df.index.min() < pd.Timestamp("2010-01-01")
