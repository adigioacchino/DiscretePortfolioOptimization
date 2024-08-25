from discrete_portfolio_optimization.portfolio import Portfolio

from pytest import mark
from pytest import approx
import pytest
import pandas as pd


@pytest.fixture
def close_df():
    return pd.read_csv('tests/data/20240825_close.csv', index_col=0)

@mark.dependency(
    depends=['tests/test_01_yf_wrapper.py::test_get_close_price_df'],
    scope='session'
)
def test_init_portfolio(close_df):
    # init with tot value
    pft = Portfolio(close_df.iloc[-1].to_list(), tot_value=10_000,
                    seed=42)

    assert pft.tot_value == 10_000
    assert pft.num_assets == 5
    assert pft.asset_value != 0
    assert pft.cash_value != 0
    assert pft.asset_value + pft.cash_value == approx(10_000)
    assert pft.weights.sum() == approx(1)
    assert all(pft.weights >= 0)
    assert all(pft.allocations >= 0)
    assert sum(pft.allocations) > 0

    # init with allocations
    pft = Portfolio(close_df.iloc[-1].to_list(),
                    allocations=[1, 2, 3, 4, 5])
    assert pft.tot_value == (close_df.iloc[-1] @ [1, 2, 3, 4, 5])

@mark.dependency(
    depends=['tests/test_01_yf_wrapper.py::test_get_close_price_df'],
    scope='session'
)
def test_random_move(close_df):
    for seed in range(10):
        pft = Portfolio(close_df.iloc[-1].to_list(), 
                        tot_value=10_000, seed=seed)

        old_allocations = pft.allocations.copy()
        old_asset_value = pft.asset_value
        old_cash_value = pft.cash_value
        old_weights = pft.weights.copy()
        pft.random_move()

        # assert that new portfolio is okay
        assert pft.tot_value == 10_000
        assert pft.num_assets == 5
        assert pft.asset_value != 0
        assert pft.cash_value != 0
        assert pft.asset_value + pft.cash_value == approx(10_000)
        assert pft.weights.sum() == approx(1)
        assert all(pft.weights >= 0)
        assert sum(pft.weights) == approx(1)
        assert all(pft.allocations >= 0)
        assert sum(pft.allocations) > 0

        # assert that things have changed
        assert any(old_allocations != pft.allocations)
        assert old_asset_value != pft.asset_value
        assert old_cash_value != pft.cash_value
        assert any(old_weights != pft.weights)


@mark.dependency(
    depends=['tests/test_01_yf_wrapper.py::test_get_close_price_df'],
    scope='session'
)
def test_computations(close_df):
    pft = Portfolio(close_df.iloc[-1].to_list(),
                    tot_value=10_000, seed=42)
    
    returns = close_df.pct_change().dropna()
    metrics = pft.portfolio_metrics(returns)

    assert 'Return' in metrics
    assert 'Volatility' in metrics
    assert 'Sharpe Ratio' in metrics

    assert isinstance(metrics['Return'], float)
    assert isinstance(metrics['Volatility'], float)
    assert isinstance(metrics['Sharpe Ratio'], float)


@mark.dependency(
    depends=['tests/test_01_yf_wrapper.py::test_get_close_price_df'],
    scope='session'
)
def test_copy(close_df):
    pft = Portfolio(close_df.iloc[-1].to_list(),
                    tot_value=10_000, seed=42)
    
    pft_copy = pft.copy()

    assert pft.tot_value == pft_copy.tot_value
    assert pft.num_assets == pft_copy.num_assets
    assert pft.asset_value == pft_copy.asset_value
    assert pft.cash_value == pft_copy.cash_value
    assert all(pft.allocations == pft_copy.allocations)
    assert all(pft.weights == pft_copy.weights)
    assert pft.allocations is not pft_copy.allocations
    assert pft.weights is not pft_copy.weights
    assert pft.current_prices is not pft_copy.current_prices
