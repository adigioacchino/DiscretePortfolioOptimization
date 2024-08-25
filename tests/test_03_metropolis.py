from discrete_portfolio_optimization.metropolis import PortfolioOptimizer
from discrete_portfolio_optimization.portfolio import Portfolio

from pytest import mark
from pytest import approx
import pytest
import pandas as pd
import numpy as np


skip_long_tests = False # set to True to skip long tests for faster (but much
                          # less accurate) testing 

dep_tests = [
    'tests/test_01_yf_wrapper.py::test_get_close_price_df',
    'tests/test_02_portfolio.py::test_init_portfolio',
    'tests/test_02_portfolio.py::test_random_move'
]

@pytest.fixture
def pft_closedf():
    close_df = pd.read_csv('tests/data/20240825_close.csv', index_col=0)
    pft = Portfolio(close_df.iloc[-1].to_list(), tot_value=10_000,
                    seed=42)
    return pft, close_df


@mark.dependency(depends=dep_tests, scope='session')
def test_init_portfolio_optimizer(pft_closedf):
    pft, close_df = pft_closedf
    t_n_alphas = 10
    t_n_steps = 5_000
    t_n_steps_per_beta = 2
    t_n_therm_steps = 1_000
    t_beta0 = 1e-3
    t_beta1 = 1
    po = PortfolioOptimizer(
        pft,
        close_df,
        n_alphas = t_n_alphas,
        n_therm_steps = t_n_therm_steps,
        beta0 = t_beta0,
        beta1 = t_beta1,
        n_steps = t_n_steps,
        n_steps_per_beta = t_n_steps_per_beta
    )

    # test alpha and beta schedules
    assert len(po.alpha_schedule) == t_n_alphas
    assert len(po.beta_schedule) == t_n_therm_steps + t_n_steps * t_n_steps_per_beta
    assert po.beta_schedule[0] == approx(t_beta0)
    assert po.beta_schedule[t_n_therm_steps] == approx(t_beta0)
    assert po.beta_schedule[-1] == approx(t_beta1)
    assert len(np.unique(po.beta_schedule)) == t_n_steps + 1


@mark.dependency(depends=dep_tests, scope='session')
def test_portfolio_minus_energy(pft_closedf):
    pft, close_df = pft_closedf
    alpha = 0.1
    gamma = 0.1
    energy = PortfolioOptimizer._portfolio_minus_energy(alpha, gamma, pft, close_df)
    assert isinstance(energy, float)

@mark.dependency(depends=dep_tests, scope='session')
@pytest.mark.skipif(skip_long_tests, reason="skipping long tests")
def test_runs(pft_closedf):
    pft, close_df = pft_closedf
    t_n_alphas = 3
    t_n_steps = 1_000
    t_n_steps_per_beta = 2
    t_n_therm_steps = 500
    po = PortfolioOptimizer(
        pft,
        close_df,
        n_alphas = t_n_alphas,
        n_therm_steps = t_n_therm_steps,
        n_steps = t_n_steps,
        n_steps_per_beta = t_n_steps_per_beta
    )

    po.full_run()

    assert len(po.best_portfolios) == t_n_alphas
    assert all([isinstance(p, Portfolio) for p in po.best_portfolios])
    assert all([p.tot_value == approx(10_000) for p in po.best_portfolios])
    assert all([p.num_assets == 5 for p in po.best_portfolios])
    assert all([p.asset_value != 0 for p in po.best_portfolios])
    assert all([p.cash_value != 0 for p in po.best_portfolios])
    assert all([p.asset_value + p.cash_value == approx(10_000) for p in po.best_portfolios])
    assert all([p.weights.sum() == approx(1) for p in po.best_portfolios])
    assert all([all(p.weights >= 0) for p in po.best_portfolios])
    assert all([sum(p.allocations) > 0 for p in po.best_portfolios])
    assert all([all(p.allocations >= 0) for p in po.best_portfolios])


@mark.dependency(depends=dep_tests, scope='session')
@pytest.mark.skipif(skip_long_tests, reason="skipping long tests")
def test_runs_fixed_alpha(pft_closedf):
    pft, close_df = pft_closedf
    po = PortfolioOptimizer(
        pft,
        close_df,
        n_therm_steps = 1_000,
        n_steps = 2_500,
        n_steps_per_beta = 2
    )

    # first optimal portfolio has low alpha, so should have high return and low volatility
    alpha0 = 1e-1
    alpha1 = 10

    po.run_fixed_alpha(alpha0)
    pft1 = po.best_portfolios[0]
    results_df = close_df.pct_change().dropna()
    metrics0 = pft.portfolio_metrics(results_df)
    metrics1 = pft1.portfolio_metrics(results_df)
    po.run_fixed_alpha(alpha1)
    pft2 = po.best_portfolios[1]
    metrics2 = pft2.portfolio_metrics(results_df)
    assert metrics1['Return'] > metrics0['Return']
    assert metrics1['Return'] > metrics2['Return']
    assert metrics2['Volatility'] < metrics0['Volatility']
    assert metrics2['Volatility'] < metrics1['Volatility']

    # now case with large and small gamma
    po = PortfolioOptimizer(
        pft,
        close_df,
        n_therm_steps = 1_000,
        n_steps = 2_500,
        n_steps_per_beta = 2,
        gamma = 100.
    )
    
    po.run_fixed_alpha(1.)
    pft3 = po.best_portfolios[0]
    assert np.linalg.norm(pft3.weights) < np.linalg.norm(pft.weights)
    assert np.linalg.norm(pft3.weights) < np.linalg.norm(pft1.weights)
    assert np.linalg.norm(pft3.weights) < np.linalg.norm(pft2.weights)


@mark.dependency(depends=dep_tests, scope='session')
@pytest.mark.skipif(skip_long_tests, reason="skipping long tests")
def test_runs_fixed_alpha_2(pft_closedf):
    pft, close_df = pft_closedf
    po = PortfolioOptimizer(
        pft,
        close_df,
        n_therm_steps = 1_000,
        n_steps = 2_500,
        n_steps_per_beta = 2,
        gamma = 0.
    )

    # alpha = 0., gamma = 0., so should be composed of the asset with highest return almost exclusively
    po.run_fixed_alpha(0.)
    pft1 = po.best_portfolios[0]
    results_df = close_df.pct_change().dropna()
    returns = results_df.mean()
    assert np.argmax(returns) == np.argmax(pft1.weights)
    assert np.max(pft1.weights) > 0.95 # will not be exactly 1 because of discrete nature of the assets

    # alpha = 100., gamma = 0., so should be composed of the asset with lowest volatility almost exclusively
    po.run_fixed_alpha(10.)
    pft2 = po.best_portfolios[1]
    cov_matrix = results_df.cov()
    volatilities = np.sqrt(np.diag(cov_matrix))
    assert np.argmin(volatilities) == np.argmax(pft2.weights)
    assert np.max(pft2.weights) > 0.95 # will not be exactly 1 because of discrete nature of the assets