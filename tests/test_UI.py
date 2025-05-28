import sys

sys.path.append("UI")  # add the UI directory (not part of the package) to the path

from DPO_UserInterface import (
    yf_download_input,
    yf_ui_fetch,
    init_po,
    run_portfolio_opt,
)

import DiscretePortfolioOptimization as dpo

import os
import pickle
import marimo as mo


#################################
# utils functions
def get_small_po():
    _, defs = init_po.run()
    po = defs["po"]
    assert isinstance(po, dpo.PortfolioOptimizer)
    # reduce the number of etas for the test
    po.eta_schedule = po.eta_schedule[:2]
    # reduce the number of thetas for the test
    po.theta_schedule = po.theta_schedule[:1000]
    return po


def get_mock_button(value: bool = True):
    # return a mock button with value "value"
    class MockButton:
        def __init__(self):
            self.value = value

    return MockButton()


#################################


def test_yf_ui_fetch():
    # get the default assets
    _, defs = yf_download_input.run()
    ticker_names = [x.strip() for x in defs["symbols_string"].value.split(",")]

    # run the YF fetch cell
    outs, defs = yf_ui_fetch.run()

    # check the output
    assert outs is None
    assert "ticker_prices" in defs.keys()
    assert "returns_df" in defs.keys()
    return_df = defs["returns_df"]
    assert set(ticker_names) == set(return_df.columns.tolist())
    assert len(return_df) > 100


def test_init_po():
    _, defs = init_po.run()
    assert "po" in defs.keys()
    po = defs["po"]
    assert isinstance(po, dpo.PortfolioOptimizer)
    assert len(po.best_portfolios) == 0


def test_run_portfolio_opt(tmp_path):
    # force_recompute = True
    po = get_small_po()
    force_recompute = get_mock_button(value=True)

    # run the portfolio optimization cell
    # re-define load_results_from_temp_file and save_results_to_temp_file
    # to fix the temp file path
    def load_results_from_temp_file():
        temp_file_path = os.path.join(tmp_path, "portfolio_optimization_results.pkl")
        if os.path.exists(temp_file_path):
            try:
                with open(temp_file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading saved results: {e}")
                return None
        return None

    def save_results_to_temp_file(portfolios):
        temp_file_path = os.path.join(tmp_path, "portfolio_optimization_results.pkl")
        with open(temp_file_path, "wb") as f:
            pickle.dump(portfolios, f)

    outs, defs = run_portfolio_opt.run(
        force_recompute=force_recompute,
        po=po,
        load_results_from_temp_file=load_results_from_temp_file,
        save_results_to_temp_file=save_results_to_temp_file,
    )

    assert isinstance(outs, mo.ui.plotly)
    assert "opt_port_plot" in defs.keys()
    assert isinstance(defs["opt_port_plot"], mo.ui.plotly)
    assert len(po.best_portfolios) == len(po.eta_schedule)

    # force_recompute = False
    force_recompute = get_mock_button(value=False)
    po = get_small_po()

    # run the portfolio optimization cell again
    outs, defs = run_portfolio_opt.run(
        force_recompute=force_recompute,
        po=po,
        load_results_from_temp_file=load_results_from_temp_file,
        save_results_to_temp_file=save_results_to_temp_file,
    )
    assert len(po.best_portfolios) == len(po.eta_schedule) * 2

    # now run with user portfolios set
    _, yf_defs = yf_ui_fetch.run()
    user_portfolios = [
        dpo.Portfolio(
            yf_defs["ticker_prices"],
            allocations=[1 for _ in range(len(yf_defs["ticker_prices"]))],
        ),
        dpo.Portfolio(
            yf_defs["ticker_prices"],
            allocations=[2 for _ in range(len(yf_defs["ticker_prices"]))],
        ),
    ]
    outs, defs = run_portfolio_opt.run(
        force_recompute=force_recompute,
        po=po,
        user_portfolios=user_portfolios,
    )

    assert isinstance(outs, mo.ui.plotly)
    assert "opt_port_plot" in defs.keys()
    assert isinstance(defs["opt_port_plot"], mo.ui.plotly)
