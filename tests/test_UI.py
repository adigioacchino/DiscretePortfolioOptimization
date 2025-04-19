from UI.DPO_UserInterface import (
    yf_download_input,
    yf_ui_fetch,
    init_po,
    run_portfolio_opt,
)

import DiscretePortfolioOptimization as dpo
import marimo as mo


#################################
# utils functions
def get_small_po():
    _, defs = init_po.run()
    po = defs["po"]
    assert isinstance(po, dpo.PortfolioOptimizer)
    # reduce the number of alphas for the test
    po.alpha_schedule = po.alpha_schedule[:2]
    # reduce the number of betas for the test
    po.beta_schedule = po.beta_schedule[:1000]
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


def test_run_portfolio_opt():
    # force_recompute = True
    po = get_small_po()
    force_recompute = get_mock_button(value=True)

    # run the portfolio optimization cell
    outs, defs = run_portfolio_opt.run(force_recompute=force_recompute, po=po)

    assert isinstance(outs, mo.ui.plotly)
    assert "opt_port_plot" in defs.keys()
    assert isinstance(defs["opt_port_plot"], mo.ui.plotly)
    assert len(po.best_portfolios) == len(po.alpha_schedule)

    # force_recompute = False
    force_recompute = get_mock_button(value=False)
    po = get_small_po()

    # run the portfolio optimization cell again
    outs, defs = run_portfolio_opt.run(force_recompute=force_recompute, po=po)
    assert len(po.best_portfolios) == len(po.alpha_schedule) * 2
