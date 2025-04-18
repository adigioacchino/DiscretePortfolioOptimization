import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    import json
    import os
    import tempfile
    import pickle

    from discrete_portfolio_optimization.yfinance_download import (
        get_close_price_df,
    )
    from discrete_portfolio_optimization.portfolio import Portfolio
    from discrete_portfolio_optimization.metropolis import PortfolioOptimizer

    import pandas as pd
    import numpy as np
    import plotly.express as px
    return (
        Portfolio,
        PortfolioOptimizer,
        get_close_price_df,
        json,
        mo,
        np,
        os,
        pd,
        pickle,
        px,
        tempfile,
    )


@app.cell
def _(os, pickle, tempfile):
    # define useful functions

    # Function to get path to temporary file
    def get_temp_file_path():
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, "portfolio_optimization_results.pkl")


    # Function to save results to temporary file
    def save_results_to_temp_file(portfolios):
        temp_file_path = get_temp_file_path()
        with open(temp_file_path, "wb") as f:
            pickle.dump(portfolios, f)


    # Function to load results from temporary file
    def load_results_from_temp_file():
        temp_file_path = get_temp_file_path()
        if os.path.exists(temp_file_path):
            try:
                with open(temp_file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading saved results: {e}")
                return None
        return None
    return (
        get_temp_file_path,
        load_results_from_temp_file,
        save_results_to_temp_file,
    )


@app.cell(column=1)
def _(mo):
    mo.md(
        """
        # Fetch data from Yahoo Finance
        Use the text box on the left to enter comma-separated symbols names (you can check them on [Yahoo Finance](https://finance.yahoo.com/)).

        ⚠️ Symbols that are not found will be marked with a ❌ and not used in the portfolio optimization.
        """
    )
    return


@app.cell
def _(mo):
    symbols_string = mo.ui.text_area(
        value="META, AMZN, AAPL, NFLX, GOOGL",
    )
    download_from_yf_button = mo.ui.run_button(
        label="Download data from Yahoo Finance",
    )
    mo.vstack(
        [symbols_string, download_from_yf_button],
        #align="center",
    )
    return download_from_yf_button, symbols_string


@app.cell
def _(mo):
    mo.md("""# Computations""")
    return


@app.cell
def _(mo, random_seed, total_value):
    mo.vstack(
        [mo.md("""## Settings for initial portfolio"""), total_value, random_seed],
        align="center",
    )
    return


@app.cell
def _(mo):
    # UI elements - initial portfolio
    total_value = mo.ui.number(
        start=100,
        stop=100_000,
        step=100,
        value=10_000,
        label="Total portfolio value:",
    )
    random_seed = mo.ui.number(start=1, stop=100, value=53, label="Random seed:")
    return random_seed, total_value


@app.cell
def _(Portfolio, random_seed, ticker_prices, total_value):
    initial_pft = Portfolio(
        ticker_prices,
        tot_value=total_value.value,
        seed=random_seed.value,
    )
    return (initial_pft,)


@app.cell
def _(mo):
    mo.md("""## Settings for Simulated Annealing""")
    return


@app.cell
def _(mo):
    # load from file the default values for sliders
    po_kwargs_file = mo.ui.file(filetypes=[".json"], kind="area")
    po_kwargs_file
    return (po_kwargs_file,)


@app.cell
def _(json, po_kwargs_file):
    # load from file the default values for sliders
    if po_kwargs_file.value:
        po_kwargs_defaults = json.loads(po_kwargs_file.value[0].contents)
    else:
        po_kwargs_defaults = {
            "alpha0": -2,
            "alpha1": 0.5,
            "n_alphas": 25,
            "gamma_switch": False,
            "gamma": -1,
            "delta_switch": False,
            "delta": -1,
            "n_therm_steps": 500,
            "beta0": 1,
            "beta1": 3,
            "n_betas": 2500,
            "n_steps_per_beta": 1,
        }
    return (po_kwargs_defaults,)


@app.cell
def _(
    alpha_slider,
    beta_slider,
    delta_switch,
    gamma_switch,
    mo,
    n_alphas,
    n_betas,
    n_steps_per_beta,
    n_therm_steps,
):
    mo.vstack(
        [
            mo.md(
                "**Alpha**: a portfolio optimized with a large alpha will have lower volatility but also lower return."
            ),
            alpha_slider,
            n_alphas,
            mo.md(
                "**Thermalization steps**: random changes to the portfolio done before starting optimization."
            ),
            n_therm_steps,
            mo.md(
                "**Beta**: beta controls the optimization dynamics. More betas means a longer, but more precise, optimization."
            ),
            beta_slider,
            n_betas,
            n_steps_per_beta,
            mo.md(
                "**Gamma**: a portfolio optimized with a large gamma will be more differentiated."
            ),
            gamma_switch,
            mo.md(
                "**Delta**: a portfolio optimized with a large delta will have less cash."
            ),
            delta_switch,
        ],
        align="center",
    )
    return


@app.cell
def _(delta, delta_switch, gamma, gamma_switch, mo):
    switchable_sliders = []
    if gamma_switch.value:
        switchable_sliders.append(gamma)
    if delta_switch.value:
        switchable_sliders.append(delta)
    mo.vstack(switchable_sliders, align="center")
    return (switchable_sliders,)


@app.cell
def _(mo, po_kwargs_defaults):
    # UI elements - simulated annealing
    alpha_slider = mo.ui.range_slider(
        start=-4,
        stop=4,
        step=0.1,
        value=[po_kwargs_defaults["alpha0"], po_kwargs_defaults["alpha1"]],
        label="Range of alpha values (exponent of 10)",
        show_value=False,
        full_width=True,
    )
    n_alphas = mo.ui.slider(
        start=2,
        stop=50,
        step=1,
        value=po_kwargs_defaults["n_alphas"],
        label="Number of alpha values",
        show_value=True,
    )
    gamma_switch = mo.ui.switch(
        value=po_kwargs_defaults["gamma_switch"], label="Use gamma"
    )
    gamma = mo.ui.slider(
        start=-2,
        stop=2,
        step=0.1,
        value=po_kwargs_defaults["gamma"],
        label="Gamma (exponent of 10)",
        show_value=True,
    )
    delta_switch = mo.ui.switch(
        value=po_kwargs_defaults["delta_switch"], label="Use delta"
    )
    delta = mo.ui.slider(
        start=-2,
        stop=2,
        step=0.1,
        value=po_kwargs_defaults["delta"],
        label="Delta (exponent of 10)",
        show_value=True,
    )
    n_therm_steps = mo.ui.slider(
        start=0,
        stop=10_000,
        step=250,
        value=po_kwargs_defaults["n_therm_steps"],
        label="Number of thermalization steps",
        show_value=True,
    )
    beta_slider = mo.ui.range_slider(
        start=-1,
        stop=8,
        step=0.5,
        value=[po_kwargs_defaults["beta0"], po_kwargs_defaults["beta1"]],
        label="Range of beta values (exponent of 10)",
        show_value=True,
    )
    n_betas = mo.ui.slider(
        start=1_000,
        stop=10_000,
        step=500,
        value=po_kwargs_defaults["n_betas"],
        label="Number of different betas used",
        show_value=True,
    )
    n_steps_per_beta = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=po_kwargs_defaults["n_steps_per_beta"],
        label="Number of SA steps done at each beta",
        show_value=True,
    )
    return (
        alpha_slider,
        beta_slider,
        delta,
        delta_switch,
        gamma,
        gamma_switch,
        n_alphas,
        n_betas,
        n_steps_per_beta,
        n_therm_steps,
    )


@app.cell
def _(
    alpha_slider,
    beta_slider,
    delta,
    delta_switch,
    gamma,
    gamma_switch,
    n_alphas,
    n_betas,
    n_steps_per_beta,
    n_therm_steps,
):
    # prepare kwarg for PortfolioOptimzer
    PO_kwargs_download = {
        "alpha0": alpha_slider.value[0],
        "alpha1": alpha_slider.value[1],
        "n_alphas": n_alphas.value,
        "n_therm_steps": n_therm_steps.value,
        "beta0": beta_slider.value[0],
        "beta1": beta_slider.value[1],
        "n_betas": n_betas.value,
        "n_steps_per_beta": n_steps_per_beta.value,
        "gamma_switch": gamma_switch.value,
        "delta_switch": delta_switch.value,
    }

    if gamma_switch.value:
        gamma_value = 10**gamma.value
        PO_kwargs_download["gamma"] = gamma_value
    else:
        gamma_value = 0
        PO_kwargs_download["gamma"] = 0
    if delta_switch.value:
        delta_value = 10**delta.value
        PO_kwargs_download["delta"] = delta_value
    else:
        delta_value = 0
        PO_kwargs_download["delta"] = 0


    PO_kwargs = {
        "alpha0": 10 ** alpha_slider.value[0],
        "alpha1": 10 ** alpha_slider.value[1],
        "n_alphas": n_alphas.value,
        "gamma": gamma_value,
        "delta": delta_value,
        "n_therm_steps": n_therm_steps.value,
        "beta0": 10 ** beta_slider.value[0],
        "beta1": 10 ** beta_slider.value[1],
        "n_betas": n_betas.value,
        "n_steps_per_beta": n_steps_per_beta.value,
    }
    return PO_kwargs, PO_kwargs_download, delta_value, gamma_value


@app.cell
def _(PO_kwargs_download, json, mo):
    # download kwargs for backup
    mo.download(
        json.dumps(PO_kwargs_download),
        "PO_kwargs.json",
        label="Download kwargs",
    )
    return


@app.cell
def _(PO_kwargs, PortfolioOptimizer, initial_pft, returns_df):
    po = PortfolioOptimizer(
        initial_portfolio=initial_pft,
        returns_df=returns_df,
        **PO_kwargs,
    )
    return (po,)


@app.cell
def _(force_recompute, mo, run_computation_button):
    mo.md(f"""
    ## Run computation
    After deciding the parameters, press this button to start the computation:

    {run_computation_button}

    {force_recompute}
    """)
    return


@app.cell
def _(mo):
    run_computation_button = mo.ui.run_button(label="Run optimization!")
    force_recompute = mo.ui.checkbox(
        value=False, label="Force recomputation even if previous results exist"
    )
    return force_recompute, run_computation_button


@app.cell
def _(mo, np, pd, po, px):
    # plot best portfolios in return vs volatility space
    def plot_portfolios(best_portfolios, pure_portfolios, returns_df):
        # portfolios just computed
        _metrics = [
            pft.portfolio_metrics(returns_df) for pft in po.best_portfolios
        ]
        _alphas = [pft.alpha for pft in po.best_portfolios]
        _returns = [m["Return"] for m in _metrics]
        _volatilities = [m["Volatility"] for m in _metrics]
        _indices = [i for i in range(len(_metrics))]

        _pure = ["Optimized" for _ in range(len(_metrics))]
        _metrics = [pft.portfolio_metrics(returns_df) for pft in pure_portfolios]
        _returns = _returns + [m["Return"] for m in _metrics]
        _volatilities = _volatilities + [m["Volatility"] for m in _metrics]
        _indices = _indices + [len(_indices) + i for i in range(len(_metrics))]
        _pure = _pure + ["Pure" for _ in range(len(_metrics))]

        # Create optimized df
        optimized_df = pd.DataFrame(
            {
                "Return": _returns[: len(_alphas)],
                "Volatility": _volatilities[: len(_alphas)],
                "Index": _indices[: len(_alphas)],
                "Type": _pure[: len(_alphas)],
                "LogAlpha": np.log10(
                    [max(a, 1e-10) for a in _alphas]
                ),  # Use a minimum value to avoid log(0)
            }
        )

        # Create pure portfolios df
        pure_df = pd.DataFrame(
            {
                "Return": _returns[len(_alphas) :],
                "Volatility": _volatilities[len(_alphas) :],
                "Index": _indices[len(_alphas) :],
                "Type": _pure[len(_alphas) :],
            }
        )

        # Create figure with optimized portfolios using blue colorscale on log scale
        _plot = px.scatter(
            optimized_df,
            x="Volatility",
            y="Return",
            color="LogAlpha",  # Use log scale for coloring
            color_continuous_scale="Blues",
            hover_data={
                "Type": True,
                "Volatility": ":.2f",
                "Return": ":.2f",
                "LogAlpha": ":.2f",  # Hide LogAlpha in hover
                "Index": True,
            },
        )

        # Add pure portfolios as orange markers
        pure_trace = px.scatter(
            pure_df,
            x="Volatility",
            y="Return",
            hover_data={
                "Type": True,
                "Volatility": ":.2f",
                "Return": ":.2f",
                "Index": True,
            },
        ).data[0]

        pure_trace.marker.color = "orange"
        pure_trace.marker.symbol = "x"
        _plot.add_trace(pure_trace)

        _plot.update_traces(
            marker_size=10,
            marker_line_width=2,
        )

        mo.output.replace_at_index(mo.ui.plotly(_plot), 1)
        return mo.ui.plotly(_plot)
    return (plot_portfolios,)


@app.cell
def _(Portfolio, np, returns_df, ticker_prices):
    # prepare "pure" portfolios
    _n = len(returns_df.columns)
    pure_portfolios = [
        Portfolio(
            ticker_prices,
            allocations=np.identity(_n, dtype=np.int64)[:, i].tolist(),
        )
        for i in range(_n)
    ]
    return (pure_portfolios,)


@app.cell
def _(
    force_recompute,
    load_results_from_temp_file,
    mo,
    plot_portfolios,
    po,
    pure_portfolios,
    returns_df,
    run_computation_button,
    save_results_to_temp_file,
):
    # here the computation is actually run
    mo.stop(not run_computation_button.value)


    def plot_portfolios_closure(
        best_portfolios, pure_portfolios=pure_portfolios, returns_df=returns_df
    ):
        return plot_portfolios(best_portfolios, pure_portfolios, returns_df)


    # Check for existing saved portfolio results
    _saved_portfolios = None
    if not force_recompute.value:
        _saved_portfolios = load_results_from_temp_file()

    if _saved_portfolios is not None and not force_recompute.value:
        # Use existing results
        po.best_portfolios = _saved_portfolios  # Restore saved portfolios
        mo.output.append(
            mo.md(
                "*Using previously computed results. Check 'Force recomputation' to run a new optimization.*"
            )
        )
    else:
        po.best_portfolios = []
        mo.output.append(
            mo.md(
                "*No previously computed results found or force recomputation is enabled.*"
            )
        )

    # plot current portfolios
    mo.output.replace_at_index(plot_portfolios_closure(po.best_portfolios), 1)

    # Run new optimization in any case
    opt_port_plot = po.full_run(callback=plot_portfolios_closure)
    # Save just the portfolios, not the UI element
    save_results_to_temp_file(po.best_portfolios)

    opt_port_plot
    return opt_port_plot, plot_portfolios_closure


@app.cell
def _(mo, opt_port_plot, pd, po, pure_portfolios, returns_df):
    # print table with selected portfolio info
    _pft_table = po.best_portfolios + pure_portfolios
    selected_idxs = [x["Index"] for x in opt_port_plot.value]
    selected_portfolios = [_pft_table[i] for i in selected_idxs]

    _returns = [pft.get_day_return(returns_df) for pft in selected_portfolios]
    _volatilities = [
        pft.get_day_volatility(returns_df) for pft in selected_portfolios
    ]
    _allocations = [pft.allocations for pft in selected_portfolios]
    _cash_values = [pft.cash_value for pft in selected_portfolios]
    _symbols = returns_df.columns.to_list()

    sp_df = pd.DataFrame(
        index=selected_idxs,
        data={
            "Return": _returns,
            "Volatility": _volatilities,
            "Cash": _cash_values,
        },
    )
    for i, symbol in enumerate(_symbols):
        sp_df[symbol] = [alloc[i] for alloc in _allocations]
    sp_df.drop_duplicates(inplace=True, keep="last")
    sp_df.reset_index(inplace=True)

    if len(sp_df) > 0:
        _out = mo.ui.table(
            sp_df.round(2),
            show_column_summaries=False,
            selection=None,
        )
    else:
        _out = mo.md("Select portfolios to see details.")

    _out
    return i, selected_idxs, selected_portfolios, sp_df, symbol


@app.cell
def _():
    return


@app.cell(column=2)
def _(download_from_yf_button, get_close_price_df, mo, pd, symbols_string):
    # fetch data from Yahoo Finance
    mo.stop(not download_from_yf_button.value)
    _close_df, _tickers_hit, _tickers_miss = get_close_price_df(
        symbols_string.value
    )

    # Create a dictionary mapping tickers to status emojis
    # Combine hit and miss tickers to get all attempted tickers
    # Use sorted list for consistent order
    _all_tickers = sorted(list(set(_tickers_hit) | set(_tickers_miss)))
    _status_map = {ticker: "✅" for ticker in _tickers_hit}
    _status_map.update({ticker: "❌" for ticker in _tickers_miss})

    # Create a pandas Series based on the sorted list to maintain order
    # Ensure pandas is imported, usually as pd
    # Assuming pd is already imported from previous cells
    _status_series = pd.Series(
        {ticker: _status_map[ticker] for ticker in _all_tickers},
        name="Historical data found in YF",
    )
    _status_series.index.name = "Ticker"

    # Convert Series to DataFrame for display
    _status_df = _status_series.to_frame()

    # Display the table (in a notebook, the last expression is displayed)
    mo.output.append(
        mo.center(
            mo.ui.table(
                _status_df,
                selection=None,
                show_download=False,
                show_column_summaries=False,
            )
        )
    )

    # define useful vars for computations
    returns_df = _close_df.pct_change().dropna() * 100
    ticker_prices = _close_df.iloc[-1].to_list()
    return returns_df, ticker_prices


if __name__ == "__main__":
    app.run()
