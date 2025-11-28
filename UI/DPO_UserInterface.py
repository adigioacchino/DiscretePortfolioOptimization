import marimo

__generated_with = "0.18.1"
app = marimo.App(width="columns")

with app.setup:
    # Define a unique session ID - a random UUID
    import marimo as mo

    import os
    import tempfile
    import pickle

    import pandas as pd
    import numpy as np
    import plotly.express as px

    session_id = os.urandom(16).hex()


@app.cell
def _():
    mo.md(
        r"""
    # Discrete Portfolio Optimization
    This tool aims at finding the **efficient risk/return frontier** of the ensemble of possible portfolios associated with a basket of assets.

    Given a set of assets $\{A_i\}_{i=1}^N$ , the algorithm seeks a portfolio $\{n_i\}_{i=1}^N$ (with $n_i$ the number of quotes to buy for the asset $i$) that maximises the following function:

    $$ \mathcal{L}\left(\{n_i\}\right) =  \sum_i w_i \bar{R}_i \ - \eta \sqrt{ \sum_{i,j}w_iw_j\Sigma_{ij}} -  \gamma  \sum_i w_i^2 - \delta \frac{\text{Cash}}{\text{Funds}} \,,$$

    where:

    - $\text{Funds}$ is the total size of the investment
    - $\text{Cash}$ is the part of the investment left in cash
    - $w_i := n_i \frac {\text{PriceAsset}_i}{\text{Funds}}$ is the weight of the investment on the asset $i$
    - $\bar{R}_i$ is the mean daily return of the asset $i$
    - $\Sigma$ is the covariance matrix of the assets' daily returns

    Overall, the algorithm aims at maximising the value of the daily return with the following penalties:

    - Portfolios with excessive volatilities are penalized ($\eta$-term)
    - Portfolios excessively concentrated in one asset are penalized ($\gamma$-term)
    - Portfolios with to much un-invested cash are penalized ($\delta$-term)
    """
    )
    return


@app.cell
def _():
    mo.md("""
    # Fetch data from Yahoo Finance
    Use the text box on the left to enter comma-separated symbols names (you can check them on [Yahoo Finance](https://finance.yahoo.com/)).

    ⚠️ Symbols that are not found will be marked with a ❌ and not used in the portfolio optimization.
    """)
    return


@app.cell
def yf_download_input():
    symbols_string = mo.ui.text_area(
        value="META, AMZN, AAPL, NFLX, GOOGL",
    )
    download_from_yf_button = mo.ui.run_button(
        label="Download data from Yahoo Finance",
    )

    target_currency = mo.ui.dropdown(
            options=["USD", "EUR", "JPY", "GBP", "CAD", "CHF"],
            value="USD",  # default
            label="Set target currency",
        )

    mo.vstack(
        [symbols_string, download_from_yf_button, target_currency],
        #align="center",
    )
    return download_from_yf_button, symbols_string, target_currency


@app.cell
def _():
    mo.md("""
    # Settings for optimization
    """)
    return


@app.cell
def _(random_seed, total_value):
    mo.vstack(
        [mo.md("""## Settings for initial portfolio"""), total_value, random_seed],
        align="start",
    )
    return


@app.cell
def _():
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
def _():
    # load from file the default values for sliders
    po_kwargs_defaults = {
        "eta0": -2,
        "eta1": 0.5,
        "n_etas": 5,
        "gamma_switch": False,
        "gamma": -1,
        "delta_switch": False,
        "delta": -1,
        "n_therm_steps": 500,
        "theta0": 1,
        "theta1": 3,
        "n_thetas": 2500,
        "n_steps_per_theta": 1,
    }
    return (po_kwargs_defaults,)


@app.cell
def _(
    delta_switch,
    eta_slider,
    eta_slider,
    gamma_switch,
    n_etas,
    n_steps_per_theta,
    n_therm_steps,
    n_thetas,
    theta_slider,
    n_thetas,
    theta_slider,
):
    _adv_settings = mo.accordion(
        {
            "### Advanced settings:": mo.vstack(
                [
                    mo.md(
                        "**Thermalization steps**: random changes to the portfolio done before starting optimization."
                    ),
                    n_therm_steps,
                    mo.md(
                        "**Theta**: theta controls the optimization dynamics. More thetas means a longer, but more precise, optimization."
                    ),
                    theta_slider,
                    n_thetas,
                    n_steps_per_theta,
                    mo.md(
                        "**Gamma**: a portfolio optimized with a large gamma will be more differentiated."
                    ),
                    gamma_switch,
                    mo.md(
                        "**Delta**: a portfolio optimized with a large delta will have less cash."
                    ),
                    delta_switch,
                ],
                align="start",
            )
        }
    )

    mo.vstack(
        [
            mo.md("## Settings for Simulated Annealing"),
            mo.md(
                "**Eta**: a portfolio optimized with a large eta will have lower volatility but also lower return."
            ),
            eta_slider,
            n_etas,
            _adv_settings,
        ],
        align="start",
    )
    return


@app.cell
def _(delta, delta_switch, gamma, gamma_switch):
    switchable_sliders = []
    if gamma_switch.value:
        switchable_sliders.append(gamma)
    if delta_switch.value:
        switchable_sliders.append(delta)
    mo.vstack(switchable_sliders, align="start")
    return


@app.cell
def _(po_kwargs_defaults):
    # UI elements - simulated annealing
    eta_slider = mo.ui.range_slider(
        start=-4,
        stop=4,
        step=0.1,
        value=[po_kwargs_defaults["eta0"], po_kwargs_defaults["eta1"]],
        label="Range of eta values (exponent of 10)",
        show_value=False,
        full_width=True,
    )
    n_etas = mo.ui.slider(
        start=2,
        stop=50,
        step=1,
        value=po_kwargs_defaults["n_etas"],
        label="Number of eta values",
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
    theta_slider = mo.ui.range_slider(
        start=-1,
        stop=8,
        step=0.5,
        value=[po_kwargs_defaults["theta0"], po_kwargs_defaults["theta1"]],
        label="Range of theta values (exponent of 10)",
        show_value=True,
    )
    n_thetas = mo.ui.slider(
        start=1_000,
        stop=10_000,
        step=500,
        value=po_kwargs_defaults["n_thetas"],
        label="Number of different thetas used",
        show_value=True,
    )
    n_steps_per_theta = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=po_kwargs_defaults["n_steps_per_theta"],
        label="Number of SA steps done at each theta",
        show_value=True,
    )
    return (
        delta,
        delta_switch,
        eta_slider,
        eta_slider,
        gamma,
        gamma_switch,
        n_etas,
        n_steps_per_theta,
        n_therm_steps,
        n_thetas,
        theta_slider,
        n_thetas,
        theta_slider,
    )


@app.cell
def _(
    delta,
    delta_switch,
    eta_slider,
    gamma,
    gamma_switch,
    n_etas,
    n_steps_per_theta,
    n_therm_steps,
    n_thetas,
    theta_slider,
):
    # prepare kwarg for PortfolioOptimzer
    if gamma_switch.value:
        gamma_value = 10**gamma.value
    else:
        gamma_value = 0
    if delta_switch.value:
        delta_value = 10**delta.value
    else:
        delta_value = 0

    PO_kwargs = {
        "eta0": 10 ** eta_slider.value[0],
        "eta1": 10 ** eta_slider.value[1],
        "n_etas": n_etas.value,
        "gamma": gamma_value,
        "delta": delta_value,
        "n_therm_steps": n_therm_steps.value,
        "theta0": 10 ** theta_slider.value[0],
        "theta1": 10 ** theta_slider.value[1],
        "n_thetas": n_thetas.value,
        "n_steps_per_theta": n_steps_per_theta.value,
    }
    return (PO_kwargs,)


@app.cell
def _(returns_df):
    _df = pd.DataFrame(columns=returns_df.columns, dtype=int)
    raw_user_portfolios = mo.ui.data_editor(
        _df, page_size=10, column_sizing_mode="fit"
    )
    _txt = "### Add your portfolios here to see them in the final plot"
    mo.accordion({_txt: raw_user_portfolios})
    return (raw_user_portfolios,)


@app.cell
def _(Portfolio, raw_user_portfolios, ticker_prices):
    # from raw to user portfolios
    user_portfolios = []
    for i in range(len(raw_user_portfolios.value)):
        user_portfolios.append(
            Portfolio(
                ticker_prices,
                allocations=raw_user_portfolios.value.iloc[i].to_list(),
            )
        )
    return (user_portfolios,)


@app.cell(column=1)
def yf_ui_fetch(
    download_from_yf_button,
    get_close_price_df,
    symbols_string,
    target_currency,
):
    # fetch data from Yahoo Finance
    mo.stop(not (download_from_yf_button.value or not mo.running_in_notebook()))
    _close_df, _tickers_hit, _tickers_miss = get_close_price_df(
        symbols_string.value, drop_missing_dates=False, target_currency=target_currency.value
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

    # add info about first available date to _status_df
    _first_valid_indices = _close_df.apply(lambda col: col.first_valid_index())
    _first_valid_dates_formatted = _first_valid_indices.dt.strftime("%Y-%m-%d")
    _status_df["First available date in YF"] = _first_valid_dates_formatted

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
    _clean_close_df = _close_df.dropna()
    returns_df = _clean_close_df.pct_change().dropna() * 100
    ticker_prices = _clean_close_df.iloc[-1].to_list()
    return returns_df, ticker_prices


@app.cell
def _(force_recompute, run_computation_button):
    mo.md(
        f"""
    # Run computation
    After deciding the parameters, press this button to start the computation:

    {run_computation_button}

    {force_recompute}
    """
    )
    return


@app.cell
def _():
    run_computation_button = mo.ui.run_button(label="Run optimization!")
    force_recompute = mo.ui.checkbox(
        value=False, label="Force recomputation even if previous results exist"
    )
    return force_recompute, run_computation_button


@app.cell
def init_po(PO_kwargs, PortfolioOptimizer, initial_pft, returns_df):
    po = PortfolioOptimizer(
        initial_portfolio=initial_pft,
        returns_df=returns_df,
        **PO_kwargs,
    )
    return (po,)


@app.cell
def _(Portfolio, returns_df, ticker_prices):
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
def _():
    # add a number input to select the number of top assets to show
    num_assets_plot = mo.ui.number(
        start=1, stop=10, label="Number of top assets to show in plot: ", value=5
    )
    num_assets_plot
    return (num_assets_plot,)


@app.cell
def run_portfolio_opt(
    force_recompute,
    load_results_from_temp_file,
    num_assets_plot,
    plot_portfolios,
    po,
    pure_portfolios,
    returns_df,
    run_computation_button,
    save_results_to_temp_file,
    user_portfolios,
):
    # here the computation is actually run
    mo.stop(not (run_computation_button.value or not mo.running_in_notebook()))


    def plot_portfolios_closure(
        po,
        pure_portfolios=pure_portfolios,
        user_portfolios=user_portfolios,
        returns_df=returns_df,
        k=num_assets_plot.value,
    ):
        return plot_portfolios(po, pure_portfolios, user_portfolios, returns_df, k)


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
    mo.output.replace_at_index(plot_portfolios_closure(po), 1)

    # Run new optimization in any case
    opt_port_plot = po.full_run(callback=plot_portfolios_closure)
    # Save just the portfolios, not the UI element
    save_results_to_temp_file(po.best_portfolios)

    opt_port_plot
    return (opt_port_plot,)


@app.cell
def _(opt_port_plot, po, pure_portfolios, returns_df, user_portfolios):
    # print table with selected portfolio info
    _pft_table = po.best_portfolios + pure_portfolios + user_portfolios
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
    for _i, _symbol in enumerate(_symbols):
        sp_df[_symbol] = [alloc[_i] for alloc in _allocations]
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
    return


@app.cell(column=2)
def _():
    # import DPO
    from DiscretePortfolioOptimization import (
        Portfolio,
        PortfolioOptimizer,
        get_close_price_df,
    )
    return Portfolio, PortfolioOptimizer, get_close_price_df


@app.cell
def _():
    # define useful functions

    # Function to get path to temporary file
    def get_temp_file_path():
        temp_dir = tempfile.gettempdir()
        # Use session_id to create a unique filename for this notebook run
        filename = f"portfolio_optimization_results_{session_id}.pkl"
        return os.path.join(temp_dir, filename)


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
    return load_results_from_temp_file, save_results_to_temp_file


@app.cell
def _():
    # plot best portfolios in return vs volatility space
    def _prepare_portfolios_plotting(portfolios, returns_df, k, is_optimized=True):
        # top k assets info
        # Initialize lists to store formatted top k asset info strings
        asset_names = returns_df.columns.to_list()
        top_k_assets_info_lists = [[] for _ in range(k)]
        for pft in portfolios:
            tot_value = pft.tot_value + 1e-2  # Avoid division by zero or near-zero
            asset_value = pft.asset_value
            weights = pft.weights
            top_k_formatted = [
                ""
            ] * k  # Initialize with empty strings for this portfolio
            # Calculate asset fractions relative to total value
            asset_frac = [
                (name, (w * asset_value / tot_value))
                for name, w in zip(asset_names, weights)
            ]
            # Sort assets by fraction, descending
            sorted_asset_frac = sorted(
                asset_frac, key=lambda item: item[1], reverse=True
            )

            # Format top k assets info, ensuring we don't go out of bounds
            num_assets_to_show = min(k, len(sorted_asset_frac))
            for i in range(num_assets_to_show):
                name, frac = sorted_asset_frac[i]
                top_k_formatted[i] = f"{name} ({frac:.1%})"

            # Append the formatted strings to the corresponding lists
            for i in range(k):
                top_k_assets_info_lists[i].append(top_k_formatted[i])

        # Create optimized df data dictionary
        if is_optimized:
            metrics = [pft.portfolio_metrics(returns_df) for pft in portfolios]
            optimized_data = {
                "Return": [m["Return"] for m in metrics],
                "Volatility": [m["Volatility"] for m in metrics],
                "Index": [i for i in range(len(metrics))],
                "Type": ["Optimized" for _ in range(len(metrics))],
                "LogEta": np.log10(
                    [max(pft.eta, 1e-10) for pft in portfolios]
                ),  # Use a minimum value to avoid log(0)
                "Cash": [pft.cash_value / pft.tot_value for pft in portfolios],
            }
            # Add top k asset info columns
            for i in range(k):
                optimized_data[f"#{i + 1} asset"] = top_k_assets_info_lists[i]
            optimized_df = pd.DataFrame(optimized_data)

            # Define hover data for optimized portfolios
            optimized_hover_data = {
                "Type": True,
                "Return": ":.2f",
                "Volatility": ":.2f",
                "Index": True,
                "LogEta": ":.2f",
                "Cash": ":.1%",
            }
            # Add top k asset info to hover data
            for i in range(k):
                optimized_hover_data[f"#{i + 1} asset"] = (
                    True  # Display the pre-formatted string
                )
        else:
            metrics = [pft.portfolio_metrics(returns_df) for pft in portfolios]
            optimized_data = {
                "Return": [m["Return"] for m in metrics],
                "Volatility": [m["Volatility"] for m in metrics],
                "Index": [i for i in range(len(metrics))],
                "Type": ["User" for _ in range(len(metrics))],
            }
            # Add top k asset info columns
            for i in range(k):
                optimized_data[f"#{i + 1} asset"] = top_k_assets_info_lists[i]
            optimized_df = pd.DataFrame(optimized_data)

            # Define hover data for optimized portfolios
            optimized_hover_data = {
                "Type": True,
                "Return": ":.2f",
                "Volatility": ":.2f",
                "Index": True,
            }
            # Add top k asset info to hover data
            for i in range(k):
                optimized_hover_data[f"#{i + 1} asset"] = (
                    True  # Display the pre-formatted string
                )

        return optimized_df, optimized_hover_data


    def plot_portfolios(po, pure_portfolios, user_portfolios, returns_df, k=5):
        # 1. optimized portfolios
        best_portfolios = po.best_portfolios
        optimized_df, optimized_hover_data = _prepare_portfolios_plotting(
            best_portfolios, returns_df, k
        )

        # 2. pure portfolios
        asset_names = po.returns_df.columns.to_list()
        metrics_pure = [
            pft.portfolio_metrics(returns_df) for pft in pure_portfolios
        ]
        _returns = [m["Return"] for m in metrics_pure]
        _volatilities = [m["Volatility"] for m in metrics_pure]
        _indices = [len(optimized_df) + i for i in range(len(metrics_pure))]
        _pure = ["Pure" for _ in range(len(metrics_pure))]

        # Create pure portfolios df
        pure_df = pd.DataFrame(
            {
                "Return": _returns,
                "Volatility": _volatilities,
                "Index": _indices,
                "Type": _pure,
                "Asset": asset_names,
            }
        )

        # 3. user portfolios
        user_df, user_hover_data = _prepare_portfolios_plotting(
            user_portfolios, returns_df, k, is_optimized=False
        )
        user_df["Index"] = [_indices[-1] + i + 1 for i in range(len(user_df))]

        ############################################################
        # Create plot
        # Create figure with optimized portfolios using blue colorscale on log scale
        _plot = px.scatter(
            optimized_df,
            x="Volatility",
            y="Return",
            color="LogEta",  # Use log scale for coloring
            color_continuous_scale="Blues",
            hover_data=optimized_hover_data,
            labels={"LogEta": "Log10(Eta)"},  # Add label for color bar
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
                "Asset": True,
            },
        ).data[0]
        pure_trace.marker.color = "orange"
        pure_trace.marker.symbol = "x"
        _plot.add_trace(pure_trace)

        # Add user portfolios as red markers
        user_trace = px.scatter(
            user_df,
            x="Volatility",
            y="Return",
            hover_data=user_hover_data,
        ).data[0]
        user_trace.marker.color = "red"
        user_trace.marker.symbol = "diamond"
        _plot.add_trace(user_trace)

        # Improve plot aesthetics
        _plot.update_layout(title="Portfolio Optimization: Return vs. Volatility [measured as %/day]")

        # Update trace properties for optimized portfolios for legend
        _plot.update_traces(
            marker_size=10,
            marker_line_width=1.5,
            marker_line_color="black",
        )

        # Ensure the pure portfolio trace also has consistent marker size
        _plot.update_traces(
            marker_size=10,
            marker_line_width=2,
        )

        ############################################################
        # deal with marimo layout and return
        mo.output.replace_at_index(mo.ui.plotly(_plot), 1)

        return mo.ui.plotly(_plot)
    return (plot_portfolios,)


if __name__ == "__main__":
    app.run()
