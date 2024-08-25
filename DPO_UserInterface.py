import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""# Imports""")
    return


@app.cell
def __():
    import marimo as mo

    from discrete_portfolio_optimization.yfinance_download import get_close_price_df
    from discrete_portfolio_optimization.portfolio import Portfolio
    from discrete_portfolio_optimization.metropolis import PortfolioOptimizer

    import pandas as pd
    import numpy as np
    import plotly.express as px
    return Portfolio, PortfolioOptimizer, get_close_price_df, mo, np, pd, px


@app.cell
def __(mo):
    mo.md("""# Load data""")
    return


@app.cell
def __(mo):
    symbols_string = mo.ui.text_area(value="META, AMZN, AAPL, NFLX, GOOGL", 
                                         label="Enter comma-separated symbols")
    symbols_string
    return symbols_string,


@app.cell
def __(get_close_price_df, symbols_string):
    close_df = get_close_price_df(symbols_string.value)
    return close_df,


@app.cell
def __(close_df):
    returns_df = close_df.pct_change().dropna() * 100
    return returns_df,


@app.cell
def __(mo):
    mo.md("""# Computations""")
    return


@app.cell
def __(mo, random_seed, total_value):
    mo.vstack([
        mo.md("""## Settings for initial portfolio"""),
        total_value,
        random_seed
    ], align="center")
    return


@app.cell
def __(mo):
    # UI elements - initial portfolio 
    total_value = mo.ui.number(start=100, stop=100_000, step=100, 
                               value=10_000,
                               label= "Total portfolio value:"
                              )
    random_seed = mo.ui.number(start=1, stop=100, 
                               value=53,
                               label= "Random seed:"
                              )
    return random_seed, total_value


@app.cell
def __(Portfolio, close_df, random_seed, total_value):
    initial_pft = Portfolio(close_df.iloc[-1].to_list(), tot_value=total_value.value, seed=random_seed.value)
    return initial_pft,


@app.cell
def __(
    alpha_slider,
    beta_slider,
    gamma,
    mo,
    n_alphas,
    n_betas,
    n_steps_per_beta,
    n_therm_steps,
):
    mo.vstack([
        mo.md("""## Settings for Simulated Annealing"""),
        alpha_slider, n_alphas, gamma, 
        n_therm_steps, beta_slider, n_betas, n_steps_per_beta
    ], align="center")
    return


@app.cell
def __(mo):
    # UI elements - simulated annealing
    alpha_slider = mo.ui.range_slider(start=-4, stop=4, step=0.5, value=[-2,2],
                                      label="Range of alpha values (exponent of 10)", show_value=True)
    n_alphas = mo.ui.slider(start=2, stop=100, step=1, value=10,
                            label="Number of alpha values", show_value=True)
    gamma = mo.ui.slider(start=-2, stop=2, step=0.5, value=1,
                         label="Gamma (exponent of 10)", show_value=True)
    n_therm_steps = mo.ui.slider(start=0, stop=10_000, step=250, value=1_000,
                                 label="Number of thermalization steps", show_value=True)
    beta_slider = mo.ui.range_slider(start=-1, stop=5, step=0.5, value=[1,3],
                                     label="Range of beta values (exponent of 10)", show_value=True)
    n_betas = mo.ui.slider(start=1_000, stop=10_000, step=500, value=5_000,
                           label="Number of different betas used", show_value=True)
    n_steps_per_beta = mo.ui.slider(start=1, stop=10, step=1, value=1,
                                    label="Number of SA steps done at each beta", show_value=True)
    return (
        alpha_slider,
        beta_slider,
        gamma,
        n_alphas,
        n_betas,
        n_steps_per_beta,
        n_therm_steps,
    )


@app.cell
def __(
    PortfolioOptimizer,
    alpha_slider,
    beta_slider,
    gamma,
    initial_pft,
    n_alphas,
    n_betas,
    n_steps_per_beta,
    n_therm_steps,
    returns_df,
):
    po = PortfolioOptimizer(
                            initial_pft, 
                            returns_df=returns_df,
                            alpha0=10**alpha_slider.value[0],
                            alpha1=10**alpha_slider.value[1],
                            n_alphas=n_alphas.value, 
                            gamma=10**gamma.value,
                            n_therm_steps=n_therm_steps.value,
                            beta0=10**beta_slider.value[0],
                            beta1=10**beta_slider.value[1],
                            n_betas=n_betas.value,
                            n_steps_per_beta=n_steps_per_beta.value
                        )
    return po,


@app.cell
def __(mo, run_computation_button):
    mo.md(f"""
    ## Run computation
    After deciding the parameters, press this button to start the computation:

    {run_computation_button}
    """)
    return


@app.cell
def __(mo):
    run_computation_button = mo.ui.run_button(label="Run optimization!")
    return run_computation_button,


@app.cell
def __(mo, po, run_computation_button):
    mo.stop(not run_computation_button.value)

    po.full_run()
    return


@app.cell
def __(mo):
    mo.md("""# Plotting""")
    return


@app.cell
def __(mo, pd, po, px, returns_df, run_computation_button):
    # plot best portfolios in return vs volatility space
    mo.stop(not run_computation_button.value)

    _metrics = [pft.portfolio_metrics(returns_df) for pft in po.best_portfolios]
    _returns = [m["Return"] for m in _metrics]
    _volatilities = [m["Volatility"] for m in _metrics]

    _plot_df = pd.DataFrame({"Return": _returns, "Volatility": _volatilities})
    _plot = px.scatter(_plot_df, x="Volatility", y="Return")

    mo.ui.plotly(_plot)
    return


if __name__ == "__main__":
    app.run()
