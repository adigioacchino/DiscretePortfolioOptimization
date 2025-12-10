# Welcome to Discrete Portfolio Optimization

This is the documentation for the Discrete Portfolio Optimization package.

## Why this package

This package differs from other portfolio optimization packages, such as [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt), in its fundamental approach to handling financial instruments.
Unlike packages that assume continuous holdings of assets, this package explicitly accounts for the discrete nature of financial instruments (i.e., you can only buy or sell whole shares).

This discrete constraint renders the optimization problem non-convex, making it unsuitable for traditional convex optimization techniques.
To address this, the Discrete Portfolio Optimization package integrates a custom optimization algorithm based on the Simulated Annealing (SA) algorithm.
SA is a probabilistic technique that can effectively explore the solution space of non-convex problems, allowing the package to find optimal or near-optimal portfolio allocations even with the real-world constraint of discrete asset quantities.

In essence, this package provides a more realistic and practical approach to portfolio optimization by directly tackling the challenges posed by the discrete nature of asset holdings, a crucial aspect often overlooked in other tools.

## How to use DPO

### Main idea and equation

The core idea of this package is to optimize a portfolio of assets where the holdings must be integers (discrete shares).
This is a non-convex optimization problem, which is solved here using the [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) algorithm.

The algorithm explores the space of possible portfolios by making random moves (swapping assets) and accepting or rejecting these moves based on a probability that depends on the change in the portfolio's "score" and a "temperature" parameter ($\theta$).

The objective function (Score) being maximized is defined as:

$$
\text{Score} = \text{Return} - \eta \cdot \text{Volatility} - \gamma \sum w_i^2 - \delta \frac{\text{Cash}}{\text{Total Value}}
$$

Where:

- **Return**: Expected daily return of the portfolio.
- **Volatility**: Expected daily volatility (risk) of the portfolio.
- **$\eta$ (eta)**: Risk aversion parameter. Larger $\eta$ leads to lower volatility portfolios.
- **$\gamma$ (gamma)**: Diversification parameter. Larger $\gamma$ penalizes concentrated portfolios (high sum of squared weights), encouraging diversification.
- **$\delta$ (delta)**: Cash penalty parameter. Larger $\delta$ penalizes holding cash, encouraging full investment.
- **$w_i$**: Weight of asset $i$ in the portfolio.

The algorithm runs for a sequence of decreasing temperatures (simulating the cooling process in annealing), allowing it to escape local optima initially and then settle into a global optimum.

### How to use the frontend

The package includes a web-based user interface built with [marimo](https://marimo.io/).
To launch the frontend, run the following command in your terminal:

```bash
uv run marimo edit UI/DPO_UserInterface.py
```

This will open a browser window where you can:

1. Enter stock symbols to download data from Yahoo Finance.
2. Configure optimization parameters (total value, risk aversion $\eta$, etc.).
3. Run the optimization and visualize the results (efficient frontier, portfolio composition, etc.).

## How to contribute

### Prepare the local environment

First, install `uv` using the command above. Then, create a new virtual environment using `uv`:

```bash
uv sync
```

(this step is not strictly necessary, it will be done automatically at the first `uv run` command)

### Pre-commit

To install pre-commit hooks, run the following command:

```bash
uv run pre-commit install
```

### Testing

To run the tests, use the following command:

```bash
uv run pytest --cov=src
```
