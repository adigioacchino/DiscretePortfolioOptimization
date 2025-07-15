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


### How to use the frontend


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
