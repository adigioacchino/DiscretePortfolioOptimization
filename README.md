# Discrete Portfolio Optimization

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/adigioacchino/a72d32f9e10c404f7738822dac068274/raw/covbadge.json)

## Running the UI

This project consists of a python package `DiscretePortfolioOptimization` and a simple UI written in [`marimo`](https://marimo.io/).

To run the UI, you need to install the package and its dependencies. The suggested way to do this is to use [`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file).

Assuming nothing is installed, you can follow [the official instructions](https://docs.astral.sh/uv/getting-started/installation/) to install `uv` and run the UI.

Then, run the following command to start the UI:

```bash
uv run marimo run UI/DPO_UserInterface.py
```

## Contribute

## Prepare the local environment

First, install `uv` using the command above. Then, create a new virtual environment using `uv`:

```bash
uv sync
```

(this step is not strictly necessary, it will be done automatically at the first `uv run` command)

## Pre-commit

To install pre-commit hooks, run the following command:

```bash
uv run pre-commit install
```

## Testing

To run the tests, use the following command:

```bash
uv run pytest --cov=src
```
