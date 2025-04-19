![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/adigioacchino/a72d32f9e10c404f7738822dac068274/raw/covbadge.json)

# Running the UI
This project consists of a python package `discrete_portfolio_optimization` and a simple UI written in [`marimo`](https://marimo.io/).

To run the UI, you need to install the package and its dependencies. The suggested way to do this is to use [`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file).

Assuming nothing is installed, you can follow the instructions below to install `uv` and run the UI.

## Linux / MacOS
Install `uv` using the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use `uv` to install python:
```bash
uv python install 3.13
```

Finally, use `uv` to run the UI (it will install the package and its dependencies):
```bash
uv run marimo run DPO_UserInterface.py
```
## Windows
The only difference is the command to install `uv`:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then you can follow the same steps as above to install python and run the UI.

# Contribute
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
