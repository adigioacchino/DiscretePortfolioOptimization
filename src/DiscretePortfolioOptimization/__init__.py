"""
Discrete Portfolio Optimization Package

This package provides tools for discrete portfolio optimization using the Metropolis algorithm.

Key components:
- Portfolio: Class for portfolio management with discrete allocations
- PortfolioOptimizer: Class for portfolio optimization using the Metropolis algorithm
- get_close_price_df: Function to download closing prices for multiple tickers
"""

# Import main classes and functions for easy access
from DiscretePortfolioOptimization.portfolio import Portfolio
from DiscretePortfolioOptimization.metropolis import PortfolioOptimizer
from DiscretePortfolioOptimization.yfinance_download import (
    get_close_price_df,
)

# Define __all__ to explicitly specify what gets imported with "from DiscretePortfolioOptimization import *"
__all__ = [
    "Portfolio",
    "PortfolioOptimizer",
    "get_close_price_df",
]
