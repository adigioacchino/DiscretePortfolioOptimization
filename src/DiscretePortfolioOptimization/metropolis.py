from DiscretePortfolioOptimization.portfolio import Portfolio

from warnings import warn
from typing import List, Optional, Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
import marimo as mo  # to have marimo-compatible progress bars


class PortfolioOptimizer:
    """
    Class to optimize a portfolio using the Metropolis algorithm.

    Args:
        initial_portfolio: Portfolio, the initial portfolio to optimize
        returns_df: pd.DataFrame, the returns of the assets in the portfolio
        eta0: float, the smallest eta to use (larger eta -> lower volatility)
        eta1: float, the largest eta to use (larger eta -> lower volatility)
        n_etas: int, the number of etas to use, so the number of final optimized portfolios
        gamma: float, the gamma parameter to use (larger gamma -> more diversified portfolio)
        delta: float, the delta parameter to use (larger delta -> less cash in the portfolio)
        n_therm_steps: int, the number of thermalization steps to use before starting the optimization
        theta0: float, the initial theta for the Simulated Annealing algorithm. It should be so that
            `theta0 * score_diff << 1` where `score_diff` is the difference between the scores of two
            portfolios separated by a random move.
        theta1: float, the final theta for the Simulated Annealing algorithm. It should be so that
            `theta1 * score_diff >> 1` where `score_diff` is the difference between the scores of two
            portfolios separated by a random move.
        n_thetas: int, the number of thetas to use in the Simulated Annealing algorithm. Larger values
            will make the algorithm more precise but slower.
        n_steps_per_theta: int, the number of steps per theta to use in the Simulated Annealing algorithm.
            Larger values will make the algorithm more precise but slower.
    """

    def __init__(
        self,
        initial_portfolio: Portfolio,
        returns_df: pd.DataFrame,
        eta0: float = 1e1,
        eta1: float = 1e3,
        n_etas: int = 10,
        gamma: float = 0.0,
        delta: float = 0.0,
        n_therm_steps: int = 1_000,
        theta0: float = 1,
        theta1: float = 1e3,
        n_thetas: int = 5_000,
        n_steps_per_theta: int = 1,
    ):
        """
        Initialize the PortfolioOptimizer.

        Args:
            initial_portfolio: Portfolio, the initial portfolio to optimize
            returns_df: pd.DataFrame, the returns of the assets in the portfolio
            eta0: float, the smallest eta to use (larger eta -> lower volatility)
            eta1: float, the largest eta to use (larger eta -> lower volatility)
            n_etas: int, the number of etas to use, so the number of final optimized portfolios
            gamma: float, the gamma parameter to use (larger gamma -> more diversified portfolio)
            delta: float, the delta parameter to use (larger delta -> less cash in the portfolio)
            n_therm_steps: int, the number of thermalization steps to use before starting the optimization
            theta0: float, the initial theta for the Simulated Annealing algorithm. It should be so that
                `theta0 * score_diff << 1` where `score_diff` is the difference between the scores of two
                portfolios separated by a random move.
            theta1: float, the final theta for the Simulated Annealing algorithm. It should be so that
                `theta1 * score_diff >> 1` where `score_diff` is the difference between the scores of two
                portfolios separated by a random move.
            n_thetas: int, the number of thetas to use in the Simulated Annealing algorithm. Larger values
                will make the algorithm more precise but slower.
            n_steps_per_theta: int, the number of steps per theta to use in the Simulated Annealing algorithm.
                Larger values will make the algorithm more precise but slower.
        """
        self.initial_portfolio = initial_portfolio.copy()
        self.rng = initial_portfolio.rng  # share seed with portfolio if was provided
        self.best_portfolios: List[Portfolio] = []
        self._current_portfolio: Portfolio
        self._current_score: float
        self.gamma = gamma
        self.delta = delta
        # check that returns df is in *daily percentage* format
        if not np.all(returns_df.mean() >= -1):
            warn(
                "Large negative returns detected,"
                " remember that returns should be in *daily percentage* format."
            )
        if not np.all(returns_df.mean() <= 1):
            warn(
                "Large positive returns detected,"
                " remember that returns should be in *daily percentage* format."
            )
        if not np.max(np.abs(returns_df.mean()) >= 0.01):
            warn(
                "Largest return is very small,"
                " remember that returns should be in *daily percentage* format."
            )
        self.returns_df = returns_df
        self.eta_schedule = np.geomspace(eta0, eta1, n_etas)
        self.theta_schedule = self._prepare_exp_theta_schedule(
            theta0, theta1, n_therm_steps, n_thetas, n_steps_per_theta
        )

    @staticmethod
    def _portfolio_minus_energy(
        eta: float,
        gamma: float,
        delta: float,
        portfolio: Portfolio,
        returns_df: pd.DataFrame,
    ) -> float:
        """
        Compute the score of a portfolio.
        The score is defined as:

        `Return - eta * Volatility - gamma * sum(weights**2) - delta * cash_value / tot_value`

        Args:
            eta: float, the eta parameter to use (larger eta -> lower volatility)
            gamma: float, the gamma parameter to use (larger gamma -> more diversified portfolio)
            delta: float, the delta parameter to use (larger delta -> less cash in the portfolio)
            portfolio: Portfolio, the portfolio to evaluate
            returns_df: pd.DataFrame, the returns of the assets in the portfolio

        Returns:
            float, the score of the portfolio
        """
        portfolio_metrics = portfolio.portfolio_metrics(returns_df)
        if gamma != 0:
            # large gamma -> diversified portfolio
            gamma_term = gamma * float(np.sum(portfolio.weights**2))
        else:
            gamma_term = 0
        if delta != 0:
            # large delta -> low cash
            delta_term = delta * portfolio.cash_value / portfolio.tot_value
        else:
            delta_term = 0
        return (
            portfolio_metrics["Return"]
            - eta * portfolio_metrics["Volatility"]  # large eta -> low volatility
            - gamma_term
            - delta_term
        )

    @staticmethod
    def _prepare_exp_theta_schedule(
        theta0: float,
        theta1: float,
        n_therm_steps: int,
        n_thetas: int,
        n_step_per_theta: int,
    ) -> np.ndarray:
        """
        Prepare the theta schedule for the Metropolis algorithm.

        Args:
            theta0: float, the initial theta for the Simulated Annealing algorithm. It should be so that
                `theta0 * score_diff << 1` where `score_diff` is the difference between the scores of two
                portfolios separated by a random move.
            theta1: float, the final theta for the Simulated Annealing algorithm. It should be so that
                `theta1 * score_diff >> 1` where `score_diff` is the difference between the scores of two
                portfolios separated by a random move.
            n_therm_steps: int, the number of thermalization steps to use before starting the optimization
            n_thetas: int, the number of thetas to use in the Simulated Annealing algorithm. Larger values
                will make the algorithm more precise but slower.
            n_steps_per_theta: int, the number of steps per theta to use in the Simulated Annealing algorithm.
                Larger values will make the algorithm more precise but slower.

        Returns:
            np.ndarray, the theta schedule to use in the Metropolis algorithm
        """
        schedule1 = np.full(n_therm_steps, theta0)
        schedule2 = np.linspace(np.log(theta0), np.log(theta1), n_thetas)
        schedule2 = np.repeat(schedule2, n_step_per_theta)
        schedule2 = np.exp(schedule2)
        return np.concatenate([schedule1, schedule2])

    def _step(self, eta: float, theta: float) -> None:
        """
        Perform a step of the Metropolis algorithm.
        It results in potentially updating the `self._current_portfolio`
        and `self._current_score` attributes.

        Args:
            eta: float, the eta parameter to use (larger eta -> lower volatility)
            theta: float, the theta parameter to use in the Metropolis algorithm

        """
        new_portfolio = self._current_portfolio.copy()
        new_portfolio.random_move()
        new_score = self._portfolio_minus_energy(
            eta, self.gamma, self.delta, new_portfolio, self.returns_df
        )
        if new_score > self._current_score:
            self._current_portfolio = new_portfolio
            self._current_score = new_score
        elif self.rng.random() < np.exp(theta * (new_score - self._current_score)):
            self._current_portfolio = new_portfolio
            self._current_score = new_score
        return

    def run_fixed_eta(self, eta: float) -> None:
        """
        Run the optimization process for a fixed eta.
        It will store the best portfolio found in `self.best_portfolios`.

        Args:
            eta: float, the eta parameter to use (larger eta -> lower volatility)
        """
        self._current_portfolio = self.initial_portfolio
        self._current_score = self._portfolio_minus_energy(
            eta, self.gamma, self.delta, self.initial_portfolio, self.returns_df
        )
        best_portfolio = self._current_portfolio
        best_score = self._current_score

        # progress bar
        # Use Any type for theta_schedule_iter to accommodate both tqdm and marimo progress bar
        theta_schedule_iter: Any
        if mo.running_in_notebook():
            theta_schedule_iter = mo.status.progress_bar(
                self.theta_schedule,
                title="Optimizing portfolio",
                subtitle=f"with eta = {eta:.2e}",
                remove_on_exit=True,
            )
        else:
            theta_schedule_iter = tqdm(
                self.theta_schedule,
                desc=f"Optimizing portfolio with eta = {eta:.2e}",
                leave=False,
            )

        for theta in theta_schedule_iter:
            self._step(eta, theta)
            if self._current_score > best_score:
                best_portfolio = self._current_portfolio
                best_score = self._current_score
        best_portfolio.eta = eta
        self.best_portfolios.append(best_portfolio)
        return

    def full_run(
        self, callback: Optional[Callable[["PortfolioOptimizer"], Any]] = None
    ) -> Any:
        """
        Run the full optimization process.

        Args:
            callback: Optional function to call after each eta optimization

        Returns:
            The result of the callback if provided, otherwise None
        """
        # progress bar
        # Use Any type for eta_schedule_iter to accommodate both tqdm and marimo progress bar
        eta_schedule_iter: Any
        if mo.running_in_notebook():
            eta_schedule_iter = mo.status.progress_bar(
                self.eta_schedule, title="Collecting optimal portfolios"
            )
        else:
            eta_schedule_iter = tqdm(
                self.eta_schedule, desc="Collecting optimal portfolios"
            )

        callback_res: Any = None
        for eta in eta_schedule_iter:
            self.run_fixed_eta(eta)
            if callback is not None:
                callback_res = callback(self)

        return callback_res
