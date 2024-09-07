from discrete_portfolio_optimization.portfolio import Portfolio

import numpy as np
import pandas as pd
from tqdm import tqdm
import marimo as mo  # to have marimo-compatible progress bars
from warnings import warn


class PortfolioOptimizer:
    def __init__(
        self,
        initial_portfolio: Portfolio,
        returns_df: pd.DataFrame,
        alpha0: float = 1e1,
        alpha1: float = 1e3,
        n_alphas: int = 10,
        gamma: float = 0.0,
        delta: float = 0.0,
        n_therm_steps: int = 1_000,
        beta0: float = 1,
        beta1: float = 1e3,
        n_betas: int = 5_000,
        n_steps_per_beta: int = 1,
    ):
        self.initial_portfolio = initial_portfolio.copy()
        self.rng = initial_portfolio.rng  # share seed with portfolio if was provided
        self.best_portfolios: list[Portfolio] = []
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
        self.alpha_schedule = np.geomspace(alpha0, alpha1, n_alphas)
        self.beta_schedule = self._prepare_exp_beta_schedule(
            beta0, beta1, n_therm_steps, n_betas, n_steps_per_beta
        )

    @staticmethod
    def _portfolio_minus_energy(
        alpha: float, gamma: float, delta: float, 
        portfolio: Portfolio, returns_df: pd.DataFrame
    ) -> float:
        portfolio_metrics = portfolio.portfolio_metrics(returns_df)
        if gamma != 0:
            # large gamma -> diversified portfolio
            gamma_term = gamma * np.sum(portfolio.weights**2)
        else:
            gamma_term = 0
        if delta != 0:
            # large delta -> low cash
            delta_term = delta * portfolio.cash_value / portfolio.tot_value
        else:
            delta_term = 0
        return (
            portfolio_metrics["Return"]
            - alpha * portfolio_metrics["Volatility"]  # large alpha -> low volatility
            - gamma_term - delta_term
        )

    @staticmethod
    def _prepare_exp_beta_schedule(
        beta0: float,
        beta1: float,
        n_therm_steps: int,
        n_betas: int,
        n_step_per_beta: int,
    ) -> np.ndarray:
        schedule1 = np.full(n_therm_steps, beta0)
        schedule2 = np.linspace(np.log(beta0), np.log(beta1), n_betas)
        schedule2 = np.repeat(schedule2, n_step_per_beta)
        schedule2 = np.exp(schedule2)
        return np.concatenate([schedule1, schedule2])

    def _step(self, alpha, beta):
        new_portfolio = self._current_portfolio.copy()
        new_portfolio.random_move()
        new_score = self._portfolio_minus_energy(
            alpha, self.gamma, self.delta, new_portfolio, self.returns_df
        )
        if new_score > self._current_score:
            self._current_portfolio = new_portfolio
            self._current_score = new_score
        elif self.rng.random() < np.exp(beta * (new_score - self._current_score)):
            self._current_portfolio = new_portfolio
            self._current_score = new_score
        return

    def run_fixed_alpha(self, alpha):
        self._current_portfolio = self.initial_portfolio
        self._current_score = self._portfolio_minus_energy(
            alpha, self.gamma, self.delta,
            self.initial_portfolio, self.returns_df
        )
        best_portfolio = self._current_portfolio
        best_score = self._current_score

        # progress bar
        if mo.running_in_notebook():
            beta_schedule_iter = mo.status.progress_bar(
                self.beta_schedule,
                title=f"Optimizing portfolio",
                subtitle=f"with alpha = {alpha:.2e}",
                remove_on_exit=True,
            )
        else:
            beta_schedule_iter = tqdm(
                self.beta_schedule,
                desc=f"Optimizing portfolio with alpha = {alpha:.2e}",
                leave=False,
            )

        for beta in beta_schedule_iter:
            self._step(alpha, beta)
            if self._current_score > best_score:
                best_portfolio = self._current_portfolio
                best_score = self._current_score
        self.best_portfolios.append(best_portfolio)
        return

    def full_run(self):
        # progress bar
        if mo.running_in_notebook():
            alpha_schedule_iter = mo.status.progress_bar(
                self.alpha_schedule, title="Collecting optimal portfolios"
            )
        else:
            alpha_schedule_iter = tqdm(
                self.alpha_schedule, desc="Collecting optimal portfolios"
            )

        for alpha in alpha_schedule_iter:
            self.run_fixed_alpha(alpha)

        return
