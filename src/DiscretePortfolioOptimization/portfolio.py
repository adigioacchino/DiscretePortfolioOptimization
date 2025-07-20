from typing import List, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd


class Portfolio:
    """
    Class to represent a portfolio of assets.

    Attributes:
        current_prices: list[float], the current prices of the assets
        tot_value: float, the total value of the portfolio
        allocations: list[float], the number of shares of each asset
        cash_value: float, the cash value of the portfolio
        seed: int, the seed for the random number generator
    """

    current_prices: NDArray[np.floating]
    allocations: NDArray[np.floating]
    weights: NDArray[np.floating]

    def __init__(
        self,
        current_prices: Union[List[float], NDArray[np.floating]],
        tot_value: Optional[float] = None,
        allocations: Optional[Union[List[float], NDArray[np.floating]]] = None,
        cash_value: float = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Portfolio object using either the total value of the portfolio
        (and random allocations) or each allocation (the total value will be computed).

        Args:
            current_prices: list[float], the current prices of the assets
            tot_value: float, the total value of the portfolio
            allocations: list[float], the number of shares of each asset
            cash_value: float, the cash value of the portfolio
            seed: int, the seed for the random number generator
        """
        self.current_prices = np.array(current_prices, dtype=float)
        self.num_assets = len(current_prices)
        self.eta: Optional[float] = (
            None  # eta is not used in this class, but it is used in the optimization class
        )
        self.asset_value: float
        self.tot_value: float
        self.cash_value: float

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        if (allocations is not None) and (tot_value is not None):
            raise RuntimeError(
                "Refusing to create a portfolio with both allocations and total value."
                "Provide allocations only and the total value will be computed."
            )
        if (allocations is None) and (tot_value is None):
            raise RuntimeError(
                "Cannot create a portfolio without `allocations` or (`tot_value`)."
            )
        elif allocations is None and tot_value is not None:
            self.tot_value = tot_value
            self.allocations = self._random_allocations()
        elif tot_value is None and allocations is not None:
            self.allocations = np.array(allocations, dtype=float)
            self.tot_value = float(self.current_prices @ self.allocations + cash_value)

        self._sync_values()  # self.asset_value, self.cash_value
        self._sync_weights()  # self.weights

    def _sync_values(self) -> None:
        """
        Synchronize the asset and cash values with the current prices and allocations.
        """
        self.asset_value = float(self.current_prices @ self.allocations)
        self.cash_value = self.tot_value - self.asset_value

    def _sync_weights(self) -> None:
        """
        Synchronize the weights with the current prices and allocations.
        """
        self.weights = np.array(
            [
                self.current_prices[i] * self.allocations[i] / self.asset_value
                for i in range(self.num_assets)
            ]
        )

    def _random_allocations(self) -> NDArray[np.floating]:
        """
        Generate random allocations for the assets.
        """
        curr_value = 0
        allocations = [0 for _ in range(self.num_assets)]
        available_poss = np.arange(self.num_assets)
        while len(available_poss) != 0:
            t_pos = self.rng.choice(available_poss)
            t_value = self.current_prices[t_pos]
            if curr_value + t_value <= self.tot_value:
                curr_value = curr_value + t_value
                allocations[t_pos] += 1
            else:
                available_poss = available_poss[available_poss != t_pos]
        return np.array(allocations, dtype=float)

    def random_move(self) -> None:
        """
        Perform a random move in the portfolio changing the allocations of two assets.
        The two assets are chosen randomly and the move is the smallest possible in terms of value.
        """
        # determine two random assets
        available_poss = np.arange(self.num_assets)
        pos1 = self.rng.choice(available_poss)
        price1 = self.current_prices[pos1]
        available_poss = available_poss[available_poss != pos1]
        pos2 = self.rng.choice(available_poss)
        price2 = self.current_prices[pos2]

        # find smallest (in term of size) move
        largest_value = np.argmax([price1, price2])
        if largest_value == 0:
            num1 = 1
            num2 = np.round(price1 / price2)
        else:
            num1 = np.round(price2 / price1)
            num2 = 1

        # determine the direction of the move
        first_2_second = self.rng.choice([True, False])
        move_done = False
        if first_2_second:
            if ((num2 * price2 - num1 * price1) <= self.cash_value) and (
                self.allocations[pos1] - num1 >= 0
            ):
                self.allocations[pos1] -= num1
                self.allocations[pos2] += num2
                move_done = True
        else:
            if ((num1 * price1 - num2 * price2) <= self.cash_value) and (
                self.allocations[pos2] - num2 >= 0
            ):
                self.allocations[pos1] += num1
                self.allocations[pos2] -= num2
                move_done = True
        if not move_done:
            if not first_2_second:
                if ((num2 * price2 - num1 * price1) <= self.cash_value) and (
                    self.allocations[pos1] - num1 >= 0
                ):
                    self.allocations[pos1] -= num1
                    self.allocations[pos2] += num2
            else:
                if ((num1 * price1 - num2 * price2) <= self.cash_value) and (
                    self.allocations[pos2] - num2 >= 0
                ):
                    self.allocations[pos1] += num1
                    self.allocations[pos2] -= num2
        self._sync_values()
        self._sync_weights()
        return

    def get_day_return(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> float:
        """
        Compute the return of the portfolio for a day.
        If `account_for_cash` is True, the return is multiplied by the fraction of the total value
        that is allocated to assets.

        Args:
            returns_df: pd.DataFrame, the returns of the assets
            account_for_cash: bool, whether to account for the cash in the portfolio

        Returns:
            float, the return of the portfolio

        """
        if account_for_cash:
            # cash has no return
            allocated_frac = (self.tot_value - self.cash_value) / self.tot_value
            return np.sum(returns_df.mean() * self.weights) * allocated_frac
        else:
            return np.sum(returns_df.mean() * self.weights)

    def get_day_volatility(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> float:
        """
        Compute the volatility of the portfolio for a day.
        The volatility is computed as the weighted sum of the square root of the covariances of the assets.
        If `account_for_cash` is True, the volatility is multiplied by the fraction of the total value
        that is allocated to assets.

        Args:
            returns_df: pd.DataFrame, the returns of the assets
            account_for_cash: bool, whether to account for the cash in the portfolio

        Returns:
            float, the volatility of the portfolio
        """
        cov_matrix = returns_df.cov()
        if account_for_cash:
            # cash has no volatility
            allocated_frac = (self.tot_value - self.cash_value) / self.tot_value
            return float(
                # self.weights.T @ np.sqrt(cov_matrix) @ self.weights * allocated_frac
                np.sqrt(self.weights.T @ (cov_matrix) @ self.weights) * allocated_frac
            )
        else:
            # return float(self.weights.T @ np.sqrt(cov_matrix) @ self.weights)
            return float(np.sqrt(self.weights.T @ (cov_matrix) @ self.weights))

    def get_sharpe(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> float:
        """
        Compute the Sharpe ratio of the portfolio.
        If `account_for_cash` is True, the Sharpe ratio is computed using the fraction of the total value
        that is allocated to assets.

        Args:
            returns_df: pd.DataFrame, the returns of the assets
            account_for_cash: bool, whether to account for the cash in the portfolio

        Returns:
            float, the Sharpe ratio of the portfolio
        """
        yret = self.get_day_return(returns_df, account_for_cash)
        ycov = self.get_day_volatility(returns_df, account_for_cash)
        return yret / ycov

    def portfolio_metrics(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> Dict[str, float]:
        """
        Compute the return, volatility, and Sharpe ratio of the portfolio.
        If `account_for_cash` is True, the metrics are computed using the fraction of the total value
        that is allocated to assets.

        Args:
            returns_df: pd.DataFrame, the returns of the assets
            account_for_cash: bool, whether to account for the cash in the portfolio

        Returns:
            dict, a dictionary with the return, volatility, and Sharpe ratio of the portfolio
        """
        ret = self.get_day_return(returns_df, account_for_cash)
        vol = self.get_day_volatility(returns_df, account_for_cash)
        return {
            "Return": ret,
            "Volatility": vol,
            "Sharpe Ratio": ret / vol,
        }

    def copy(self) -> "Portfolio":
        """
        Return a copy of the Portfolio object.
        """
        # Convert numpy arrays to lists and explicitly cast them for type safety
        # current_prices_list: List[float] = [float(x) for x in self.current_prices]
        # allocations_list: List[float] = [float(x) for x in self.allocations]

        return Portfolio(
            self.current_prices,
            None,
            self.allocations,
            self.cash_value,
        )

    def __str__(self) -> str:
        return (
            f"Portfolio with total value of {round(float(self.tot_value), ndigits=1)}"
            f" (of which {round(self.asset_value, ndigits=1)} in assets and {round(self.cash_value, ndigits=1)} cash)"
            f" and allocations {self.allocations}."
        )

    def __repr__(self) -> str:
        return self.__str__()
