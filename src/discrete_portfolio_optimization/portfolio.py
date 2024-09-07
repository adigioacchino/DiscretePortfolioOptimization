import numpy as np
import pandas as pd


class Portfolio:
    def __init__(
        self,
        current_prices: list[float],
        tot_value: float | None = None,
        allocations: list[float] | None = None,
        cash_value: float = 0,
        seed: int | None = None,
    ):
        self.current_prices = np.array(current_prices)
        self.num_assets = len(current_prices)

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
        elif allocations is None:
            self.tot_value = tot_value
            self.allocations = self._random_allocations()
        elif tot_value is None:
            self.allocations = np.array(allocations)
            self.tot_value = self.current_prices @ self.allocations + cash_value
        self._sync_values()  # self.asset_value, self.cash_value
        self._sync_weights()  # self.weights

    def _sync_values(self):
        self.asset_value = self.current_prices @ self.allocations
        self.cash_value = self.tot_value - self.asset_value

    def _sync_weights(self):
        self.weights = np.array(
            [
                self.current_prices[i] * self.allocations[i] / self.asset_value
                for i in range(self.num_assets)
            ]
        )

    def _random_allocations(self):
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
        return np.array(allocations)

    def random_move(self):
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
            num2 = round(price1 / price2)
        else:
            num1 = round(price2 / price1)
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
        if account_for_cash:
            # cash has no return
            allocated_frac = (self.tot_value - self.cash_value) / self.tot_value
            return np.sum(returns_df.mean() * self.weights) * allocated_frac
        else:
            return np.sum(returns_df.mean() * self.weights)

    def get_day_volatility(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> float:
        cov_matrix = returns_df.cov()
        if account_for_cash:
            # cash has no volatility
            allocated_frac = (self.tot_value - self.cash_value) / self.tot_value
            return self.weights.T @ np.sqrt(cov_matrix) @ self.weights * allocated_frac
        else:
            return self.weights.T @ np.sqrt(cov_matrix) @ self.weights

    def get_sharpe(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> float:
        yret = self.get_day_return(returns_df, account_for_cash)
        ycov = self.get_day_volatility(returns_df, account_for_cash)
        return yret / ycov

    def portfolio_metrics(
        self, returns_df: pd.DataFrame, account_for_cash: bool = True
    ) -> dict:
        ret = self.get_day_return(returns_df, account_for_cash)
        vol = self.get_day_volatility(returns_df, account_for_cash)
        return {
            "Return": ret,
            "Volatility": vol,
            "Sharpe Ratio": ret / vol,
        }

    def copy(self):
        return Portfolio(
            self.current_prices,
            cash_value=self.cash_value,
            allocations=self.allocations.copy(),
        )

    def __str__(self):
        return (
            f"Portfolio with total value of {round(self.tot_value, ndigits=1)}"
            f" (of which {round(self.asset_value, ndigits=1)} in assets and {round(self.cash_value, ndigits=1)} cash)"
            f" and allocations {self.allocations}."
        )

    def __repr__(self):
        return self.__str__()
