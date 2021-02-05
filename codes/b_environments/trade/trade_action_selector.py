import numpy as np
import random

from codes.e_utils.actions import ActionSelector
from .trade_constant import Action, TimeUnit


class ArgmaxTradeActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __init__(self, env=None):
        self.env = env

    def __call__(self, q_values):
        if self.env.step_idx == (335 if self.env.time_unit == TimeUnit.ONE_HOUR else 13):
            return np.array([Action.MARKET_SELL.value] * len(q_values))
        else:
            if self.env.hold_coin_quantity == 0.0:
                q_values[:, Action.MARKET_SELL.value] = np.nan
            return np.nanargmax(q_values, axis=1)


class EpsilonGreedyTradeDQNActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, env=None):
        self.epsilon = epsilon
        self.env = env
        self.default_action_selector = ArgmaxTradeActionSelector(env=env)

    def __call__(self, q_values):
        if self.env.step_idx == (335 if self.env.time_unit == TimeUnit.ONE_HOUR else 13):
            actions = np.array([Action.MARKET_SELL.value] * len(q_values))
        else:
            if random.random() < self.epsilon:
                if self.env.hold_coin_quantity == 0.0:
                    return np.array(
                        [random.choice([Action.HOLD.value, Action.MARKET_BUY.value])] * len(q_values)
                    )
                else:
                    return np.array(
                        [random.choice([Action.HOLD.value, Action.MARKET_BUY.value, Action.MARKET_SELL.value])] * len(q_values)
                    )
            else:
                actions = self.default_action_selector(q_values)

        return actions


class RandomTradeDQNActionSelector(ActionSelector):
    def __init__(self, env=None):
        self.env = env

    def __call__(self, q_values):
        if self.env.step_idx == (335 if self.env.time_unit == TimeUnit.ONE_HOUR else 13):
            actions = np.array([Action.MARKET_SELL.value] * len(q_values))
        else:
            if self.env.hold_coin_quantity == 0.0:
                return np.array(
                    [random.choice([Action.HOLD.value, Action.MARKET_BUY.value])] * len(q_values)
                )
            else:
                return np.array(
                    [random.choice([Action.HOLD.value, Action.MARKET_BUY.value, Action.MARKET_SELL.value])] * len(q_values)
                )

        return actions