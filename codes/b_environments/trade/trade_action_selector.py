import numpy as np
import random

from .trade_constant import Action, TimeUnit
from codes.e_utils.actions import ActionSelector


class TradeActionSelector(ActionSelector):
    def __init__(self, env=None):
        self.env = env

    def __call__(self, q_values):
        # if self.env.step_idx == (335 if self.env.time_unit == TimeUnit.ONE_HOUR else 13):
        #     return np.array([Action.MARKET_SELL.value] * len(q_values))
        # else:
        #     return None
        return None


class ArgmaxTradeActionSelector(TradeActionSelector):
    """
    Selects actions using argmax
    """
    def __init__(self, env=None):
        super(ArgmaxTradeActionSelector, self).__init__(env)

    def __call__(self, q_values):
        actions = super(q_values)
        if actions is None:
            if self.env.hold_coin_quantity == 0.0:
                q_values[:, Action.MARKET_SELL.value] = np.nan
            actions = np.nanargmax(q_values, axis=1)
        return actions


class EpsilonGreedyTradeDQNActionSelector(TradeActionSelector):
    def __init__(self, epsilon=0.05, env=None):
        super(EpsilonGreedyTradeDQNActionSelector, self).__init__(env)
        self.epsilon = epsilon
        self.default_action_selector = ArgmaxTradeActionSelector(env=env)

    def __call__(self, q_values):
        actions = super(q_values)
        if actions is None:
            if random.random() < self.epsilon:
                if self.env.hold_coin_quantity == 0.0:
                    actions = np.array(
                        [random.choice([Action.HOLD.value, Action.MARKET_BUY.value])] * len(q_values)
                    )
                else:
                    actions = np.array(
                        [random.choice([Action.HOLD.value, Action.MARKET_BUY.value, Action.MARKET_SELL.value])] * len(q_values)
                    )
            else:
                if self.env.hold_coin_quantity == 0.0:
                    q_values[:, Action.MARKET_SELL.value] = np.nan
                actions = np.nanargmax(q_values, axis=1)

        return actions


class RandomTradeDQNActionSelector(TradeActionSelector):
    def __init__(self, env=None):
        super(RandomTradeDQNActionSelector, self).__init__(env)

    def __call__(self, q_values):
        actions = super(q_values)
        if actions is None:
            if self.env.hold_coin_quantity == 0.0:
                actions = np.array(
                    [random.choice([Action.HOLD.value, Action.MARKET_BUY.value])] * len(q_values)
                )
            else:
                actions = np.array(
                    [random.choice([Action.HOLD.value, Action.MARKET_BUY.value, Action.MARKET_SELL.value])] * len(q_values)
                )

        return actions