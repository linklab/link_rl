import random

import gym
import numpy as np


from common.environments.trade.trade_constant import EnvironmentType, WINDOW_SIZE, TimeUnit, MAX_BUY_SIZE, \
    INITIAL_TOTAL_KRW, Action, BUY_AMOUNT, COMMISSION_RATE, SLIPPAGE_COUNT
from common.environments.trade.trade_data import get_data
from common.environments.trade.trade_utils import get_history_entry, get_order_unit, get_previous_one_unit_date_time


class UpbitEnvironment(gym.Env):
    def __init__(self, coin_name, time_unit, environment_type=EnvironmentType.TRAIN, previous_one_datetime=None):
        super(UpbitEnvironment, self).__init__()
        self.coin_name = coin_name
        self.time_unit = time_unit
        self.environment_type = environment_type
        self.previous_one_datetime = previous_one_datetime

        if self.environment_type == EnvironmentType.LIVE:
            assert self.previous_one_datetime

        self.history = None
        self.input_size = (2, WINDOW_SIZE + 1, 6)

        self.data, self.state_data, self.last_data_datetime_krw = get_data(self.coin_name, self.time_unit)
        self.data_size = len(self.data)

        self.transaction_state_idx = None
        self.transaction_start_datetime = None

        self.observation_space = gym.spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=self.input_size,
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(Action))

        ##### STATISTICS: BEGIN #####
        self.hold_coin_quantity = None
        self.profits = None
        self.position_value = None
        self.hold_coin_krw = None
        self.sold_coin_quantity = None
        self.sold_profit = None
        self.balance = None
        self.positions = None
        ##### STATISTICS: END   #####

    def reset(self):
        ##### STATISTICS: BEGIN #####
        self.hold_coin_quantity = 0.0
        self.profits = 0.0
        self.position_value = 0.0
        self.hold_coin_krw = 0.0
        self.sold_coin_quantity = 0.0
        self.sold_profit = 0.0
        self.balance = 0.0
        self.positions = []
        ##### STATISTICS: END   #####

        if self.environment_type in [EnvironmentType.TRAIN, EnvironmentType.TEST]:
            self.balance = INITIAL_TOTAL_KRW
        elif self.environment_type == EnvironmentType.LIVE:
            self.balance = 0
        else:
            raise ValueError()

        self.history = []

        if self.environment_type in [EnvironmentType.TRAIN, EnvironmentType.TEST]:
            self.transaction_state_idx = random.randint(
                a=WINDOW_SIZE, b=self.data_size - WINDOW_SIZE - (MAX_BUY_SIZE + 1)
            )
        elif self.environment_type == EnvironmentType.LIVE:
            self.transaction_state_idx = self.data_size - 1
        else:
            raise ValueError()

        self.transaction_start_datetime = self.data.iloc[self.transaction_state_idx]['datetime_krw']
        state_idx = self.transaction_state_idx - WINDOW_SIZE + 1

        for idx in range(WINDOW_SIZE):
            state_one_data = self.state_data[state_idx + idx]
            state_previous_one_data = self.state_data[state_idx + idx - 1]
            self.history.append(
                get_history_entry(state_one_data, state_previous_one_data)
            )

        assert state_idx + idx == self.transaction_state_idx, (state_idx + idx, self.transaction_state_idx)

        initial_state = self.get_state(0.0, 0.0, self.history)

        return initial_state  # obs

    def step(self, action):
        data = self.data.iloc[self.transaction_state_idx, :]
        effective_action = False

        if action == Action.HOLD.value:
            # Action이 HOLD이면 현 시점에서 모든 코인을 매도한다고 가정할 때의 정보를 transaction_info에 담음
            transaction_info = self.get_transaction_sell_info(data=data, num_buys=len(self.positions))
            reward = 0.0
        elif action == Action.MARKET_BUY.value:
            transaction_info = self.get_transaction_buy_info(data=data)

            if len(self.positions) < 10:
                self.positions.append(transaction_info['coin_krw'] + transaction_info['commission_fee'])
                self.hold_coin_krw += transaction_info['coin_krw']
                self.hold_coin_quantity += transaction_info['coin_quantity']
                self.balance = self.balance - (transaction_info['coin_krw'] + transaction_info['commission_fee'])
                effective_action = True

            reward = 0.0
        elif action == Action.MARKET_SELL.value:
            transaction_info = self.get_transaction_sell_info(data=data, num_buys=len(self.positions))

            if len(self.positions) > 0:
                sum_position = 0.0
                for p in self.positions:
                    sum_position += p

                self.sold_profit = transaction_info["coin_krw"] - sum_position

                self.profits += self.sold_profit
                self.positions.clear()

                self.balance += transaction_info["coin_krw"]
                self.sold_coin_quantity = self.hold_coin_quantity
                self.hold_coin_quantity = 0
                self.hold_coin_krw = 0
                effective_action = True
                reward = self.sold_profit / 1000000
            else:
                reward = 0.0
        else:
            raise ValueError()

        done_conditions = [
            action == Action.MARKET_SELL,
            data['datetime_krw'] == self.last_data_datetime_krw
        ]

        if self.environment_type in [EnvironmentType.TRAIN, EnvironmentType.TEST]:
            if any(done_conditions):
                done = True
                next_state = None
            else:
                done = False

                self.position_value = self.get_position_value(data)

                self.history.pop(0)
                state_one_data = self.state_data[self.transaction_state_idx]
                state_previous_one_data = self.state_data[self.transaction_state_idx - 1]
                self.history.append(
                    get_history_entry(state_one_data, state_previous_one_data)
                )
                next_state = self.get_state(self.hold_coin_krw, self.position_value, self.history)

            info = self.get_info(action, effective_action, data, transaction_info)
            self.transaction_state_idx += 1

        elif self.environment_type == EnvironmentType.LIVE:
            done = True
            next_state = None
            info = self.get_info(action, effective_action, data, transaction_info)
        else:
            raise ValueError()

        return next_state, reward, done, info

    def render(self, mode='human'):
        pass

    def get_position_value(self, data):
        # 포지션에 따른 가치 --> position_value
        # 현 시점에 매도하였을 때 이득을 얻는다면 position_value > 0
        if len(self.positions) > 0:
            sum_position = 0.0
            for p in self.positions:
                sum_position += p

            slippage = get_order_unit(data['final']) * SLIPPAGE_COUNT * len(self.positions)
            coin_unit_price = data['open'] - slippage
            commission_fee = self.hold_coin_quantity * coin_unit_price * COMMISSION_RATE
            return self.hold_coin_quantity * coin_unit_price - commission_fee - sum_position
        else:
            return 0.0

    def get_state(self, hold_coin_krw, position_value, np_history):
        next_state = np.concatenate((
            np.array([[
                [hold_coin_krw, hold_coin_krw],
                [hold_coin_krw, hold_coin_krw],
                [hold_coin_krw, hold_coin_krw],
                [position_value, position_value],
                [position_value, position_value],
                [position_value, position_value]
            ]], dtype=np.float32) / (INITIAL_TOTAL_KRW * 10),
            np_history
        ))

        next_state = np.transpose(next_state, (2, 0, 1))

        return next_state

    def get_transaction_buy_info(self, data):
        if self.environment_type in [EnvironmentType.TRAIN, EnvironmentType.TEST]:
            commission_fee = BUY_AMOUNT * COMMISSION_RATE
            coin_krw = BUY_AMOUNT - commission_fee
            slippage = get_order_unit(data['final']) * SLIPPAGE_COUNT
            coin_unit_price = data['final'] + slippage
            coin_quantity = coin_krw / coin_unit_price
        elif self.environment_type == EnvironmentType.LIVE:
            commission_fee = data['commission_fee']
            coin_krw = data['bought_coin_krw']
            slippage = data['bought_coin_unit_price'] - data['final']
            coin_unit_price = data['bought_coin_unit_price']
            coin_quantity = data['bought_coin_quantity']
        else:
            raise ValueError()

        transaction_info = {
            "slippage": slippage,
            "coin_krw": coin_krw,
            "coin_unit_price": coin_unit_price,
            "coin_quantity": coin_quantity,
            "commission_fee": commission_fee
        }
        return transaction_info

    def get_transaction_sell_info(self, data, num_buys=0):
        if self.environment_type in [EnvironmentType.TRAIN, EnvironmentType.TEST]:
            slippage = get_order_unit(data['final']) * SLIPPAGE_COUNT * num_buys
            coin_unit_price = data['final'] - slippage
            commission_fee = self.hold_coin_quantity * coin_unit_price * COMMISSION_RATE
            coin_krw = self.hold_coin_quantity * coin_unit_price - commission_fee
            coin_quantity = self.hold_coin_quantity
        elif self.environment_type == EnvironmentType.LIVE:
            slippage = data['final'] - data['sold_coin_unit_price']
            coin_unit_price = data['sold_coin_unit_price']
            commission_fee = data['commission_fee']
            coin_krw = data['total_sold_krw']
            coin_quantity = 0.0
        else:
            raise ValueError()

        transaction_info = {
            "slippage": slippage,
            "coin_krw": coin_krw,
            "coin_unit_price": coin_unit_price,
            "coin_quantity": coin_quantity,
            "commission_fee": commission_fee
        }
        return transaction_info

    def get_info(self, action, effective_action, data, transaction_info):
        if action in [Action.HOLD.value, Action.MARKET_BUY.value]:
            info = {
                "datetime_krw": data['datetime_krw'],
                "action": action,
                "effective_action": effective_action,
                "close_price": data['final'],
                "open_price": data['open'],
                "hold_coin": self.hold_coin_quantity,
                "balance": self.balance
            }
        elif action == Action.MARKET_SELL.value:
            info = {
                "datetime_krw": data['datetime_krw'],
                "action": action,
                "effective_action": effective_action,
                "close_price": data['final'],
                "open_price": data['open'],
                "hold_coin": self.hold_coin_quantity,
                "sold_coin": self.sold_coin_quantity,
                "sold_profit": self.sold_profit,
                "balance": self.balance
            }
        else:
            raise ValueError()

        info = {**info, **transaction_info}
        return info

    def get_action_meanings(self):
        return [Action.HOLD, Action.MARKET_BUY, Action.MARKET_SELL]


if __name__ == "__main__":
    train_env = UpbitEnvironment(
        coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=EnvironmentType.TRAIN
    )

    test_env = UpbitEnvironment(
        coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=EnvironmentType.TEST
    )

    live_env = UpbitEnvironment(
        coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=EnvironmentType.LIVE,
        previous_one_datetime=get_previous_one_unit_date_time(TimeUnit.ONE_HOUR)
    )

    # print(train_env.data)
    # print(train_env.state_data)
    # print(train_env.state_data.shape)

    state = train_env.reset()
    print(state.shape)

    state = test_env.reset()
    print(state.shape)

    state = live_env.reset()
    print(state.shape)
