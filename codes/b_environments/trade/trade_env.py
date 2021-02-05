import math
import random
import gym
import numpy as np


from .trade_constant import TradeEnvironmentType, WINDOW_SIZE, TimeUnit, INITIAL_TOTAL_KRW, Action, BUY_AMOUNT, COMMISSION_RATE, SLIPPAGE_COUNT
from .trade_utils import get_history_entry, get_order_unit, get_previous_one_unit_date_time


class UpbitEnvironment(gym.Env):
    def __init__(self, coin_name, time_unit, data_info, environment_type=TradeEnvironmentType.TRAIN):
        super(UpbitEnvironment, self).__init__()
        self.coin_name = coin_name
        self.time_unit = time_unit
        self.environment_type = environment_type

        if self.environment_type == TradeEnvironmentType.LIVE:
            self.previous_one_datetime = get_previous_one_unit_date_time(time_unit)

        self.history = None
        self.input_size = (2, WINDOW_SIZE + 1, 6)

        self.data = data_info["data"]
        self.state_data = data_info["state_data"]
        self.first_datetime_krw = data_info["first_datetime_krw"]
        self.last_datetime_krw = data_info["last_datetime_krw"]

        self.data_size = len(self.data)

        print("DATA SIZE: {0}".format(self.data_size))
        print("FIRST DATETIME KRW: {0}".format(self.first_datetime_krw))
        print("LAST DATETIME KRW: {0}".format(self.last_datetime_krw))

        # [NOTE] 전체 데이터에서 마지막 데이터는 단지 open 가격을 가져오는 용도로만 사용하며,
        # [NOTE] 다음 행동을 추출하기 위한 주요 데이터로 활용하지 않음.
        self.data_size = self.data_size - 1

        ############## [NOTE] Transaction 시작 인덱스 ################
        self.transaction_state_idx = None
        self.transaction_start_datetime = None
        ###########################################################

        self.observation_space = gym.spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=self.input_size,
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(Action))

        ##### STATISTICS: BEGIN #####
        self.hold_coin_quantity = None
        self.profit = None
        self.position_value = None
        self.hold_coin_krw = None
        self.sold_coin_quantity = None
        self.sold_profit = None
        self.balance = None
        self.positions = None
        ##### STATISTICS: END   #####

        self.step_idx = None
        self.action_count = np.zeros(shape=self.action_space.n)

    def reset(self):
        ##### STATISTICS: BEGIN #####
        self.hold_coin_quantity = 0.0
        self.profit = 0.0
        self.position_value = 0.0
        self.hold_coin_krw = 0.0
        self.sold_coin_quantity = 0.0
        self.sold_profit = 0.0
        self.balance = 0.0
        self.positions = []
        ##### STATISTICS: END   #####

        if self.environment_type in [TradeEnvironmentType.TRAIN, TradeEnvironmentType.TEST_RANDOM, TradeEnvironmentType.TEST_SEQUENTIAL]:
            self.balance = INITIAL_TOTAL_KRW
        elif self.environment_type == TradeEnvironmentType.LIVE:
            self.balance = 0
        else:
            raise ValueError()

        self.history = []

        if self.environment_type == TradeEnvironmentType.TRAIN:
            self.transaction_state_idx = random.randint(
                a=WINDOW_SIZE - 1,
                b=self.data_size - (336 if self.time_unit == TimeUnit.ONE_HOUR else 14)
            )
        elif self.environment_type == TradeEnvironmentType.TEST_RANDOM:
            self.transaction_state_idx = random.randint(
                a=0,
                b=self.data_size - (336 if self.time_unit == TimeUnit.ONE_HOUR else 14)
            )
        elif self.environment_type == TradeEnvironmentType.TEST_SEQUENTIAL:
            assert self.transaction_state_idx >= 0
        elif self.environment_type == TradeEnvironmentType.LIVE:
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

        self.step_idx = 0

        initial_state = self.get_state(hold_coin_krw=0.0, position_value=0.0, history=self.history)

        return initial_state  # obs

    def step(self, action):
        self.step_idx += 1
        self.action_count[action] += 1

        candle_data = self.data.iloc[self.transaction_state_idx, :]

        # [NOTE] current_trade_data["trade_price"]는 다음 타임 유닛의 open 가격으로 결정
        current_trade_data = dict()
        current_trade_data["trade_price"] = self.data.iloc[self.transaction_state_idx + 1]["open"]

        effective_action = False

        if action == Action.HOLD.value:
            # Action이 HOLD이면 현 시점에서 모든 코인을 매도한다고 가정할 때의 정보를 transaction_info에 담음
            transaction_info = self.get_transaction_sell_info(
                current_trade_data=current_trade_data, num_buys=len(self.positions)
            )

            if len(self.positions) > 0:
                reward = -0.01
            else:
                reward = 0.0

            effective_action = False
        elif action == Action.MARKET_BUY.value:
            transaction_info = self.get_transaction_buy_info(
                current_trade_data=current_trade_data
            )

            if len(self.positions) < 10:
                self.positions.append(transaction_info['coin_krw'] + transaction_info['commission_fee'])
                self.hold_coin_krw += transaction_info['coin_krw']
                self.hold_coin_quantity += transaction_info['coin_quantity']
                self.balance = self.balance - (transaction_info['coin_krw'] + transaction_info['commission_fee'])
                effective_action = True

            reward = 0.0
        elif action == Action.MARKET_SELL.value:
            transaction_info = self.get_transaction_sell_info(
                current_trade_data=current_trade_data, num_buys=len(self.positions)
            )

            if len(self.positions) > 0:
                sum_position = 0.0
                for p in self.positions:
                    sum_position += p

                self.sold_profit = transaction_info["coin_krw"] - sum_position

                self.profit += self.sold_profit
                self.positions.clear()

                self.balance += transaction_info["coin_krw"]
                self.sold_coin_quantity = self.hold_coin_quantity
                self.hold_coin_quantity = 0.0
                self.hold_coin_krw = 0.0
                effective_action = True

                reward = 100 * self.sold_profit / INITIAL_TOTAL_KRW
            else:
                reward = 0.0
        else:
            raise ValueError()

        reward = max(0.0, reward)

        done_conditions = [
            action == Action.MARKET_SELL.value,
            self.step_idx == (336 if self.time_unit == TimeUnit.ONE_HOUR else 14),
            self.transaction_state_idx >= self.data_size - 1
        ]

        if self.environment_type in [TradeEnvironmentType.TRAIN, TradeEnvironmentType.TEST_RANDOM, TradeEnvironmentType.TEST_SEQUENTIAL]:
            self.transaction_state_idx += 1
            if any(done_conditions):
                done = True
                next_state = None
            else:
                done = False

                self.position_value = self.get_position_value(current_trade_data)

                self.history.pop(0)
                state_one_data = self.state_data[self.transaction_state_idx]
                state_previous_one_data = self.state_data[self.transaction_state_idx - 1]
                self.history.append(
                    get_history_entry(state_one_data, state_previous_one_data)
                )
                next_state = self.get_state(self.hold_coin_krw, self.position_value, self.history)
        elif self.environment_type == TradeEnvironmentType.LIVE:
            self.transaction_state_idx += 1
            done = True
            next_state = None
        else:
            raise ValueError()

        info = self.get_info(action, effective_action, candle_data, transaction_info)

        return next_state, reward, done, info

    def render(self, mode='human'):
        pass

    def get_position_value(self, current_trade_data):
        # 포지션에 따른 가치 --> position_value
        # 현 시점에 매도하였을 때 이득을 얻는다면 position_value > 0
        if len(self.positions) > 0:
            sum_position = 0.0
            for p in self.positions:
                sum_position += p

            slippage = get_order_unit(current_trade_data['trade_price']) * SLIPPAGE_COUNT * math.ceil(len(self.positions) / 3)
            coin_unit_price = current_trade_data['trade_price'] - slippage
            commission_fee = self.hold_coin_quantity * coin_unit_price * COMMISSION_RATE
            coin_krw = self.hold_coin_quantity * coin_unit_price - commission_fee
            return coin_krw - sum_position
        else:
            return 0.0

    def get_state(self, hold_coin_krw, position_value, history):
        next_state = np.concatenate((
            np.array([[
                [hold_coin_krw, hold_coin_krw],
                [hold_coin_krw, hold_coin_krw],
                [hold_coin_krw, hold_coin_krw],
                [position_value, position_value],
                [position_value, position_value],
                [position_value, position_value]
            ]], dtype=np.float32) / (INITIAL_TOTAL_KRW * 10),
            history
        ))

        next_state = np.transpose(next_state, (2, 0, 1))

        return next_state

    def get_transaction_buy_info(self, current_trade_data):
        if self.environment_type in [TradeEnvironmentType.TRAIN, TradeEnvironmentType.TEST_RANDOM, TradeEnvironmentType.TEST_SEQUENTIAL]:
            commission_fee = BUY_AMOUNT * COMMISSION_RATE
            coin_krw = BUY_AMOUNT - commission_fee
            slippage = get_order_unit(current_trade_data['trade_price']) * SLIPPAGE_COUNT
            coin_unit_price = current_trade_data['trade_price'] + slippage
            coin_quantity = coin_krw / coin_unit_price
        elif self.environment_type == TradeEnvironmentType.LIVE:
            commission_fee = current_trade_data['commission_fee']
            coin_krw = current_trade_data['bought_coin_krw']
            slippage = current_trade_data['bought_coin_unit_price'] - current_trade_data['trade_price']
            coin_unit_price = current_trade_data['bought_coin_unit_price']
            coin_quantity = current_trade_data['bought_coin_quantity']
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

    def get_transaction_sell_info(self, current_trade_data, num_buys=0):
        if self.environment_type in [TradeEnvironmentType.TRAIN, TradeEnvironmentType.TEST_RANDOM, TradeEnvironmentType.TEST_SEQUENTIAL]:
            slippage = get_order_unit(current_trade_data['trade_price']) * SLIPPAGE_COUNT * math.ceil(num_buys / 3)
            coin_unit_price = current_trade_data['trade_price'] - slippage
            commission_fee = self.hold_coin_quantity * coin_unit_price * COMMISSION_RATE
            coin_krw = self.hold_coin_quantity * coin_unit_price - commission_fee
            coin_quantity = self.hold_coin_quantity
        elif self.environment_type == TradeEnvironmentType.LIVE:
            slippage = current_trade_data['trade_price'] - current_trade_data['sold_coin_unit_price']
            coin_unit_price = current_trade_data['sold_coin_unit_price']
            commission_fee = current_trade_data['commission_fee']
            coin_krw = current_trade_data['total_sold_krw']
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

    def get_info(self, action, effective_action, candle_data, transaction_info):
        info = {
            "datetime_krw": candle_data['datetime_krw'],
            "action": action,
            "effective_action": effective_action,
            "hold_coin": self.hold_coin_quantity,
            "balance": self.balance,
            "profit": self.profit,
            "close_price": candle_data['final'],
            "open_price": candle_data['open'],
            "action_count": self.action_count
        }

        if action == Action.MARKET_SELL.value:
            info["sold_coin"] = self.sold_coin_quantity
            info["sold_profit"] = self.sold_profit

        info = {**info, **transaction_info}
        return info

    def get_action_meanings(self):
        return ["HOLD", "BUY", "SELL"]


if __name__ == "__main__":
    pass
    # train_env = UpbitEnvironment(
    #     coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=TradeEnvironmentType.TRAIN
    # )
    #
    # test_env = UpbitEnvironment(
    #     coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=TradeEnvironmentType.TEST
    # )
    #
    # live_env = UpbitEnvironment(
    #     coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=TradeEnvironmentType.LIVE,
    #     previous_one_datetime=get_previous_one_unit_date_time(TimeUnit.ONE_HOUR)
    # )

    # print(train_env.data)
    # print(train_env.state_data)
    # print(train_env.state_data.shape)
    #
    # state = train_env.reset()
    # print(state.shape)
    #
    # state = test_env.reset()
    # print(state.shape)
    #
    # state = live_env.reset()
    # print(state.shape)
