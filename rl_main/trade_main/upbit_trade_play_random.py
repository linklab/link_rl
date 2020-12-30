import torch
import warnings
import sys, os

from common.environments.trade.trade_action_selector import RandomTradeDQNActionSelector

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.environments.trade.trade_constant import TimeUnit, EnvironmentType, Action
from common.environments.trade.trade_data import get_data
from common.environments.trade.trade_env import UpbitEnvironment
from common.fast_rl import experience_single, rl_agent
from rl_main.trade_main.upbit_trade_main import test_random, test_sequential_all

##### NOTE #####
from config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    coin_name = "OMG"
    time_unit = TimeUnit.ONE_DAY

    train_data_info, test_data_info = get_data(coin_name=coin_name, time_unit=time_unit)

    print(train_data_info["first_datetime_krw"], train_data_info["last_datetime_krw"])
    print(test_data_info["first_datetime_krw"], test_data_info["last_datetime_krw"])

    print("#### TEST RANDOM 100")
    test_random_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=test_data_info,
        environment_type=EnvironmentType.TEST_RANDOM,
    )
    random_action_selector = RandomTradeDQNActionSelector(env=test_random_env)
    random_agent = rl_agent.DQNAgent(dqn_model=None, action_selector=random_action_selector, device=device)
    test_random(test_random_env, random_agent, num_episodes=100)

    print()

    print("#### TEST SEQUENTIALLY")
    test_sequential_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=test_data_info,
        environment_type=EnvironmentType.TEST_SEQUENTIAL,
    )
    random_action_selector = RandomTradeDQNActionSelector(env=test_sequential_env)
    random_agent = rl_agent.DQNAgent(dqn_model=None, action_selector=random_action_selector, device=device)
    test_sequential_all(test_sequential_env, random_agent, data_size=len(test_data_info["data"]))