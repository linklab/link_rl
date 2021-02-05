import torch
import warnings
import sys, os

from common.environments import RandomTradeDQNActionSelector

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.environments import TimeUnit, TradeEnvironmentType
from common.environments import get_data
from common.environments import UpbitEnvironment
from common.fast_rl import rl_agent
from rl_main.trade_main.upbit_trade_main import evaluate_random, evaluate_sequential_all

##### NOTE #####
from codes.a_config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    coin_name = "OMG"
    time_unit = TimeUnit.ONE_DAY

    train_data_info, evaluate_data_info = get_data(coin_name=coin_name, time_unit=time_unit)

    print(train_data_info["first_datetime_krw"], train_data_info["last_datetime_krw"])
    print(evaluate_data_info["first_datetime_krw"], evaluate_data_info["last_datetime_krw"])

    print("#### TEST RANDOM 100")
    evaluate_random_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=evaluate_data_info,
        environment_type=TradeEnvironmentType.TEST_RANDOM,
    )
    random_action_selector = RandomTradeDQNActionSelector(env=evaluate_random_env)
    random_agent = rl_agent.DQNAgent(dqn_model=None, action_selector=random_action_selector, device=device)
    evaluate_random("DQN", evaluate_random_env, random_agent, num_episodes=100)

    print()

    print("#### TEST SEQUENTIALLY")
    evaluate_sequential_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=evaluate_data_info,
        environment_type=TradeEnvironmentType.TEST_SEQUENTIAL,
    )
    random_action_selector = RandomTradeDQNActionSelector(env=evaluate_sequential_env)
    random_agent = rl_agent.DQNAgent(dqn_model=None, action_selector=random_action_selector, device=device)
    evaluate_sequential_all("DQN", evaluate_sequential_env, random_agent, data_size=len(evaluate_data_info["data"]))