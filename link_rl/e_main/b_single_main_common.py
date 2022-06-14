import warnings
from link_rl.g_utils.types import OffPolicyAgentTypes
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from gym import logger
logger.set_level(level=40)

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from link_rl.e_main.config_single import config
from link_rl.a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME

config.SYSTEM_USER_NAME = SYSTEM_USER_NAME
config.SYSTEM_COMPUTER_NAME = SYSTEM_COMPUTER_NAME

import random
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

if config.SEED is not None:
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if config.USE_HER:
    assert config.AGENT_TYPE in OffPolicyAgentTypes