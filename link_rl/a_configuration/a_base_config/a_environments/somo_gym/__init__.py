import sys
from pathlib import Path
import os
import yaml

from link_rl.c_encoders.a_encoder import ENCODER


PROJECT_HOME = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir)
)

sys.path.append(os.path.join(
    PROJECT_HOME, "link_rl", "b_environments", "somo_gym"
))


class ConfigSomoGym:
    def __init__(self, env_name):
        run_config_file = os.path.join(
            PROJECT_HOME, "link_rl", "b_environments", "somo_gym", "environments", env_name, "benchmark_run_config.yaml"
        )

        with open(run_config_file, "r") as config_file:
            self.RUN_CONFIG = yaml.safe_load(config_file)

        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
