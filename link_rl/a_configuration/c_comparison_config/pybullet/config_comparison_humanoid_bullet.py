from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.pybullet.config_humanoid_bullet import ConfigHumanoidBulletSac


class ConfigComparisonHumanoidBulletSac(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "HumanoidBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigHumanoidBulletSac(),
            ConfigHumanoidBulletSac(),
            ConfigHumanoidBulletSac(),
            ConfigHumanoidBulletSac(),
        ]
