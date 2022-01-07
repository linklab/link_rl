from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.mujoco.parameter_ant_mujoco import ParameterAntMujocoSac
from a_configuration.c_parameters.mujoco.parameter_hopper_mujoco import ParameterHopperMujocoSac
from a_configuration.c_parameters.mujoco.parameter_walker2d_mujoco import ParameterWalker2dMujocoSac
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletSac


class ParameterComparisonAntBulletSac(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "AntBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ParameterAntBulletSac(),
            ParameterAntBulletSac(),
            ParameterAntBulletSac(),
            ParameterAntBulletSac(),
        ]