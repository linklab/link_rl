from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.mujoco.parameter_hopper_mujoco import ParameterHopperMujocoSac


class ParameterComparisonHopperMujocoSac(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "Hopper-v2"

        self.AGENT_PARAMETERS = [
            ParameterHopperMujocoSac(),
            ParameterHopperMujocoSac(),
            ParameterHopperMujocoSac()
        ]