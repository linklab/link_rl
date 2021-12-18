from a_configuration.base.a_environments.pybullet.gym_pybullet import ParameterCartPoleBullet
from a_configuration.base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.base.c_models.linear_layers import ParameterSmallLinearLayer
from a_configuration.base.parameter_base import ParameterBase


class ParameterCartPoleBulletDqn(
    ParameterBase, ParameterCartPoleBullet, ParameterDqn, ParameterSmallLinearLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterDqn.__init__(self)
        ParameterSmallLinearLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS = 200


# OnPolicy