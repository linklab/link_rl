from a_configuration.b_base.b_agents.agents import ParameterAgent
from g_utils.commons import AgentType


class ParameterReinforce(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.REINFORCE

        self.LEARNING_RATE = 0.0001
        self.BUFFER_CAPACITY = 1_000
        self.BATCH_SIZE = None


class ParameterA2c(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A2C

        self.ACTOR_LEARNING_RATE = 0.00005
        self.LEARNING_RATE = 0.0001
        self.BUFFER_CAPACITY = 1_000
        self.BATCH_SIZE = 128
        self.ENTROPY_BETA = 0.0001
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 200


class ParameterPpo(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO

        self.ACTOR_LEARNING_RATE = 0.00005
        self.LEARNING_RATE = 0.0001
        self.ENTROPY_BETA = 0.0001
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 200

        self.PPO_EPSILON_CLIP = 0.2
        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.PPO_K_EPOCH = 3
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10 * 3
