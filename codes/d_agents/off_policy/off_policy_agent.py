from codes.d_agents.a0_base_agent import BaseAgent
from codes.e_utils import replay_buffer


class OffPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, params, device):
        super(OffPolicyAgent, self).__init__()
        self.params = params
        self.device = device

        if hasattr(self.params, "PER_PROPORTIONAL") and self.params.PER_PROPORTIONAL:
            self.buffer = replay_buffer.PrioritizedReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        elif hasattr(self.params, "PER_RANK_BASED") and self.params.PER_RANK_BASED:
            self.buffer = replay_buffer.RankBasedPrioritizedReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                params=self.params, alpha=0.7, beta_start=0.5, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        else:
            self.buffer = replay_buffer.ExperienceReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )