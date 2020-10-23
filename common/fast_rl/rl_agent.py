"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
import os, glob

from common.fast_rl.common.noise import OrnsteinUhlenbeckActionNoise
from . import actions


def save_model(model_save_dir, env_name, net_name, net, step, mean_episode_reward):
    model_save_filename = os.path.join(
        model_save_dir, "{0}_{1}_{2}_{3}.pth".format(
            env_name, net_name, step, mean_episode_reward
        )
    )
    torch.save(net.state_dict(), model_save_filename)
    return model_save_filename


def load_model(model_save_dir, env_name, net_name, net, step=None):
    if step:
        saved_models = glob.glob(os.path.join(
            model_save_dir, "{0}_{1}_{2}_*.pth".format(env_name, net_name, step)
        ))

    else:
        saved_models = glob.glob(os.path.join(
            model_save_dir, "{0}_{1}_*.pth".format(env_name, net_name)
        ))

    saved_models.sort(key=lambda filename: int(filename.split("/")[-1].split("_")[-2]))
    assert len(saved_models) > 0, "※※※※※※※※※※ There is no model !!!: {0} ※※※※※※※※※※".format(saved_models)

    saved_model = saved_models[-1]
    print("MODEL FILE NAME: {0}".format(saved_model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = torch.load(saved_model, map_location=device)

    net.load_state_dict(model_params)


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_agent_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v = self.dqn_model(states)
        # q = q_v.data.cpu().numpy()
        q = q_v.detach().cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    # TODO: unify code with DQNAgent, as only action action_selector is differs.
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


class ActorCriticAgent(BaseAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, critics=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        probs_v, values_v = self.model(states)

        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)

        probs = probs_v.data.cpu().numpy()
        actions = np.array(self.action_selector(probs))
        critics = [values_v.data.squeeze().cpu().numpy()]
        return actions, critics


class ContinuousActorCriticAgent(BaseAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """
    def __init__(self, model, action_min, action_max, action_selector=actions.ContinuousNormalActionSelector(),
                 device="cpu", preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.action_min = action_min
        self.action_max = action_max
        self.device = device
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, critics=None):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        mu_v, var_v, values_v = self.model(states)
        actions = self.action_selector(mu_v, var_v, self.action_min, self.action_max)
        critics = [values_v.data.squeeze().cpu().numpy()]
        return actions, critics


class AgentDDPG(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, model, n_actions, action_selector, action_min, action_max, device="cpu", ou_enabled=True,
                 preprocessor=default_states_preprocessor):
        self.model = model
        self.device = device
        self.ou_enabled = ou_enabled
        self.preprocessor = preprocessor
        self.action_selector = action_selector
        self.action_min = action_min
        self.action_max = action_max
        self.n_actions = n_actions
        self.step_idx = 0

    def initial_agent_state(self):
        return 0.0

    def __call__(self, states, agent_states=None):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        if self.step_idx < 1000:
            actions = np.random.normal(size=(1, 1))
            noises = np.zeros_like(actions)
            new_agent_states = np.zeros_like(actions)
        else:
            mu_v = self.model(states)
            mu = mu_v.data.cpu().numpy()
            ####################################

            # if agent_states is None:
            #     new_agent_states = [None] * len(states)
            # else:
            #     new_agent_states = agent_states
            #
            # noises_v = torch.Tensor(self.ou_noise.noise()).unsqueeze(dim=-1).to(self.device)
            # noises = noises_v.data.cpu().numpy()
            #
            # actions = mu + noises
            # actions = np.clip(actions, self.action_min, self.action_max)

            ####################################

            actions, new_agent_states = self.action_selector(mu, agent_states)
            noises = new_agent_states

            #####################################

        actions = np.clip(actions, self.action_min, self.action_max)
        self.step_idx += 1

        return actions, noises, new_agent_states


class AgentD4PG(BaseAgent):
    def __init__(self, model, n_actions, action_selector, action_min, action_max, device="cpu",
                 preprocessor=default_states_preprocessor):
        self.model = model
        self.device = device
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.action_min = action_min
        self.action_max = action_max
        self.n_actions = n_actions

    def __call__(self, states, agent_states):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        mu_v = self.model(states)
        new_agent_states = agent_states

        actions = mu_v.data.cpu().numpy()
        actions += self.action_selector(actions)
        actions = np.clip(actions, self.action_min, self.action_max)
        return actions, new_agent_states