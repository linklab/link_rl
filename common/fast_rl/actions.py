import numpy as np
from typing import Union

import torch


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ContinuousActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, mu_v, var_v, action_min, action_max):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, action_selector=None):
        self.epsilon = epsilon
        self.action_selector = action_selector if action_selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.action_selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class EpsilonGreedyDDPGActionSelector:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def __call__(self, mu, agent_states, ou_enabled=True, ou_rho=0.15, ou_mu=0.0, ou_dt=0.1, ou_sigma=0.2):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        if ou_enabled and self.epsilon > 0:
            new_agent_states = []
            for agent_state, action in zip(agent_states, actions):
                agent_state = np.zeros(shape=action.shape, dtype=np.float32)
                agent_state += ou_rho * (ou_mu - actions)
                agent_state += ou_sigma * np.sqrt(ou_dt) * np.random.normal(size=action.shape)
                action += self.epsilon * agent_state
                new_agent_states.append(agent_state)
        else:
            new_agent_states = agent_states
        return actions, new_agent_states


class DDPGActionSelector:
    def __init__(self, epsilon, ou_enabled):
        self.epsilon = epsilon
        self.ou_enabled = ou_enabled

    def __call__(self, mu, agent_states, ou_rho=0.15, ou_mu=0.0, ou_dt=0.1, ou_sigma=0.75):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        if isinstance(agent_states, list):
            agent_states = np.asarray(agent_states)

        if self.ou_enabled:
            agent_states = agent_states + ou_rho * (ou_mu - agent_states) + ou_sigma * np.sqrt(ou_dt) * np.random.normal(size=actions.shape)
            actions = actions + self.epsilon * agent_states

        new_agent_states = agent_states

        return actions, new_agent_states


class EpsilonGreedyD4PGActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def __call__(self, actions):
        assert isinstance(actions, np.ndarray)
        actions += self.epsilon * np.random.normal(size=actions.shape)
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, var_v, action_min, action_max):
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, action_min, action_max)
        return actions


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(
        self, action_selector: Union[EpsilonGreedyActionSelector, EpsilonGreedyDDPGActionSelector, EpsilonGreedyD4PGActionSelector],
        eps_start: Union[int, float], eps_final: Union[int, float], eps_frames: int
    ):
        self.action_selector = action_selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.udpate(0)

    def udpate(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.action_selector.epsilon = max(self.eps_final, eps)