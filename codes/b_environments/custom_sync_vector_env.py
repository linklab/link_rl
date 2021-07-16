from abc import abstractmethod
from copy import deepcopy

from gym.vector import SyncVectorEnv
from gym.vector.utils import concatenate
import numpy as np


class CustomSyncVectorEnv(SyncVectorEnv):
    def stop(self):
        self.envs[0].stop()

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            # if self._dones[i]:
            #     observation = env.reset()
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(observations, self.observations,
                                        self.single_observation_space)

        return (deepcopy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards), np.copy(self._dones), infos)
