from gym.vector import SyncVectorEnv


class CustomSyncVectorEnv(SyncVectorEnv):
    def stop(self):
        self.envs[0].stop()