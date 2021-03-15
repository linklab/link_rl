from gym.vector import SyncVectorEnv


class CustomSyncVectorEnv(SyncVectorEnv):
    def stop(self):
        self.envs.env[0].stop()
