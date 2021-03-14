from gym.vector import SyncVectorEnv


class CustomSyncVectorEnv(SyncVectorEnv):
    def stop(self):
        pass