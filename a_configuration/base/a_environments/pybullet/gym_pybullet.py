# [NOTE] 바로 아래 CartPoleBulletEnv import 구문 삭제하지 말것
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
class ParameterCartPoleBullet:
    def __init__(self):
        self.ENV_NAME = "CartPoleBulletEnv-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.N_STEP = 1
