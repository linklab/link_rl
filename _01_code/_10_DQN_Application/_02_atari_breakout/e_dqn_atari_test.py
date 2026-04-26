# https://ale.farama.org/environments/pong/
import os

import gymnasium as gym
import ale_py

from _01_code._09_DQN.d_dqn_train_test import DqnTester
from _01_code._10_DQN_Application._02_atari_breakout.b_atari_preprocessing_with_dummy_agent import \
    CroppedAtariPreprocessing
from c_qnet import QNetCNN

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    ENV_NAME = "PongNoFrameskip-v4"

    test_env = gym.make(ENV_NAME, render_mode="rgb_array")
    test_env = CroppedAtariPreprocessing(
        test_env,
        noop_max=30,
        top_crop=34,  # 상단 점수판 영역
        bottom_crop=16,  # 하단 여백
        screen_size=(84, 84),
        grayscale_obs=True,
        grayscale_newaxis=False,
        frame_skip=4,
        scale_obs=True
    )

    env = gym.wrappers.FrameStackObservation(test_env, stack_size=4)

    qnet = QNetCNN(n_actions=4)

    dqn_tester = DqnTester(env=test_env, qnet = qnet, env_name=ENV_NAME, current_dir=CURRENT_DIR)
    dqn_tester.test()

    test_env.close()

if __name__ == "__main__":
    main()