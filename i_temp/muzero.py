import math
import numpy as np
import gym
from muzero_models import *
from muzero_selfplay import *
from muzero_train import *
from muzero_parameter import *
from muzero_replay_buffer import *

def main():
    print("*" * 80)
    config = MuZeroConfig()
    replay_buffer = ReplayBuffer({}, config)
    training_step = 0
    max_training_steps = 10000
    learner = Trainer(config)
    worker = SelfPlay(config, 0)
    global_steps = 0
    while training_step < max_training_steps:
        game_history = worker.play_game(
            config.visit_softmax_temperature_fn(training_step),
            None,
            False,
            replay_buffer
        )
        replay_buffer.save_game(game_history)

        if replay_buffer.num_played_games % 10 == 0:
            training_step += 1
            batch = replay_buffer.get_batch()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = learner.update_weights(batch[-1])
        print(replay_buffer.num_played_games)
        if replay_buffer.num_played_games % 100 == 0:
            for _ in range(5):
                worker.play_game(
                0,
                None,
                True,
                replay_buffer
            )

if __name__ == "__main__":
    main()