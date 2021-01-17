import time
import numpy as np
from icecream import ic

from codes.e_utils.common_utils import load_model, remove_models, save_model
from codes.e_utils.names import EnvironmentName


class RewardTracker:
    def __init__(self, params, frame=True, stat=None, worker_id=None, early_stopping=None):
        self.params = params
        self.min_ts_diff = 1    # 1 second
        self.stat = stat
        self.frame = frame
        self.episode_reward_list = None
        self.done_episodes = 0
        self.mean_episode_reward = 0.0
        self.count_stop_condition_episode = 0
        self.worker_id = worker_id
        self.early_stopping = early_stopping

    def __enter__(self):
        self.start_ts = time.time()
        self.ts = time.time()
        self.ts_frame = 0
        self.episode_reward_list = []
        return self

    def start_reward_track(self):
        self.__enter__()

    def __exit__(self, *args):
        pass

    def set_episode_reward(
            self, episode_reward, episode_done_step, epsilon, last_info=None, mean_loss=None, model=None, wandb=None
    ):
        self.done_episodes += 1

        self.episode_reward_list.append(episode_reward)
        self.mean_episode_reward = np.mean(self.episode_reward_list[-self.params.AVG_EPISODE_SIZE_FOR_STAT:])

        current_ts = time.time()
        elapsed_time = current_ts - self.start_ts
        ts_diff = current_ts - self.ts

        is_print_performance = False

        if ts_diff > self.min_ts_diff:
            is_print_performance = True
            self.print_performance(
                episode_done_step, self.done_episodes, episode_reward, current_ts, ts_diff, self.mean_episode_reward, epsilon,
                elapsed_time, last_info, mean_loss, wandb
            )

        solved = False
        if self.early_stopping:
            solved = self.early_stopping(self.mean_episode_reward, model, episode_done_step)
        else:
            if self.mean_episode_reward > self.params.STOP_MEAN_EPISODE_REWARD:
                self.count_stop_condition_episode += 1
            else:
                self.count_stop_condition_episode = 0

            if self.count_stop_condition_episode >= self.params.STOP_PATIENCE_COUNT:
                if not is_print_performance:
                    self.print_performance(
                        episode_done_step, self.done_episodes, episode_reward, current_ts, ts_diff, self.mean_episode_reward, epsilon,
                        elapsed_time, last_info, mean_loss, wandb
                    )
                solved = True
        if solved:
            print("Solved in {0} {1} and {2} episodes!".format(
                episode_done_step,
                "frame" if self.frame else "step",
                self.done_episodes
            ))
            return True, self.mean_episode_reward
        else:
            return False, self.mean_episode_reward

    def print_performance(self, episode_done_step, done_episodes, episode_reward, current_ts, ts_diff,
                          mean_episode_reward, epsilon, elapsed_time, last_info, mean_loss, wandb=None):
        speed = (episode_done_step - self.ts_frame) / ts_diff
        self.ts_frame = episode_done_step
        self.ts = current_ts

        if self.worker_id is not None:
            prefix = "[Worker ID: {0}]".format(self.worker_id)
        else:
            prefix = ""

        if isinstance(epsilon, tuple) or isinstance(epsilon, list):
            epsilon_str = " eps.: {0:5.3f}, {1:5.3f},".format(
                epsilon[0] if epsilon[0] else 0.0,
                epsilon[1] if epsilon[1] else 0.0
            )
        elif isinstance(epsilon, float):
            epsilon_str = " eps.: {0:5.3f},".format(
                epsilon if epsilon else 0.0,
            )
        else:
            epsilon_str = ""

        if self.mean_episode_reward > self.params.STOP_MEAN_EPISODE_REWARD:
            mean_episode_reward_str = "{0:7.3f} (SOLVED COUNT: {1})".format(
                mean_episode_reward,
                self.count_stop_condition_episode + 1
            )
        else:
            mean_episode_reward_str = "{0:7.3f}".format(
                mean_episode_reward
            )

        if isinstance(episode_reward, np.ndarray):
            episode_reward = episode_reward[0]

        print(
            "{0}[{1:6}/{2}] Ep. {3}, ep._reward: {4:7.3f}, mean_{5}_ep._reward: {6},{7} "
            "speed: {8:7.2f} {9}, {10}".format(
                prefix,
                episode_done_step,
                self.params.MAX_GLOBAL_STEP,
                done_episodes,
                episode_reward,
                self.params.AVG_EPISODE_SIZE_FOR_STAT,
                mean_episode_reward_str,
                epsilon_str,
                speed,
                "fps" if self.frame else "steps/sec.",
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
        ), end="")

        if last_info and "action_count" in last_info:
            print(", {0}".format(last_info["action_count"]), end="")

        if mean_loss is not None:
            print(", mean (critic) loss {0:7.4f}".format(mean_loss), end="")

        if self.params.ENVIRONMENT_ID == EnvironmentName.TRADE_V0:
            print(", profit {0:8.1f}".format(last_info['profit']), end="")

        print("", flush=True)

        if self.params.WANDB:
            wandb_info = {
                "episode reward": episode_reward,
                "mean_loss": mean_loss,
                "speed": speed,
                "step_idx": episode_done_step,
                "episode": done_episodes
            }
            if epsilon:
                wandb_info["epsilon"] = epsilon
            wandb.log(wandb_info)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, evaluation_min_threshold=0.0, evaluation_min_step_idx=0,
                 verbose=False, delta=0.0, model_save_dir=".", model_save_file_prefix=None, agent=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.evaluation_min_threshold = evaluation_min_threshold
        self.evaluation_min_step_idx = evaluation_min_step_idx
        self.verbose = verbose
        self.counter = 0
        self.best_evaluation_value = -1.0e10
        self.early_stop = False
        self.delta = delta
        self.model_save_dir = model_save_dir
        self.model_save_file_prefix = model_save_file_prefix
        self.agent = agent

    def __call__(self, evaluation_value, model, step_idx):
        solved = False

        if step_idx < self.evaluation_min_step_idx:
            print(f"Current step {step_idx} is less than {self.evaluation_min_step_idx}. "
                  f"No early stopping (and no saving) processed")
        else:
            if self.best_evaluation_value == -1.0e10:
                self.best_evaluation_value = evaluation_value

            if evaluation_value < self.evaluation_min_threshold or evaluation_value < self.best_evaluation_value + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}. '
                      f'Best evaluation value is still {self.best_evaluation_value:.2f}')
                if self.counter >= self.patience:
                    solved = True
                    load_model(
                        self.model_save_dir,
                        self.model_save_file_prefix,
                        self.agent
                    )
            elif evaluation_value >= self.best_evaluation_value + self.delta:
                self.save_checkpoint(evaluation_value, step_idx)
                self.best_evaluation_value = evaluation_value
                self.counter = 0
            else:
                raise ValueError()

        return solved

    def save_checkpoint(self, evaluation_value, step_idx):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.best_evaluation_value == -1.0e10:
                print(f'evaluation_value recorded first ({evaluation_value:.2f}).  Saving model ...')
            else:
                print(f'evaluation_value increased ({self.best_evaluation_value:.2f} --> {evaluation_value:.2f}).  Saving model ...')

        remove_models(
            self.model_save_dir,
            self.model_save_file_prefix,
            self.agent
        )

        save_model(
            self.model_save_dir,
            self.model_save_file_prefix,
            self.agent,
            step_idx,
            evaluation_value
        )