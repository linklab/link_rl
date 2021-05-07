import time

import subprocess

from codes.e_utils.names import RLAlgorithmName

subprocess.call('', shell=True)
import colorama
from termcolor import colored
colorama.init(autoreset=False)

from codes.e_utils.common_utils import load_model, remove_models, save_model


class SpeedTracker:
    def __init__(self, params, worker_id=None):
        self.params = params
        self.min_ts_diff = 1    # 1 second
        self.done_episodes = 0
        self.count_stop_condition_episode = 0
        self.worker_id = worker_id

    def __enter__(self):
        self.start_ts = time.time()
        self.ts = time.time()
        self.ts_frame = 0
        return self

    def start_reward_track(self):
        self.__enter__()

    def __exit__(self, *args):
        pass

    def set_episode_reward(self, episode_done_step):
        self.done_episodes += 1

        current_ts = time.time()
        elapsed_time = current_ts - self.start_ts
        ts_diff = current_ts - self.ts

        speed = (episode_done_step - self.ts_frame) / (ts_diff + 1.0e-8)

        self.ts_frame = episode_done_step
        self.ts = current_ts

        return speed, elapsed_time


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
            self, patience=7, evaluation_value_min_threshold=0.0, evaluation_std_max_threshold=0.0,
            delta=0.0, model_save_dir=".", model_save_file_prefix=None, agent=None, params=None
    ):
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
        self.evaluation_value_min_threshold = evaluation_value_min_threshold
        self.evaluation_std_max_threshold = evaluation_std_max_threshold
        self.evaluation_min_step_idx = 0
        self.counter = 0
        self.best_evaluation_value = -1.0e10
        self.early_stop = False
        self.delta = delta
        self.model_save_dir = model_save_dir
        self.model_save_file_prefix = model_save_file_prefix
        self.agent = agent
        self.params = params
        # self.periodic_save_count = 1
        # self.next_periodic_save_step_idx = int(self.params.MAX_GLOBAL_STEP * self.periodic_save_count / 10)

    def evaluate(self, evaluation_value, evaluation_value_std, episode_done_step):
        solved = False
        good_model_saved = False
        msg = ""

        if evaluation_value_std < self.evaluation_std_max_threshold:
            std_msg = colored(
                f'STD {evaluation_value_std:.2f} is less than {self.evaluation_std_max_threshold}.', "yellow"
            )
        else:
            std_msg = colored(
                f'STD {evaluation_value_std:.2f} is more than {self.evaluation_std_max_threshold}.', "yellow"
            )

        if self.best_evaluation_value == -1.0e10:
            # if episode_done_step >= self.next_periodic_save_step_idx:
            #     evaluation_str = colored(
            #         f'STEP {episode_done_step} is more than {self.next_periodic_save_step_idx}.',
            #         "magenta"
            #     )
            #     msg = f"Periodic Save!!! - {evaluation_str}."
            #     self.save_checkpoint(evaluation_value, episode_done_step)
            #     self.periodic_save_count = self.periodic_save_count + 1
            #     self.next_periodic_save_step_idx = int(self.params.MAX_GLOBAL_STEP * self.periodic_save_count / 10)
            # else:
            #     msg = ""

            if episode_done_step < self.evaluation_min_step_idx and hasattr(self.agent, 'epsilon_tracker') and self.agent.epsilon_tracker:
                evaluation_str = colored(
                    f'STEP {episode_done_step} is less than {self.evaluation_min_step_idx}. {std_msg}',
                    "magenta"
                )
                msg += f"{evaluation_str}. No early stopping (and no saving) processed"
            elif evaluation_value < self.evaluation_value_min_threshold:
                evaluation_str = colored(
                    f'{evaluation_value:.2f} is less than {self.evaluation_value_min_threshold}. {std_msg}',
                    'blue'
                )
                msg += f"{evaluation_str}. No early stopping (and no saving) processed"
            elif evaluation_value >= self.evaluation_value_min_threshold and evaluation_value_std > self.evaluation_std_max_threshold:
                evaluation_str = colored(
                    f'{evaluation_value:.2f} is more than {self.evaluation_value_min_threshold}. '
                    f'But, std {evaluation_value_std:.2f} is more than {self.evaluation_std_max_threshold}',
                    'blue'
                )
                msg += f"{evaluation_str}. No early stopping (and no saving) processed"
            else:
                saving_str = colored(f"Evaluation value is good!!! Saving model ...", 'green')
                evaluation_str = colored(f'{evaluation_value:.2f} recorded first.', 'green')
                msg += f'*** Evaluation value {evaluation_str}. {saving_str}'

                self.save_checkpoint(evaluation_value, episode_done_step)
                good_model_saved = True
                self.best_evaluation_value = evaluation_value
                self.counter = 0
        else:
            if evaluation_value < self.best_evaluation_value + self.delta or evaluation_value_std > self.evaluation_std_max_threshold:
                self.counter += 1
                counter_str = colored(f'{self.counter} out of {self.patience}', 'red')
                best_str = colored(f'{self.best_evaluation_value:.2f}', 'green')
                msg = f'EarlyStopping counter: {counter_str}. Best evaluation value is still {best_str}'

                if self.counter >= self.patience:
                    solved = True
                    load_model(
                        self.model_save_dir,
                        self.model_save_file_prefix,
                        self.agent
                    )
            else:
                saving_str = colored(f"Saving model ...", 'green')
                evaluation_str = colored(
                    f'{self.best_evaluation_value:.2f} is increased into {evaluation_value:.2f}', 'green'
                )

                msg = f'*** Evaluation value {evaluation_str}. {saving_str}'

                self.save_checkpoint(evaluation_value, episode_done_step)
                good_model_saved = True
                self.best_evaluation_value = evaluation_value
                self.counter = 0

        return solved, good_model_saved, msg

    def save_checkpoint(self, evaluation_value, episode_done_step):
        '''Saves model when validation loss decrease.'''
        remove_models(
            self.model_save_dir,
            self.model_save_file_prefix,
            self.agent
        )

        save_model(
            self.model_save_dir,
            self.model_save_file_prefix,
            self.agent,
            episode_done_step,
            evaluation_value
        )