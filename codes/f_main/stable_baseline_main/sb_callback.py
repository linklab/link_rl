from stable_baselines3.common.callbacks import BaseCallback


class Callback(BaseCallback):
    def __init__(self, model):
        super(Callback, self).__init__()
        self.count = 0
        self.model = model

    def _init_callback(self) -> None:
        print(self.training_env)

    def _on_step(self) -> bool:
        #print(self.count)
        # print(self.locals)
        # print()

        return True
