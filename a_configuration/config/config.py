import os
import sys


class Config:
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
    )
    if PROJECT_HOME not in sys.path:
        sys.path.append(PROJECT_HOME)

    MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "f_play", "models")
    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)
