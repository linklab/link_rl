import numpy as np
from evogym import sample_robot

from link_rl.c_encoders.a_encoder import ENCODER


class ConfigEvolutionGym:
    def __init__(self):
        self.ROBOT_SHAPE = (3, 3)
        #self.ROBOT_STRUCTURE, self.ROBOT_CONNECTIONS = sample_robot(self.ROBOT_SHAPE)

        self.ROBOT_STRUCTURE = np.asarray([
            [0, 0, 0],
            [4, 3, 4],
            [2, 0, 2],
        ])

        self.ROBOT_CONNECTIONS = np.asarray([
            [3, 3, 4, 5],
            [4, 6, 5, 8]
        ])

        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
