import os
import sys
from threading import Thread
import socket
import json
import numpy as np
import logging
import torch
import torch.nn.functional as F

from link_rl.b_environments.ai_birds.computer_vision.GroundTruthReader import NotVaildStateError, GroundTruthReader
from link_rl.b_environments.ai_birds.utils.agent_client import AgentClient, PlayingMode
from link_rl.b_environments.ai_birds.utils.trajectory_planner import SimpleTrajectoryPlanner

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir, os.pardir, os.pardir, os.pardir))

log_dir = os.path.join(
    PROJECT_HOME, 'link_rl', 'b_environments', 'ai_birds', 'log'
)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


class StateMaker:
    def __init__(self):
        # Crops 480x840x3 picture to 310x770x3 and
        # then resizes it to 84x84x3
        # also normalizes the pixel values to -1,1 range
        # Important: pass png without alpha channel
        pass

    def make(self, raw_state):
        state = raw_state.transpose((2, 0, 1))
        state = state.astype(np.float32)
        state /= 255

        tensor_state = torch.from_numpy(state).unsqueeze(0)
        tensor_state = F.interpolate(tensor_state, size=84)

        return tensor_state


class ClientRLAgent:
    def __init__(self):
        self.id = 0

        # Wrapper of the communicating messages
        with open(
                os.path.join(
                    PROJECT_HOME, 'link_rl', 'b_environments', 'ai_birds', 'utils', 'json', 'server_client_config.json'
                ),
                'r'
        ) as config:
            sc_json_config = json.load(config)

        self.logger = logging.getLogger(name="Agent {0}".format(self.id))
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        self.logger.addHandler(stream_handler)

        formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s : %(message)s")
        file_handler = logging.FileHandler(os.path.join(log_dir, "%s.log" % (self.id)))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.agent_client = AgentClient(logger=self.logger, **sc_json_config[0])

        try:
            self.agent_client.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        self.level_count = 0
        self.failed_counter = 0
        self.solved = []
        self.tp = SimpleTrajectoryPlanner()
        self.sling_center = None
        self.sling_mbr = None

        self.first_shot = True
        self.prev_target = None

        self.model = np.loadtxt(
            os.path.join(
                PROJECT_HOME, 'link_rl', 'b_environments', 'ai_birds', 'utils', 'txt', 'model'
            ),
            delimiter=","
        )

        self.target_class = list(map(
            lambda x: x.replace("\n", ""),
            open(os.path.join(
                PROJECT_HOME, 'link_rl', 'b_environments', 'ai_birds', 'utils', 'txt', 'target_class'
            )).readlines()
        ))

    def get_slingshot_center(self):
        try:
            self.agent_client.fully_zoom_out()
            ground_truth = self.agent_client.get_ground_truth_without_screenshot()
            ground_truth_reader = GroundTruthReader(ground_truth, self.model, self.target_class)
            sling = ground_truth_reader.find_slingshot_mbr()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.tp.get_reference_point(sling)
            self.sling_mbr = sling
        except NotVaildStateError:
            self.agent_client.fully_zoom_out()
            ground_truth = self.agent_client.get_ground_truth_without_screenshot()
            ground_truth_reader = GroundTruthReader(ground_truth, self.model, self.target_class)
            sling = ground_truth_reader.find_slingshot_mbr()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.tp.get_reference_point(sling)
            self.sling_mbr = sling

    def update_no_of_levels(self):
        # check the number of levels in the game
        n_levels = self.agent_client.get_number_of_levels()

        # if number of levels has changed make adjustments to the solved array
        if n_levels > len(self.solved):
            for n in range(len(self.solved), n_levels):
                self.solved.append(0)

        if n_levels < len(self.solved):
            self.solved = self.solved[:n_levels]

        # self.logger.info('No of Levels: ' + str(n_levels))

        return n_levels

    def check_my_score(self):
        """
         * Run the Client (Naive Agent)
        *"""
        scores = self.agent_client.get_all_level_scores()
        #print(" My score: ")
        level = 1
        for i in scores:
            self.logger.info(" level ", level, "  ", i)
            if i > 0:
                self.solved[level - 1] = 1
            level += 1
        return scores
