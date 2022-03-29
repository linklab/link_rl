import csv
import enum
import os
import random
import numpy as np


class GeneratorType(enum.Enum):
    TYPE_1 = "Random Instances"
    TYPE_2 = "Fixed Instances"
    TYPE_3 = "Hard Instances"


class KnapsackGenerator():
    def __init__(self):
        self.NUM_ITEM = None # N
        self.LIMIT_WEIGHT_KNAPSACK = None # Wp

        self.MIN_WEIGHT_ITEM = None
        self.MAX_WEIGHT_ITEM = None

        self.MIN_VALUE_ITEM = None
        self.MAX_VALUE_ITEM = None

    def pre_generator(self, instances_infos, generator_type=None):
        if generator_type == GeneratorType.TYPE_1:
            self.NUM_ITEM = instances_infos[0]
            self.LIMIT_WEIGHT_KNAPSACK = random.randrange(instances_infos[1]/10, 3 * instances_infos[1] + 1)
            self.MIN_WEIGHT_ITEM = 1
            self.MAX_WEIGHT_ITEM = instances_infos[1] + 1

            self.MIN_VALUE_ITEM = 1
            self.MAX_VALUE_ITEM = instances_infos[1] + 1

        elif generator_type == GeneratorType.TYPE_2:
            self.NUM_ITEM = instances_infos[0]

            if self.NUM_ITEM == 50:
                self.LIMIT_WEIGHT_KNAPSACK = 12.5
            elif self.NUM_ITEM in [300, 500]:
                self.LIMIT_WEIGHT_KNAPSACK = 37.5
            else:
                raise ValueError()

            self.MIN_WEIGHT_ITEM = 0
            self.MAX_WEIGHT_ITEM = 2

            self.MIN_VALUE_ITEM = 0
            self.MAX_VALUE_ITEM = 2

        elif generator_type == GeneratorType.TYPE_3:
            self.NUM_ITEM = instances_infos[0]

            self.MIN_WEIGHT_ITEM = 1
            self.MAX_WEIGHT_ITEM = instances_infos[1] + 1

        else:
            raise ValueError()

    def generator(self, generator_type):
        knapsack = np.zeros(shape=(self.NUM_ITEM + 1, 2), dtype=float)

        if generator_type == GeneratorType.TYPE_3:
            for item_idx in range(self.NUM_ITEM):
                item_weight = np.random.randint(
                    low=self.MIN_WEIGHT_ITEM, high=self.MAX_WEIGHT_ITEM, size=(1, 1)
                )
                knapsack[item_idx][0] = item_weight + (self.MAX_WEIGHT_ITEM - 1) / 10
                knapsack[item_idx][1] = item_weight

            knapsack[-1][1] = self.NUM_ITEM / 1001 * sum(knapsack[:][1])

        elif generator_type in [GeneratorType.TYPE_1, GeneratorType.TYPE_2]:
            for item_idx in range(self.NUM_ITEM):
                item_weight = np.random.randint(
                    low=self.MIN_WEIGHT_ITEM, high=self.MAX_WEIGHT_ITEM, size=(1, 1)
                )
                item_value = np.random.randint(
                    low=self.MIN_VALUE_ITEM, high=self.MAX_VALUE_ITEM, size=(1, 1)
                )
                knapsack[item_idx][0] = item_value
                knapsack[item_idx][1] = item_weight

            knapsack[-1][1] = self.LIMIT_WEIGHT_KNAPSACK
        else:
            raise ValueError()

        return knapsack


M = 1000
INDEX = 3


def generate_random_instamces():
    instance_info_keys = ["n_50_r_100", "n_300_r_600", "n_500_r_1800"]
    instance_info_values = [[50, 100], [300, 600], [500, 1800]]

    for n in range(M):
        for idx in range(INDEX):
            instance0 = KnapsackGenerator()
            instance0.pre_generator(instance_info_values[idx], GeneratorType.TYPE_1)
            state = instance0.generator(GeneratorType.TYPE_1)
            file_path = os.path.join("random_instances2", instance_info_keys[idx])

            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            f = open(file_path + "/instance" + str(n) + ".csv", "w", newline='')
            writer = csv.writer(f)

            for s in state:
                writer.writerow(s)

            f.close()
            print("done", n)


def generate_fixed_instamces():
    instance_info_keys = ["n_50_wp_12.5", "n_300_wp_37.5", "n_500_wp_37.5"]
    instance_info_values = [[50, 12.5], [300, 37.5], [500, 37.5]]

    for n in range(M):
        for idx in range(INDEX):
            instance0 = KnapsackGenerator()
            instance0.pre_generator(instance_info_values[idx], GeneratorType.TYPE_2)
            state = instance0.generator(GeneratorType.TYPE_2)
            file_path = os.path.join("fixed_instances2", instance_info_keys[idx])

            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            f = open(file_path + "/instance" + str(n) + ".csv", "w", newline='')
            writer = csv.writer(f)

            for s in state:
                writer.writerow(s)

            f.close()
            print("done", n)


def generate_hard_instamces():
    instance_info_keys = ["n_50_r_100", "n_300_r_600", "n_500_r_1000"]
    instance_info_values = [[50, 100], [300, 600], [500, 1000]]

    for n in range(M):
        for idx in range(INDEX):
            instance0 = KnapsackGenerator()
            instance0.pre_generator(instance_info_values[idx], GeneratorType.TYPE_3)
            state = instance0.generator(GeneratorType.TYPE_3)
            file_path = os.path.join("random_instances2", instance_info_keys[idx])

            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            f = open(file_path + "/instance" + str(n) + ".csv", "w", newline='')
            writer = csv.writer(f)

            for s in state:
                writer.writerow(s)

            f.close()
            print("done", n)


if __name__ == "__main__":
    generate_random_instamces()
    generate_fixed_instamces()
    generate_hard_instamces()

