import csv
import os

import numpy as np
from gurobipy import * # conda install gurobi


def random_instance_reader(key, num):
    instance_info_keys = ["n_50_r_100", "n_300_r_600", "n_500_r_1800"]

    file_path = os.path.join("random_instances", instance_info_keys[key])

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    f = open(file_path + "/instance" + str(num) + ".csv", "r", )
    reader = csv.reader(f)
    data = list(reader)
    data = np.array(data)
    f.close()

    return data


def fixed_instance_reader(key, num):
    instance_info_keys = ["n_50_wp_12.5", "n_300_wp_37.5", "n_500_wp_37.5"]

    file_path = os.path.join("fixed_instances", instance_info_keys[key])

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    f = open(file_path + "/instance" + str(num) + ".csv", "r", )
    reader = csv.reader(f)
    data = list(reader)
    data = np.array(data)
    f.close()

    return data


def hard_instance_reader(key, num):
    instance_info_keys = ["n_50_r_100", "n_300_r_600", "n_500_r_1000"]

    file_path = os.path.join("hard_instances", instance_info_keys[key])

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    f = open(file_path + "/instance" + str(num) + ".csv", "r", )
    reader = csv.reader(f)
    data = list(reader)
    data = np.array(data)
    f.close()

    return data


def model_kp(capacity, value, weight, LogToConsole = True, TimeLimit=10):
    num_item = len(value)
    assert num_item == len(weight)
    model = Model()
    model.params.LogToConsole = LogToConsole
    #model.params.TimeLimit = TimeLimit
    x = model.addVars(num_item, vtype=GRB.BINARY)
    model.setObjective(quicksum(value[i] * x[i] for i in range(num_item)), GRB.MAXIMIZE)
    model.addConstr(quicksum(weight[i] * x[i] for i in range(num_item)) <= capacity)
    model.optimize()

    items_selected = [i for i in range(num_item) if x[i].X > 0.5]
    total_value = int(model.ObjVal)

    return items_selected, total_value


if __name__ == "__main__":
    # Knapsack_capacity = 10
    # print(Knapsack_capacity)
    # values = [1, 10, 1, 2, 9]
    # weights = [10, 1, 10, 1, 9]

    INDEX = 3
    M = 1000
    for idx in range(INDEX):
        for m in range(M):
            data = random_instance_reader(idx, m)
            #data = fixed_instance_reader(idx, m)
            #data = hard_instance_reader(idx, m)
            Knapsack_capacity = float(data[-1][1])
            values = data[:-1, 0]
            weights = data[:-1, 1]

            items_selected, total_value = model_kp(Knapsack_capacity, values, weights, False)
            print("instance_reader", idx, m)
            print("items_selected", items_selected)
            print("total_value", total_value)