import time
import csv
import random
import numpy as np
from gurobipy import * # conda install gurobi

RUNS = 20


def model_kp(capacity, value, weight, LogToConsole = True, TimeLimit=10):
    num_item = len(value)
    assert num_item == len(weight)
    model = Model()
    model.params.LogToConsole = LogToConsole
    x = model.addVars(num_item, vtype=GRB.BINARY)
    model.setObjective(quicksum(value[i] * x[i] for i in range(num_item)), GRB.MAXIMIZE)
    model.addConstr(quicksum(weight[i] * x[i] for i in range(num_item)) <= capacity)
    model.optimize()

    items_selected = [i for i in range(num_item) if x[i].X > 0.5]
    total_value = int(model.ObjVal)

    return items_selected, total_value


def dummy_test():
    Knapsack_capacity = 500
    values = [
        12.000, 19.000, 11.000, 14.000, 11.000, 2.000, 7.000, 11.000, 17.000,
        17.000, 6.000, 13.000, 2.000, 1.000, 13.000, 15.000, 19.000, 7.000,
        15.000, 10.000, 7.000, 13.000, 2.000, 15.000, 15.000, 6.000, 3.000,
        2.000, 6.000, 16.000, 11.000, 15.000, 12.000, 4.000, 18.000, 6.000,
        16.000, 2.000, 13.000, 12.000, 1.000, 5.000, 18.000, 3.000, 5.000,
        18.000, 18.000, 8.000, 3.000, 13.000
    ]
    weights = [
        12.000, 11.000, 4.000, 3.000, 4.000, 19.000, 10.000, 1.000, 10.000,
        6.000, 18.000, 5.000, 2.000, 9.000, 19.000, 8.000, 8.000, 12.000, 13.000,
        2.000, 19.000, 3.000, 8.000, 10.000, 8.000, 14.000, 17.000, 15.000,
        4.000, 12.000, 14.000, 11.000, 1.000, 18.000, 15.000, 4.000, 6.000,
        3.000, 16.000, 12.000, 11.000, 11.000, 19.000, 17.000, 15.000, 7.000,
        1.000, 8.000, 14.000, 10.000
    ]

    items_selected, total_value = model_kp(Knapsack_capacity, values, weights, False)
    print("items_selected", items_selected)
    print("total_value", total_value)


def test_gurobi(capacity, nums, max_v, min_v, max_w, min_w):

    elapsed_time_list = []
    for run in range(RUNS):
        values = []
        weights = []
        items = []
        for item_idx in range(nums):
            item_value = random.randint(min_v, max_v)
            item_weight = random.randint(min_w, max_w)

            items.append([item_value, item_weight])

            values.append(item_value)
            weights.append(item_weight)

        start_time = time.process_time()
        items_selected, total_value = model_kp(capacity, values, weights, False)
        end_time = time.process_time()

        elapsed_time_in_ms = 1000 * (end_time - start_time)
        elapsed_time_list.append(elapsed_time_in_ms)

    #print("items : ", items)
    #print("items_selected : ", items_selected)
    #print("total_value : ", total_value)
    print("Capacity: {0:4}, Num_items: {1:4}, Elapsed_time: {2:7.5f} ms".format(
        capacity, nums, np.mean(elapsed_time_list))
    )

    #write_csv(items, items_selected, total_value, elapsed_time_in_ms)
    #print("end!!!")


def write_csv(items, items_selected, total_value, time):
    with open('gurobi_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar=' ')

        writer.writerow(['items'] + [":"] + items)
        writer.writerow(['items_selected'] + [":"] + items_selected)
        writer.writerow(['total_value'] + [":"] + [total_value])
        writer.writerow(['time'] + [":"] + [time])


if __name__ == "__main__":

    #Hyperparameter
    Knapsack_capacity = 500
    Num_items = 50
    Max_value = 20
    Min_value = 10
    Max_weight = 20
    Min_weight = 10

    capacity_list  = [500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000]
    num_items_list = [50,    100,   150,   200,   250,   300,   350,   400,   450,   500]

    print("#" * 100)
    for capacity in capacity_list:
        print()
        for num_items in num_items_list:
            test_gurobi(capacity, num_items, Max_value, Min_value, Max_weight, Min_weight)

    print("#" * 100)
    for num_items in num_items_list:
        print()
        for capacity in capacity_list:
            test_gurobi(capacity, num_items, Max_value, Min_value, Max_weight, Min_weight)

