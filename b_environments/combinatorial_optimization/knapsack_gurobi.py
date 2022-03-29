import csv
import numpy as np
from gurobipy import *

f = open("random_instances/n_50_r_100/instance0.csv", "r")
reader = csv.reader(f)

data = list(reader)

f.close()

data = np.array(data)

Knapsack_capacity = float(data[-1][1])
values = data[:-1, 0]
weights = data[:-1, 1]


def model_kp(capacity, value, weight, LogToConsole = True, TimeLimit=10):
    num_item = 50
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
    items_selected, total_value = model_kp(Knapsack_capacity, values, weights, False)
    print("items_selected", items_selected)
    print("total_value", total_value)



