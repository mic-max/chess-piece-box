from pulp import *

# Inputs
ceiling_height = 347
cylinders = [
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
    {'r': 55, 'h': 170, 'mount': 'floor'},
]

n = len(cylinders)
M = 10000  # Big-M value for constraint relaxation

# MILP problem
prob = LpProblem("MinimizeBoundingBox", LpMinimize)

# Variables for cylinder centers
x = [LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
y = [LpVariable(f"y_{i}", lowBound=0) for i in range(n)]

# Bounding box limits
x_min = LpVariable("x_min", lowBound=0)
x_max = LpVariable("x_max", lowBound=0)
y_min = LpVariable("y_min", lowBound=0)
y_max = LpVariable("y_max", lowBound=0)

# Bounding box containment
for i, c in enumerate(cylinders):
    r = c['r']
    prob += x_min <= x[i] - r
    prob += x_max >= x[i] + r
    prob += y_min <= y[i] - r
    prob += y_max >= y[i] + r

# Non-overlapping constraints with disjunctive separation
for i in range(n):
    for j in range(i + 1, n):
        r1, r2 = cylinders[i]['r'], cylinders[j]['r']
        b1 = LpVariable(f"sep_x_pos_{i}_{j}", cat="Binary")
        b2 = LpVariable(f"sep_x_neg_{i}_{j}", cat="Binary")
        b3 = LpVariable(f"sep_y_pos_{i}_{j}", cat="Binary")
        b4 = LpVariable(f"sep_y_neg_{i}_{j}", cat="Binary")
        prob += b1 + b2 + b3 + b4 == 1  # Exactly one active

        prob += x[i] + r1 <= x[j] - r2 + M * (1 - b1)
        prob += x[j] + r2 <= x[i] - r1 + M * (1 - b2)
        prob += y[i] + r1 <= y[j] - r2 + M * (1 - b3)
        prob += y[j] + r2 <= y[i] - r1 + M * (1 - b4)

# Objective: minimize perimeter
prob += (x_max - x_min) + (y_max - y_min)

# Solve
prob.solve(PULP_CBC_CMD(msg=1))

# Output
print("Status:", LpStatus[prob.status])
for i in range(n):
    print(f"Cylinder {i}: x = {value(x[i]):.2f}, y = {value(y[i]):.2f}")
print(f"Bounding Box: x [{value(x_min):.2f}, {value(x_max):.2f}], y [{value(y_min):.2f}, {value(y_max):.2f}]")
