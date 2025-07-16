"""Optimizer: CollegeStudyPlanOptimizer, GeneralizedAssignmentOptimizer, KnapsackOptimizer"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class Optimizer:

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_optimizer(self, name: str):
        assert name in {"CSPP", "GAP", "KP"}

        if name == "CSPP":
            return CollegeStudyPlanOptimizer(self.gurobi_environment)
        elif name == "GAP":
            return GeneralizedAssignmentOptimizer(self.gurobi_environment)
        else:  # name == "KP"
            return KnapsackOptimizer(self.gurobi_environment)

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list):
        pass


class CollegeStudyPlanOptimizer(Optimizer):

    def __init__(self, gurobi_environment = None):
        super().__init__("CSPP", gurobi_environment)

        df = pd.read_csv(os.path.join(file_path, "../data/csv/MIT_courses.csv"), sep="|", index_col=False)
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list = None):
        v = v[0]
        w_hat = w_hat[0]

        n = v.size

        model = gp.Model("CSPP_model", env=self.gurobi_environment)
        x_variables = model.addVars(range(n), vtype=gp.GRB.BINARY)
        model.addConstr(gp.quicksum(w_hat[j] * x_variables[j] for j in range(n)) <= 1)
        for point in labeled_points["positive"]:
            model.addConstr(gp.quicksum(x_variables[j] * (point[0, j] < 0.5) for j in range(n)) >= 1 - 1e-8)
        for point in labeled_points["negative"]:
            model.addConstr(gp.quicksum(x_variables[j] * (point[0, j] > 0.5) for j in range(n)) <=
                            round(point.sum()) - 1 + 1e-8)
        for j, p in enumerate(self.prerequisites):
            for code_list in p:
                model.addConstr(x_variables[j] <= gp.quicksum(x_variables[code] for code in code_list))
        for j, c in enumerate(self.corequisites):
            for code_list in c:
                model.addConstr(x_variables[j] <= gp.quicksum(x_variables[code] for code in code_list))
        for j, a in enumerate(self.alternatives):
            model.addConstr(x_variables[j] + gp.quicksum(x_variables[code] for code in a) <= 1)
        model.setObjective(gp.quicksum(v[j] * x_variables[j] for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        if model.Status == 2:
            return np.array([[round(x_variables[j].X) for j in range(n)]])
        else:
            return None


class GeneralizedAssignmentOptimizer(Optimizer):

    def __init__(self, gurobi_environment = None):
        super().__init__("GAP", gurobi_environment)

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list):
        m = w_hat.shape[0]
        n = w_hat.shape[1]

        model = gp.Model("GAP_model", env=self.gurobi_environment)
        x_variables = model.addVars([(i, j) for i in range(m) for j in range(n)], vtype=gp.GRB.BINARY)
        model.addConstrs((gp.quicksum(w_hat[i, j] * x_variables[i, j] for j in range(n)) <= 1
                          for i in range(m)))
        for point in labeled_points["positive"]:
            model.addConstr(gp.quicksum(gp.quicksum(x_variables[i, j] * (point[i, j] < 0.5) for i in range(m))
                                        for j in range(n)) >= 1 - 1e-8)
        for i in range(m):
            for sub_point in labeled_sub_points[i]["negative"]:
                model.addConstr(gp.quicksum(x_variables[i, j] * (sub_point[j] > 0.5) for j in range(n)) <=
                                round(sub_point.sum()) - 1 + 1e-8)
        model.addConstrs((gp.quicksum(x_variables[i, j] for i in range(m)) <= 1 for j in range(n)))
        model.setObjective(gp.quicksum(v[i, j] * x_variables[i, j] for i in range(m) for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        if model.Status == 2:
            return np.array([[round(x_variables[i, j].X, 0) for j in range(n)] for i in range(m)])
        else:
            return None


class KnapsackOptimizer(Optimizer):

    def __init__(self, gurobi_environment = None):
        super().__init__("KP", gurobi_environment)

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list = None):
        v = v[0]
        w_hat = w_hat[0]

        n = v.size

        model = gp.Model("KP_model", env=self.gurobi_environment)
        x_variables = model.addVars(range(n), vtype=gp.GRB.BINARY)
        model.addConstr(gp.quicksum(w_hat[j] * x_variables[j] for j in range(n)) <= 1)
        for point in labeled_points["positive"]:
            model.addConstr(gp.quicksum(x_variables[j] * (point[0, j] < 0.5) for j in range(n)) >= 1 - 1e-8)
        for point in labeled_points["negative"]:
            model.addConstr(gp.quicksum(x_variables[j] * (point[0, j] > 0.5) for j in range(n)) <=
                            round(point.sum()) - 1 + 1e-8)
        model.setObjective(gp.quicksum(v[j] * x_variables[j] for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        if model.Status == 2:
            return np.array([[round(x_variables[j].X, 0) for j in range(n)]])
        else:
            return None
