"""Optimizer: CollegeStudyPlanOptimizer, GeneralizedAssignmentOptimizer, KnapsackOptimizer"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class Problem:

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_problem(self, name: str):
        assert name in {"CSPP", "GAP", "KP"}

        if name == "CSPP":
            return CollegeStudyPlanProblem(self.gurobi_environment)
        elif name == "GAP":
            return GeneralizedAssignmentProblem(self.gurobi_environment)
        else:  # name == "KP"
            return KnapsackProblem(self.gurobi_environment)

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        pass

    def check(self, u: np.ndarray):
        pass


class CollegeStudyPlanProblem(Problem):

    def __init__(self, gurobi_environment = None):
        super().__init__("CSPP", gurobi_environment)

        df = pd.read_csv(os.path.join(file_path, "../data/csv/MIT_courses.csv"), sep="|", index_col=False)
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        v = v[0]
        w_bar = w_bar[0]

        n = v.size

        model = gp.Model("CSPP_model", env=self.gurobi_environment)
        x_variables = model.addVars(range(n), vtype=gp.GRB.BINARY)
        model.addConstr(gp.quicksum(w_bar[j] * x_variables[j] for j in range(n)) <= 1)
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

    def check(self, u: np.ndarray):
        u = u[0]
        valid = True
        for j, p in enumerate(self.prerequisites):
            for code_list in p:
                if not u[j] <= np.sum([u[code] for code in code_list]):
                    valid = False
        for j, c in enumerate(self.corequisites):
            for code_list in c:
                if not u[j] <= np.sum([u[code] for code in code_list]):
                    valid = False
        for j, a in enumerate(self.alternatives):
            if not u[j] + np.sum([u[code] for code in a]) <= 1:
                valid = False
        return valid

class GeneralizedAssignmentProblem(Problem):

    def __init__(self, gurobi_environment = None):
        super().__init__("GAP", gurobi_environment)

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        m = w_bar.shape[0]
        n = w_bar.shape[1]

        model = gp.Model("GAP_model", env=self.gurobi_environment)
        x_variables = model.addVars([(i, j) for i in range(m) for j in range(n)], vtype=gp.GRB.BINARY)
        model.addConstrs((gp.quicksum(w_bar[i, j] * x_variables[i, j] for j in range(n)) <= 1 for i in range(m)))
        model.addConstrs((gp.quicksum(x_variables[i, j] for i in range(m)) <= 1 for j in range(n)))
        model.setObjective(gp.quicksum(v[i, j] * x_variables[i, j] for i in range(m) for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        if model.Status == 2:
            return np.array([[round(x_variables[i, j].X, 0) for j in range(n)] for i in range(m)])
        else:
            return None

    def check(self, u: np.ndarray):
        return (u.sum(axis=1) <= 1).all()

class KnapsackProblem(Problem):

    def __init__(self, gurobi_environment = None):
        super().__init__("KP", gurobi_environment)

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        v = v[0]
        w_bar = w_bar[0]

        n = v.size

        model = gp.Model("KP_model", env=self.gurobi_environment)
        x_variables = model.addVars(range(n), vtype=gp.GRB.BINARY)
        model.addConstr(gp.quicksum(w_bar[j] * x_variables[j] for j in range(n)) <= 1)
        model.setObjective(gp.quicksum(v[j] * x_variables[j] for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        if model.Status == 2:
            return np.array([[round(x_variables[j].X, 0) for j in range(n)]])
        else:
            return None

    def check(self, u: np.ndarray):
        return True
