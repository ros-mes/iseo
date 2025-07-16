"""Sampling strategy: CUT, SIM."""

import gurobipy as gp
import numpy as np

from algorithm.exception import ModelResolutionFailedError


class Sampler:

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_sampling_strategy(self, name: str):
        assert name in {"CUT", "SIM"}

        if name == "CUT":
            return CUT(self.gurobi_environment)
        else:  # name == "SIM"
            return SIM(self.gurobi_environment)

    def sample(self, w_hat: np.ndarray, labeled_sub_points: dict):
        pass


class CUT(Sampler):

    def __init__(self, gurobi_environment = None):
        super().__init__("CUT", gurobi_environment)

    def sample(self, w_hat: np.ndarray, labeled_sub_points: dict):
        n = labeled_sub_points["positive"].shape[1]

        model = gp.Model("CUT_model", env=self.gurobi_environment)
        mu_variables = model.addVars(range(n), vtype=gp.GRB.BINARY)
        p_variables = model.addVars(range(n), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                    vtype=gp.GRB.CONTINUOUS)
        model.addConstr(gp.quicksum(p_variables[j] * mu_variables[j] for j in range(n)) == 1)
        for point in labeled_sub_points["positive"]:
            model.addConstr(gp.quicksum(mu_variables[j] * (point[j] < 0.5) for j in range(n)) >= 1)
        for point in labeled_sub_points["negative"]:
            model.addConstr(gp.quicksum(mu_variables[j] * (point[j] > 0.5) for j in range(n)) <=
                            round(point.sum()) - 1)
        model.setObjective(gp.quicksum((p_variables[j] - w_hat[j]) ** 2 for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.setParam("NodeLimit", 50000)
        model.ModelSense = gp.GRB.MINIMIZE
        model.optimize()

        if model.Status in {2: "OPTIMAL", 8: "NODE_LIMIT"}:
            try:
                return np.array([round(mu_variables[j].X, 0) for j in range(n)])
            except:
                raise ModelResolutionFailedError("Sampling failed")
        return None


class SIM(Sampler):

    def __init__(self, gurobi_environment = None):
        super().__init__("SIM", gurobi_environment)

    def sample(self, w_hat: np.ndarray, labeled_sub_points: dict):
        n = labeled_sub_points["positive"].shape[1]

        model = gp.Model("SIM_model", env=self.gurobi_environment)
        mu_variables = model.addVars(range(n), vtype=gp.GRB.BINARY)
        u_variable = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        abs_u_variable = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        model.addConstr(u_variable == 1 + gp.quicksum(w_hat[j] * mu_variables[j] for j in range(n)))
        model.addConstr(abs_u_variable == gp.abs_(u_variable))
        for point in labeled_sub_points["positive"]:
            model.addConstr(gp.quicksum(mu_variables[j] * (point[j] < 0.5) for j in range(n)) >= 1)
        for point in labeled_sub_points["negative"]:
            model.addConstr(gp.quicksum(mu_variables[j] * (point[j] > 0.5) for j in range(n)) <=
                            round(point.sum()) - 1)
        model.setObjective(abs_u_variable)

        model.setParam("OutputFlag", 0)
        model.setParam("NodeLimit", 50000)
        model.ModelSense = gp.GRB.MINIMIZE
        model.optimize()

        if model.Status in {2: "OPTIMAL", 8: "NODE_LIMIT"}:
            try:
                return np.array([round(mu_variables[j].X, 0) for j in range(n)])
            except:
                raise ModelResolutionFailedError("Sampling failed")
        return None
