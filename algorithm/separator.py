"""Linear separator: SEP, SVM."""

import gurobipy as gp
import numpy as np

from algorithm.exception import ModelResolutionFailedError


class Separator:

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_linear_separator(self, name: str):
        assert name in {"SEP", "SVM"}

        if name == "SEP":
            return SEP(self.gurobi_environment)
        else:  # name == "SVM"
            return SVM(self.gurobi_environment)

    def separate(self, labeled_sub_points: dict, verbose: int = 0):
        pass


class SEP(Separator):

    def __init__(self, gurobi_environment = None):
        super().__init__("SEP", gurobi_environment)

    @ staticmethod
    def dual_norm(x: np.ndarray):
        return (1 / np.sqrt(2)) * np.linalg.norm(np.append(x[:-1] * np.sqrt(x.size), 1))

    def separate(self, labeled_sub_points: dict, verbose: int = 0):
        n = labeled_sub_points["positive"].shape[1]
        
        x = np.append(np.zeros(n), 1.0)
        Lambda = np.array([x / self.dual_norm(x)])
        x = np.append(- np.ones(n), - 1.0)
        Lambda = np.concatenate([Lambda, [x / self.dual_norm(x)]], axis=0)
        for j in range(n):
            x = np.append(np.zeros(n), 0.0)
            x[j] = - 1.0
            Lambda = np.concatenate([Lambda, [x / self.dual_norm(x)]], axis=0)
            x = np.append(np.zeros(n), 1.0)
            x[j] = 1.0
            Lambda = np.concatenate([Lambda, [x / self.dual_norm(x)]], axis=0)
        for x in labeled_sub_points["positive"]:
            x = np.append(x, 1)
            Lambda = np.concatenate([Lambda, [x / self.dual_norm(x)]], axis=0)
        for x in labeled_sub_points["negative"]:
            x = np.append(-x, -1)
            Lambda = np.concatenate([Lambda, [x / self.dual_norm(x)]], axis=0)

        model = gp.Model("SEP_model", env=self.gurobi_environment)
        p_variables = model.addVars(range(n + 1), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                                    ub=gp.GRB.INFINITY)
        c_variables = model.addVars(range(Lambda.shape[0]), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
        model.addConstr(c_variables.sum() == 1)
        model.addConstrs((gp.quicksum(c_variables[i] * Lambda[i, j] for i in range(Lambda.shape[0])) == p_variables[j]
                          for j in range(n + 1)))
        model.setObjective(0.25 * (np.power(np.sqrt(n), 2) * gp.quicksum(p_variables[j] * p_variables[j]
                                                                     for j in range(n)) +
                                   p_variables[n] * p_variables[n]))

        model.setParam('OutputFlag', verbose)
        model.ModelSense = gp.GRB.MINIMIZE
        model.optimize()

        if model.Status == 2:
            gamma = np.array([p_variables[j].X for j in range(n + 1)])
            gradient = np.append(0.5 * np.power(np.sqrt(n), 2) * gamma[:-1], 0.5 * gamma[-1])
            return - gradient[:-1] / gradient[-1]
        raise ModelResolutionFailedError("Linear separation failed")


class SVM(Separator):

    def __init__(self, gurobi_environment = None):
        super().__init__("SVM", gurobi_environment)

    def separate(self, labeled_sub_points: dict, verbose: int = 0):
        n = labeled_sub_points["positive"].shape[1]

        for method_code in {2: "barrier", 1: "dual simplex", 0: "primal simplex"}:
            model = gp.Model("SVM_model", env=self.gurobi_environment)
            w_variables = model.addVars([j for j in range(n)], vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY,
                                        name="w")
            b_variable = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="b")
            model.addConstrs((b_variable - gp.quicksum(labeled_sub_points["positive"][i, j] * w_variables[j]
                                                       for j in range(n)) >= 1
                              for i in range(labeled_sub_points["positive"].shape[0])),
                             "feasible_point_separation_constraints")
            model.addConstrs((b_variable - gp.quicksum(labeled_sub_points["negative"][i, j] * w_variables[j]
                                                       for j in range(n)) <= - 1
                              for i in range(labeled_sub_points["negative"].shape[0])),
                             "infeasible_point_separation_constraints")
            model.addConstrs((w_variables[j] <= b_variable for j in range(n)), "w_leq_b_constraints")
            model.setObjective(gp.quicksum(w_variables[j] * w_variables[j] for j in range(n)))

            model.setParam("OutputFlag", verbose)
            model.ModelSense = gp.GRB.MINIMIZE

            model.setParam("Method", method_code)
            model.optimize()
            if model.Status == 2:
                return np.array([w_variables[j].X for j in range(n)]) / b_variable.X

        for method_code in {2: "barrier", 1: "dual simplex", 0: "primal simplex"}:
            model = gp.Model("SVM_model", env=self.gurobi_environment)
            w_variables = model.addVars([j for j in range(n)], vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name="w")
            b_variable = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="b")
            model.addConstrs((b_variable - gp.quicksum(labeled_sub_points["positive"][i, j] * w_variables[j]
                                                       for j in range(n)) >= 0
                              for i in range(labeled_sub_points["positive"].shape[0])),
                             "feasible_point_separation_constraints")
            model.addConstrs((b_variable - gp.quicksum(labeled_sub_points["negative"][i, j] * w_variables[j]
                                                       for j in range(n)) <= 0
                              for i in range(labeled_sub_points["negative"].shape[0])),
                             "infeasible_point_separation_constraints")
            model.addConstrs((w_variables[j] <= b_variable for j in range(n)), "w_leq_b_constraints")
            model.addConstr(gp.quicksum(w_variables[j] * w_variables[j] for j in range(n)) == 1)
            model.setObjective(gp.LinExpr())

            model.setParam("OutputFlag", verbose)
            model.ModelSense = gp.GRB.MINIMIZE

            model.setParam("Method", method_code)
            model.optimize()
            if model.Status == 2:
                return np.array([w_variables[j].X for j in range(n)]) / b_variable.X

        raise ModelResolutionFailedError("Linear separation failed")
