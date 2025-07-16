"""Bounder: CollegeStudyPlanBounder, GeneralizedAssignmentBounder, KnapsackBounder"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class Bounder:
        
    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_bounder(self, name: str):
        assert name in {"CSPP", "GAP", "KP"}

        if name == "CSPP":
            return CollegeStudyPlanBounder(self.gurobi_environment)
        elif name == "GAP":
            return GeneralizedAssignmentBounder(self.gurobi_environment)
        else:  # name == "KP"
            return KnapsackBounder(self.gurobi_environment)

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        pass
        
        
class CollegeStudyPlanBounder(Bounder):
    
    def __init__(self, gurobi_environment = None):
        super().__init__("CSPP", gurobi_environment)

        df = pd.read_csv(os.path.join(file_path, "../data/csv/MIT_courses.csv"), sep="|", index_col=False)
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        v = v[0]

        n = v.size

        model = gp.Model("CSPP_bound_model", env=self.gurobi_environment)
        x_variables = model.addVars(range(n), vtype=gp.GRB.BINARY, name="x")
        w_variables = model.addVars(range(n), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                    name="w")
        model.addConstr(gp.quicksum(w_variables[j] * x_variables[j]for j in range(n)) <= 1)
        model.addConstrs(
            (gp.quicksum(labeled_sub_points[0]["positive"][k, j] * w_variables[j] for j in range(n)) <= 1
             for k in range(labeled_sub_points[0]["positive"].shape[0])), "feasible_point_separation_constraints")
        model.addConstrs(
            (gp.quicksum(labeled_sub_points[0]["negative"][k, j] * w_variables[j] for j in range(n)) >= 1
             for k in range(labeled_sub_points[0]["negative"].shape[0])), "infeasible_point_separation_constraints")
        for point in labeled_sub_points[0]["negative"]:
            model.addConstr(gp.quicksum(x_variables[j] * (point[j] > 0.5)
                                        for j in range(n)) <= round(point.sum()) - 1 + 1e-8)
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
        model.setParam("NonConvex", 2)
        model.setParam("NodeLimit", 50000)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        return model.ObjBound

class GeneralizedAssignmentBounder(Bounder):
    
    def __init__(self, gurobi_environment = None):
        super().__init__("GAP", gurobi_environment)

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        m = v.shape[0]
        n = v.shape[1]

        model = gp.Model("GAP_bound_model", env=self.gurobi_environment)
        x_variables = model.addVars([(i, j) for i in range(m) for j in range(n)], vtype=gp.GRB.BINARY, name="x")
        w_variables = model.addVars([(i, j) for i in range(m) for j in range(n)], vtype=gp.GRB.CONTINUOUS, 
                                    lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="w")
        for i in range(m):
            model.addConstr(gp.quicksum(w_variables[i, j] * x_variables[i, j] for j in range(n)) <= 1)
            model.addConstrs(
                (gp.quicksum(labeled_sub_points[i]["positive"][k, j] * w_variables[i, j] for j in range(n)) <= 1
                 for k in range(labeled_sub_points[i]["positive"].shape[0])),
                "feasible_point_separation_constraints")
            model.addConstrs(
                (gp.quicksum(labeled_sub_points[i]["negative"][k, j] * w_variables[i, j] for j in range(n)) >= 1
                 for k in range(labeled_sub_points[i]["negative"].shape[0])),
                "infeasible_point_separation_constraints")
            for point in labeled_sub_points[i]["negative"]:
                model.addConstr(gp.quicksum(x_variables[i, j] * (point[j] > 0.5)
                                            for j in range(n)) <= round(point.sum()) - 1 + 1e-8)
        model.addConstrs((gp.quicksum(x_variables[i, j] for i in range(m)) <= 1 for j in range(n)))
        model.setObjective(gp.quicksum(v[i, j] * x_variables[i, j] for i in range(m) for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.setParam("NonConvex", 2)
        model.setParam("NodeLimit", 50000)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        return model.ObjBound


class KnapsackBounder(Bounder):

    def __init__(self, gurobi_environment = None):
        super().__init__("KP", gurobi_environment)

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        v = v[0]

        n = v.size

        model = gp.Model("KP_bound_model", env=self.gurobi_environment)
        x_variables = model.addVars(range(n), vtype=gp.GRB.BINARY, name="x")
        w_variables = model.addVars(range(n), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                    name="w")
        model.addConstr(gp.quicksum(w_variables[j] * x_variables[j] for j in range(n)) <= 1)
        model.addConstrs(
            (gp.quicksum(labeled_sub_points[0]["positive"][k, j] * w_variables[j] for j in range(n)) <= 1
             for k in range(labeled_sub_points[0]["positive"].shape[0])), "feasible_point_separation_constraints")
        model.addConstrs(
            (gp.quicksum(labeled_sub_points[0]["negative"][k, j] * w_variables[j] for j in range(n)) >= 1
             for k in range(labeled_sub_points[0]["negative"].shape[0])), "infeasible_point_separation_constraints")
        for point in labeled_sub_points[0]["negative"]:
            model.addConstr(gp.quicksum(x_variables[j] * (point[j] > 0.5)
                                        for j in range(n)) <= round(point.sum()) - 1 + 1e-8)
        model.setObjective(gp.quicksum(v[j] * x_variables[j] for j in range(n)))

        model.setParam("OutputFlag", 0)
        model.setParam("NonConvex", 2)
        model.setParam("NodeLimit", 50000)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        return model.ObjBound
