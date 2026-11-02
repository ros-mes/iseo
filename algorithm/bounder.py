"""Bounders: CollegeStudyPlanBounder, GeneralizedAssignmentBounder, KnapsackBounder.

A bounder computes an upper bound on the optimal value of a test problem with unknown knapsack constraints. It
maximizes the objective over the solutions ``x`` that satisfy the known constraints, do not dominate any negative
sub-point, and for which some weights ``w`` consistent with the labeled sub-points (``w @ p <= 1`` for every
positive sub-point ``p`` and ``w @ q >= 1`` for every negative sub-point ``q``) satisfy ``w @ x <= 1``. Since the
true weights are consistent with the labels, the optimal value of this nonconvex mixed-integer quadratic program
is an upper bound on the optimal value of the problem. The model is solved by Gurobi with a limit of 50,000
branch-and-bound nodes, and its dual bound, which remains valid when the node limit is reached, is returned.
"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class Bounder:
    """Base class and factory of the bounders.

    A bounder is obtained with ``Bounder(gurobi_environment=env).get_bounder(name)``.

    Args:
        name: Name of the problem class.
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """
        
    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_bounder(self, name: str):
        """Returns the bounder of the given problem class.

        Args:
            name: Name of the problem class, ``"CSPP"``, ``"GAP"`` or ``"KP"``.

        Returns:
            Bounder: Instance of :class:`CollegeStudyPlanBounder`, :class:`GeneralizedAssignmentBounder` or
            :class:`KnapsackBounder` using the same Gurobi environment.
        """
        assert name in {"CSPP", "GAP", "KP"}

        if name == "CSPP":
            return CollegeStudyPlanBounder(self.gurobi_environment)
        elif name == "GAP":
            return GeneralizedAssignmentBounder(self.gurobi_environment)
        else:  # name == "KP"
            return KnapsackBounder(self.gurobi_environment)

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        """Computes an upper bound on the optimal value of the problem. Implemented by the subclasses.

        Args:
            v: Objective coefficients, shape ``(m, n)``.
            labeled_sub_points: List of length ``m`` whose element ``i`` is a dictionary with keys ``"positive"``
                and ``"negative"``, mapping to the arrays of positive and negative sub-points of constraint ``i``,
                of shapes ``(k_pos_i, n)`` and ``(k_neg_i, n)``.

        Returns:
            float: Upper bound on the optimal value (dual bound of the bounding model).
        """
        pass
        
        
class CollegeStudyPlanBounder(Bounder):
    """Bounder of the college study plan problem (CSPP), with a single unknown constraint (``m = 1``).

    The known constraints are the prerequisites, corequisites and alternatives read from
    ``data/csv/MIT_courses.csv``.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).

    Attributes:
        prerequisites (list): For each course, list of groups of course codes; if the course is taken, at least
            one course of each group must be taken.
        corequisites (list): For each course, list of groups of course codes, with the same meaning as
            ``prerequisites``.
        alternatives (list): For each course, list of course codes that cannot be taken together with it.
    """
    
    def __init__(self, gurobi_environment = None):
        super().__init__("CSPP", gurobi_environment)

        df = pd.read_csv(os.path.join(file_path, "../data/csv/MIT_courses.csv"), sep="|", index_col=False)
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        """Computes an upper bound on the optimal value of the college study plan problem.

        Args:
            v: Credits of the courses, shape ``(1, n)``.
            labeled_sub_points: List of length 1 whose element is a dictionary with keys ``"positive"`` and
                ``"negative"``, mapping to the arrays of positive and negative sub-points of the time budget
                constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.

        Returns:
            float: Upper bound on the optimal value (dual bound of the bounding model).
        """
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
    """Bounder of the generalized assignment problem (GAP), with one unknown capacity constraint per agent.

    The weights ``w[i]`` of the capacity constraints are chosen independently for every agent ``i``, and the
    known constraints assign every job to at most one agent.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """
    
    def __init__(self, gurobi_environment = None):
        super().__init__("GAP", gurobi_environment)

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        """Computes an upper bound on the optimal value of the generalized assignment problem.

        Args:
            v: Revenues of assigning each job to each agent, shape ``(m, n)``.
            labeled_sub_points: List of length ``m`` whose element ``i`` is a dictionary with keys ``"positive"``
                and ``"negative"``, mapping to the arrays of positive and negative sub-points of the capacity
                constraint of agent ``i``, of shapes ``(k_pos_i, n)`` and ``(k_neg_i, n)``.

        Returns:
            float: Upper bound on the optimal value (dual bound of the bounding model).
        """
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
    """Bounder of the knapsack problem (KP), with a single unknown constraint (``m = 1``) and no known
    constraints.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("KP", gurobi_environment)

    def bound(self, v: np.ndarray, labeled_sub_points: list):
        """Computes an upper bound on the optimal value of the knapsack problem.

        Args:
            v: Values of the items, shape ``(1, n)``.
            labeled_sub_points: List of length 1 whose element is a dictionary with keys ``"positive"`` and
                ``"negative"``, mapping to the arrays of positive and negative sub-points of the knapsack
                constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.

        Returns:
            float: Upper bound on the optimal value (dual bound of the bounding model).
        """
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
