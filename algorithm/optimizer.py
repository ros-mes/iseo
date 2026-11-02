"""Optimizers: CollegeStudyPlanOptimizer, GeneralizedAssignmentOptimizer, KnapsackOptimizer.

An optimizer solves the 0-1 integer program of a test problem in which every unknown knapsack constraint
``w_bar[i] @ x[i] <= 1`` is replaced by its surrogate ``w_hat[i] @ x[i] <= 1``. The solutions whose label can be
inferred by monotonicity from the labeled points are excluded, so that the optimal solution is a new point worth
submitting to the oracles.
"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class Optimizer:
    """Base class and factory of the optimizers.

    An optimizer is obtained with ``Optimizer(gurobi_environment=env).get_optimizer(name)``.

    Args:
        name: Name of the problem class.
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_optimizer(self, name: str):
        """Returns the optimizer of the given problem class.

        Args:
            name: Name of the problem class, ``"CSPP"``, ``"GAP"`` or ``"KP"``.

        Returns:
            Optimizer: Instance of :class:`CollegeStudyPlanOptimizer`, :class:`GeneralizedAssignmentOptimizer`
            or :class:`KnapsackOptimizer` using the same Gurobi environment.
        """
        assert name in {"CSPP", "GAP", "KP"}

        if name == "CSPP":
            return CollegeStudyPlanOptimizer(self.gurobi_environment)
        elif name == "GAP":
            return GeneralizedAssignmentOptimizer(self.gurobi_environment)
        else:  # name == "KP"
            return KnapsackOptimizer(self.gurobi_environment)

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list):
        """Solves the problem with the unknown constraints replaced by their surrogates. Implemented by the
        subclasses.

        Args:
            v: Objective coefficients, shape ``(m, n)``.
            w_hat: Surrogate weights of the unknown constraints, shape ``(m, n)``.
            labeled_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative points, of shapes ``(k_pos, m, n)`` and ``(k_neg, m, n)``.
            labeled_sub_points: List of length ``m`` whose element ``i`` is a dictionary with keys ``"positive"``
                and ``"negative"``, mapping to the arrays of positive and negative sub-points of constraint ``i``,
                of shapes ``(k_pos_i, n)`` and ``(k_neg_i, n)``.

        Returns:
            np.ndarray | None: Optimal binary solution, shape ``(m, n)``, or ``None`` if the model is not solved
            to optimality.
        """
        pass


class CollegeStudyPlanOptimizer(Optimizer):
    """Optimizer of the college study plan problem (CSPP), with a single unknown constraint (``m = 1``).

    Maximizes the total credits ``v[0] @ x`` subject to the surrogate time budget constraint
    ``w_hat[0] @ x <= 1``, the prerequisites, corequisites and alternatives read from
    ``data/csv/MIT_courses.csv``, and the exclusion of the solutions dominated by a positive point or dominating
    a negative point.

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

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list = None):
        """Solves the college study plan problem with the surrogate time budget constraint.

        Args:
            v: Credits of the courses, shape ``(1, n)``.
            w_hat: Surrogate weights of the time budget constraint, shape ``(1, n)``.
            labeled_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative points, of shapes ``(k_pos, 1, n)`` and ``(k_neg, 1, n)``.
            labeled_sub_points: Unused.

        Returns:
            np.ndarray | None: Optimal binary solution, shape ``(1, n)``, or ``None`` if the model is not solved
            to optimality.
        """
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
    """Optimizer of the generalized assignment problem (GAP), with one unknown capacity constraint per agent.

    Maximizes the total revenue ``sum(v * x)`` over the assignments ``x`` of shape ``(m, n)`` (``m`` agents,
    ``n`` jobs) subject to the surrogate capacity constraints ``w_hat[i] @ x[i] <= 1``, the assignment of every
    job to at most one agent, the exclusion of the solutions dominated by a positive point, and, for every agent
    ``i``, the exclusion of the rows ``x[i]`` dominating a negative sub-point of constraint ``i``.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("GAP", gurobi_environment)

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list):
        """Solves the generalized assignment problem with the surrogate capacity constraints.

        Args:
            v: Revenues of assigning each job to each agent, shape ``(m, n)``.
            w_hat: Surrogate weights of the capacity constraints, shape ``(m, n)``.
            labeled_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative points, of shapes ``(k_pos, m, n)`` and ``(k_neg, m, n)``.
            labeled_sub_points: List of length ``m`` whose element ``i`` is a dictionary with keys ``"positive"``
                and ``"negative"``, mapping to the arrays of positive and negative sub-points of constraint ``i``,
                of shapes ``(k_pos_i, n)`` and ``(k_neg_i, n)``.

        Returns:
            np.ndarray | None: Optimal binary assignment, shape ``(m, n)``, or ``None`` if the model is not
            solved to optimality.
        """
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
    """Optimizer of the knapsack problem (KP), with a single unknown constraint (``m = 1``).

    Maximizes ``v[0] @ x`` subject to the surrogate constraint ``w_hat[0] @ x <= 1`` and the exclusion of the
    solutions dominated by a positive point or dominating a negative point.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("KP", gurobi_environment)

    def optimize(self, v: np.ndarray, w_hat: np.ndarray, labeled_points: dict, labeled_sub_points: list = None):
        """Solves the knapsack problem with the surrogate knapsack constraint.

        Args:
            v: Values of the items, shape ``(1, n)``.
            w_hat: Surrogate weights of the knapsack constraint, shape ``(1, n)``.
            labeled_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative points, of shapes ``(k_pos, 1, n)`` and ``(k_neg, 1, n)``.
            labeled_sub_points: Unused.

        Returns:
            np.ndarray | None: Optimal binary solution, shape ``(1, n)``, or ``None`` if the model is not solved
            to optimality.
        """
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
