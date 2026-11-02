"""Problems: CollegeStudyPlanProblem, GeneralizedAssignmentProblem, KnapsackProblem.

A problem gives access to the 0-1 integer program of a test problem with all the constraints known. It is used to
compute the true optimum of an instance when evaluating the results, and to check whether a point satisfies the
known constraints of the problem.
"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class Problem:
    """Base class and factory of the problems.

    A problem is obtained with ``Problem(gurobi_environment=env).get_problem(name)``.

    Args:
        name: Name of the problem class.
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_problem(self, name: str):
        """Returns the problem of the given class.

        Args:
            name: Name of the problem class, ``"CSPP"``, ``"GAP"`` or ``"KP"``.

        Returns:
            Problem: Instance of :class:`CollegeStudyPlanProblem`, :class:`GeneralizedAssignmentProblem` or
            :class:`KnapsackProblem` using the same Gurobi environment.
        """
        assert name in {"CSPP", "GAP", "KP"}

        if name == "CSPP":
            return CollegeStudyPlanProblem(self.gurobi_environment)
        elif name == "GAP":
            return GeneralizedAssignmentProblem(self.gurobi_environment)
        else:  # name == "KP"
            return KnapsackProblem(self.gurobi_environment)

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        """Solves the problem with all the constraints known. Implemented by the subclasses.

        Args:
            v: Objective coefficients, shape ``(m, n)``.
            w_bar: Weights of the knapsack constraints ``w_bar[i] @ x[i] <= 1``, shape ``(m, n)``.

        Returns:
            np.ndarray | None: Optimal binary solution, shape ``(m, n)``, or ``None`` if the model is not solved
            to optimality.
        """
        pass

    def check(self, u: np.ndarray):
        """Checks whether a point satisfies the known constraints of the problem. Implemented by the subclasses.

        Args:
            u: Binary point, shape ``(m, n)``.

        Returns:
            bool: ``True`` if ``u`` satisfies the known constraints, ``False`` otherwise.
        """
        pass


class CollegeStudyPlanProblem(Problem):
    """College study plan problem (CSPP), with a single knapsack constraint (``m = 1``).

    Maximizes the total credits ``v[0] @ x`` subject to the time budget constraint ``w_bar[0] @ x <= 1`` and
    the prerequisites, corequisites and alternatives read from ``data/csv/MIT_courses.csv``.

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

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        """Solves the college study plan problem with the time budget constraint known.

        Args:
            v: Credits of the courses, shape ``(1, n)``.
            w_bar: Weights of the time budget constraint, shape ``(1, n)``.

        Returns:
            np.ndarray | None: Optimal binary solution, shape ``(1, n)``, or ``None`` if the model is not solved
            to optimality.
        """
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
        """Checks whether a point satisfies the prerequisites, corequisites and alternatives.

        Args:
            u: Binary point, shape ``(1, n)``.

        Returns:
            bool: ``True`` if ``u`` satisfies all the prerequisites, corequisites and alternatives, ``False``
            otherwise.
        """
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
    """Generalized assignment problem (GAP), with one capacity constraint per agent.

    Maximizes the total revenue ``sum(v * x)`` over the assignments ``x`` of shape ``(m, n)`` (``m`` agents,
    ``n`` jobs) subject to the capacity constraints ``w_bar[i] @ x[i] <= 1`` and the assignment of every job to
    at most one agent.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("GAP", gurobi_environment)

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        """Solves the generalized assignment problem with the capacity constraints known.

        Args:
            v: Revenues of assigning each job to each agent, shape ``(m, n)``.
            w_bar: Weights of the capacity constraints (resources divided by capacities), shape ``(m, n)``.

        Returns:
            np.ndarray | None: Optimal binary assignment, shape ``(m, n)``, or ``None`` if the model is not
            solved to optimality.
        """
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
        """Checks whether every row of a point sums to at most 1.

        Args:
            u: Binary point, shape ``(m, n)``.

        Returns:
            bool: ``True`` if every row of ``u`` sums to at most 1, ``False`` otherwise.
        """
        return (u.sum(axis=1) <= 1).all()

class KnapsackProblem(Problem):
    """Knapsack problem (KP), with a single knapsack constraint (``m = 1``).

    Maximizes ``v[0] @ x`` subject to the knapsack constraint ``w_bar[0] @ x <= 1``.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("KP", gurobi_environment)

    def solve(self, v: np.ndarray, w_bar: np.ndarray):
        """Solves the knapsack problem with the knapsack constraint known.

        Args:
            v: Values of the items, shape ``(1, n)``.
            w_bar: Weights of the knapsack constraint (weights divided by the capacity), shape ``(1, n)``.

        Returns:
            np.ndarray | None: Optimal binary solution, shape ``(1, n)``, or ``None`` if the model is not solved
            to optimality.
        """
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
        """Checks whether a point satisfies the known constraints, of which the knapsack problem has none.

        Args:
            u: Binary point, shape ``(1, n)``.

        Returns:
            bool: Always ``True``.
        """
        return True
