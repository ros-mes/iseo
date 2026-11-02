"""Sampling strategies: CUT, SIM.

A sampling strategy selects, for an unknown knapsack constraint, a new point to be labeled. The candidates are
the binary points whose label cannot be inferred by monotonicity from the labeled sub-points: they are not
dominated by any positive sub-point and do not dominate any negative sub-point. The selection model is solved
by Gurobi with a limit of 50,000 branch-and-bound nodes.
"""

import gurobipy as gp
import numpy as np

from algorithm.exception import ModelResolutionFailedError


class Sampler:
    """Base class and factory of the sampling strategies.

    A sampling strategy is obtained with ``Sampler(gurobi_environment=env).get_sampling_strategy(name)``.

    Args:
        name: Name of the sampling strategy.
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_sampling_strategy(self, name: str):
        """Returns the sampling strategy with the given name.

        Args:
            name: Name of the sampling strategy, ``"CUT"`` or ``"SIM"``.

        Returns:
            Sampler: Instance of :class:`CUT` or :class:`SIM` using the same Gurobi environment.
        """
        assert name in {"CUT", "SIM"}

        if name == "CUT":
            return CUT(self.gurobi_environment)
        else:  # name == "SIM"
            return SIM(self.gurobi_environment)

    def sample(self, w_hat: np.ndarray, labeled_sub_points: dict):
        """Selects a new point to be labeled for an unknown constraint. Implemented by the subclasses.

        Args:
            w_hat: Current surrogate weights of the constraint, shape ``(n,)``.
            labeled_sub_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative sub-points of the constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.

        Returns:
            np.ndarray | None: Selected binary point, shape ``(n,)``, or ``None`` if the model terminates with
            a status other than optimal or node limit (e.g. when the label of every point can be inferred).

        Raises:
            ModelResolutionFailedError: If the model terminates without providing a solution.
        """
        pass


class CUT(Sampler):
    """Closest cutting plane sampling strategy (CUT).

    Every candidate point ``mu`` induces the hyperplane ``{p : p @ mu = 1}`` in the space of the weights, which
    its label would use to cut the set of weights consistent with the labeled sub-points. CUT selects the
    candidate whose hyperplane is closest, in Euclidean distance, to the current surrogate weights ``w_hat``, by
    solving the nonconvex mixed-integer quadratic program ``min ||p - w_hat||^2`` subject to ``p @ mu = 1``.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("CUT", gurobi_environment)

    def sample(self, w_hat: np.ndarray, labeled_sub_points: dict):
        """Selects a new point to be labeled for an unknown constraint with CUT.

        Args:
            w_hat: Current surrogate weights of the constraint, shape ``(n,)``.
            labeled_sub_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative sub-points of the constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.

        Returns:
            np.ndarray | None: Selected binary point, shape ``(n,)``, or ``None`` if the model terminates with
            a status other than optimal or node limit.

        Raises:
            ModelResolutionFailedError: If the model terminates without providing a solution.
        """
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
    """Simple margin sampling strategy (SIM).

    Selects the candidate point ``mu`` minimizing ``|1 + w_hat @ mu|``, by solving a mixed-integer program.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("SIM", gurobi_environment)

    def sample(self, w_hat: np.ndarray, labeled_sub_points: dict):
        """Selects a new point to be labeled for an unknown constraint with SIM.

        Args:
            w_hat: Current surrogate weights of the constraint, shape ``(n,)``.
            labeled_sub_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative sub-points of the constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.

        Returns:
            np.ndarray | None: Selected binary point, shape ``(n,)``, or ``None`` if the model terminates with
            a status other than optimal or node limit.

        Raises:
            ModelResolutionFailedError: If the model terminates without providing a solution.
        """
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
