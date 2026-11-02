"""Linear separators: SEP, SVM.

A linear separator learns, from the labeled sub-points of an unknown knapsack constraint, surrogate weights
``w_hat`` such that the hyperplane ``w_hat @ x = 1`` separates the positive (feasible) sub-points from the
negative (infeasible) ones. The surrogate constraint ``w_hat @ x <= 1`` then replaces the unknown one.
"""

import gurobipy as gp
import numpy as np

from algorithm.exception import ModelResolutionFailedError


class Separator:
    """Base class and factory of the linear separators.

    A separator is obtained with ``Separator(gurobi_environment=env).get_linear_separator(name)``.

    Args:
        name: Name of the separator.
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, name: str = None, gurobi_environment = None):
        self.name = name
        self.gurobi_environment = gurobi_environment

    def get_linear_separator(self, name: str):
        """Returns the linear separator with the given name.

        Args:
            name: Name of the separator, ``"SEP"`` or ``"SVM"``.

        Returns:
            Separator: Instance of :class:`SEP` or :class:`SVM` using the same Gurobi environment.
        """
        assert name in {"SEP", "SVM"}

        if name == "SEP":
            return SEP(self.gurobi_environment)
        else:  # name == "SVM"
            return SVM(self.gurobi_environment)

    def separate(self, labeled_sub_points: dict, verbose: int = 0):
        """Computes the surrogate weights of an unknown constraint. Implemented by the subclasses.

        Args:
            labeled_sub_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative sub-points of the constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.
            verbose: Gurobi output flag (``0`` silent, ``1`` verbose).

        Returns:
            np.ndarray: Surrogate weights ``w_hat``, shape ``(n,)``.

        Raises:
            ModelResolutionFailedError: If the separation model cannot be solved to optimality.
        """
        pass


class SEP(Separator):
    """Linear separator inspired by convex optimization in the oracle model (SEP).

    The labeled sub-points and the prior knowledge on the weights are encoded as vectors of size ``n + 1``:
    ``(x, 1)`` for every positive sub-point ``x``, ``(-x, -1)`` for every negative sub-point ``x``, the same
    encoding for the all-zeros point and the unit vectors (positive) and for the all-ones point (negative), and
    ``(-e_j, 0)`` for the nonnegativity of every weight ``j``. After normalizing these vectors with
    :meth:`dual_norm`, the point ``gamma`` of their convex hull minimizing
    ``0.25 * (n * ||gamma[:n]||^2 + gamma[n]^2)`` is computed by solving a convex quadratic program, and the
    surrogate weights are obtained from the gradient ``g`` of this function at ``gamma`` as
    ``w_hat = -g[:n] / g[n]``. When the minimum is nonzero, ``g`` has a positive inner product with every
    encoded vector, so ``w_hat`` strictly separates the labeled sub-points and satisfies the prior knowledge.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("SEP", gurobi_environment)

    @ staticmethod
    def dual_norm(x: np.ndarray):
        """Computes the norm used to normalize the vectors of the separation model.

        Args:
            x: Vector of size ``n + 1``.

        Returns:
            float: ``sqrt(((n + 1) * ||x[:n]||^2 + 1) / 2)``.
        """
        return (1 / np.sqrt(2)) * np.linalg.norm(np.append(x[:-1] * np.sqrt(x.size), 1))

    def separate(self, labeled_sub_points: dict, verbose: int = 0):
        """Computes the surrogate weights of an unknown constraint with SEP.

        Args:
            labeled_sub_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative sub-points of the constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.
            verbose: Gurobi output flag (``0`` silent, ``1`` verbose).

        Returns:
            np.ndarray: Surrogate weights ``w_hat``, shape ``(n,)``.

        Raises:
            ModelResolutionFailedError: If the quadratic program is not solved to optimality.
        """
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
    """Hard-margin support vector machine with nonnegative weights (SVM).

    Solves the convex quadratic program ``min ||w||^2`` subject to ``w @ x <= b - 1`` for every positive
    sub-point ``x``, ``w @ x >= b + 1`` for every negative sub-point ``x`` and ``0 <= w <= b``, and returns
    ``w_hat = w / b``, so that the separating hyperplane is ``w_hat @ x = 1``. If this program cannot be solved
    to optimality, a fallback model without margin is solved instead: find ``w`` and ``b`` with
    ``w @ x <= b`` for every positive sub-point, ``w @ x >= b`` for every negative sub-point, ``0 <= w <= b``
    and ``||w||^2 = 1``. Each model is attempted with the barrier, dual simplex and primal simplex methods, in
    this order, until one of them returns an optimal solution.

    Args:
        gurobi_environment: Gurobi environment in which the models are built (``None`` for the default
            environment).
    """

    def __init__(self, gurobi_environment = None):
        super().__init__("SVM", gurobi_environment)

    def separate(self, labeled_sub_points: dict, verbose: int = 0):
        """Computes the surrogate weights of an unknown constraint with SVM.

        Args:
            labeled_sub_points: Dictionary with keys ``"positive"`` and ``"negative"``, mapping to the arrays of
                positive and negative sub-points of the constraint, of shapes ``(k_pos, n)`` and ``(k_neg, n)``.
            verbose: Gurobi output flag (``0`` silent, ``1`` verbose).

        Returns:
            np.ndarray: Surrogate weights ``w_hat``, shape ``(n,)``.

        Raises:
            ModelResolutionFailedError: If neither the margin model nor the fallback model is solved to
                optimality.
        """
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
