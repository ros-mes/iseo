"""ISEO: Interactive Sampling-Enhanced Optimization"""

import logging
import os
import time

import gurobipy as gp
import h5py
import numpy as np

from algorithm.bounder import Bounder
from algorithm.exception import ModelResolutionFailedError
from algorithm.separator import Separator
from algorithm.optimizer import Optimizer
from algorithm.problem import Problem
from algorithm.oracle import Oracle
from algorithm.sampler import Sampler

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)


class ISEO:

    def __init__(self, N: int, thr: float, T: int, separator_name: str, sampler_name: str, gurobi_environment = None):

        assert N >= 0
        assert thr >= 0
        assert separator_name in {"SEP", "SVM"}
        assert sampler_name in {"CUT", "SIM"}

        self.N = N
        self.thr = thr
        self.T = T
        self.separator_name = separator_name
        self.sampler_name = sampler_name
        self.gurobi_environment = gurobi_environment

    def init_log(self, problem_name, instance_name):

        log_dir = os.path.join(file_path, "../output/log/{}_{}/{}".format(
            self.separator_name, self.sampler_name, problem_name))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        root_logger = logging.getLogger()

        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler("{}/{}.log".format(log_dir, instance_name))
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.setLevel(logging.INFO)

    def run(self, problem_name: str, instance_name: str, v: np.ndarray, w_bar: np.ndarray, labeled_points: dict):

        start_time = time.time()

        # Initialization

        m = v.shape[0]
        n = v.shape[1]

        separator = Separator(gurobi_environment=self.gurobi_environment).get_linear_separator(self.separator_name)
        sampler = Sampler(gurobi_environment=self.gurobi_environment).get_sampling_strategy(self.sampler_name)
        optimizer = Optimizer(gurobi_environment=self.gurobi_environment).get_optimizer(problem_name)
        problem = Problem(gurobi_environment=self.gurobi_environment).get_problem(problem_name)
        oracles = [Oracle(w_bar[i]) for i in range(m)]
        bounder = Bounder(gurobi_environment=self.gurobi_environment).get_bounder(problem_name)

        t = 0
        flag = True
        UB = np.inf
        LB = np.max([(v * x).sum() for x in labeled_points["positive"]])
        labeled_sub_points = [{"positive": np.array([point[i] for point in labeled_points["positive"]]),
                               "negative": np.empty((0, n))} for i in range(m)]
        for i in range(m):
            for point in labeled_points["negative"]:
                if oracles[i].label(point[i]) == 1:
                    labeled_sub_points[i]["positive"] = labeled_sub_points[i]["positive"][
                        np.where([not x for x in (labeled_sub_points[i]["positive"] <= point[i]).all(axis=1)])]
                    labeled_sub_points[i]["positive"] = np.concat(
                        [labeled_sub_points[i]["positive"], [point[i]]])
                else:  # oracles[i].label(point[i]) == -1
                    labeled_sub_points[i]["negative"] = labeled_sub_points[i]["negative"][
                        np.where([not x for x in (labeled_sub_points[i]["negative"] >= point[i]).all(axis=1)])]
                    labeled_sub_points[i]["negative"] = np.concat(
                        [labeled_sub_points[i]["negative"], [point[i]]])

        # Logging

        print(f"Instance: {instance_name}")

        self.init_log(problem_name, instance_name)

        logging.info("Instance: {}".format(instance_name))
        logging.info("Oracle budget: {}".format(self.N))
        logging.info("Threshold: {}".format(self.thr))
        logging.info("Linear separator: {}".format(self.separator_name))
        logging.info("Sampling strategy: {}".format(self.sampler_name))

        logging.info("-" * 80)
        logging.info("Initialization")
        logging.info("Upper bound: {}".format(UB))
        logging.info("Lower bound: {}".format(LB))
        logging.info("Gap: {}".format((UB - LB) / LB if LB > 0 else np.inf))
        logging.info("N. labeled points: {} positive, {} negative".format(
            labeled_points["positive"].shape[0], labeled_points["negative"].shape[0]))
        logging.info("N. labeled sub-points: {}".format(" | ".join(["{} positive, {} negative".format(
            labeled_sub_points[i]["positive"].shape[0], labeled_sub_points[i]["negative"].shape[0])
            for i in range(m)])))
        logging.info("N. oracle calls: {}".format([oracles[i].n_calls for i in range(m)]))

        h5_dir = os.path.join(file_path, "../output/h5/{}_{}/{}".format(
            self.separator_name, self.sampler_name, problem_name))
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)
        with (h5py.File("{}/{}.h5".format(h5_dir, instance_name), "w") as f):

            group = f.create_group("initial")
            group.attrs["UB"] = UB
            group.attrs["LB"] = LB
            group.attrs["n_labeled_points"] = [labeled_points["positive"].shape[0],
                                               labeled_points["negative"].shape[0]]
            group.attrs["n_labeled_sub_points"] = [[labeled_sub_points[i]["positive"].shape[0],
                                                    labeled_sub_points[i]["negative"].shape[0]] for i in range(m)]
            group.attrs["n_oracle_calls"] = [oracles[i].n_calls for i in range(m)]
            group.create_dataset("labeled_points_positive", data=labeled_points["positive"])
            group.create_dataset("labeled_points_negative", data=labeled_points["negative"])
            for i in range(m):
                group.create_dataset("labeled_sub_points_{}_positive".format(i), data=labeled_sub_points[i]["positive"])
                group.create_dataset("labeled_sub_points_{}_negative".format(i), data=labeled_sub_points[i]["negative"])

            # Main loop

            while (((UB - LB) / LB if LB > 0 else np.inf) > self.thr and flag is True and t < self.T
                and time.time() - start_time < 500000):  # Stopping criterion

                iteration_start_time = time.time()

                t = t + 1

                print(f"Iteration {t}")

                logging.info("-" * 80)
                logging.info(f"Iteration {t}")
                group = f.create_group(f"iter_{t}")

                w_hat = []
                failure = False
                for i in range(m):
                    try:
                        w_hat.append(separator.separate(labeled_sub_points[i]))
                    except ModelResolutionFailedError as e:
                        logging.error(str(e) + f"(i)")
                        failure = True
                if failure:
                    break
                w_hat = np.array(w_hat)
                logging.info("Linear separations completed")
                group.create_dataset("w_hat", data=w_hat)

                new_labeled_sub_points = [{"positive": np.empty((0, n)), "negative": np.empty((0, n))}
                                          for _ in range(m)]

                for theta in ["sampling", "optimization"]:

                    if (np.array([oracles[i].n_calls for i in range(m)]) >= self.N).any():
                        logging.info("An oracle budget was reached")
                        flag = False
                        break

                    if theta == "sampling":
                        u = []
                        failure = False
                        for i in range(m):
                            try:
                                u.append(sampler.sample(w_hat[i], labeled_sub_points[i]))
                            except ModelResolutionFailedError as e:
                                logging.error(str(e) + f"(i)")
                                failure = True
                        if failure:
                            break
                        if any(x is None for x in u):
                            logging.info("Some samplings returned nothing: {}".format(
                                [i for i in range(m) if u[i] is None]))
                            group.create_dataset("u_sampling",
                                                 data=[(x if x is not None else -1 * np.ones(n)) for x in u])
                        else:
                            u = np.array(u)
                            logging.info("Samplings completed")
                            group.create_dataset("u_sampling", data=u)
                    else:  # theta == "optimization"
                        u = optimizer.optimize(v, w_hat, labeled_points, labeled_sub_points)  # Optimization
                        if u is None:
                            logging.info("Optimization returned nothing")
                            group.create_dataset("u_optimization", data=-1 * np.ones((m, n)))
                        else:
                            logging.info("Optimization completed")
                            group.create_dataset("u_optimization", data=u)

                    if u is not None:
                        l = np.zeros(m)
                        n_labeled_sub_points = {"positive": 0, "negative": 0}
                        for i in range(m):
                            if u[i] is not None:
                                if (u[i] <= labeled_sub_points[i]["positive"]).all(axis=1).any():
                                    l[i] = 1
                                    n_labeled_sub_points["positive"] += 1
                                elif (u[i] >= labeled_sub_points[i]["negative"]).all(axis=1).any():
                                    l[i] = -1
                                    n_labeled_sub_points["negative"] += 1
                                else:
                                    l[i] = oracles[i].label(u[i])  # Oracle call
                                    if l[i] == 1:
                                        new_labeled_sub_points[i]["positive"] = np.concat(
                                            [new_labeled_sub_points[i]["positive"], [u[i]]])
                                        n_labeled_sub_points["positive"] += 1
                                    else:
                                        new_labeled_sub_points[i]["negative"] = np.concat(
                                            [new_labeled_sub_points[i]["negative"], [u[i]]])
                                        n_labeled_sub_points["negative"] += 1
                        logging.info("{} sub-points were labeled positive".format(n_labeled_sub_points["positive"]))
                        logging.info("{} sub-points were labeled negative".format(n_labeled_sub_points["negative"]))
                        if theta == "optimization":
                            group.create_dataset("labels_optimization", data=l)
                            if m == 1:
                                new_LB_found = False
                                for mu in np.unique(new_labeled_sub_points[0]["positive"], axis=0):
                                    if problem.check([mu]):
                                        labeled_points["positive"] = np.concat([labeled_points["positive"], [[mu]]])
                                        new_LB = (v * [mu]).sum()
                                        if new_LB > LB:
                                            LB = new_LB
                                    else:
                                        labeled_points["negative"] = np.concat([labeled_points["positive"], [[mu]]])
                                if new_LB_found is True:
                                    logging.info("New lower bound")
                                if new_labeled_sub_points[0]["negative"].shape[0] > 0:
                                    labeled_points["negative"] = np.concat([
                                        labeled_points["negative"],
                                        [[mu] for mu in np.unique(new_labeled_sub_points[0]["negative"], axis=0)]])
                                    logging.info("New infeasible point(s)")
                            else:
                                if (l == 1).all():
                                    labeled_points["positive"] = np.concat([labeled_points["positive"], [u]])
                                    logging.info("New feasible point")
                                    if (v * u).sum() > LB:
                                        LB = (v * u).sum()
                                        logging.info("New lower bound")
                                else:
                                    labeled_points["negative"] = np.concat([labeled_points["negative"], [u]])
                                    logging.info("New infeasible point")
                        else:  # theta == "sampling"
                            group.create_dataset("labels_sampling", data=l)

                for i in range(m):
                    labeled_sub_points[i]["positive"] = np.concat(
                        [labeled_sub_points[i]["positive"], np.unique(new_labeled_sub_points[i]["positive"], axis=0)])
                    labeled_sub_points[i]["negative"] = np.concat(
                        [labeled_sub_points[i]["negative"], np.unique(new_labeled_sub_points[i]["negative"], axis=0)])
                for i in range(m):
                    group.create_dataset("new_labeled_sub_points_{}_positive".format(i),
                                         data=new_labeled_sub_points[i]["positive"])
                    group.create_dataset("new_labeled_sub_points_{}_negative".format(i),
                                         data=new_labeled_sub_points[i]["negative"])

                new_UB = bounder.bound(v, labeled_sub_points)  # Bounding
                if new_UB < UB:
                    UB = new_UB
                    logging.info("New upper bound")
                if UB == LB:
                    logging.info("Optimal solution found")

                logging.info("Upper bound: {}".format(UB))
                logging.info("Lower bound: {}".format(LB))
                logging.info("Gap: {}".format((UB - LB) / LB if LB > 0 else np.inf))
                logging.info("N. labeled points: {} positive, {} negative".format(
                    labeled_points["positive"].shape[0], labeled_points["negative"].shape[0]))
                logging.info("N. labeled sub-points: {}".format(" | ".join(["{} positive, {} negative".format(
                    labeled_sub_points[i]["positive"].shape[0], labeled_sub_points[i]["negative"].shape[0])
                    for i in range(m)])))
                logging.info("N. oracle calls: {}".format([oracles[i].n_calls for i in range(m)]))

                group.attrs["UB"] = UB
                group.attrs["LB"] = LB
                group.attrs["gap"] = (UB - LB) / LB if LB > 0 else np.inf
                group.attrs["flag"] = flag
                group.attrs["n_labeled_points"] = [labeled_points["positive"].shape[0],
                                                   labeled_points["negative"].shape[0]]
                group.attrs["n_labeled_sub_points"] = [[labeled_sub_points[i]["positive"].shape[0],
                                                        labeled_sub_points[i]["negative"].shape[0]] for i in range(m)]
                group.attrs["n_oracle_calls"] = [oracles[i].n_calls for i in range(m)]

                iteration_running_time = time.time() - iteration_start_time
                logging.info("Running time: {}".format(iteration_running_time))
                group.attrs["running_time"] = iteration_running_time

            x_hat = labeled_points["positive"][np.argmax([(v * x).sum() for x in labeled_points["positive"]])]
            w_hat = np.array([separator.separate(labeled_sub_points[i]) for i in range(m)])

            group = f.create_group("final")
            group.create_dataset("x_hat", data=x_hat)
            group.create_dataset("w_hat", data=w_hat)

        logging.info("-" * 80)
        logging.info("End. Total running time: {} s".format(time.time() - start_time))

        return x_hat, w_hat
