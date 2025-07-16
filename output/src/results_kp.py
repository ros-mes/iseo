import os
import pickle

import h5py
import numpy as np

from algorithm.problem import Problem


# Log function
def log(f, message):
    print(message)
    f.write(message + "\n")

# Problem
problem = Problem().get_problem(name="KP")

# Threshold
thr = 0.01

# Metrics directory
metrics_dir = f"../metrics"
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)
    
with (open("{}/KP.txt".format(metrics_dir), "w") as f):

    log(f, "Type | Separator | Sampler | # Solved | # Thresholds | # Feas. solutions on thr. | Gap | Error | # Calls | # Iterations | Running time | "
           "# Calls to thr. | # Iterations to thr. | Running time to thr. | # Iterations to opt. | # Calls to opt. | Running time to opt.")

    for instance_type in ["uncorrelated", "weakly-correlated", "strongly-correlated"]:
        for separator_name, sampler_name in [("SEP", "CUT"), ("SEP", "SIM"), ("SVM", "CUT"), ("SVM", "SIM")]:

            # Metrics
            n_thresholds_reached = 0
            n_solved_instances = 0
            n_feas_solution_on_thr = 0
            gap = []
            error = []
            n_calls = []
            n_iterations = []
            running_time = []
            n_calls_to_thr = []
            n_iterations_to_thr = []
            running_time_to_thr = []
            n_calls_to_opt = []
            n_iterations_to_opt = []
            running_time_to_opt = []

            for instance_number in range(1, 31):

                # Instance
                with open(f"../../data/instances/KP/KP_{instance_type}_{instance_number}.pickle", "br") as g:
                    instance = pickle.load(g)
                w_bar = np.array([instance["weights"] / instance["capacity"]])
                v = np.array([instance["values"]])
                n = v.shape[1]
                opt_solution = problem.solve(v, w_bar)
                opt_value = (v * opt_solution).sum()

                # H5 file
                file = \
                h5py.File(f"../output/h5/{separator_name}_{sampler_name}/KP/KP_{instance_type}_{instance_number}.h5", "r")

                # Extraction
                iteration_numbers = sorted([eval(name.split("_")[1]) for name in file.keys()
                                            if name.startswith("iter")])
                n_iterations.append(iteration_numbers[-1])

                labeled_points = {"positive": np.empty((0, 1, n)), "negative": np.empty((0, 1, n))}
                labeled_points["positive"] = np.concat(
                    [labeled_points["positive"], file["initial"][f"labeled_points_positive"]])
                LB_it = [np.max([np.sum(v * p) for p in labeled_points["positive"]])]
                gap_it = [(file[f"initial"].attrs["UB"] - LB_it[0]) / LB_it[0]]
                for it in iteration_numbers:
                    if np.array(file[f"iter_{it}"][f"new_labeled_sub_points_0_positive"]).shape[0] > 0:
                        labeled_points["positive"] = np.concat(
                            [labeled_points["positive"],
                             [[p] for p in file[f"iter_{it}"][f"new_labeled_sub_points_0_positive"]]])
                        LB_it.append(np.max([np.sum(v * p) for p in labeled_points["positive"]]))
                        gap_it.append((file[f"iter_{it}"].attrs["UB"] - LB_it[it]) / LB_it[it])
                    else:
                        LB_it.append(LB_it[-1])
                        gap_it.append(gap_it[-1])

                gap.append(gap_it[-1])
                error.append((opt_value - LB_it[-1]) / opt_value)
                n_calls.append(file[f"iter_{n_iterations[-1]}"].attrs["n_oracle_calls"].sum())
                running_time.append(np.array([file[f"iter_{it}"].attrs["running_time"]
                                              for it in iteration_numbers]).sum())

                thr_reached = False
                opt_reached = False
                error_it = []
                for iteration in iteration_numbers:
                    if thr_reached is False and gap_it[iteration] < thr:
                        thr_reached = True
                        n_thresholds_reached += 1
                        n_calls_to_thr.append(file[f"iter_{iteration}"].attrs["n_oracle_calls"].sum())
                        n_iterations_to_thr.append(iteration)
                        running_time_to_thr.append(sum([file[f"iter_{it}"].attrs["running_time"]
                                                        for it in range(1, iteration + 1)]))
                        w_hat = np.array(file[f"iter_{iteration}"]["w_hat"])
                        n_feas_solution_on_thr += int((w_bar * problem.solve(v, w_hat) <= 1).all())
                    if opt_reached is False and LB_it[iteration] == opt_value:
                        opt_reached = True
                        n_solved_instances += 1
                        n_calls_to_opt.append(file[f"iter_{iteration}"].attrs["n_oracle_calls"].sum())
                        n_iterations_to_opt.append(iteration)
                        running_time_to_opt.append(sum([file[f"iter_{it}"].attrs["running_time"]
                                                        for it in range(1, iteration + 1)]))
                    if opt_reached is False:
                        error_it.append((opt_value - LB_it[iteration]) / opt_value)

            # List to array
            gap = np.array(gap)
            error = np.array(error)
            n_calls = np.array(n_calls)
            n_iterations = np.array(n_iterations)
            running_time = np.array(running_time)
            n_calls_to_thr = np.array(n_calls_to_thr)
            n_iterations_to_thr = np.array(n_iterations_to_thr)
            running_time_to_thr = np.array(running_time_to_thr)
            n_calls_to_opt = np.array(n_calls_to_opt)
            n_iterations_to_opt = np.array(n_iterations_to_opt)
            running_time_to_opt = np.array(running_time_to_opt)

            # Log
            log(f, "{} | {} | {} | {}/30 | {}/30 | {}/30 | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}".format(
                instance_type,
                separator_name,
                sampler_name,
                n_solved_instances,
                n_thresholds_reached,
                n_feas_solution_on_thr,
                round(gap.mean(), 5),
                round(error.mean(), 5),
                round(n_calls.mean(), 2),
                round(n_iterations.mean(), 2),
                round(running_time.mean(), 2),
                None if n_calls_to_thr.size == 0 else round(n_calls_to_thr.mean(), 2),
                None if n_iterations_to_thr.size == 0 else round(n_iterations_to_thr.mean(), 2),
                None if running_time_to_thr.size == 0 else round(running_time_to_thr.mean(), 2),
                None if n_calls_to_opt.size == 0 else round(n_calls_to_opt.mean(), 2),
                None if n_iterations_to_opt.size == 0 else round(n_iterations_to_opt.mean(), 2),
                None if running_time_to_opt.size == 0 else round(running_time_to_opt.mean(), 2)))
