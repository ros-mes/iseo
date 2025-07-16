"""Test: generalized assignment problem"""

import argparse
import ast
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gurobipy as gp
import numpy as np
import pickle

from algorithm.iseo import ISEO

file_path = os.path.join(os.path.dirname(__file__))
file_path = os.path.abspath(file_path)

parser = argparse.ArgumentParser(prog="Test CSPP")
parser.add_argument("--separator")
parser.add_argument("--sampler")
parser.add_argument("--agents")
parser.add_argument("--first")
parser.add_argument("--last")
args = parser.parse_args()

N = 2000
thr = 0.0
T = 2 * N
separator_name = args.separator
sampler_name = args.sampler

assert separator_name in {"SEP", "SVM"}
assert sampler_name in {"CUT", "SIM"}

gurobi_environment = gp.Env(empty=True)
gurobi_environment.setParam("ThreadLimit", 2)
gurobi_environment.start()

first = ast.literal_eval(args.first) if args.first else 1
last = ast.literal_eval(args.last) if args.last else 5

assert 1 <= first <= 5
assert 1 <= last <= 5
assert first <= last

if args.agents is None:
    n_agents = None
    n_agents_values = [5, 8, 10]
else:
    n_agents = ast.literal_eval(args.agents)
    assert n_agents in {5, 8, 10}
    n_agents_values = [n_agents]
instance_numbers = range(first, last + 1)

iseo = ISEO(N, thr, T, separator_name, sampler_name, gurobi_environment)

for n_agents in n_agents_values:
    n_jobs_values = n_agents * np.array([3, 4, 5, 6])
    for n_jobs in n_jobs_values:
        for instance_number in instance_numbers:
            h5_file = os.path.join(file_path, "../output/h5/{}_{}/GAP/GAP_{}_{}_{}.h5".format(
                separator_name, sampler_name, n_agents, n_jobs, instance_number))
            if os.path.exists(h5_file):
                os.remove(h5_file)
            log_file = os.path.join(file_path, "../output/log/{}_{}/GAP/GAP_{}_{}_{}.log".format(
                separator_name, sampler_name, n_agents, n_jobs, instance_number))
            if os.path.exists(log_file):
                os.remove(log_file)
            with open(os.path.join(file_path, "../data/instances/GAP/GAP_{}_{}_{}.pickle".format(
                    n_agents, n_jobs, instance_number)), "br") as f:
                instance = pickle.load(f)
            v = instance["revenues"]
            w_bar = instance["resources"] / instance["capacities"].reshape(-1, 1)
            labeled_points = {"positive": np.array([np.zeros(v.shape)]),
                              "negative": np.array([np.ones(v.shape)])}
            iseo.run(problem_name="GAP", instance_name=instance["name"], v=v, w_bar=w_bar,
                     labeled_points=labeled_points)
