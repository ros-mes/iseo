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
parser.add_argument("--first")
parser.add_argument("--last")
args = parser.parse_args()

N = 2000
thr = 0.0
T = 2 * N
separator_name = args.separator if args.separator else "SEP"
sampler_name = args.sampler if args.separator else "CUT"

assert separator_name in {"SEP", "SVM"}
assert sampler_name in {"CUT", "SIM"}

gurobi_environment = gp.Env(empty=True)
gurobi_environment.setParam("ThreadLimit", 2)
gurobi_environment.start()

first = ast.literal_eval(args.first) if args.first else 1
last = ast.literal_eval(args.last) if args.last else 30

assert 1 <= first <= 30
assert 1 <= last <= 30
assert first <= last

instance_numbers = range(first, last + 1)

iseo = ISEO(N, thr, T, separator_name, sampler_name, gurobi_environment)

for instance_number in instance_numbers:
    h5_file = os.path.join(file_path, "../output/h5/{}_{}/CSPP/CSPP_{}.h5".format(
        separator_name, sampler_name, instance_number))
    if os.path.exists(h5_file):
        os.remove(h5_file)
    log_file = os.path.join(file_path, "../output/log/{}_{}/CSPP/CSPP_{}.log".format(
        separator_name, sampler_name, instance_number))
    if os.path.exists(log_file):
        os.remove(log_file)
    with open(os.path.join(file_path, "../data/instances/CSPP/CSPP_{}.pickle".format(instance_number)), "br") as f:
        instance = pickle.load(f)
    v = np.array([instance["credits"]])
    w_bar = np.array([instance["time_consumption"] / instance["time_budget"]])
    labeled_points = {"positive": np.array([np.zeros(v.shape)]), "negative": np.array([np.ones(v.shape)])}
    iseo.run(problem_name="CSPP", instance_name=instance["name"], v=v, w_bar=w_bar, labeled_points=labeled_points)
