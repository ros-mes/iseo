<div align="center">

# ISEO

### Interactive Sampling-Enhanced Optimization

**Solving 0–1 integer programs with unknown knapsack constraints using membership oracles**

[![Journal](https://img.shields.io/badge/Computational_Optimization_and_Applications-2026-1f4e79.svg)](https://doi.org/10.1007/s10589-026-00853-6)
[![arXiv](https://img.shields.io/badge/arXiv-2405.14090-b31b1b.svg)](https://arxiv.org/abs/2405.14090)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![Gurobi](https://img.shields.io/badge/solver-Gurobi_12-EE3524.svg)](https://www.gurobi.com/)

Rosario Messana · Rui Chen · Andrea Lodi · Alberto Ceselli

</div>

---

This repository contains the official implementation of **ISEO**, the algorithmic framework introduced in

> Rosario Messana, Rui Chen, Andrea Lodi, Alberto Ceselli.
> *Solving 0–1 integer programs with unknown knapsack constraints using membership oracles.*
> **Computational Optimization and Applications**, 2026. DOI: [10.1007/s10589-026-00853-6](https://doi.org/10.1007/s10589-026-00853-6). Preprint: [arXiv:2405.14090](https://arxiv.org/abs/2405.14090).

together with the benchmark instances, the experiment drivers and the scripts that produce the metrics reported in the paper.

ISEO optimizes over binary decisions when some of the constraints are **hidden**: their coefficients cannot be read, and the only way to learn about them is to ask an oracle whether a candidate solution satisfies them. Think of a planner who must build the best study plan for a student without knowing how much time each course really takes, and who can only ask the student "*can you manage this plan?*". Drawing on active learning, ISEO learns surrogate linear constraints from the answers, chooses which solutions to query next, and optimizes over the surrogates. At every iteration it maintains both a solution that is verified to be feasible and a valid upper bound on the optimum.

### Highlights

- **Black-box knapsack constraints**, accessed only through yes/no membership queries under a fixed query budget.
- **Active-learning loop**: *separate → sample → optimize → bound*, with interchangeable components (2 separators × 2 sampling strategies).
- **Certified progress**: a lower bound from verified feasible solutions and a valid upper bound from a mixed-integer quadratic relaxation.
- **Query-efficient**: labels implied by monotonicity are inferred for free, so oracle calls are spent only on genuinely uncertain points.
- **180 benchmark instances** from three problem classes, including a realistic study-plan problem built on MIT course data.
- **Full traces** of every run (bounds, labeled points, surrogate weights, timings) stored in HDF5 for analysis.

## Contents

- [The problem](#the-problem)
- [How ISEO works](#how-iseo-works)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Output files](#output-files)
- [Benchmark data](#benchmark-data)
- [Citation](#citation)

## The problem

ISEO addresses 0-1 integer programs of the form

```math
\max_{x}\ \sum_{i=1}^{m} v_i^\top x_i
\qquad \text{s.t.} \qquad
x \in X \subseteq \{0,1\}^{m \times n},
\qquad
\bar{w}_i^\top x_i \le 1 \quad (i = 1, \dots, m),
```

where

- the objective coefficients $v$ and the set $X$ of **known** constraints (e.g. course prerequisites, or "each job is assigned at most once") are given explicitly;
- the nonnegative (normalized) weights $\bar{w}_i$ of the $m$ **unknown** knapsack constraints are hidden;
- each unknown constraint $i$ can only be probed through a **membership oracle**

```math
\mathcal{O}_i(x_i) =
\begin{cases}
+1 & \text{if } \bar{w}_i^\top x_i \le 1 \quad \text{(feasible: positive label)}, \\
-1 & \text{otherwise} \quad \text{(infeasible: negative label)},
\end{cases}
```

  which may be called at most $N$ times.

The goal is to find the best possible feasible solution within the query budget, and to certify how far it is from the optimum.

Because the weights are nonnegative, feasibility is monotone: any solution dominated by a feasible one is feasible, and any solution dominating an infeasible one is infeasible. ISEO exploits this to label many points without calling the oracle.

## How ISEO works

```mermaid
flowchart TD
    init(["Initial labels: all-zeros solution feasible, all-ones solution infeasible"]) --> sep
    sep["<b>1. Separate</b><br/>learn surrogate weights ŵᵢ from the labeled points<br/><i>SEP or SVM</i>"] --> sam
    sam["<b>2. Sample</b><br/>pick an informative unlabeled point for each constraint<br/><i>CUT or SIM</i>"] --> opt
    opt["<b>3. Optimize</b><br/>solve the 0-1 ILP with ŵᵢ in place of the unknown w̄ᵢ"] --> bnd
    bnd["<b>4. Bound</b><br/>LB: best verified feasible solution<br/>UB: MIQP over all weights consistent with the labels"] --> stop{"gap ≤ threshold<br/>or budget exhausted?"}
    stop -->|no| sep
    stop -->|yes| ret(["Return the best feasible solution&nbsp;$$\hat{x}$$&nbsp;and the surrogate weights&nbsp;$$\hat{w}$$"])
    sam -.->|query| ora{{"Membership oracles"}}
    opt -.->|query| ora
```

Each iteration consists of four steps:

1. **Separate.** For every unknown constraint, a linear separator is trained on the labeled points (feasible vs. infeasible) and yields surrogate weights $\hat{w}_i$.
   - **SEP**: a separation method inspired by convex optimization in the oracle model, computed as a convex quadratic program over the labeled data.
   - **SVM**: a hard-margin support vector machine with nonnegative weights (convex QP), with a fallback formulation when the margin problem cannot be solved.
2. **Sample.** For every unknown constraint, an unlabeled point is selected among those whose label cannot be inferred by monotonicity, and the oracle is queried on it.
   - **CUT** (closest cutting plane): the point whose induced hyperplane in weight space is closest, in Euclidean distance, to $\hat{w}_i$ (a mixed-integer quadratic program).
   - **SIM** (simple margin): the point closest to the current decision boundary, in the spirit of margin-based active learning (a mixed-integer program).
3. **Optimize.** The original problem is solved with $\hat{w}_i$ in place of $\bar{w}_i$, excluding solutions dominated by known feasible points and solutions dominating known infeasible ones. The optimal solution is submitted to the oracles; if every oracle accepts it, it is a verified feasible solution and may improve the lower bound.
4. **Bound.** An upper bound is obtained by maximizing the objective over all $x \in X$ for which *some* weight vector consistent with all labels makes $x$ feasible. This is a nonconvex MIQP; Gurobi's dual bound is used, so the upper bound stays valid even when the solver stops early.

The algorithm stops when the relative gap $(\mathrm{UB} - \mathrm{LB}) / \mathrm{LB}$ falls below the threshold, when an oracle reaches its budget $N$, after $T$ iterations, or when the time limit is reached.

In the paper's experiments, **SEP + CUT** gives the most accurate solutions across the three problem classes. **CUT** is markedly more expensive than **SIM**, because it solves a harder MIQP at each query, so the preferable sampler depends on the available time and query budgets. The theoretical analysis, the precise formulations and the full computational study are in the [paper](https://doi.org/10.1007/s10589-026-00853-6).

## Repository structure

```
iseo/
├── algorithm/                  # The ISEO framework
│   ├── iseo.py                 #   main loop (class ISEO)
│   ├── separator.py            #   linear separators: SEP, SVM
│   ├── sampler.py              #   sampling strategies: CUT, SIM
│   ├── optimizer.py            #   0-1 ILPs with surrogate constraints (one per problem class)
│   ├── bounder.py              #   upper bounds via MIQP (one per problem class)
│   ├── problem.py              #   ground-truth models with all constraints known
│   ├── oracle.py               #   membership oracle
│   └── exception.py            #   ModelResolutionFailedError
├── data/
│   ├── instances/              # Benchmark instances (pickled Python dictionaries)
│   │   ├── KP/                 #   90 knapsack instances
│   │   ├── CSPP/               #   30 college study plan instances
│   │   └── GAP/                #   60 generalized assignment instances
│   ├── csv/MIT_courses.csv     # Courses, credits and precedence relations used by CSPP
│   └── src/                    # Knapsack instance generator
├── test/                       # Experiment drivers (one per problem class) and tmux batch launchers
├── output/
│   ├── src/                    # Scripts computing the metrics tables from the HDF5 traces
│   ├── metrics/                # Metrics tables (KP.txt, CSPP.txt, GAP.txt)
│   ├── h5/                     # (generated) numerical trace of each run
│   └── log/                    # (generated) human-readable log of each run
└── requirements.txt
```

The implementation consists of a central class and several satellite classes:

| File | Role |
|---|---|
| [`iseo.py`](algorithm/iseo.py) | Core of the ISEO algorithm; orchestrates all the other components. |
| [`separator.py`](algorithm/separator.py) | Applies the selected separation algorithm (**SEP** or **SVM**) to find a separating hyperplane between positive and negative points for each unknown constraint. |
| [`sampler.py`](algorithm/sampler.py) | Applies the selected sampling strategy (**CUT** or **SIM**) to select a new unlabeled point for each unknown constraint. |
| [`optimizer.py`](algorithm/optimizer.py) | Solves the 0-1 linear programming models of the test problems with the unknown constraints replaced by their surrogates. |
| [`bounder.py`](algorithm/bounder.py) | Computes the upper bound on the value of the best labeled positive solution. |
| [`problem.py`](algorithm/problem.py) | Solves the 0-1 linear programming models of the test problems when all the constraints are known (used to evaluate the results). |
| [`oracle.py`](algorithm/oracle.py) | Labels a point as positive (+1) or negative (−1). |
| [`exception.py`](algorithm/exception.py) | Exception raised when the solution process of a programming model fails. |

## Installation

**Requirements:** Python 3.10 or later and [Gurobi](https://www.gurobi.com/) 12 with a valid license.

```bash
git clone https://github.com/ros-mes/iseo.git
cd iseo
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt   # gurobipy, numpy, h5py, pandas
```

> [!NOTE]
> The `gurobipy` package from PyPI includes a size-limited license, which may suffice for small trials. Runs with the budgets of the paper generate models that can exceed its limits, so a full Gurobi license is recommended; it is [free for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Quick start

The following script runs ISEO on a knapsack instance with a small budget of 100 oracle calls, which takes a few minutes on a laptop. Run it from the repository root.

```python
import pickle

import gurobipy as gp
import numpy as np

from algorithm.iseo import ISEO

with open("data/instances/KP/KP_uncorrelated_1.pickle", "rb") as f:
    instance = pickle.load(f)

# Arrays have shape (m, n): m unknown constraints (here 1) over n binary variables (here 100 items)
v = np.array([instance["values"]])                              # objective coefficients
w_bar = np.array([instance["weights"] / instance["capacity"]])  # hidden weights, only seen by the oracle

# Initial labels: the empty solution is feasible, the full solution is infeasible
labeled_points = {"positive": np.zeros((1, *v.shape)), "negative": np.ones((1, *v.shape))}

env = gp.Env(empty=True)
env.setParam("ThreadLimit", 2)
env.start()

iseo = ISEO(N=100, thr=0.01, T=200, separator_name="SEP", sampler_name="CUT", gurobi_environment=env)
x_hat, w_hat = iseo.run(problem_name="KP", instance_name="KP_uncorrelated_1_demo",
                        v=v, w_bar=w_bar, labeled_points=labeled_points)

print("Value of the best verified feasible solution:", (v * x_hat).sum())
```

The progress of the run (bounds, gap, labeled points and oracle calls at each iteration) is written to `output/log/SEP_CUT/KP/KP_uncorrelated_1_demo.log`, and the complete numerical trace to `output/h5/SEP_CUT/KP/KP_uncorrelated_1_demo.h5`.

**Parameters of `ISEO`:**

| Parameter | Description |
|---|---|
| `N` | Oracle budget: maximum number of calls to each membership oracle. |
| `thr` | Relative gap threshold below which the algorithm stops. |
| `T` | Maximum number of iterations. |
| `separator_name` | `"SEP"` or `"SVM"`. |
| `sampler_name` | `"CUT"` or `"SIM"`. |
| `gurobi_environment` | Optional `gurobipy.Env` shared by all the models. |

**Arguments of `ISEO.run`:** `problem_name` (`"KP"`, `"CSPP"` or `"GAP"`), `instance_name` (used to name the output files), `v` and `w_bar` (arrays of shape `(m, n)`), and `labeled_points`, a dictionary with the initial `"positive"` and `"negative"` points as arrays of shape `(k, m, n)`. It returns the best verified feasible solution `x_hat` and the final surrogate weights `w_hat`. In the experiments the oracle is simulated from `w_bar`, which ISEO never accesses directly.

## Reproducing the experiments

The scripts in [`test/`](test/) run ISEO on the benchmark instances with the settings of the paper:

| Setting | Value |
|---|---|
| Oracle budget `N` | 2000 calls per unknown constraint |
| Gap threshold `thr` | 0 (runs until the budget is exhausted or optimality is proven) |
| Iteration limit `T` | 2N = 4000 |
| Initial labels | all-zeros solution positive, all-ones solution negative |
| Gurobi | 2 threads; node limit of 50,000 for the sampling and bounding MIQPs |
| Time limit | 500,000 s per run |

Run them as modules from the repository root, always specifying both `--separator` and `--sampler`:

```bash
# Knapsack: --type uncorrelated | weakly-correlated | strongly-correlated, instances 1–30
python -m test.test_kp --separator SEP --sampler CUT --type uncorrelated --first 1 --last 10

# College study plan: instances 1–30
python -m test.test_cspp --separator SEP --sampler CUT --first 1 --last 30

# Generalized assignment: --agents 5 | 8 | 10 (all sizes if omitted), instances 1–5
python -m test.test_gap --separator SEP --sampler CUT --agents 5 --first 1 --last 5
```

To launch the full experimental campaign, the `run_*.bash` scripts start one [tmux](https://github.com/tmux/tmux) session per combination of separator, sampler and block of instances. Activate the Python environment first, then run:

```bash
bash test/run_kp.bash     # 36 sessions: {SEP, SVM} × {CUT, SIM} × 3 types × 3 blocks of 10 instances
bash test/run_cspp.bash   # 12 sessions: {SEP, SVM} × {CUT, SIM} × 3 blocks of 10 instances
bash test/run_gap.bash    # 12 sessions: {SEP, SVM} × {CUT, SIM} × 3 numbers of agents
```

> [!WARNING]
> The full campaign is computationally heavy: with the CUT sampler, a single knapsack run takes about two days on average (see `output/metrics/KP.txt`). Re-running an instance overwrites its previous output files.

Once the runs are complete, the scripts in [`output/src/`](output/src/) process the HDF5 traces and write the metrics tables to [`output/metrics/`](output/metrics/). They use paths relative to their own directory and read the traces from `output/output/h5/` (the path `../output/h5/` seen from `output/src/`), so move the traces there first. From the repository root:

```bash
mkdir -p output/output
mv output/h5 output/output/h5
cd output/src
PYTHONPATH=../.. python results_kp.py
PYTHONPATH=../.. python results_cspp.py
PYTHONPATH=../.. python results_gap.py
```

Each row of a metrics table aggregates the instances of one group for one separator–sampler combination:

| Column | Meaning |
|---|---|
| # Solved | Instances whose best verified feasible solution is optimal (the optimum is computed with all constraints known). |
| # Thresholds | Instances where the relative gap fell below 1%. |
| # Feas. solutions on thr. | Among those, instances where the solution of the surrogate problem was feasible for the true constraints when the threshold was reached. |
| Gap | Mean final relative gap (UB − LB) / LB. |
| Error | Mean final relative error (OPT − LB) / OPT. |
| # Calls, # Iterations, Running time | Mean totals per run (running time in seconds). |
| … to thr. / … to opt. | Mean oracle calls, iterations and running time needed to reach the 1% gap / the optimum, over the instances that reached it. |

## Output files

Each execution of ISEO generates a log file and an HDF5 file, named after the instance:

```
output/log/<separator>_<sampler>/<problem>/<instance>.log
output/h5/<separator>_<sampler>/<problem>/<instance>.h5
```

The **log files** contain a compact and descriptive account of the execution: bounds, gap, number of labeled points and oracle calls at every iteration, and running times.

The **HDF5 files** contain the numerical information produced by the algorithm:

| Group | Content |
|---|---|
| `initial` | Attributes `UB`, `LB`, `n_labeled_points`, `n_labeled_sub_points`, `n_oracle_calls`; datasets with the initial labeled points and sub-points. |
| `iter_<t>` | Datasets `w_hat` (surrogate weights), `u_sampling` / `labels_sampling` (sampled points and their labels), `u_optimization` / `labels_optimization` (optimized solution and its labels), `new_labeled_sub_points_<i>_positive` / `_negative`; attributes `UB`, `LB`, `gap`, `flag`, `n_labeled_points`, `n_labeled_sub_points`, `n_oracle_calls`, `running_time`. |
| `final` | Datasets `x_hat` (best verified feasible solution) and `w_hat` (final surrogate weights). |

For example, to retrieve the evolution of the gap:

```python
import h5py

with h5py.File("output/h5/SEP_CUT/KP/KP_uncorrelated_1.h5", "r") as f:
    iterations = sorted((k for k in f if k.startswith("iter_")), key=lambda k: int(k.split("_")[1]))
    gaps = [f[k].attrs["gap"] for k in iterations]
    x_hat = f["final/x_hat"][()]
```

## Benchmark data

All instances are stored in [`data/instances/`](data/instances/) as pickled Python dictionaries:

```python
import pickle

with open("data/instances/GAP/GAP_5_15_1.pickle", "rb") as f:
    instance = pickle.load(f)
```

### Knapsack problem (KP)

90 instances in [`data/instances/KP/`](data/instances/KP/): 30 for each of three correlation types between values and weights. A single knapsack constraint is unknown.

| Key | Type | Description |
|---|---|---|
| `name` | `str` | `KP_<instance_type>_<instance_number>` |
| `instance_type` | `str` | `"uncorrelated"`, `"weakly-correlated"` or `"strongly-correlated"` |
| `instance_number` | `int` | 1, …, 30 |
| `values` | `np.ndarray`, shape `(100,)` | Item values (objective) |
| `weights` | `np.ndarray`, shape `(100,)` | Item weights (unknown constraint) |
| `capacity` | `float` | `instance_number * sum(weights) / 31` |

The instances were generated with the scripts in [`data/src/`](data/src/), which follow Pisinger's classical generator with coefficient range $R = 10{,}000$. Weights are uniform in $[1, R]$; values are uniform in $[1, R]$ (uncorrelated), within $\pm R/10$ of the weight (weakly correlated), or equal to the weight plus 10 (strongly correlated).

- [`knapsack_generator.py`](data/src/knapsack_generator.py): class `BinaryKnapsackGenerator`.
- [`generate_kp_instances.py`](data/src/generate_kp_instances.py): generates the knapsack instances with `BinaryKnapsackGenerator`.

### College study plan problem (CSPP)

30 instances in [`data/instances/CSPP/`](data/instances/CSPP/). A student selects courses to maximize the total number of credits, subject to prerequisites, corequisites and mutually exclusive alternatives (known constraints) and to a time budget whose per-course time consumption is unknown.

| Key | Type | Description |
|---|---|---|
| `name` | `str` | `CSPP_<instance_number>` |
| `instance_number` | `int` | 1, …, 30 |
| `credits` | `np.ndarray`, shape `(150,)` | Credits (MIT units) of each course (objective) |
| `time_consumption` | `np.ndarray`, shape `(150,)` | Time required by each course (unknown constraint) |
| `time_budget` | `int` | 1700 |

The known constraints come from [`data/csv/MIT_courses.csv`](data/csv/MIT_courses.csv), a `|`-separated list of 150 MIT courses centered on the undergraduate Mathematics curriculum, with columns:

| Column | Description |
|---|---|
| `code` | Course index (0–149), matching the position in the instance arrays |
| `name` | Course title |
| `credits` | MIT units as (lecture, lab, preparation); their sum is the course's credits |
| `prerequisites` | List of groups of course codes; at least one course from each group must be taken |
| `corequisites` | Same format as `prerequisites` |
| `alternatives` | Course codes that cannot be taken together with this course |

### Generalized assignment problem (GAP)

60 instances in [`data/instances/GAP/`](data/instances/GAP/), from the OR-Library sets gap1–gap12. Each job is assigned to at most one agent (known constraint), and the capacity constraint of each agent is unknown, giving one membership oracle per agent.

| Key | Type | Description |
|---|---|---|
| `name` | `str` | `GAP_<n_agents>_<n_jobs>_<instance_number>` |
| `n_agents` | `int` | 5, 8 or 10 |
| `n_jobs` | `int` | `3 * n_agents`, `4 * n_agents`, `5 * n_agents` or `6 * n_agents` |
| `instance_number` | `str` | `"1"`, …, `"5"` |
| `revenues` | `np.ndarray`, shape `(n_agents, n_jobs)` | Revenue of assigning each job to each agent (objective) |
| `resources` | `np.ndarray`, shape `(n_agents, n_jobs)` | Resources consumed by each job on each agent (unknown constraints) |
| `capacities` | `np.ndarray`, shape `(n_agents,)` | Capacity of each agent (unknown constraints) |

## Citation

If you use this code or the benchmark instances in your research, please cite:

```bibtex
@article{messana2026iseo,
  title   = {Solving 0--1 Integer Programs with Unknown Knapsack Constraints Using Membership Oracles},
  author  = {Messana, Rosario and Chen, Rui and Lodi, Andrea and Ceselli, Alberto},
  journal = {Computational Optimization and Applications},
  year    = {2026},
  doi     = {10.1007/s10589-026-00853-6}
}
```

For additional details about the algorithm and its implementation, see the paper referenced above. Questions and bug reports are welcome on the [issue tracker](https://github.com/ros-mes/iseo/issues).
