# ISEO

## Paper

Anonymous author(s), Solving 0-1 integer programs with unknown knapsack constraints using membership oracles.

## Data

### Knapsack problem

The directory "data/inst/knapsack_problem" contains 90 files, each containing a Python dictionary with the following structure:

  - name: str | pattern = KP_<instance_type>_<instance_number>
  - instance_type: str | values = ["uncorrelated", "weakly-correlated", "strongly-correlated"]
  - instance_number: int | values = [1, ..., 30]
  - values: np.ndarray | shape = (60,)
  - weights: np.ndarray | shape = (60,)
  - capacity: int = (instance_number * sum(weights)) / 31

The directory "data/src" contains the scripts used to generate the knapsack instances:

  - knapsack_generator.py | Class KnapsackGenerator
  - generate_kp_instances.py | Generates the knapsack instances by using the class KnapsackGenerator

### College study plan problem

The directory "data/inst/college_study_plan_problem" contains 30 files, each containing a Python dictionary with the following structure:

  - name: str | pattern = CSPP_<instance_number>
  - instance_number: int | values = [1, ..., 100]
  - credits: np.ndarray | shape = (150,)
  - time_consumption: np.ndarray | shape = (150,)
  - time_budget: int = 1700

The directory "data/csv" contains the file MIT_courses.csv with the list of undergraduate courses in Mathematics at the MIT with their respective number of credits, prerequisites, corequisites and alternatives.

### Generalized assignment problem

The directory "data/inst/generalized_assignment_problem" contains 60 files, each containing a Python dictionary with the following structure:

  - name: str | pattern = GAP_<n_agents>_<n_jobs>_<instance_number>
  - n_agents: int | values = [5, 8, 10]
  - n_jobs: int | values = [n_agents * 3, n_agents * 4, n_agents * 5, n_agents * 6]
  - instance_number: int | values = [1, ..., 5]
  - revenues: np.ndarray | shape = (n_agents, n_jobs)
  - resources: np.ndarray | shape = (n_agents, n_jobs)
  - capacities: np.ndarray | shape = (n_agents,)

## Algorithm

The implementation of ISEO algorithm is composed of a central class and several other satellite classes. They are distributed over the following Python files:

  - **bounder.py**: computes the upper bound to the value of the best labeled positive solution;
  - **exception.py**: contains an exception raised when a programming model solution process fails;
  - **iseo.py**: contains the core of the iseo algorithm, which relies on all the other components;
  - **optimizer.py**: solves the 0-1 linear programming models corresponding to the test problems with unknown constraints;
  - **oracle.py**: labels an unlabeled point by assigning the label positive or negative;
  - **problem.py**: solves the 0-1 linear programming models corresponding to the test problems when all the constraints are known;
  - **sampler.py**: applies the selected sampling strategy (CUT or SIM) to select a new unlabeled point with respect to each unknown constraint;
  - **separator.py**: applies the selected separation algorithm (SEP or SVM) to find a separating hyperplane between positive and negative points with respect to each unknown constraint.

For additional details about the algorithm and its implementation, see the paper referenced above.

Each execution of the ISEO algorithm generates a h5 file and a log file, whose names are the filename stem of the instance the algorithm runs on followed by ".h5" and ".log" respectively.

The h5 files contain numerical information produced by the algorithm such as labeled points, gap, number of calls to the oracle, upper and lower bound, and running time by iteration. They are saved in "output/h5/<separator_name>\_<sampler_name>/".

The log files contain a compact and descriptive account of the execution. They are saved in "output/log/<separator_name>\_<sampler_name>/".

## Tests and results

The files used to run tests and to process the test results for each of the 3 types of instances are inside the directories "test/" and "output/src/", respectively.