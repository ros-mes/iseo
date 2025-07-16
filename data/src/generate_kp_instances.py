import pickle

from data.src.knapsack_generator import BinaryKnapsackGenerator

generator = BinaryKnapsackGenerator(n_items=100, coefficient_range=10000, instances_in_series=30)

for instance_type in ["uncorrelated", "weakly-correlated", "strongly-correlated"]:
    for instance_number in range(1, 31):
        instance = generator.generate(instance_type=instance_type, instance_number=instance_number)
        name = "KP_{}_{}.pickle".format(instance_type, instance_number)
        instance_with_name = {"name": name}
        for key, value in instance.items():
            instance_with_name[key] = value
        with open("../instances/KP/{}".format(name), "bw") as f:
            pickle.dump(instance_with_name, f)
