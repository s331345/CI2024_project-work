import numpy as np
from icecream import ic
from TreeNode import *
from data_load import *
from Tree import *
from tqdm import tqdm

class Population:
    def __init__(self, size, initial_population_size):
        # I create a population object with a defined size and an initial population size.
        self.size = size
        self.initial_population_size = initial_population_size
        self.individuals = []

    def generate_initial_population(self):
        # I generate the initial population by creating random trees and calculating their fitness.
        while len(self.individuals) < self.initial_population_size:
            root = create_random_tree(x.shape[0], [i for i in range(x.shape[0])], 0)
            parameter_values = []

            def collect_parameters(node):
                # I collect parameter values from leaf nodes in the tree.
                if node is None:
                    return
                if node.label == "parameter":
                    parameter_values.append(node.data)
                collect_parameters(node.left_child)
                collect_parameters(node.right_child)

            collect_parameters(root)

            for i in range(len(parameter_values)):
                parameter_values[i] = np.random.uniform(-90.0, 90.0)

            tree = Tree(root, count_operations(root), parameter_values)
            tree.improve()
            tree.calculate_fitness()

            if (
                tree.fitness_score is not None
                and not np.isnan(tree.fitness_score)
                and tree.fitness_score != np.inf
                and tree.fitness_score != -np.inf
            ):
                self.individuals.append(tree)

        self.individuals.sort(key=lambda t: t.fitness_score, reverse=True)

    def select_parent(self):
        # I select a parent tree from the population using a selection method.
        return select_parent(self.individuals)

    def update_population(self, offspring):
        # I update the population by adding offspring and keeping the best individuals.
        self.individuals.extend(offspring)
        self.individuals.sort(key=lambda t: t.fitness_score, reverse=True)
        self.individuals = self.individuals[:self.size]
