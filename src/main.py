import numpy as np
from TreeNode import TreeNode
from data_load import BINARY_OPS, UNARY_OPS, x, y
from Tree import Tree
from Population import Population
from gp import GeneticAlgorithm

# I set the population size and the initial population size.
population_size = 80
initial_population_size = 50

# I create the initial population of random trees.
population = Population(population_size, initial_population_size)
population.generate_initial_population()

# I check if the population is not empty; otherwise, I stop the program with an error.
if not population.individuals:
    raise ValueError("Error: The initial population is empty. Check tree generation.")

# I set the number of offspring and the maximum number of generations for the algorithm.
offspring_size = 10
max_generations = 1000

# I create the genetic algorithm object with the population and parameters.
genetic_algorithm = GeneticAlgorithm(population, offspring_size, max_generations)

# I run the genetic algorithm to evolve the population and get the best tree.
best_tree = genetic_algorithm.evolve()

# I print the best tree if found; otherwise, I print an error message.
if best_tree:
    print("\nBest individual found:")
    print(best_tree)
else:
    print("No valid individual found.")
