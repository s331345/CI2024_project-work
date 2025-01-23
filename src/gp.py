import numpy as np
from icecream import ic
from TreeNode import *
from data_load import *
from Tree import *
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(self, population, offspring_size, max_generations):
        # I initialize the genetic algorithm with a population, the number of offspring, and the maximum generations.
        self.population = population
        self.offspring_size = offspring_size
        self.max_generations = max_generations

    def evolve(self):
        # I evolve the population over a defined number of generations to find better solutions.
        for generation in tqdm(range(self.max_generations)):
            offspring = []
            for _ in range(self.offspring_size // 2):
                parent1 = self.population.select_parent()
                parent2 = self.population.select_parent()

                child1 = crossover_trees(parent1, parent2)
                child2 = crossover_trees(parent2, parent1)

                child1.mutate()
                child2.mutate()

                child1.calculate_fitness()
                child2.calculate_fitness()

                if not any(tree.fitness_score == child1.fitness_score for tree in self.population.individuals):
                    offspring.append(child1)
                if not any(tree.fitness_score == child2.fitness_score for tree in self.population.individuals):
                    offspring.append(child2)

            self.population.update_population(offspring)

        return self.population.individuals[0]
