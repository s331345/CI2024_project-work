import numpy as np  
from icecream import ic
from TreeNode import *
from data_load import * 
from Tree import *  
from tqdm import tqdm
class GeneticAlgorithm:
    def __init__(self, population, offspring_size, max_generations):
        self.population = population
        self.offspring_size = offspring_size
        self.max_generations = max_generations

    def evolve(self):
        """Evolve la popolazione attraverso le generazioni."""
        for generation in tqdm(range(self.max_generations)):
            offspring = []
            for _ in range(self.offspring_size // 2):
                # Selezione dei genitori
                parent1 = self.population.select_parent()
                parent2 = self.population.select_parent()

                # Crossover
                child1 = crossover_trees(parent1, parent2)
                child2 = crossover_trees(parent2, parent1)

                # Mutazione e semplificazione
                child1.mutate()
                child2.mutate()

                # Calcolo del fitness
                child1.calculate_fitness()
                child2.calculate_fitness()

                # Evita duplicati
                if not any(tree.fitness_score == child1.fitness_score for tree in self.population.individuals):
                    offspring.append(child1)
                if not any(tree.fitness_score == child2.fitness_score for tree in self.population.individuals):
                    offspring.append(child2)

            # Aggiorna la popolazione
            self.population.update_population(offspring)

        # Restituisce il miglior individuo
        return self.population.individuals[0]
