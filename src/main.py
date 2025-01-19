import numpy as np  
from icecream import ic
from TreeNode import *
from data_load import * 
from Tree import *  
from tqdm import tqdm
from Population import *
from gp import * 




# Creazione della popolazione iniziale
population = Population(60, 50)
population.generate_initial_population()

# Creazione e avvio dell'algoritmo genetico
genetic_algorithm = GeneticAlgorithm(population, 10, 1500)
best_tree = genetic_algorithm.evolve()




# Output del miglior individuo
if best_tree:
    print(best_tree)
else:
    print("Nessun individuo valido trovato.")
