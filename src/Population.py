import numpy as np  
from icecream import ic
from TreeNode import *
from data_load import * 
from Tree import *  
from tqdm import tqdm

class Population:
    def __init__(self, size, initial_population_size):
        self.size = size
        self.initial_population_size = initial_population_size
        self.individuals = []

    def generate_initial_population(self):
        """Genera la popolazione iniziale assicurandosi che i parametri siano popolati."""
        while len(self.individuals) < self.initial_population_size:
            root = create_random_tree(x.shape[0], [i for i in range(x.shape[0])], 0)
            param_indices = []
            new_numbers = []

            # Raccogli parametri dai nodi foglia
            _ = collect_negative_numbers(root, param_indices)
            if param_indices:
                for index in param_indices:
                    if index < 0:
                        new_numbers.append(np.random.randint(0, 10))  # Aggiungi un valore casuale
                        modify_tree_node(root, index, -len(new_numbers))  # Modifica il nodo per riflettere l'indice del nuovo parametro

            # Crea un oggetto Tree con i parametri aggiornati
            tree = Tree(root, count_operations(root), new_numbers)

            # Semplifica l'albero dopo aver riempito i parametri
            tree.improve()

            # Calcola il fitness del tree
            tree.calculate_fitness()

            # Aggiungi l'albero alla popolazione solo se il fitness è valido
            if (
                tree.fitness_score is not None
                and not np.isnan(tree.fitness_score)
                and tree.fitness_score != np.inf
                and tree.fitness_score != -np.inf
            ):
                self.individuals.append(tree)

        # Ordina la popolazione per fitness decrescente
        self.individuals.sort(key=lambda t: t.fitness_score, reverse=True)  
      
    def select_parent(self):
        """Seleziona un genitore dalla popolazione."""
        return select_parent(self.individuals)

    def update_population(self, offspring):
        """Aggiorna la popolazione con nuovi individui."""
        self.individuals.extend(offspring)
        self.individuals.sort(key=lambda t: t.fitness_score, reverse=True)
        self.individuals = self.individuals[:self.size]
