
from TreeNode import TreeNode
from data_load import BINARY_OPS, UNARY_OPS, x, y
import numpy as np  
from icecream import ic
from TreeNode import *
import matplotlib.pyplot as plt
import networkx as nx

class Tree:
    def __init__(self, root: TreeNode, identifier: int, numbers: list, fitness_score: float = None):
        self.root = root
        self.identifier = identifier
        self.numbers = numbers
        self.fitness_score = fitness_score

   
    def __repr__(self):
        # Usa tree_to_string per generare la formula dall'albero
        formula = tree_to_string(self.root, self.numbers) if self.root else "None"
        return (f"Tree(formula={formula}, score={self.fitness_score})")


    def calculate_fitness(self):
        """
        Calcola il valore di fitness basato sulla Mean Squared Error (MSE).
        Penalizza alberi complessi per promuovere soluzioni più semplici.
        """
        try:
            total_error = 0
            n_samples = x.shape[0]  # Numero di campioni

            # Calcola l'errore quadratico totale
            for i in range(n_samples):
                prediction = evaluate_expression(self.root, x[i], self.numbers)
                if np.isnan(prediction) or np.isinf(prediction):
                    # Penalizza risultati non validi
                    self.fitness_score = -np.inf
                    return

                total_error += np.square(y[i] - prediction)

            # Calcola la Mean Squared Error (MSE)
            mse = total_error / n_samples

            # Penalità per la complessità dell'albero
            complexity_penalty = self.identifier * 10  # Moltiplicatore per la complessità

            # Fitness come valore negativo della MSE penalizzato
            self.fitness_score = -mse - complexity_penalty

        except Exception as e:
            # Penalizza in caso di errori
            print(f"Errore durante il calcolo del fitness: {e}")
            self.fitness_score = -np.inf
    
    def mutate(self):
        """Applica una mutazione all'albero e lo semplifica."""
        mutated_expression = duplicate_tree_node(self.root)
        new_numbers = self.numbers.copy()
        if np.random.random() < 0.4 and self.identifier != 0:
            # Modifica un'operazione con probabilità 1/tree.identifier
            mutate_tree_nodes(mutated_expression, 1 / self.identifier)
        else:
            if np.random.random() < 0.1:
                # Aggiunge una nuova operazione all'albero
                parent_node = None
                current_node = mutated_expression
                is_leaf_found = False
                # Seleziona casualmente una foglia
                while not is_leaf_found:
                    if not isinstance(current_node.data, str):
                        is_leaf_found = True
                    else:
                        if np.random.random() < 0.5 and current_node.left_child is not None:
                            parent_node = current_node
                            current_node = current_node.left_child
                        elif current_node.right_child is not None:
                            parent_node = current_node
                            current_node = current_node.right_child
                # Decide dove aggiungere la nuova operazione
                if parent_node.left_child == current_node:
                    if np.random.random() > 0.5:
                        new_leaf = TreeNode(-1, None, None)
                        leaf_values = []
                        _ = collect_positive_values(mutated_expression, leaf_values)
                        if leaf_values:
                            for i in range(x.shape[0]):
                                if i not in leaf_values:
                                    new_leaf.data = i
                                    break
                        if new_leaf.data == -1:
                            new_param_value = np.random.randint(0, 10)
                            new_numbers.append(new_param_value)
                            new_leaf.data = -len(new_numbers)
                        parent_node.left_child = TreeNode(
                            np.random.choice(list(BINARY_OPS.keys())), current_node, new_leaf
                        )
                    else:
                        parent_node.left_child = TreeNode(
                            np.random.choice(list(UNARY_OPS.keys())), current_node, None
                        )
                else:
                    if np.random.random() > 0.5:
                        new_leaf = TreeNode(-1, None, None)
                        leaf_values = []
                        _ = collect_positive_values(mutated_expression, leaf_values)
                        if leaf_values:
                            for i in range(x.shape[0]):
                                if i not in leaf_values:
                                    new_leaf.data = i
                                    break
                        if new_leaf.data == -1:
                            new_param_value = np.random.randint(0, 10)
                            new_numbers.append(new_param_value)
                            new_leaf.data = -len(new_numbers)
                        parent_node.right_child = TreeNode(np.random.choice(list(BINARY_OPS.keys())), current_node, new_leaf)
                    else:
                        parent_node.right_child = TreeNode(np.random.choice(list(UNARY_OPS.keys())), current_node, None)

                return Tree(mutated_expression, self.identifier + 1, new_numbers)
            else:
                if np.random.random() < 0.5 and len(self.numbers) != 0:
                    # Modifica un parametro incrementandolo o decrementandolo
                    param_index = np.random.randint(0, len(self.numbers))
                    if np.random.random() < 0.5:
                        new_numbers[param_index] += 1
                    else:
                        new_numbers[param_index] -= 1
                else:
                    # Sostituisce una foglia con un'altra foglia
                    candidates = np.random.choice(
                        [i for i in range(-len(new_numbers), x.shape[0])], 2
                    )
                    modify_tree_node(mutated_expression, candidates[0], candidates[1])
        self = Tree(mutated_expression, self.identifier, new_numbers)
        old_tree = duplicate_tree_node(self.root)
        if improve_tree(self.root, self.numbers) > 0:
            updated_numbers = []
            param_indices = []
            collect_negative_numbers(self.root, param_indices)
            for index in param_indices:
                updated_numbers.append(self.numbers[(-index) - 1])
                modify_tree_node(self.root, index, -len(updated_numbers))
            self.numbers = updated_numbers
        self.identifier = count_operations(self.root)
        if self.identifier == 0:
            self.root = old_tree
    
    def improve(self):
        """Semplifica l'albero eliminando operazioni ridondanti e aggiorna i parametri."""
        old_tree = duplicate_tree_node(self.root)
        if improve_tree(self.root, self.numbers) > 0:
            updated_numbers = []
            param_indices = []
            collect_negative_numbers(self.root, param_indices)
            for index in param_indices:
                updated_numbers.append(self.numbers[(-index) - 1])
                modify_tree_node(self.root, index, -len(updated_numbers))
            self.numbers = updated_numbers
        self.identifier = count_operations(self.root)
        if self.identifier == 0:
            self.root = old_tree



#functions



def duplicate_tree_node(tree_node):
    left_child = None
    right_child = None

    if tree_node.left_child is not None:
        left_child = duplicate_tree_node(tree_node.left_child)
    if tree_node.right_child is not None:
        right_child = duplicate_tree_node(tree_node.right_child)

    return TreeNode(tree_node.data, left_child, right_child)

def create_random_tree(max_depth, available_values, depth=0):
    if depth >= max_depth or (depth > 0 and np.random.random() < 0.3):
        if len(available_values) > 0:
            index = np.random.randint(0, len(available_values))
            value = available_values.pop(index)
        else:
            value = -np.random.randint(1, 10)
        return TreeNode(value, None, None)

    if np.random.random() > 0.5:
        # Create a binary operation node
        operation = np.random.choice(list(BINARY_OPS.keys()))
        node = TreeNode(operation)
        node.left_child = create_random_tree(max_depth, available_values, depth + 1)
        node.right_child = create_random_tree(max_depth, available_values, depth + 1)
    else:
        # Create a unary operation node
        operation = np.random.choice(list(UNARY_OPS.keys()))
        node = TreeNode(operation)
        node.left_child = create_random_tree(max_depth, available_values, depth + 1)

    return node


def modify_tree_node(tree_node, target_value, new_value):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            if modify_tree_node(tree_node.left_child, target_value, new_value) == 1:
                return 1
            if modify_tree_node(tree_node.right_child, target_value, new_value) == 1:
                return 1
        elif tree_node.data in UNARY_OPS:
            if modify_tree_node(tree_node.left_child, target_value, new_value) == 1:
                return 1
    else:
        if tree_node.data == target_value:
            tree_node.data = new_value
            return 1
    return 0


def collect_negative_numbers(tree_node, numbers_list):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            return (collect_negative_numbers(tree_node.left_child, numbers_list) +
                    collect_negative_numbers(tree_node.right_child, numbers_list))
        elif tree_node.data in UNARY_OPS:
            return collect_negative_numbers(tree_node.left_child, numbers_list)
    elif tree_node.data < 0:
        numbers_list.append(tree_node.data)
        return 1
    else:
        return 0

def collect_positive_values(tree_node, values_list):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            return (collect_positive_values(tree_node.left_child, values_list) +
                    collect_positive_values(tree_node.right_child, values_list))
        elif tree_node.data in UNARY_OPS:
            return collect_positive_values(tree_node.left_child, values_list)
    elif tree_node.data >= 0:
        values_list.append(tree_node.data)
        return 1
    else:
        return 0

def count_operations(tree_node):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            return (count_operations(tree_node.left_child) +
                    count_operations(tree_node.right_child) + 1)
        elif tree_node.data in UNARY_OPS:
            return count_operations(tree_node.left_child) + 1
    else:
        return 0

def tree_to_string(tree_node, numbers):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            if tree_node.data == "add":
                return "(" + tree_to_string(tree_node.left_child, numbers) + "+" + tree_to_string(tree_node.right_child, numbers) + ")"
            elif tree_node.data == "subtract":
                return "(" + tree_to_string(tree_node.left_child, numbers) + "-" + tree_to_string(tree_node.right_child, numbers) + ")"
            elif tree_node.data == "multiply":
                return "(" + tree_to_string(tree_node.left_child, numbers) + "*" + tree_to_string(tree_node.right_child, numbers) + ")"
            elif tree_node.data == "divide":
                return "(" + tree_to_string(tree_node.left_child, numbers) + "/" + tree_to_string(tree_node.right_child, numbers) + ")"
            elif tree_node.data == "power":
                return "(" + tree_to_string(tree_node.left_child, numbers) + "**" + tree_to_string(tree_node.right_child, numbers) + ")"
        elif tree_node.data in UNARY_OPS:
            if tree_node.data == "sin":
                return "sin(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "cos":
                return "cos(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "tan":
                return "tan(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "exp":
                return "exp(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "log":
                return "log(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "sqrt":
                return "sqrt(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "abs":
                return "abs(" + tree_to_string(tree_node.left_child, numbers) + ")"
            elif tree_node.data == "square":
                return "square(" + tree_to_string(tree_node.left_child, numbers) + ")"
    if tree_node.data < 0:
        return str(numbers[(-tree_node.data) - 1])
    return "x" + str(tree_node.data)


def evaluate_expression(tree_node, values_list, numbers):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            q1 = evaluate_expression(tree_node.left_child, values_list, numbers)
            q2 = evaluate_expression(tree_node.right_child, values_list, numbers)
            if np.isnan(q1) or np.isnan(q2) or np.isinf(q1) or np.isinf(q2):
                return np.inf
            if tree_node.data == "add":
                return q1 + q2
            elif tree_node.data == "subtract":
                return q1 - q2
            elif tree_node.data == "multiply":
                return q1 * q2
            elif tree_node.data == "divide":
                if q2 == 0:
                    return np.inf
                return q1 / q2
            elif tree_node.data == "power":
                if q1 < 0:
                    return np.inf
                return np.float_power(q1, q2)
        elif tree_node.data in UNARY_OPS:
            q = evaluate_expression(tree_node.left_child, values_list, numbers)
            if np.isnan(q) or np.isinf(q):
                return np.inf
            if tree_node.data == "sin":
                return np.sin(q)
            elif tree_node.data == "cos":
                return np.cos(q)
            elif tree_node.data == "tan":
                return np.tan(q)
            elif tree_node.data == "exp":
                return np.exp(q)
            elif tree_node.data == "log":
                if q <= 0:
                    return np.inf
                return np.log(q)
            elif tree_node.data == "sqrt":
                if q < 0:
                    return np.inf
                return np.sqrt(q)
            elif tree_node.data == "abs":
                return np.abs(q)
            elif tree_node.data == "square":
                return np.power(q, 2)
    else:
        if tree_node.data < 0:
            return numbers[(-tree_node.data) - 1]
        return values_list[tree_node.data]



def crossover_trees(parent1: Tree, parent2: Tree):
    new_numbers = []
    param_indices = []

    # Copia l'albero di parent1, trova il nodo per il crossover e imposta un valore temporaneo
    tree1 = duplicate_tree_node(parent1.root)
    current_node = tree1
    while isinstance(current_node.data, str) and current_node.data not in BINARY_OPS:
        current_node = current_node.left_child
    current_node.right_child = TreeNode(0, None, None)

    # Trova i parametri utilizzati nell'albero troncato di tree1 e aggiorna gli indicatori dei parametri
    _ = collect_negative_numbers(tree1, param_indices)
    if param_indices:
        for index in param_indices:
            new_numbers.append(parent1.numbers[(-index) - 1])
            modify_tree_node(tree1, index, -len(new_numbers))

    # Copia il sottoalbero destro di parent2 e trova i parametri utilizzati in esso
    param_indices.clear()
    subtree_candidate = parent2.root
    if isinstance(subtree_candidate.data, str) and subtree_candidate.data in BINARY_OPS:
        if np.random.random() > 0.5:
            subtree = duplicate_tree_node(subtree_candidate.left_child)
        else:
            subtree = duplicate_tree_node(subtree_candidate.right_child)
    elif isinstance(subtree_candidate.data, str):
        subtree = duplicate_tree_node(subtree_candidate.left_child)
    else:
        subtree = TreeNode(
            np.random.choice(list(UNARY_OPS.keys())),
            duplicate_tree_node(subtree_candidate),
            None,
        )

    _ = collect_negative_numbers(subtree, param_indices)
    if param_indices:
        for index in param_indices:
            new_numbers.append(parent2.numbers[(-index) - 1])
            modify_tree_node(subtree, index, -len(new_numbers))

    # Unisce l'albero troncato tree1 con il sottoalbero subtree
    current_node.right_child = subtree
    num_operations = count_operations(tree1)
    return Tree(tree1, num_operations, new_numbers)



def select_parent(trees_population):
    candidates = sorted(np.random.choice(trees_population, 4), key=lambda tree: tree.fitness_score, reverse=True)
    return candidates[0]

def improve_tree(tree_node, numbers):
    if tree_node is None:
        return 0

    if isinstance(tree_node.data, str):
        # Operazioni binarie
        if tree_node.data in BINARY_OPS:
            binary_rules = {
                "add": lambda left, right: left is not None and left < 0 and numbers[-left - 1] == 0,
                "subtract": lambda left, right: right is not None and right < 0 and numbers[-right - 1] == 0,
                "multiply": lambda left, right: left is not None and left < 0 and numbers[-left - 1] == 1,
                "divide": lambda left, right: right is not None and right < 0 and numbers[-right - 1] == 1,
            }
            rule = binary_rules.get(tree_node.data)

            if rule:
                # Verifica il figlio sinistro
                if (
                    tree_node.left_child
                    and not isinstance(tree_node.left_child.data, str)
                    and rule(tree_node.left_child.data, None)
                ):
                    tree_node.data = tree_node.right_child.data
                    tree_node.left_child = tree_node.right_child.left_child
                    tree_node.right_child = tree_node.right_child.right_child
                    return 1

                # Verifica il figlio destro
                if (
                    tree_node.right_child
                    and not isinstance(tree_node.right_child.data, str)
                    and rule(None, tree_node.right_child.data)
                ):
                    tree_node.data = tree_node.left_child.data
                    tree_node.right_child = tree_node.left_child.right_child
                    tree_node.left_child = tree_node.left_child.left_child
                    return 1

            left_improve = improve_tree(tree_node.left_child, numbers) if tree_node.left_child else 0
            right_improve = improve_tree(tree_node.right_child, numbers) if tree_node.right_child else 0
            return left_improve + right_improve

        # Operazioni unarie
        elif tree_node.data in UNARY_OPS:
            unary_rules = {
                "exp": lambda child: child == "log",
                "log": lambda child: child == "exp",
                "sqrt": lambda child: child == "square",
                "square": lambda child: child == "sqrt",
            }
            rule = unary_rules.get(tree_node.data)

            if (
                rule
                and tree_node.left_child
                and isinstance(tree_node.left_child.data, str)
                and rule(tree_node.left_child.data)
            ):
                tree_node.data = tree_node.left_child.left_child.data
                tree_node.right_child = tree_node.left_child.left_child.right_child
                tree_node.left_child = tree_node.left_child.left_child.left_child
                return 0

            return improve_tree(tree_node.left_child, numbers) if tree_node.left_child else 0

    return 0


def improve_tree_structure(tree: Tree):
    old_tree = duplicate_tree_node(tree.root)
    simplification_result = improve_tree(tree.root, tree.numbers)

    if simplification_result > 0:
        # Aggiorna i parametri
        updated_numbers = []
        parameter_indices = []
        _ = collect_negative_numbers(tree.root, parameter_indices)

        if parameter_indices:
            for index in parameter_indices:
                updated_numbers.append(tree.numbers[(-index) - 1])
                modify_tree_node(tree.root, index, -len(updated_numbers))

        tree.numbers = updated_numbers

    tree.identifier = count_operations(tree.root)

    # Se il numero di operazioni è 0, ripristina l'albero precedente
    if tree.identifier == 0:
        tree.root = old_tree

    return tree
