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
        formula = tree_to_string(self.root, self.numbers) if self.root else "None"
        return (f"Tree(formula={formula}, score={self.fitness_score})")



    def improve(self):
        #i use it for removing redundant operations and updating parameters.
        old_tree = duplicate_tree_node(self.root)

        if improve_tree(self.root, self.numbers):
            
            updated_numbers = []
            parameter_indices = []

            # indexes
            collect_nodes_by_label(self.root, "parameter", parameter_indices)

            # update numbers
            for index in parameter_indices:
                if index < len(self.numbers):  # valid index?
                    updated_numbers.append(self.numbers[index])

            self.numbers = updated_numbers

        #new identifier
        self.identifier = count_operations(self.root)

        # no valid operations, restore old tree
        if self.identifier == 0:
            self.root = old_tree


    def mutate(self):
        #to make random changes to the tree to explore new solutions
        mutated_expression = duplicate_tree_node(self.root)
        new_numbers = self.numbers.copy()
        
        if np.random.random() < 0.4 and self.identifier != 0:
            
            mutate_tree_nodes(mutated_expression, 1 / self.identifier)
        else:
            if np.random.random() < 0.1:
                
                current_node = mutated_expression
                parent_node = None
                is_leaf_found = False

                # search for a leaf
                while not is_leaf_found:
                    if current_node.label in ["parameter", "variable"]:
                        is_leaf_found = True
                    else:
                        if np.random.random() < 0.5 and current_node.left_child:
                            parent_node = current_node
                            current_node = current_node.left_child
                        elif current_node.right_child:
                            parent_node = current_node
                            current_node = current_node.right_child

                
                new_leaf_label = "variable" if np.random.random() > 0.5 else "parameter"
                new_leaf = TreeNode(
                    np.random.randint(len(x)) if new_leaf_label == "variable" else np.random.randint(0, 10),
                    label=new_leaf_label,
                )

                if parent_node.left_child == current_node:
                    parent_node.left_child = TreeNode(
                        np.random.choice(list(BINARY_OPS.keys())), current_node, new_leaf, label="binary"
                    )
                else:
                    parent_node.right_child = TreeNode(
                        np.random.choice(list(BINARY_OPS.keys())), current_node, new_leaf, label="binary"
                    )
            else:
                
                if len(self.numbers) > 0:
                    param_index = np.random.randint(0, len(self.numbers))
                    if np.random.random() < 0.5:
                        new_numbers[param_index] += 1
                    else:
                        new_numbers[param_index] -= 1

        self.root = mutated_expression
        self.numbers = new_numbers
        self.identifier = count_operations(mutated_expression)


    def calculate_fitness(self):
        #to measure how good the tree is by calculating fitness with MSE and a penalty.
        try:
            total_error = 0
            n_samples = x.shape[0]  

            for i in range(n_samples):
                prediction = evaluate_expression(self.root, x[i], self.numbers)
                if np.isnan(prediction) or np.isinf(prediction):
                    self.fitness_score = -np.inf
                    return

                total_error += np.square(y[i] - prediction)

            mse = total_error / n_samples

            complexity_penalty = self.identifier * 10 
            self.fitness_score = -mse - complexity_penalty

        except Exception as e:
            print(f"Errore durante il calcolo del fitness: {e}")
            self.fitness_score = -np.inf



#I use it to simplify the tree and update its parameters after structural changes.

def improve_tree_structure(tree: Tree):
    
    old_tree = duplicate_tree_node(tree.root)
    simplification_result = improve_tree(tree.root, tree.numbers)

    if simplification_result > 0:
        tree.root = simplify_tree(tree.root)

        updated_numbers = []
        parameter_indices = []

        def collect_parameters(node):
            if node is None:
                return
            if node.label == "parameter":
                parameter_indices.append(node.data)
            collect_parameters(node.left_child)
            collect_parameters(node.right_child)

        collect_parameters(tree.root)

        updated_numbers = list(set(parameter_indices))
        tree.numbers = updated_numbers

    tree.identifier = count_operations(tree.root)

    if tree.identifier == 0:
        tree.root = old_tree

    return tree

# removing constant operations and neutral elements.
def simplify_tree(tree_node):
   
    if tree_node is None:
        return tree_node

    tree_node.left_child = simplify_tree(tree_node.left_child)
    tree_node.right_child = simplify_tree(tree_node.right_child)

    if tree_node.label == "binary":
        left = tree_node.left_child
        right = tree_node.right_child

        if left and right and left.label == "parameter" and right.label == "parameter":
            if tree_node.data == "add":
                return TreeNode(left.data + right.data, label="parameter")
            elif tree_node.data == "subtract":
                return TreeNode(left.data - right.data, label="parameter")
            elif tree_node.data == "multiply":
                return TreeNode(left.data * right.data, label="parameter")
            elif tree_node.data == "divide" and right.data != 0:
                return TreeNode(left.data / right.data, label="parameter")

        if tree_node.data == "add":
            if left and left.label == "parameter" and left.data == 0:
                return right
            if right and right.label == "parameter" and right.data == 0:
                return left
        elif tree_node.data == "multiply":
            if left and left.label == "parameter" and left.data == 1:
                return right
            if right and right.label == "parameter" and right.data == 1:
                return left
            if left and left.label == "parameter" and left.data == 0:
                return TreeNode(0, label="parameter")
            if right and right.label == "parameter" and right.data == 0:
                return TreeNode(0, label="parameter")
        elif tree_node.data == "subtract":
            if right and right.label == "parameter" and right.data == 0:
                return left
        elif tree_node.data == "divide":
            if right and right.label == "parameter" and right.data == 1:
                return left

    if tree_node.label == "unary" and tree_node.left_child:
        child = tree_node.left_child

        if child.label == "parameter":
            if tree_node.data == "sin":
                return TreeNode(np.sin(child.data), label="parameter")
            elif tree_node.data == "cos":
                return TreeNode(np.cos(child.data), label="parameter")
            elif tree_node.data == "exp":
                return TreeNode(np.exp(child.data), label="parameter")
            elif tree_node.data == "log" and child.data > 0:
                return TreeNode(np.log(child.data), label="parameter")
            elif tree_node.data == "sqrt" and child.data >= 0:
                return TreeNode(np.sqrt(child.data), label="parameter")

    return tree_node



def improve_tree(tree_node, numbers):
    if tree_node is None:
        return 0

    if tree_node.label == "binary":
        binary_rules = {
            "add": lambda left, right: left and left.label == "parameter" and numbers[left.data] == 0,
            "subtract": lambda left, right: right and right.label == "parameter" and numbers[right.data] == 0,
            "multiply": lambda left, right: left and left.label == "parameter" and numbers[left.data] == 1,
            "divide": lambda left, right: right and right.label == "parameter" and numbers[right.data] == 1,
        }

        rule = binary_rules.get(tree_node.data)

        if rule:
            if (
                tree_node.left_child
                and tree_node.left_child.label == "parameter"
                and 0 <= tree_node.left_child.data < len(numbers)
                and rule(tree_node.left_child, None)
            ):
                tree_node.data = tree_node.right_child.data
                tree_node.label = tree_node.right_child.label
                tree_node.left_child = tree_node.right_child.left_child
                tree_node.right_child = tree_node.right_child.right_child
                return 1

            if (
                tree_node.right_child
                and tree_node.right_child.label == "parameter"
                and 0 <= tree_node.right_child.data < len(numbers)
                and rule(None, tree_node.right_child)
            ):
                tree_node.data = tree_node.left_child.data
                tree_node.label = tree_node.left_child.label
                tree_node.right_child = tree_node.left_child.right_child
                tree_node.left_child = tree_node.left_child.left_child
                return 1

        left_improve = improve_tree(tree_node.left_child, numbers) if tree_node.left_child else 0
        right_improve = improve_tree(tree_node.right_child, numbers) if tree_node.right_child else 0
        return left_improve + right_improve

    elif tree_node.label == "unary":
        unary_rules = {
            "exp": lambda child: child.label == "unary" and child.data == "log",
            "log": lambda child: child.label == "unary" and child.data == "exp",
            "sqrt": lambda child: child.label == "unary" and child.data == "square",
            "square": lambda child: child.label == "unary" and child.data == "sqrt",
        }
        rule = unary_rules.get(tree_node.data)

        if (
            rule
            and tree_node.left_child
            and rule(tree_node.left_child)
        ):
            tree_node.data = tree_node.left_child.left_child.data
            tree_node.label = tree_node.left_child.left_child.label
            tree_node.right_child = tree_node.left_child.left_child.right_child
            tree_node.left_child = tree_node.left_child.left_child.left_child
            return 1

        return improve_tree(tree_node.left_child, numbers) if tree_node.left_child else 0

    return 0



def select_parent(trees_population):
    candidates = sorted(np.random.choice(trees_population, 4), key=lambda tree: tree.fitness_score, reverse=True)
    return candidates[0]

def tree_to_string(tree_node, numbers):
    if tree_node.label == "binary":
        return f"({tree_to_string(tree_node.left_child, numbers)} {tree_node.data} {tree_to_string(tree_node.right_child, numbers)})"
    elif tree_node.label == "unary":
        return f"{tree_node.data}({tree_to_string(tree_node.left_child, numbers)})"
    elif tree_node.label == "parameter":
        return str(tree_node.data)  
    elif tree_node.label == "variable":
        return f"x{tree_node.data}"  
    return ""

def evaluate_expression(tree_node, values_list, numbers):
    if tree_node.label == "binary":
        q1 = evaluate_expression(tree_node.left_child, values_list, numbers)
        q2 = evaluate_expression(tree_node.right_child, values_list, numbers)
        if np.isnan(q1) or np.isnan(q2) or np.isinf(q1) or np.isinf(q2):
            return np.inf
        return BINARY_OPS[tree_node.data](q1, q2)
    elif tree_node.label == "unary":
        q = evaluate_expression(tree_node.left_child, values_list, numbers)
        if np.isnan(q) or np.isinf(q):
            return np.inf
        return UNARY_OPS[tree_node.data](q)
    elif tree_node.label == "parameter":
        return tree_node.data  
    elif tree_node.label == "variable":
        return values_list[tree_node.data]  
    return 0

def crossover_trees(parent1: Tree, parent2: Tree):
    new_numbers = []

    tree1 = duplicate_tree_node(parent1.root)
    current_node = tree1
    while current_node.label != "binary" and current_node.left_child:
        current_node = current_node.left_child
    current_node.right_child = TreeNode(0, label="parameter") 

    def collect_parameters(node, numbers):
        if node is None:
            return
        if node.label == "parameter":
            if node.data not in numbers:
                numbers.append(node.data)
        collect_parameters(node.left_child, numbers)
        collect_parameters(node.right_child, numbers)

    collect_parameters(tree1, new_numbers)

    subtree_candidate = parent2.root
    if subtree_candidate.label == "binary":
        if np.random.random() > 0.5:
            subtree = duplicate_tree_node(subtree_candidate.left_child)
        else:
            subtree = duplicate_tree_node(subtree_candidate.right_child)
    elif subtree_candidate.label == "unary":
        subtree = duplicate_tree_node(subtree_candidate.left_child)
    else:
        subtree = TreeNode(
            np.random.choice(list(UNARY_OPS.keys())), 
            duplicate_tree_node(subtree_candidate), 
            None, 
            label="unary"
        )

    collect_parameters(subtree, new_numbers)

    current_node.right_child = subtree
    num_operations = count_operations(tree1)
    return Tree(tree1, num_operations, new_numbers)

def duplicate_tree_node(tree_node):
    left_child = None
    right_child = None

    if tree_node.left_child is not None:
        left_child = duplicate_tree_node(tree_node.left_child)
    if tree_node.right_child is not None:
        right_child = duplicate_tree_node(tree_node.right_child)

    return TreeNode(tree_node.data, left_child, right_child, label=tree_node.label)

def create_random_tree(max_depth, available_values, depth=0):
    if depth >= max_depth or (depth > 0 and np.random.random() < 0.3):
        if np.random.random() > 0.5:  
            index = np.random.randint(len(available_values))
            return TreeNode(index, label="variable")
        else:  
            param_value = np.random.randint(0, 10)
            return TreeNode(param_value, label="parameter")

    if np.random.random() > 0.5:  
        operation = np.random.choice(list(BINARY_OPS.keys()))
        node = TreeNode(operation, label="binary")
        node.left_child = create_random_tree(max_depth, available_values, depth + 1)
        node.right_child = create_random_tree(max_depth, available_values, depth + 1)
    else: 
        operation = np.random.choice(list(UNARY_OPS.keys()))
        node = TreeNode(operation, label="unary")
        node.left_child = create_random_tree(max_depth, available_values, depth + 1)

    return node

def modify_tree_node(tree_node, target_value, new_value):
    if tree_node.label in ["binary", "unary"]:
        if tree_node.left_child and modify_tree_node(tree_node.left_child, target_value, new_value) == 1:
            return 1
        if tree_node.right_child and modify_tree_node(tree_node.right_child, target_value, new_value) == 1:
            return 1
    elif tree_node.label in ["parameter", "variable"]:
        if tree_node.data == target_value:
            tree_node.data = new_value
            return 1
    return 0

def collect_nodes_by_label(tree_node, label, nodes_list):
    if tree_node is None:
        return 0

    if tree_node.label == label:
        nodes_list.append(tree_node.data)
        return 1

    left_count = collect_nodes_by_label(tree_node.left_child, label, nodes_list)
    right_count = collect_nodes_by_label(tree_node.right_child, label, nodes_list)

    return left_count + right_count

def count_operations(tree_node):
    if tree_node is None:
        return 0

    if tree_node.label == "binary":
        return count_operations(tree_node.left_child) + count_operations(tree_node.right_child) + 1
    elif tree_node.label == "unary":
        return count_operations(tree_node.left_child) + 1

    return 0

