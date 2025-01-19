from data_load import BINARY_OPS, UNARY_OPS, x, y
import numpy as np  


class TreeNode:
    def __init__(self, data: str | int, left_child=None, right_child=None):
        self.data = data
        self.left_child = left_child
        self.right_child = right_child

    def __repr__(self):
        return f"TreeNode(data={self.data}, left_child={self.left_child}, right_child={self.right_child})"

def mutate_tree_nodes(tree_node, mutation_prob):
    if isinstance(tree_node.data, str):
        if tree_node.data in BINARY_OPS:
            if np.random.random() < mutation_prob:
                tree_node.data = np.random.choice(list(BINARY_OPS.keys()))
            if tree_node.left_child is not None:
                mutate_tree_nodes(tree_node.left_child, mutation_prob)
            if tree_node.right_child is not None:
                mutate_tree_nodes(tree_node.right_child, mutation_prob)
        elif tree_node.data in UNARY_OPS:
            if np.random.random() < mutation_prob:
                tree_node.data = np.random.choice(list(UNARY_OPS.keys()))
            if tree_node.left_child is not None:
                mutate_tree_nodes(tree_node.left_child, mutation_prob)
    else:
        return 0
