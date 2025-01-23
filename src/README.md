# CI2024_project-work

Final project of the 'Computational intelligence' course 2024-2025

The project is organized into separate files, and each file has a specific job. Here's a quick explanation of what each file does and how they work together:

data_load.py: This file loads the dataset and defines safe math operations (like addition, multiplication, division, and logarithms) to avoid errors like dividing by zero or using negative numbers in logs. It outputs the dataset (x and y) and provides operators used in the program, stored in two dictionaries.

Tree_node.py: This file has the TreeNode class, which represents parts of an expression tree. Each node can be a variable, a constant, or an operation (like addition or sine). It also includes ways to change, copy, or evaluate the nodes.

Tree.py: This file has the Tree class, which manages expression trees as a whole. These trees are potential solutions (formulas) for the problem. The class allows things like mutation, crossover, simplification, and checking how good a solution is. It uses data_load.py for math calculations and fitness checks.

Population.py: This file manages groups (populations) of trees. It creates the first group of trees, picks "parents" for making new trees, and updates the group with new ones. It uses Tree.py for all tree-related operations and uses a method called tournament selection to pick the best parents.

gp.py: This file is where the genetic programming algorithm is written. It handles the whole process of evolving trees step by step—selection, crossover, mutation, and updating the population. It uses Population.py to manage the population across generations.

main.py: This is the starting point of the program. It sets up the population, chooses parameters (like how many trees and generations to use), and runs the genetic algorithm. In the end, it gives the best solution found, which is an understandable formula for the regression task.