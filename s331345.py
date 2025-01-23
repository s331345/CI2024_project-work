# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5

def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray:
    return (x[1] + 10) * (x[2] + 1) - x[0] * (x[1] - 1)

def f3(x: np.ndarray) -> np.ndarray:
    return ((9) * (8 - x[1])) + (((4 / (x[2] + 2)) - 5) * 8)

def f4(x: np.ndarray) -> np.ndarray:
    return np.sin(np.exp(x[0] * x[1])) + np.exp(np.log(((x[0] + 2) / (2 * x[1])) + 1))

def f5(x: np.ndarray) -> np.ndarray:
    return (np.sin((x[1] / 100) * (x[0] / 50)) - (x[1] / 100) * (x[0] / 50)) * (np.cos(np.cos(x[1])) * np.cos(np.cos(x[1])))

def f6(x: np.ndarray) -> np.ndarray:
    return ((x[1] * 1000 / 616) * 1000 / 958) - (x[0] * 1000 / 1440)


def f7(x: np.ndarray) -> np.ndarray:
    return np.abs((71 * x[0] * x[0] // 24 * x[1]) // (np.abs(x[0] - x[1]) + np.sin(66)))

def f8(x: np.ndarray) -> np.ndarray:
    return (((x[5] * x[5] * x[5]) * np.abs(x[5] * x[5] * x[5])) + (x[5] * np.sqrt(np.abs(x[5])) * 85)) + (x[4] * x[4] * 80)