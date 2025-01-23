import numpy as np
from icecream import ic


BINARY_OPS = {
    'add': lambda a, b: np.add(a, b, dtype=np.float64),
    'subtract': lambda a, b: np.subtract(a, b, dtype=np.float64),
    'multiply': lambda a, b: np.multiply(a, b, dtype=np.float64),
    'divide': lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan, dtype=np.float64), where=(b != 0)),
    'power': lambda a, b: np.power(a, b, out=np.full_like(a, np.nan, dtype=np.float64), where=(a >= 0) & (b >= 0)),
}

UNARY_OPS = {
    'sin': lambda a: np.sin(np.nan_to_num(a, nan=0.0)),
    'cos': lambda a: np.cos(np.nan_to_num(a, nan=0.0)),
    'exp': lambda a: np.exp(np.clip(a, None, 700)),  # Evita overflow
    'log': lambda a: np.log(np.where(a > 0, a, np.nan)),
    'sqrt': lambda a: np.sqrt(np.where(a >= 0, a, np.nan)),
}


data = np.load('./data/problem_1.npz')
x = data['x']
y = data['y']
ic(x.shape, y.shape) 