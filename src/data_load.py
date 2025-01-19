import numpy as np
from icecream import ic


BINARY_OPS = {
    'subtract': lambda a, b: np.subtract(a, b, where=~np.isnan(a) & ~np.isnan(b)),
    'add': lambda a, b: np.add(a, b, where=~np.isnan(a) & ~np.isnan(b)),
    'divide': lambda a, b: np.divide(a, b, out=np.full_like(a, float('inf')), where=(b != 0) & ~np.isnan(a) & ~np.isnan(b)),
    'multiply': lambda a, b: np.multiply(a, b, where=~np.isnan(a) & ~np.isnan(b)),
    'power': lambda a, b: np.power(a, b, out=np.full_like(a, float('inf')), where=(a >= 0) & ~np.isnan(a) & ~np.isnan(b))
}

UNARY_OPS = {
    'cos': lambda a: np.cos(np.nan_to_num(a)),
    'sin': lambda a: np.sin(np.nan_to_num(a)),
    'exp': lambda a: np.exp(np.clip(a, None, 700)),  # Avoid overflow
    'tan': lambda a: np.tan(np.nan_to_num(a)),
    'sqrt': lambda a: np.sqrt(np.where(a >= 0, a, float('inf'))),
    'log': lambda a: np.log(np.where(a > 0, a, float('inf'))),
    'square': lambda a: np.square(np.nan_to_num(a)),
    'abs': lambda a: np.abs(np.nan_to_num(a))
}

data = np.load('./data/problem_8.npz')
x = data['x']
y = data['y']
ic(x.shape, y.shape) 