import random
from typing import Iterable
import numpy as np
from negmas.preferences.value_fun import TableFun, AffineFun, LinearFun, LambdaFun

__ = TableFun, AffineFun, LinearFun, LambdaFun

FloatRange = tuple[float, float] | tuple[float, float, int] | list[float] | float
IntRange = tuple[int, int] | tuple[int, int, int] | list[int] | int
FloatIssueRange = tuple[float, float] | tuple[float, float, float] | list[float]

# random.seed(1234)
# np.random.seed(1234)


def range_in(x: FloatIssueRange):
    if isinstance(x, tuple) and len(x) == 3:
        x = np.round(np.linspace(x[0], x[1], num=x[2], endpoint=True)).tolist()  # type: ignore
    if isinstance(x, tuple) and len(x) == 2:
        return range_in((*x, 11))  # type: ignore
    if isinstance(x, list):
        return x
    raise ValueError(f"Unsupported iterable {x}")


def float_in(x: FloatRange):
    if isinstance(x, tuple) and len(x) == 3:
        num = (x[1] - x[0]) / x[2]
        x = np.linspace(x[0], x[1], num=num).tolist()  # type: ignore
    if isinstance(x, tuple) and len(x) == 2:
        return x[0] + (x[1] - x[0]) * random.random()
    if isinstance(x, list):
        return random.choice(x)
    if isinstance(x, Iterable):
        raise ValueError(f"Unsupported iterable {x}")
    return x


def int_in(x: IntRange):
    if isinstance(x, tuple) and len(x) == 2:
        return random.randint(x[0], x[-1])
    if isinstance(x, tuple) and len(x) == 3:
        x = list(range(x[0], x[1], x[2]))
    if isinstance(x, list):
        return random.choice(list(x))
    if isinstance(x, Iterable):
        raise ValueError(f"Unsupported iterable {x}")
    return x
