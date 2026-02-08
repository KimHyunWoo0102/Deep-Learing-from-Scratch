import numpy as np


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3

    return 1 if np.sum(w * x) + b > 0 else 0


if __name__ == "__main__":
    print(OR(0, 0))
    print(OR(0, 1))
    print(OR(1, 0))
    print(OR(1, 1))
