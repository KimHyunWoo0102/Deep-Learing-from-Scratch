import numpy as np


def softmax(a):
    # 숫자가 커지면 오버플로우가 발생하니 지수 크기 조절
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
