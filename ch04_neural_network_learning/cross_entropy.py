import numpy as np


def cross_entropy_error(y, t):
    y = np.array(y)
    t = np.array(t)

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + delta)) / batch_size


if __name__ == "__main__":
    import sys, os

    parent_dir = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(parent_dir)

    from dataset.mnist import load_mnist

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(cross_entropy_error([0.1, 0.2, 0.7], [0, 0, 1]))

    # TODO : x를 훈련시켜서 y 예측값을 뽑아냈다고 칩시다.
