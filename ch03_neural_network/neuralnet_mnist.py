import os
import sys
import pickle
import numpy as np

# --- [1] 절대 경로 및 시스템 경로 설정 ---
# 현재 파일(neuralnet_mnist.py)의 절대 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 부모 폴더(프로젝트 루트: Deep-Learning-from-Scratch) 경로 계산
# os.pardir는 '..'와 같으며, 어떤 폴더 이름이든 상관없이 한 단계 위로 올라갑니다.
root_path = os.path.abspath(os.path.join(current_dir, os.pardir))

# 파이썬 모듈 검색 경로에 루트 추가
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# --- 경로 진입 로직 끝 ---

from dataset.mnist import load_mnist
from ch03_neural_network.activate_functions.sigmoid_function import sigmoid
from ch03_neural_network.softmax import softmax
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


def init_network():
    weight_file_path = os.path.join(current_dir, "sample_weight.pkl")
    with open(weight_file_path, "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i : i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)

        accuracy_cnt += np.sum(p == t[i : i + batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
