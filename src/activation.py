import math

import numpy as np


class Activation:
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"

    def __init__(self, mode) -> None:
        self.mode = mode

    def __linear_calculate(self, res):
        return res

    def __sigmoid_calculate(self, res):
        res = np.array([(1 / (1 + pow(math.e, -x))) for x in res])
        return res

    def __relu_calculate(self, res):
        res[res < 0] = 0
        return res

    def __softmax_calculate(self, res):
        numerator = np.array([pow(math.e, x) for x in res])
        denominator = np.sum([pow(math.e, x) for x in res])
        return numerator / denominator

    def calculate(self, x, w, b):
        res = np.matmul(x, w)
        res = np.add(res, b)
        if self.mode == Activation.LINEAR:
            return self.__linear_calculate(res)
        elif self.mode == Activation.RELU:
            return self.__relu_calculate(res)
        elif self.mode == Activation.SIGMOID:
            return self.__sigmoid_calculate(res)
        elif self.mode == Activation.SOFTMAX:
            return self.__softmax_calculate(res)
        else:
            raise Exception(
                "Mode is not implemented, please select correct mode")

    def predict(self, res):
        if self.mode == Activation.LINEAR:
            return res
        elif self.mode == Activation.RELU:
            return res
        elif self.mode == Activation.SIGMOID:
            return res
        elif self.mode == Activation.SOFTMAX:
            return res
        else:
            raise Exception(
                "Mode is not implemented, please select correct mode")


if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    w1 = np.array([[1, 1], [1, 1]])
    b1 = np.array([0, -1])
    res = np.matmul(x, w1)
    res = np.add(res, b1)
    print(res)
    res[res < 0] = 0
    print(res)

    # w = np.array([1, -2])
    # res = np.matmul(res, w)
    # Max 0

    # numerator = np.array([pow(math.e, x) for x in [3, 4, 1]])
    # denominator = np.sum([pow(math.e, x) for x in [3, 4, 1]])
    # print(numerator / denominator)
