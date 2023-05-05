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

    def __linear_derivative(self, res):
        return np.ones(np.shape(res))

    def __sigmoid_calculate(self, res):
        res = np.array([(1 / (1 + pow(math.e, -x))) for x in res])
        return res

    def __sigmoid_derivative(self, res):
        return res * (1 - res)

    def __relu_calculate(self, res):
        res[res < 0] = 0
        return res

    def __relu_derivative(self, res):
        res_copy = res.copy()
        res_copy[res_copy <= 0] = 0
        res_copy[res_copy > 0] = 1
        return res_copy

    def __softmax_calculate(self, res):
        numerator = np.array([pow(math.e, x) for x in res])
        denominator = np.array([np.sum(pow(math.e, x)) for x in res])
        result = np.empty(numerator.shape)

        for index, ele in enumerate(numerator):
            result[index] = ele / denominator[index]

        return result

    def __softmax_derivative(self, res, target=[]):
        """ 
        if t == 1 then -(1-o) -> o - 1
        """
        subtract = np.subtract(res, target)
        return np.array(subtract)

        return res

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

    def derivative(self, res, target=[]):
        if self.mode == Activation.LINEAR:
            return self.__linear_derivative(res)
        elif self.mode == Activation.RELU:
            return self.__relu_derivative(res)
        elif self.mode == Activation.SIGMOID:
            return self.__sigmoid_derivative(res)
        elif self.mode == Activation.SOFTMAX:
            return self.__softmax_derivative(res, target)
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
    # Softmax test
    output = np.array([9.85e-1, 1.419e-2, 2.04e-4, 2.94e-6,
                      4.24e-8, 6.11e-10, 8.81e-12, 6.11e-10, 4.24e-8, 1.82e-15])

    target = np.zeros(output.shape)
    target[-1] = 1

    a = Activation(Activation.SOFTMAX)
    res = a.derivative(output, target)

    print(res)
