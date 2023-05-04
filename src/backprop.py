import numpy as np

from .activation import Activation

MAX_SSE = 1e-8


class Backpropagation:
    def __init__(self, model, expected) -> None:
        self.input_size = model["model"]["input_size"]
        self.layers = model["model"]["layers"]
        self.input = model["input"]
        self.weights = model["initial_weights"]
        self.target = model["target"]
        self.learning_rate = model["learning_parameters"]["learning_rate"]
        self.batch_size = model["learning_parameters"]["batch_size"]
        self.max_iteration = model["learning_parameters"]["max_iteration"]
        self.error_threshold = model["learning_parameters"]["error_threshold"]
        self.expected = expected

    def calculate(self):
        error = 1
        
        for i in range(len(self.max_iteration)):
            pass

    def __loss(self, target: list, pred: list, activation):
        """ 
        For linear, sigmoid, relu use SSE.

        Softmax use cross entropy 
        """
        result = 0
        if activation == Activation.SOFTMAX:
            result = self.__cross_entropy(target, pred)
        else:
            result = self._sse(target, pred)

        return result

    def __sse(self, target: list, pred: list):
        """ 
        Calculate errors using sse

        sum squared of (target - pred )/ 2
        """
        total = 0
        length = len(pred)
        for i in range(length):
            total += pow(target[i] - pred[i], 2)

        return total / 2

    def __cross_entropy(self, target: list, pred: list):
        """ 
        Calculate cross entropy

        If target == 1 then -np log pj

        """
        for index, val in enumerate(target):
            if val == 1:
                return -np.log(pred[index])
