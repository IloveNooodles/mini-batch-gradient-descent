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
        self.learning_params = model["learning_parameters"]
        self.expected = expected

    def __loss(target: list, y: list, activation) -> float:
        """ 
        For linear, sigmoid, relu use SSE.
        Softmax use cross entropy 
        """
        result = 0
        if activation == Activation.SOFTMAX:
            # cross entropy
            pass

        else:
            # SSE
            total = 0
            length = len(y)
            for i in range(length):
                total += pow(target[i] - y[i], 2)

            result = total / 2

        return result
