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

    def transform_to_ffnn_model(self):
        model = {}
        model["layers"] = len(self.layers) + 1
        model["activation_functions"] = [x["activation_function"]
                                         for x in self.layers]
        model["neurons"] = [self.input_size] + [x["number_of_neurons"]
                                                for x in self.layers]
        model["weights"] = [np.transpose(x) for x in self.weights]
        model["rows"] = len(self.input)
        model["data"] = self.input
        model["target"] = self.target
        model["max_sse"] = MAX_SSE

        return model

    def __loss(target: list, y: list, activation) -> float:
        # For linear, sigmoid, relu use SSE
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
