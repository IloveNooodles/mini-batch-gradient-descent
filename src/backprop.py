import numpy as np

from .activation import Activation


class Backprop:
    def __init__(self) -> None:
        pass

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
