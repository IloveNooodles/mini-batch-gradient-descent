from typing import List

import numpy as np

from .activation import Activation
from .ffnn import FFNN

MAX_SSE = 1e-8


class Backpropagation:
    def __init__(self, model, expected, ffnn_model) -> None:
        self.input_size = model["model"]["input_size"]
        self.layers = model["model"]["layers"]
        self.input = np.array(model["input"])
        self.weights = np.array(model["initial_weights"])
        self.target = np.array(model["target"])
        self.learning_rate = model["learning_parameters"]["learning_rate"]
        self.batch_size = model["learning_parameters"]["batch_size"]
        self.max_iteration = model["learning_parameters"]["max_iteration"]
        self.error_threshold = model["learning_parameters"]["error_threshold"]
        self.expected = expected
        self.ffnn_model = ffnn_model
        self.output = None

    def forward_propagation(self):
        """ 
        Propagate Forward using FFNN
        """
        ffnn = FFNN(self.ffnn_model)
        res = ffnn.compute()
        # ffnn.predict()
        # self.output = res
        self.output = ffnn.get_all_output_layer()

    def back_propagate(self):
        """ 
        epoch: banyaknya iterasi pada setiap 

        batch_size paramater ukuran minibatch utk Stochastic gradient descent

        step: pemrosesan satu minibatch

        """
        # Untuk setiap layer tapi dari belakang

        splitted_input, splitted_target = self.split_input_targets(
            self.input, self.target, self.batch_size)

        epoch = 0

        for i in range(len(self.layers) - 1, -1, -1):
            output_layer = self.output[i]

            # print(self.target[i], self.output[i],
            #       self.layers[i]["activation_function"])
            error = self.__loss(
                self.target[i], output_layer, self.layers[i]["activation_function"])

            while epoch < self.max_iteration and error >= self.error_threshold:
                # Selama belum gg, backpropagate , update bobot trs feed lagi kedepan
                pass

    def split_input_targets(self, inputs: List, targets: List, batch_size: int) -> List[List]:
        total_input = []
        total_target = []
        ctr = 0
        cur_input = []
        cur_target = []

        length = len(inputs)
        for i in range(length):
            cur_target.append(targets[i])
            cur_input.append(inputs[i])

            ctr += 1

            if ctr == batch_size:
                total_input.append(cur_input)
                total_target.append(cur_target)
                ctr = 0
                cur_target = []
                cur_input = []

        return total_input, total_target

    def mini_batch(self, activation=Activation.LINEAR):
        """ 
        tergantung batch_size
        1. batch_size == 1 then incremental
        2. batch_size == |len tranining data| then  stochastic gradient descent
        3. else mini batch

        Update weight, weight = weight - ndeltaError
        """

        # Until termination condition is met

        # TEST MODEL
        initial_weight = np.array([
            [0.5, 0.5],
            [0.0, 0.0]
        ])
        input_model = np.array([0.0])
        ouput_model = np.array([0.5, 0.5])
        target = np.array([0, 1])
        error = 0
        max_iterasi = 1000
        batch_size = 1

        # Init delta w jadi nol
        delta_weight = np.zeros((4,))

        # untuk setiap instance lakukan
        for epoch in range(max_iterasi):

            # Untuk seiap batch
            for i in range(batch_size):
                # untuk setiap instance update delta bobot dan error

                # error += self.__loss()

                # Update
                pass
            # Update bobot

        return error

    def __loss(self, target, pred, activation):
        """ 
        For linear, sigmoid, relu use SSE.

        Softmax use cross entropy 
        """
        if activation == Activation.SOFTMAX:
            return self.__cross_entropy(target, pred)
        else:
            return self.__sse(target, pred)

    def __sse(self, target, pred):
        """ 
        Calculate errors using sse

        sum squared of (target - pred )/ 2
        """
        total = 0
        length = len(pred)
        for i in range(length):
            total += pow(target[i] - pred[i], 2)
        return np.sum(total) / 2

    def __cross_entropy(self, target, pred):
        """ 
        Calculate cross entropy

        If target == 1 then -np log pj

        """
        res = 0
        for index, val in enumerate(target):
            for j, t in enumerate(val):
                if t == 1:
                    res += np.sum(-np.log(pred[index][j]))

        return res
