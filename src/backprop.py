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
        # print(self.output)

    def back_propagate(self):
        """ 
        epoch: banyaknya iterasi pada setiap 

        batch_size paramater ukuran minibatch utk Stochastic gradient descent

        step: pemrosesan satu minibatch

        """
        # Untuk setiap layer tapi dari belakang

        total_batch = self.split_input_targets_to_batch(
            self.input, self.target, self.batch_size)

        epoch = 0

      # while epoch < self.max_iteration and error >= self.error_threshold:
      #     # Selama belum gg, backpropagate , update bobot trs feed lagi kedepan
      #     pass
      #
      # print(self.target[i], self.output[i],
      #       self.layers[i]["activation_function"])

      #  error = self.__loss(
      #   self.target[i], output_layer, self.layers[i]["activation_function"])
      #
        # propagate dari belakang, updatenya perminibatch
        total_layer = len(self.layers)
        prev_error = None
        for index_layer in range(total_layer - 1, -1, -1):
            current_activation = self.layers[index_layer]["activation_function"]
            output_layer = self.output[index_layer]
            a = Activation(current_activation)
            for index, batch_instance in enumerate(total_batch):
                inputs = np.array(batch_instance["inputs"])
                targets = np.array(batch_instance["targets"])
                gradient = None

                # Kalau dia punya output (layer paling ujung)
                if index_layer == total_layer - 1:
                    # Calculate dho error / dho weight
                    # First part: -(t - o) -> (o - t)
                    first_part = np.subtract(output_layer, targets)

                    # Second part: derivative
                    second_part = a.derivative(output_layer, targets)

                    # Third part: input di layer sebelumnya
                    third_part = None

                    # Kalau cuma 1 layer, pake input layer
                    # add bias
                    if len(self.output) == 1:
                        third_part = np.insert(
                            inputs, 0, np.ones(inputs[index].shape[0]))
                    else:
                        third_part = np.insert(
                            self.output[index_layer - 1], 0, np.ones(self.output[index_layer-1].shape[0]))

                    # Third part must same as len each inital weights transposed
                    cur_weights = np.transpose(self.weights[index_layer])
                    third_parts = np.zeros(cur_weights.shape)
                    for i in range(len(cur_weights)):
                        third_parts[i] = third_part

                    third_parts = np.transpose(third_parts)

                    prev_error = np.multiply(first_part, second_part)
                    gradient = np.multiply(prev_error, third_parts)
                    """ 
                    b b
                    w5 w7
                    w6 w8
                    [[ 0.13849856 -0.03809824]
                    [ 0.08216704 -0.02260254]
                    [ 0.08266763 -0.02274024]]
                    """
                    # print(first_part, second_part, third_part)
                    # print(third_part)
                    # gradient = np.dot(
                    #     np.dot(first_part, second_part), third_part)
                    # print(gradient)
                # Kalau hidden layer
                else:
                    # Calculate dho error / dho weight
                    # First part: dho error / dho net

                    # dho Ed / dho net_o; prev error
                    # print(prev_error)

                    # dho net_o / dho h
                    next_layer_cur_weights = self.weights[index_layer + 1][1:]

                    # dho Ed / dho h
                    temp = np.multiply(prev_error, next_layer_cur_weights)

                    # calculate sum of the error outputs layer
                    output_layer_sum_error = np.array(
                        [np.sum(x) for x in temp])

                    derivative = a.derivative(output_layer, targets)

                    # calculate prev error
                    prev_error = np.multiply(
                        output_layer_sum_error, derivative)

                    # Second part: dho net / dho weight
                    second_part = None

                    # Kalau udah paling ujung
                    if index_layer == 0:
                        second_part = np.insert(
                            inputs, 0, np.ones(inputs.shape[0]))
                        pass
                    else:
                        second_part = np.insert(
                            self.output[index_layer], 0, np.ones(self.output[index_layer].shape[0]))

                    cur_weights = np.transpose(self.weights[i])
                    second_parts = np.zeros(cur_weights.shape)
                    for i in range(len(cur_weights)):
                        second_parts[i] = second_part

                    second_parts = np.transpose(second_parts)

                    gradient = np.multiply(prev_error, second_parts)
                    # print(gradient)

            # Update weight
            self.weights[i] = self.weights[index_layer] - \
                np.dot(self.learning_rate, gradient)

    def split_input_targets_to_batch(self, inputs: List, targets: List, batch_size: int) -> List[List]:
        total_batch = []
        ctr = 0
        cur_input = []
        cur_target = []

        length = len(inputs)
        for i in range(length):
            cur_target.append(targets[i])
            cur_input.append(inputs[i])

            ctr += 1

            if ctr == batch_size:
                total_batch.append({
                    "inputs": cur_input,
                    "targets": cur_target
                })

                ctr = 0
                cur_target = []
                cur_input = []

        return total_batch

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
