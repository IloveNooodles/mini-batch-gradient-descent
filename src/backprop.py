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
        self.single_output = None
        self.output = None

    def forward_propagation(self):
        """ 
        Propagate Forward using FFNN
        """
        ffnn = FFNN(self.ffnn_model)
        res = ffnn.compute()
        self.single_output = res
        self.output = np.array(ffnn.get_all_output_layer())

    def back_propagate(self):
        """ 
        epoch: banyaknya iterasi pada setiap 

        batch_size paramater ukuran minibatch utk Stochastic gradient descent

        step: pemrosesan satu minibatch

        """
        total_batch = self.split_input_targets_to_batch(
            self.input, self.target, self.batch_size)

        epoch = 0

        self.forward_propagation()
        current_epoch_error = self.__loss(
            self.target, self.single_output, self.layers[-1]["activation_function"])
        print("="*6 + " Backpropagation start " + "="*6)
        # Selama belum gg, backpropagate , update bobot trs feed lagi kedepan
        while (epoch < self.max_iteration and current_epoch_error >= self.error_threshold):
            total_layer = len(self.layers)
            prev_error = None
            print("=" * 8 + f" EPOCH {epoch + 1} " + "=" * 8)
            print(f"ERROR: {current_epoch_error}")
            print(f"Output: {self.single_output}")
            delta_weights_total = []
            # back propagate
            for index_layer in range(total_layer - 1, -1, -1):
                current_activation = self.layers[index_layer]["activation_function"]
                output_layer = self.output[index_layer]
                a = Activation(current_activation)
                # delta_weights_batch = []
                delta_weights = np.zeros(self.weights[index_layer].shape)
                for index, batch_instance in enumerate(total_batch):
                    inputs = np.array(batch_instance["inputs"])
                    targets = np.array(batch_instance["targets"])
                    gradient = None
                    delta_weights = np.zeros(self.weights[index_layer].shape)
                    # Kalau dia punya output (layer paling ujung)
                    for instance_index, instance in enumerate(inputs):
                        if index_layer == total_layer - 1:
                            # Calculate dho error / dho weight
                            # First part: -(t - o) -> (o - t)
                            first_part = np.subtract(
                                output_layer[instance_index], targets[instance_index])

                            # Second part: derivative
                            second_part = a.derivative(
                                output_layer[instance_index], targets[instance_index])

                            # print("FIRST: ", first_part)
                            # print("SECOND: ", second_part)
                            # print(self.output)
                            # Third part: input di layer sebelumnya
                            third_part = None

                            # Kalau cuma 1 layer, pake input layer
                            # add bias
                            if len(self.output) == 1:
                                third_part = np.ones(
                                    instance.shape[0] + 1)
                                third_part[1:] = instance
                            else:
                                shape = self.output[index_layer -
                                                    1][instance_index].shape
                                third_part = np.ones(shape[0] + 1)
                                third_part[1:] = self.output[index_layer -
                                                             1][instance_index]
                            # Third part must same as len each inital weights transposed
                            cur_weights = np.transpose(
                                self.weights[index_layer])
                            third_parts = np.zeros(cur_weights.shape)

                            for i in range(len(cur_weights)):
                                third_parts[i] = third_part

                            third_parts = np.transpose(third_parts)

                            if current_activation == Activation.SOFTMAX:
                                prev_error = second_part
                            else:
                                prev_error = np.multiply(
                                    first_part, second_part)
                            # w = w + nabla * delta * x
                            #
                            gradient = np.multiply(prev_error, third_parts)

                        # Kalau hidden layer
                        else:
                            # Calculate dho error / dho weight
                            # First part: dho error / dho net

                            # dho Ed / dho net_o; prev error
                            # print(prev_error)
                            # dho net_o / dho h
                            # remove bias
                            next_layer_cur_weights = self.weights[index_layer + 1][1:]
                            # print(next_layer_cur_weights)

                            # dho Ed / dho h
                            temp = np.multiply(
                                prev_error, next_layer_cur_weights)

                            # calculate sum of the error outputs layer
                            output_layer_sum_error = np.array(
                                [np.sum(x) for x in temp])
                            derivative = a.derivative(
                                output_layer[instance_index], targets[instance_index])

                            # calculate prev error
                            prev_error = np.multiply(
                                output_layer_sum_error, derivative)

                            # Second part: dho net / dho weight
                            second_part = None

                            # Kalau udah paling ujung
                            if index_layer == 0:
                                second_part = np.ones(instance.shape[0] + 1)
                                second_part[1:] = instance
                            else:
                                shape = self.output[index_layer][instance_index].shape
                                second_part = np.ones(shape[0] + 1)
                                second_part[1:] = self.output[index_layer][instance_index]

                            cur_weights = np.transpose(self.weights[i])
                            second_parts = np.zeros(cur_weights.shape)
                            for i in range(len(cur_weights)):
                                second_parts[i] = second_part

                            second_parts = np.transpose(second_parts)
                            gradient = np.multiply(prev_error, second_parts)
                            # print(gradient)

                        # Update delta weights
                        delta_weights = delta_weights - \
                            np.dot(self.learning_rate, gradient)

                    # delta_weights_batch.append(delta_weights)

                delta_weights_total.append(delta_weights)
            # Update weight
            print("DELTA WEGIHTS CUY", np.array(delta_weights_total.reverse()))
            self.weights = self.weights + \
                np.array(delta_weights_total)
            print("WEIGHTS GAMING", self.weights)
            print(
                f"Layer {index_layer + 1} Batch {index + 1} completed")

            new_weights = [np.transpose(x) for x in self.weights]
            self.ffnn_model["weights"] = new_weights
            self.forward_propagation()
            current_epoch_error = self.__loss(
                self.target,  self.single_output, self.layers[-1]["activation_function"])

            epoch += 1

        print("Backpropagation complete")
        if current_epoch_error < self.error_threshold:
            print(
                f"Reason: error ({current_epoch_error}) < threshold ({self.error_threshold})")
        else:
            print("Reason: Max iteration reached")

        print("Final output")
        print(self.single_output)
        print("Final weights")
        print(self.weights)
        print("Excpected")
        print(np.array(self.expected["final_weights"]))

    def split_input_targets_to_batch(self, inputs: List, targets: List, batch_size: int) -> List[List]:
        """ 
        Split input and targets into batch
        """

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
