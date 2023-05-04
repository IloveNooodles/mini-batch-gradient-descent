import json
import os
from typing import Dict

import numpy as np

from .activation import Activation

ACTIVATION_LIST = [Activation.LINEAR, Activation.RELU,
                   Activation.SIGMOID, Activation.SOFTMAX]

MAX_SSE = 1e-8
BASE_FFNN_PATH = "test/test_case_ffnn/"
BASE_BACKPROP_PATH = "test/test_case_backprop/"


class Reader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_ffnn(filename: str) -> Dict:
        """ 
        Read ffnn models from json
        """
        try:
            with open(BASE_FFNN_PATH + filename, "rb") as f:
                json_file = json.load(f)
                # Return models
                if validate_data(json_file):
                    return json_file
                return None
        except OSError as e:
            print("File not found")
            os._exit(-1)

    @staticmethod
    def read_backprop(filename: str) -> Dict:
        try:
            with open(BASE_BACKPROP_PATH + filename, "rb") as f:
                json_file = json.load(f)
                raw_model = json_file["case"]
                transformed_model = transform_to_ffnn_model(raw_model)
                expected = json_file["expect"]

                return raw_model, transformed_model, expected

        except OSError as e:
            print("File not found")
            os._exit(-1)


def transform_to_ffnn_model(input_model: dict):
    """ 
    Needed to trasnsform to the current FFNN model
    """
    model = {}
    model["layers"] = len(input_model["model"]["layers"]) + 1
    model["activation_functions"] = [x["activation_function"]
                                     for x in input_model["model"]["layers"]]
    model["neurons"] = [input_model["model"]["input_size"]] + [x["number_of_neurons"]
                                                               for x in input_model["model"]["layers"]]
    model["weights"] = [np.transpose(x)
                        for x in input_model["initial_weights"]]
    model["rows"] = len(input_model["input"])
    model["data"] = input_model["input"]
    model["target"] = input_model["target"]
    model["max_sse"] = MAX_SSE

    return model


def validate_data(json_data) -> bool:
    # Validate layers
    layers = json_data['layers']
    activation_functions = np.array(
        json_data['activation_functions'], dtype=np.string_)

    # Neurons are input hidden output
    neurons = np.array(json_data['neurons'], dtype=np.int32)
    weights = json_data['weights']
    rows = json_data['rows']
    data = np.array(json_data['data'], dtype=np.float64)
    data_names = np.array(json_data['data_names'], dtype=np.string_)
    target_names = np.array(json_data['target_names'], dtype=np.string_)
    target = np.array(json_data['target'], dtype=np.int32)
    max_sse = json_data['max_sse']

    if not isinstance(layers, int):
        raise Exception("Layers is not integer")

    # Validate activation function per layers
    if activation_functions.shape[0] != layers - 1:
        raise Exception("Length of activation functions is not the same")

    for function in activation_functions:
        if function.decode() not in ACTIVATION_LIST:
            raise Exception("Invalid activation functions")

    # # Validate neurons
    if neurons.shape[0] != layers:
        raise Exception("Neurons number don't match with layers")

    assert neurons.dtype == np.int32

    # Validate weights, weights must be layers - 1
    if len(weights) != layers - 1:
        raise Exception("Please input correct weights")

    for index, weight_per_neuron in enumerate(weights):
        for weight_neuron in weight_per_neuron:
            if len(weight_neuron) != neurons[index] + 1:
                raise Exception(
                    f"Invalid number of weights parameter in weight {index}")
            np.array(weight_neuron, dtype=np.float64)

    # Validate rows
    if not isinstance(rows, int):
        raise Exception("Rows is not integer")

    # data_names
    len_data_features = data_names.shape[0]
    assert np.issubdtype(data_names.dtype, np.string_) == True

    # data attr
    if data.shape[0] != rows:
        raise Exception("Number of data doesn't match with rows")

    if data.shape[1] != len_data_features:
        raise Exception("Number of data columns doesnt match")

    assert np.issubdtype(target_names.dtype, np.string_) == True
    assert target.shape[0] == rows

    if not isinstance(max_sse, float):
        raise Exception("Please input correct sse")

    return True


if __name__ == "__main__":
    data = Reader.read_ffnn("./test/test.json")
    # data2 = datasets.load_breast_cancer()
    print(data)
    # print(data2)
