import json
import os

import numpy as np

from .activation import Activation

ACTIVATION_LIST = [Activation.LINEAR, Activation.RELU,
                   Activation.SIGMOID, Activation.SOFTMAX]


class Reader:
    def __init__(self) -> None:
        pass

    """ 
    FFNN models are json like
    """
    @staticmethod
    def read_ffnn(filepath):
        try:
            with open(filepath, "rb") as f:
                json_file = json.loads(f.read())
                # Return models
                if validate_data(json_file):
                    return json_file
                return None
        except OSError as e:
            print("File not found")
            os._exit(-1)


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
