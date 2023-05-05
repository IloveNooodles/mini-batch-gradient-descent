import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from src.activation import Activation
from src.backprop import Backpropagation
from src.ffnn import FFNN
from src.graph import Graph
from src.reader import Reader

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
    model["max_sse"] = 1e-8

    return model

def current_model_predict():

    iris = load_iris()
    x, y = iris.data, iris.target

    encoder = OneHotEncoder(sparse_output=False)
    y_reshaped = y.reshape(len(y), 1)
    y = encoder.fit_transform(y_reshaped)    
    y = np.array(y) 

    input_size = x.shape[1]
    neuron = 3

    initial_weights = [[0.5 for _ in range(neuron)] for _ in range(input_size + 1)]

    model = {
        "case":{
           "model": {
                "input_size": input_size,
                "layers": [
                    {
                        "number_of_neurons": neuron,
                        "activation_function": "sigmoid",
                    },
                ]
           },
           "input": x,
           "initial_weights": [initial_weights],
           "target": y,
           "learning_parameters": {
                "learning_rate": 0.0001,
                "batch_size": 150,
                "max_iteration": 20000,
                "error_threshold": 0.0
            }
        },
        "expect":{
            "stopped_by": "max_iteration",
        }
    }

    raw_model = model["case"]
    ffnn_model = transform_to_ffnn_model(raw_model)
    expected = model["expect"]

    b = Backpropagation(raw_model, expected, ffnn_model)
    b.back_propagate()

if __name__ == "__main__":
    current_model_predict()

