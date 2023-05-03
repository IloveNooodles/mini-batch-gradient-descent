import numpy as np

from .activation import Activation
from .reader import Reader


class FFNN:
    def __init__(self, model) -> None:
        self.layers = model['layers']
        self.activation_functions = np.array(model['activation_functions'])
        self.neurons = np.array(model['neurons'])
        self.weights = model['weights']
        self.rows = model['rows']
        self.data_names = np.array(model['data_names'])
        self.data = np.array(model['data'])
        self.target = np.array(model['target'])
        self.target_names = np.array(model['target_names'])
        self.output = None
        self.max_sse = model["max_sse"]
        pass

    def __str__(self) -> str:
        return f"\
  Layers: {self.layers}\n\
  Activations: {self.activation_functions}\n\
  Neurons: {self.neurons}\n\
  Weights: {self.weights}\n\
  Rows: {self.rows}\n\
  Data: {self.data}\n\
  Data_names: {self.data_names}\n\
  target: {self.target}\n\
  target_names: {self.target_names}\n\
  max_sse: {self.max_sse}\n"

    # Will return output functions
    def compute(self):
        res = self.data
        for i in range(self.layers - 1):
            activation_function = Activation(self.activation_functions[i])
            transposed_weights = np.transpose(np.array(self.weights[i]))
            weights, bias = self.separate_bias(transposed_weights)
            res = activation_function.calculate(res, weights, bias)

        self.output = res
        return res

    def separate_bias(self, data):
        bias = data[0, :]
        weight = data[1:, :]
        return weight, bias

    def predict(self):
        A = Activation(self.activation_functions[-1])
        res = A.predict(self.output)
        sse = self._calculate_sse()
        print(f"\
  Data Names: {self.data_names}\n\
  Data: {self.data}\n\
  Target Names: {self.target_names}\n\
  Target: {self.target}\n\
  Predictions: {res}\n\
  SSE: {sse}\n\
  isValid: {[s < self.max_sse for s in sse]} ( < 1e-6)\n")

    def _calculate_sse(self):
        sse = 0
        for i in range(len(self.output)):
            sse += pow(self.output[i] - self.target[i], 2)
        sse = sse / 2
        return sse


if __name__ == "__main__":
    model = Reader.read_ffnn("./test/xor_linear_relu.json")
    a = FFNN(model=model)
    a.compute()
    a.predict()
