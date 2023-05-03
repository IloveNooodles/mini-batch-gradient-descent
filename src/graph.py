import numpy as np
from graphviz import Digraph

from .ffnn import FFNN


class Graph:
    def __init__(self, ffnn: FFNN, filename: str) -> None:
        self.ffnn = ffnn
        self.filename = filename
        self.f = Digraph(
            'G', filename=f'./res/{self.filename}_graph', format='png')

    def draw(self):
        self.f.attr('node', shape='circle')
        self._add_node()
        self._add_edge()
        self.f.view()
        return self.f

    def _add_node(self):
        # Add input nodes
        self.f.node("b1")
        for data_name in self.ffnn.data_names:
            self.f.node(data_name)

        # Add nodes in the next layers
        for layer in range(1, self.ffnn.layers - 1):
            self.f.node(f"b{layer+1}")
            for j in range(self.ffnn.neurons[layer]):
                self.f.node(f"h{layer}{j+1}")

    def _add_edge(self):
        # Add edges between layers
        layers = self.ffnn.layers

        for layer in range(layers - 1):
            transposed_weights = np.transpose(
                np.array(self.ffnn.weights[layer]))
            weights, bias = self.ffnn.separate_bias(transposed_weights)

            # Add edges between bias and hidden layer or output
            for i in range(len(bias)):
                if layers == 2:
                    bias_name = f"b{layer+1}"
                    end = f"y{i+1}"
                else:
                    bias_name = f"b{layer+1}"
                    end = f"h{layer+1}{i+1}"
                    if layer == self.ffnn.layers - 2:
                        end = f"y{i+1}"

                self.f.edge(bias_name, end, label=f"{bias[i]}")

            # Add edges between input and hidden layer or output
            for i in range(self.ffnn.neurons[layer]):
                for j in range(len(weights[i])):
                    if layers == 2:
                        start = self.ffnn.data_names[i]
                        end = f"y{j+1}"
                    elif layer == 0:
                        start = self.ffnn.data_names[i]
                        end = f"h{layer+1}{j+1}"
                    elif layer == self.ffnn.layers - 2:
                        start = f"h{layer}{i+1}"
                        end = f"y{j+1}"
                    else:
                        start = f"h{layer}{i+1}"
                        end = f"h{layer+1}{j+1}"
                    self.f.edge(start, end, label=f"{weights[i][j]}")


if __name__ == "__main__":
    ps = graphviz.Digraph('petshop', node_attr={
                          'shape': 'plaintext'}, format='png')
    ps.node('x1')
    ps.node('x2')
    ps.node('x3')
    ps.node('h1')
    ps.edge('x1', 'h1', label='20')
    ps.edge('x2', 'h1', label='10')
    ps.edge('x3', 'h1', label='30')
    print(ps.render(directory="/img", view=True))
