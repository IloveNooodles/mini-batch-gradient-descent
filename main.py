from src.activation import Activation
from src.backprop import Backpropagation
from src.ffnn import FFNN
from src.graph import Graph
from src.reader import Reader


def main():
    print("=============================================")
    print("                Backpropagation              ")
    print("=============================================")
    filename = input("Input filename inside test folder: ")

    # path = "./test/"
    # transformed_model = Reader.read_ffnn("softmax.json")
    raw_model, ffnn_model, expected = Reader.read_backprop(
        "mlp.json")  # Ganti ke file

    b = Backpropagation(raw_model, expected, ffnn_model)
    b.back_propagate()

    ffnn = FFNN(model=ffnn_model)
    filename = filename.split(".")[0]
    graph = Graph(ffnn, filename)

    print("=============================================")
    print("Graph is saved in folder res with name: " +
          filename + "_graph.png")
    print("=============================================")
    graph.draw()


if __name__ == "__main__":
    main()
