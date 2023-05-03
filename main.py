from src.activation import Activation
from src.ffnn import FFNN
from src.graph import Graph
from src.reader import Reader


def main():
    print("=============================================")
    print("        FEED FORWARD NEURAL NETWORK")
    print("=============================================")
    filename = input("Input filename inside test folder: ")

    path = "./test/"
    model = Reader.read_ffnn(path + filename)

    ffnn = FFNN(model=model)
    ffnn.compute()
    ffnn.predict()

    filename = filename.split(".")[0]
    graph = Graph(ffnn, filename)

    print("=============================================")
    print("Graph is saved in folder res with name: " +
          filename + "_graph.png")
    print("=============================================")
    graph.draw()


if __name__ == "__main__":
    main()
