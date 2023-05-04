from src.activation import Activation
from src.backprop import Backpropagation
from src.ffnn import FFNN
from src.graph import Graph
from src.reader import Reader
from src.reader_backprop import ReaderBackprop


def main():
    print("=============================================")
    print("                Backpropagation              ")
    print("=============================================")
    filename = input("Input filename inside test folder: ")

    # path = "./test/"
    # model = Reader.read_ffnn(path + filename)
    model, expected = ReaderBackprop.read_backprop("linear.json")
    # print(model, expected)

    b = Backpropagation(model, expected)
    transformed_model = b.transform_to_ffnn_model()

    ffnn = FFNN(model=transformed_model)
    ffnn.compute()
    ffnn.predict()

    # filename = filename.split(".")[0]
    # graph = Graph(ffnn, filename)

    # print("=============================================")
    # print("Graph is saved in folder res with name: " +
    #       filename + "_graph.png")
    # print("=============================================")
    # graph.draw()


if __name__ == "__main__":
    main()
