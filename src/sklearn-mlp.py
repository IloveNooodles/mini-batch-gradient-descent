
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


def predict_mlp():
    """
    Sklearn multilayer perceptron
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
    When testing:
    1. Change activation
    2. Change batch_size
    """

    iris = load_iris()
    x, y = iris.data, iris.target

    # Encode dataset
    encoder = OneHotEncoder(sparse_output=False)
    y_reshaped = y.reshape(len(y), 1)
    y = encoder.fit_transform(y_reshaped)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=100)

    print("=" * 8 + " Dataset information " + "="*8)
    print("X Train: ", x_train)
    print("Y Train: ", y_train)
    print("X Test: ", x_test)
    print("Y Test: ", y_test)

    # mlp = MLPClassifier(
    #     # hidden_layer_sizes=(3, 3),
    #     activation="identity",
    #     solver="sgd",
    #     batch_size=200,
    #     learning_rate="invscaling",
    #     learning_rate_init=0.2,
    #     power_t=0.5,
    #     max_iter=200,
    #     shuffle=True,
    #     random_state=100,
    #     verbose=True,
    #     tol=1e-6,
    #     momentum=0.6,
    # )

    # mlp.fit(x_train, y_train)

    # print("=" * 8 + " Result fit " + "="*8)
    # # for key in dir(mlp):
    # #     print("Key: ", key, "\nValue: ", mlp.__getattribute__(key))

    # res = mlp.predict(x_test)
    # print(classification_report(y_test, res))
    # print(confusion_matrix(y_test, res))


if __name__ == "__main__":
    predict_mlp()
