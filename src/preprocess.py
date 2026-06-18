import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist


def load_data(dataset="mnist"):

    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    elif dataset == "fashion":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion'")

    # Binary classification (0 vs 1)
    train_filter = (y_train == 0) | (y_train == 1)
    test_filter = (y_test == 0) | (y_test == 1)

    x_train = x_train[train_filter]
    y_train = y_train[train_filter]

    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Keep dataset small for QNN
    x_train = x_train[:500]
    y_train = y_train[:500]

    x_test = x_test[:100]
    y_test = y_test[:100]

    return x_train, y_train, x_test, y_test