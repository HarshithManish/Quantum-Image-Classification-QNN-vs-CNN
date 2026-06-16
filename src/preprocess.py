import numpy as np
from tensorflow.keras.datasets import mnist


def load_data():

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Use only digits 0 and 1
    train_filter = (y_train == 0) | (y_train == 1)
    test_filter = (y_test == 0) | (y_test == 1)

    x_train = x_train[train_filter]
    y_train = y_train[train_filter]

    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    # Normalize images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reduce dataset size
    x_train = x_train[:500]
    y_train = y_train[:500]

    x_test = x_test[:100]
    y_test = y_test[:100]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Training Images:", x_train.shape)
    print("Training Labels:", y_train.shape)

    print("Testing Images:", x_test.shape)
    print("Testing Labels:", y_test.shape)