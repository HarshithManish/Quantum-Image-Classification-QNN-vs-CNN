from preprocess import load_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


def build_cnn():

    model = Sequential([
        Conv2D(
            16,
            (3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),

        MaxPooling2D(),

        Flatten(),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = build_cnn()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )

    loss, accuracy = model.evaluate(
        x_test,
        y_test
    )

    print("\nCNN Accuracy:", accuracy)