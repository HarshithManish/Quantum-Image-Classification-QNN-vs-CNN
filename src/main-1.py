from preprocess import load_data
from feature_extraction import extract_features

from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from qiskit_algorithms.optimizers import COBYLA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from evaluation import evaluate_model


# ------------------------
# QNN
# ------------------------

def build_qnn():

    feature_map = ZZFeatureMap(
        feature_dimension=4,
        reps=1
    )

    ansatz = RealAmplitudes(
        4,
        reps=1
    )

    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )

    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=20)
    )

    return classifier


# ------------------------
# CNN
# ------------------------

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


# ------------------------
# MAIN
# ------------------------

if __name__ == "__main__":

    print("\nLoading Data...")

    x_train, y_train, x_test, y_test = load_data()

    # =====================
    # QNN
    # =====================

    print("\nTraining QNN...")

    train_features = extract_features(x_train)
    test_features = extract_features(x_test)

    qnn = build_qnn()

    qnn.fit(
        train_features,
        y_train
    )

    qnn_pred = qnn.predict(
        test_features
    )

    print("\nQNN Results")

    evaluate_model(
        y_test,
        qnn_pred
    )

    qnn_acc = accuracy_score(
        y_test,
        qnn_pred
    )

    # =====================
    # CNN
    # =====================

    print("\nTraining CNN...")

    x_train_cnn = x_train[..., None]
    x_test_cnn = x_test[..., None]

    cnn = build_cnn()

    cnn.fit(
        x_train_cnn,
        y_train,
        epochs=5,
        verbose=0
    )

    cnn_pred = cnn.predict(
        x_test_cnn
    )

    cnn_pred = (cnn_pred > 0.5).astype(int)

    print("\nCNN Results")

    evaluate_model(
        y_test,
        cnn_pred
    )

    cnn_acc = accuracy_score(
        y_test,
        cnn_pred
    )

    print("\n" + "=" * 40)
    print("FINAL COMPARISON")
    print("=" * 40)

    print(f"QNN Accuracy : {qnn_acc:.4f}")
    print(f"CNN Accuracy : {cnn_acc:.4f}")