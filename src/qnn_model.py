import numpy as np

from preprocess import load_data
from feature_extraction import extract_features

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from qiskit_algorithms.optimizers import COBYLA


def build_qnn():

    num_qubits = 4

    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=1
    )

    ansatz = RealAmplitudes(
        num_qubits,
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


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    train_features = extract_features(x_train)
    test_features = extract_features(x_test)

    print("Building QNN...")

    classifier = build_qnn()

    print("Training...")

    classifier.fit(
        train_features,
        y_train
    )

    print("Training Complete")

    predictions = classifier.predict(
        test_features
    )

    accuracy = np.mean(predictions == y_test)

    print("Accuracy:", accuracy)