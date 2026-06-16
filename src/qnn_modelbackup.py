from qiskit import QuantumCircuit

from preprocess import load_data
from feature_extraction import extract_features


def create_quantum_circuit(features):

    qc = QuantumCircuit(4)

    for i in range(4):
        qc.ry(float(features[i]), i)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    return qc


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    train_features = extract_features(x_train)

    sample = train_features[0]

    print("Feature Vector:")
    print(sample)

    circuit = create_quantum_circuit(sample)

    print("\nQuantum Circuit:\n")
    print(circuit)