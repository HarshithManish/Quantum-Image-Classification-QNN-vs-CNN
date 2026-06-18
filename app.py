import subprocess
import streamlit as st

st.set_page_config(
    page_title="Quantum Image Classification",
    page_icon="⚛️",
    layout="wide"
)

# --------------------
# LOGIN
# --------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error("Invalid Credentials")

# --------------------
# DASHBOARD
# --------------------

else:

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go To",
        [
            "Project Overview",
            "QNN",
            "CNN",
            "Datasets",
            "Run Experiment",
            "Results"
        ]
    )

    # --------------------
    # PROJECT
    # --------------------

    if page == "Project Overview":

        st.title("Quantum Image Classification")

        st.markdown("""
        ⚛️ Quantum Image Classification

Project Goal
-------------
Compare Quantum Neural Networks (QNN)
with Convolutional Neural Networks (CNN)
for image classification.

Objectives
------------
✓ Implement a QNN using Qiskit
✓ Implement a CNN using TensorFlow
✓ Evaluate on MNIST
✓ Evaluate on Fashion-MNIST
✓ Compare Accuracy, Precision, Recall and F1 Score

Technology Stack
----------------
Python
Qiskit
TensorFlow
Scikit-Learn
Streamlit
        """)

    # --------------------
    # QNN
    # --------------------

    elif page == "QNN":

        st.title("⚛️ Quantum Neural Network")

        st.markdown("""
        A Quantum Neural Network (QNN) uses quantum circuits
        to perform machine learning tasks.

        Project Configuration:

        - 4 Qubits
        - ZZFeatureMap
        - RealAmplitudes Ansatz
        - COBYLA Optimizer

        Workflow:

        Image → Features → Quantum Circuit → Classification
        """)

    # --------------------
    # CNN
    # --------------------

    elif page == "CNN":

        st.title("🧠 Convolutional Neural Network")

        st.markdown("""
        CNNs are deep learning models specialized for images.

        Architecture Used:

        Conv2D
        ↓
        MaxPooling
        ↓
        Flatten
        ↓
        Dense
        ↓
        Output

        CNN is used as the benchmark model.
        """)

    # --------------------
    # DATASETS
    # --------------------

    elif page == "Datasets":

        st.title("📊 Datasets")

        st.markdown("""
        ### MNIST

        Handwritten digit dataset.

        Classes Used:

        - Digit 0
        - Digit 1

        ---

        ### Fashion-MNIST

        Fashion image dataset.

        Classes Used:

        - T-Shirt
        - Trouser
        """)

    # --------------------
    # RUN
    # --------------------

    elif page == "Run Experiment":

        st.title("🚀 Run Experiment")

    st.info("Run the QNN vs CNN comparison.")

    if st.button("Run Model"):

        output_box = st.empty()

        process = subprocess.Popen(
            ["python", "src/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_text = ""

        for line in process.stdout:

            output_text += line

            output_box.code(
                output_text,
                language="text"
            )

        process.wait()

        st.success("Experiment Completed!")

    # --------------------
    # RESULTS
    # --------------------

    elif page == "Results":

        st.title("📈 Results")

    try:

        with open(
            "results/experiment_results.txt",
            "r"
        ) as f:

            results = f.read()

        st.code(
            results,
            language="text"
        )

    except FileNotFoundError:

        st.warning(
            "No experiment results found. Run the model first."
        )