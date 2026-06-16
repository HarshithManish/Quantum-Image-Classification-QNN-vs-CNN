import numpy as np
from preprocess import load_data


def extract_features(images):

    features = []

    for img in images:

        # Split image into 4 equal regions
        top_left = np.mean(img[:14, :14])
        top_right = np.mean(img[:14, 14:])
        bottom_left = np.mean(img[14:, :14])
        bottom_right = np.mean(img[14:, 14:])

        features.append([
            top_left,
            top_right,
            bottom_left,
            bottom_right
        ])

    return np.array(features)


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    train_features = extract_features(x_train)
    test_features = extract_features(x_test)

    print("Train Feature Shape:", train_features.shape)
    print("Test Feature Shape:", test_features.shape)

    print("\nFirst Feature Vector:")
    print(train_features[0])