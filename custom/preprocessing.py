from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def import_mnist():
    return fetch_openml('mnist_784')

def draw_digit(digit):
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap = matplotlib.cm.binary, # pylint: disable=maybe-no-member
    interpolation="nearest")
    plt.axis("off")
    plt.show()

def get_labels(data_frame):
    return data_frame["label"]

def get_features(data_frame):
    return data_frame.drop(["label"], axis=1)


def main():
    mnist = import_mnist()
    X, y = mnist["data"], mnist["target"]
    print(X.shape)
    print(y.shape)
    draw_digit(X[36000])

if __name__ == "__main__":
    main()
