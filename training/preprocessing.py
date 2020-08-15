from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def import_mnist():
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]
    y = y.astype('int32')
    X = X.astype(np.float64)
    
    return X, y


def draw_digit(digit):
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


def main():
    mnist = import_mnist()
    X, y = mnist["data"], mnist["target"]
    print(X.shape)
    print(y.shape)
    draw_digit(X[36000])


if __name__ == "__main__":
    main()
