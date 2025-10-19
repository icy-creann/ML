import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":    # Example usage
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_train = np.array([300.0, 500.0, 700.0, 900.0, 1100.0])

    m = len(x_train)
    print(f"Number of training examples: {m}")

    i = 0
    res = np.zeros(m)

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        print(f"Training example {i}: x = {x_i}, y = {y_i}")

        w = 200.0
        b = 90.0
        res[i] = w * x_i + b
        print(f"Prediction for x = {x_i}: {res[i]}")

    plt.scatter(x_train, y_train, marker='x',c='r')
    plt.plot(x_train, res, marker='o', c='b')
    plt.title("Linear Regression")
    plt.ylabel('Price')
    plt.xlabel('Size')
    plt.show()
