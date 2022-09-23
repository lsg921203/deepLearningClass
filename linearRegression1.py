import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create random input and output data
    x = np.linspace(-3.0, 3.0, 2000)
    mu, sigma = 0, 0.5  # mean and standard deviation
    random_noise = np.random.normal(mu, sigma, 2000)
    y = 3.6 * x - 1.5 + random_noise

    # Randomly initialize weights
    m = np.random.randn()
    c = np.random.randn()

    learning_rate = 1e-6

    plt.plot(x, y, 'r+')
    plt.plot(x, m * x + c)
    plt.show()

    for t in range(2001):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        y_pred = m * x + c

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t % 100 == 0:
            plt.plot(x, y, 'r+')
            plt.plot(x, m * x + c)
            plt.show()
            print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_c = grad_y_pred.sum()
        grad_m = (grad_y_pred * x).sum()

        # Update weights
        m -= learning_rate * grad_m
        c -= learning_rate * grad_c

    print(f'Result pred: y = ({m}) x + ({c})')


if __name__ == '__main__':
    main()