import torch
import matplotlib.pyplot as plt


def main():

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # Create random input and output data
    # x = torch.linspace(-3.0, 3.0, 2000, device=device, dtype=dtype)
    # random_noise = torch.normal(mean=torch.zeros(2000), std=0.5)
    #
    # y = 3.6 * x - 1.5 + random_noise

    # Randomly initialize weights
    m = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(400001):
        x = torch.normal(mean=torch.zeros(20), std=0.5)
        random_noise = torch.normal(mean=torch.zeros(20), std=0.1)

        y = 3.6 * x - 1.5 + random_noise
        # Forward pass: compute predicted y
        y_pred = m * x + c
        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 10000 == 0:
            plt.plot(x, y, 'r+')
            plt.plot(x, m * x + c)
            plt.connect('key_press_event', keyEvent)
            plt.show()
            print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_c = grad_y_pred.sum()
        grad_m = (grad_y_pred * x).sum()

        # Update weights using gradient descent
        m -= learning_rate * grad_m
        c -= learning_rate * grad_c

    print(f'Result pred: y = ({m}) x + ({c})')


def keyEvent():
    plt.close()


if __name__ == '__main__':
    main()