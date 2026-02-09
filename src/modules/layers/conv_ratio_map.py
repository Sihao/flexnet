#!/Users/donyin/miniconda3/envs/imperial/bin/python

import numpy as np

from pathlib import Path
from scipy import optimize
from natsort import natsorted
import matplotlib.pyplot as plt


"""
- This is my attempt to perform symbolic regression on the Conv Ratio data
- i.e., to see what is the simplest polynomial to represent the conv ratio pattern.
"""


def plot_polynomial_fit(x, y, order, verbose=False):
    def poly_func(x, *params):
        return sum(param * x**i for i, param in enumerate(params[::-1]))

    popt, _ = optimize.curve_fit(poly_func, x, y, p0=[1] * (order + 1))

    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = poly_func(x_fit, *popt)

    mse = np.mean((y - poly_func(x, *popt)) ** 2)

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label="Original Data")
        plt.plot(x_fit, y_fit, "r-", label=f"Fitted {order}th Order Polynomial")
        plt.xlabel("Conv Ratio Index")
        plt.ylabel("Conv Ratio Value")
        plt.title(
            f"Symbolic Regression: {order}th Order Polynomial Fit to Conv Ratio Data\nMSE: {mse:.4e}"
        )
        plt.legend()
        plt.grid(True)

        plt.savefig(
            f"conv_ratio_regression_polynomial_{order}th_order_mse_{mse:.4e}.png"
        )
        plt.close()

    if verbose:
        func_str = " + ".join([f"{popt[i]:.4e}x^{order-i}" for i in range(order + 1)])
        print(f"Fitted function: f(x) = {func_str}")
        print(f"Mean Squared Error: {mse:.4e}")
        print(f"Plot saved as 'conv_ratio_{order}th_order_mse_{mse:.4e}.png'")

    def fitted_function(x):
        return poly_func(x, *popt)

    return fitted_function


# This is from experiment 7 / threshold + hard sigmoid on CIFAR-10 with 120 epochs
data_dict = {
    "Conv Ratio 0": 0.6970062255859375,
    "Conv Ratio 1": 0.6411237716674805,
    "Conv Ratio 2": 0.6306915283203125,
    "Conv Ratio 3": 0.7519168853759766,
    "Conv Ratio 4": 0.7327690124511719,
    "Conv Ratio 5": 0.6576423645019531,
    "Conv Ratio 6": 0.6048927307128906,
    "Conv Ratio 7": 0.4195098876953125,
    "Conv Ratio 8": 0.3528213500976562,
    "Conv Ratio 9": 0.2884140014648437,
    "Conv Ratio 10": 0.210357666015625,
    "Conv Ratio 11": 0.323089599609375,
    "Conv Ratio 12": 0.360198974609375,
}

x = np.array([int(key.split()[-1]) for key in natsorted(data_dict.keys())])
y = np.array(list(data_dict.values()))

fitted_func_vgg16_get_conv_ratio = plot_polynomial_fit(x, y, 5)

if __name__ == "__main__":
    print(fitted_func_vgg16_get_conv_ratio(5))  # e.g., predict the value at x=5
