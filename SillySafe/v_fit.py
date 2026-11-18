"""
Fit V-shaped data with gradient descent on mean-squared error.

Model: 1 input -> hidden layer (ReLU) -> 1 linear output

    h_j   = relu(w1_j * x + b1_j)          hidden neurons
    y_hat = sum_j w2_j * h_j + b2          output

A single line y = m*x + b cannot make a V, but two ReLU neurons can: one
turns on to the right of the vertex and the other to the left.

The gradients are the chain rule from the class notes,

    dLoss/dm = dLoss/dy * dy/dm = (2/N) * sum (y_i - m*x_i - b) * (-x_i)
    dLoss/db = dLoss/dy * dy/db = (2/N) * sum (y_i - m*x_i - b) * (-1)

applied once to the output layer (where the "x" is the hidden activation h)
and once more through the ReLU to the hidden layer.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / "annacode-v.tsv"

HIDDEN = 2            # number of hidden neurons (>= 2 needed for a V)
LEARNING_RATE = 0.01
EPOCHS = 5000
SEED = 0

# ---------------------------------------------------------------- data
data = np.loadtxt(DATA_FILE, delimiter="\t", skiprows=1)
x_raw = data[:, 1]
y = data[:, 2]
N = len(x_raw)

# Scale x to zero mean / unit spread so one learning rate works for all weights
x_mean, x_std = x_raw.mean(), x_raw.std()
x = (x_raw - x_mean) / x_std

# ---------------------------------------------------------------- parameters
rng = np.random.default_rng(SEED)
w1 = rng.normal(0, 1, HIDDEN)
w1[0], w1[1] = abs(w1[0]), -abs(w1[1])   # one neuron per arm of the V
b1 = np.zeros(HIDDEN)
w2 = rng.normal(0, 1, HIDDEN)
b2 = 0.0


def forward(x):
    """Return (prediction, hidden activations, hidden pre-activations)."""
    z = np.outer(x, w1) + b1          # (N, HIDDEN)
    h = np.maximum(z, 0)              # ReLU
    return h @ w2 + b2, h, z


# ---------------------------------------------------------------- training
losses = []
for epoch in range(EPOCHS):
    y_hat, h, z = forward(x)
    error = y - y_hat
    losses.append(np.mean(error ** 2))

    # dLoss/dy_hat for each sample: (2/N) * (y_i - y_hat_i) * (-1)
    dL_dy = (2 / N) * error * (-1)

    # Output layer: y_hat = w2 . h + b2  (h plays the role of x, w2 of m, b2 of b)
    dL_dw2 = h.T @ dL_dy
    dL_db2 = dL_dy.sum()

    # Hidden layer: chain through w2 and the ReLU slope (1 if z > 0 else 0)
    dL_dz = np.outer(dL_dy, w2) * (z > 0)
    dL_dw1 = x @ dL_dz
    dL_db1 = dL_dz.sum(axis=0)

    w2 -= LEARNING_RATE * dL_dw2
    b2 -= LEARNING_RATE * dL_db2
    w1 -= LEARNING_RATE * dL_dw1
    b1 -= LEARNING_RATE * dL_db1

    if epoch % 500 == 0:
        print(f"epoch {epoch:5d}   loss {losses[-1]:.4f}")

final_loss = np.mean((y - forward(x)[0]) ** 2)
print(f"final loss (MSE): {final_loss:.4f}")

# Each neuron's kink, converted back to the original x units
kinks = -b1 / w1 * x_std + x_mean
print("hidden-neuron kinks at x =", np.round(kinks, 2))

# ---------------------------------------------------------------- plots
x_line_raw = np.linspace(x_raw.min(), x_raw.max(), 400)
y_line = forward((x_line_raw - x_mean) / x_std)[0]

fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_fit.scatter(x_raw, y, s=18, label="data")
ax_fit.plot(x_line_raw, y_line, color="tab:red", linewidth=2,
            label=f"fit ({HIDDEN} ReLU neurons)")
ax_fit.set_xlabel("x")
ax_fit.set_ylabel("output 1")
ax_fit.set_title("Data and fit")
ax_fit.legend()

ax_loss.plot(losses)
ax_loss.set_yscale("log")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss (MSE)")
ax_loss.set_title("Loss vs. epoch")

fig.tight_layout()
fig.savefig(Path(__file__).parent / "v_fit.png", dpi=150)
plt.show()
