"""
Backpropagation on the sklearn diabetes dataset: y = ReLU(X @ w + b)

Same idea as backprop.py, but with 10 input features instead of one x, so
"m" becomes a weight vector w (one weight per feature) and the per-row dm
becomes a per-feature gradient.

  forward :  z = X @ w + b ;  guess = relu(z)
  backward:  dz = (guess - y)/N * (z > 0)
             dw = X.T @ dz        (one entry per feature, replaces "dm")
             db = dz.sum()
"""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

torch.manual_seed(0)

LR = 0.1
EPOCHS = 500
L2 = 0.0   # set > 0 for Ridge-style weight decay (adds L2 * w to dw)

# ---------------------------------------------------------------------------
# Data: 442 samples, 10 features. sklearn's features are already centered and
# scaled, but with tiny magnitudes (~0.05), which makes gradient descent
# slow. We re-standardize to unit variance using *training* statistics only.
# ---------------------------------------------------------------------------
data = load_diabetes()
X_all = torch.tensor(data.data, dtype=torch.float32)
y_all = torch.tensor(data.target, dtype=torch.float32)   # range ~25..346, always > 0

perm = torch.randperm(len(X_all))
n_train = int(0.8 * len(X_all))
tr, te = perm[:n_train], perm[n_train:]
mean, std = X_all[tr].mean(0), X_all[tr].std(0)
X_train, y_train = (X_all[tr] - mean) / std, y_all[tr]
X_test, y_test = (X_all[te] - mean) / std, y_all[te]


def forward(X, w, b):
    z = X @ w + b
    return z, torch.relu(z)


def backprop_step(w, b, X, y, lr, l2=0.0):
    """One manual forward + backward + gradient descent step. Returns new (w, b, loss)."""
    z, guess = forward(X, w, b)
    loss = 0.5 * ((guess - y) ** 2).mean() + 0.5 * l2 * (w ** 2).sum()

    dloss_dguess = (guess - y) / len(X)     # d(loss)/d(guess), shape (N,)
    dguess_dz = (z > 0).float()             # ReLU derivative
    dz = dloss_dguess * dguess_dz           # d(loss)/d(z)
    dw = X.T @ dz + l2 * w                  # dz/dw = X  -> shape (10,)
    db = dz.sum()                           # dz/db = 1

    return w - lr * dw, b - lr * db, loss.item(), dw, db


def mse(X, y, w, b):
    return ((forward(X, w, b)[1] - y) ** 2).mean().item()


# Init: small random weights; bias at mean(y) so z > 0 for (almost) every
# sample. Starting with z <= 0 everywhere would be a "dead ReLU" (no gradient).
w = torch.randn(10) * 0.01
b = y_train.mean().clone()

# ---------------------------------------------------------------------------
# Check manual gradients against autograd on the first step
# ---------------------------------------------------------------------------
w_ag = w.clone().requires_grad_(True)
b_ag = b.clone().requires_grad_(True)
loss_ag = 0.5 * ((forward(X_train, w_ag, b_ag)[1] - y_train) ** 2).mean() + 0.5 * L2 * (w_ag ** 2).sum()
loss_ag.backward()
_, _, _, dw0, db0 = backprop_step(w, b, X_train, y_train, LR, L2)
print(f"grad check  max|dw manual - autograd| = {(dw0 - w_ag.grad).abs().max():.2e}, "
      f"|db diff| = {(db0 - b_ag.grad).abs():.2e}\n")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
train_losses, test_losses = [], []
for epoch in range(EPOCHS):
    w, b, loss, _, _ = backprop_step(w, b, X_train, y_train, LR, L2)
    train_losses.append(mse(X_train, y_train, w, b))
    test_losses.append(mse(X_test, y_test, w, b))
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:4d}  train MSE {train_losses[-1]:8.1f}  test MSE {test_losses[-1]:8.1f}")

# Baselines for context: always predicting the training mean, and the
# closed-form least squares solution (no ReLU).
baseline = ((y_test - y_train.mean()) ** 2).mean().item()
A = torch.cat([X_train, torch.ones(n_train, 1)], 1)
sol = torch.linalg.lstsq(A, y_train.unsqueeze(1)).solution.squeeze()
ols_test = ((torch.cat([X_test, torch.ones(len(X_test), 1)], 1) @ sol - y_test) ** 2).mean().item()
print(f"\ntest MSE: ours {test_losses[-1]:.1f} | closed-form OLS {ols_test:.1f} | predict-the-mean {baseline:.1f}")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
fig, (ax_loss, ax_pred, ax_w) = plt.subplots(1, 3, figsize=(17, 5))

ax_loss.plot(train_losses, label="train")
ax_loss.plot(test_losses, label="test")
ax_loss.axhline(ols_test, color="gray", ls="--", label="OLS test (closed form)")
ax_loss.set(title="Loss vs epoch", xlabel="epoch", ylabel="MSE")
ax_loss.legend()

pred = forward(X_test, w, b)[1]
ax_pred.scatter(y_test, pred, alpha=0.7)
lims = [y_all.min().item(), y_all.max().item()]
ax_pred.plot(lims, lims, "k--", label="perfect")
ax_pred.set(title="Test: predicted vs actual", xlabel="actual progression", ylabel="predicted")
ax_pred.legend()

ax_w.barh(data.feature_names, w.numpy())
ax_w.set(title="Learned weight per feature", xlabel="weight")

fig.tight_layout()
fig.savefig("backprop_diabetes.png", dpi=120)
print("saved backprop_diabetes.png")
plt.show()
