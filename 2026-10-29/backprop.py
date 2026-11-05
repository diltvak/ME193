"""
Backpropagation for y = ReLU(m*x + b), mirroring the spreadsheet.

Spreadsheet column      ->  Here
  guess                 ->  forward pass:  z = m*x + b ; guess = relu(z)
  db, dm (per row)      ->  backward pass: gradient of loss w.r.t. b and m
  sum of rows * lr      ->  gradient descent step: m -= lr * dm ; b -= lr * db

Part 1 does the chain rule by hand. Part 2 lets PyTorch autograd do it.
Part 3 trains with torch.nn and an optimizer. All three agree.
"""
import torch

# Data from the spreadsheet
X = torch.tensor([6, 6, 7, 7, 7, 8, 8, 9, 10, 10] + [11] * 14, dtype=torch.float32)
Y = torch.tensor([32.5, 29.5, 25, 23, 21.5, 21, 20, 19, 17, 15.5, 15.5, 12,
                  12, 9.5, 7.5, 6, 4.5, 3.5, 3, 2, 2, 1, 1, 1])

LR = 0.005
EPOCHS = 3000
# NOTE: the spreadsheet starts at m=b=0, but with a ReLU that gives z=0 and
# relu'(0)=0, so no gradient flows ("dead ReLU") and nothing would ever learn.
# Start with z > 0 instead.
M0, B0 = 0.1, 1.0


# ---------------------------------------------------------------------------
# Part 1: manual backprop (the chain rule, exactly what the spreadsheet does)
# ---------------------------------------------------------------------------
def manual_step(m, b, x, y, lr):
    # forward
    z = m * x + b                # linear layer
    guess = torch.relu(z)        # ReLU layer
    loss = 0.5 * ((guess - y) ** 2).mean()

    # backward: chain rule, one factor per layer, applied right to left
    dloss_dguess = (guess - y) / len(x)       # d(loss)/d(guess)
    dguess_dz = (z > 0).float()               # ReLU derivative: 1 if z>0 else 0
    dz = dloss_dguess * dguess_dz             # d(loss)/d(z)
    db = dz.sum()                             # dz/db = 1   (spreadsheet "db")
    dm = (dz * x).sum()                       # dz/dm = x   (spreadsheet "dm")

    # gradient descent update
    return m - lr * dm, b - lr * db, loss.item()


m, b = torch.tensor(M0), torch.tensor(B0)
history = []   # (epoch, m, b, loss) every epoch, used for the plots at the bottom
for epoch in range(EPOCHS):
    history.append((epoch, m.item(), b.item(), None))
    m, b, loss = manual_step(m, b, X, Y, LR)
    history[-1] = (epoch, history[-1][1], history[-1][2], loss)
    if epoch % 500 == 0:
        print(f"[manual]   epoch {epoch:4d}  loss {loss:9.4f}  m {m:.4f}  b {b:.4f}")
print(f"[manual]   final: m={m:.4f} b={b:.4f}\n")


# ---------------------------------------------------------------------------
# Part 2: autograd. We only write the forward pass; .backward() runs the chain
# rule for us and fills in m.grad and b.grad.
# ---------------------------------------------------------------------------
m2 = torch.tensor(M0, requires_grad=True)
b2 = torch.tensor(B0, requires_grad=True)
for epoch in range(EPOCHS):
    guess = torch.relu(m2 * X + b2)
    loss = 0.5 * ((guess - Y) ** 2).mean()

    loss.backward()                  # compute m2.grad, b2.grad
    if epoch == 0:
        # Check against the manual gradients for the very first step
        z = M0 * X + B0
        dz = (torch.relu(z) - Y) / len(X) * (z > 0).float()
        print(f"[autograd] epoch 0 grads: dm={m2.grad.item():.4f} (manual {(dz * X).sum():.4f}), "
              f"db={b2.grad.item():.4f} (manual {dz.sum():.4f})")

    with torch.no_grad():            # update without tracking the update itself
        m2 -= LR * m2.grad
        b2 -= LR * b2.grad
    m2.grad.zero_()                  # grads accumulate by default, so reset
    b2.grad.zero_()
    if epoch % 500 == 0:
        print(f"[autograd] epoch {epoch:4d}  loss {loss.item():9.4f}  m {m2.item():.4f}  b {b2.item():.4f}")
print(f"[autograd] final: m={m2.item():.4f} b={b2.item():.4f}\n")


# ---------------------------------------------------------------------------
# Part 3: the idiomatic PyTorch version (nn.Linear + nn.ReLU + optimizer)
# ---------------------------------------------------------------------------
model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
with torch.no_grad():
    model[0].weight.fill_(M0)
    model[0].bias.fill_(B0)
opt = torch.optim.SGD(model.parameters(), lr=LR)
x_col, y_col = X.unsqueeze(1), Y.unsqueeze(1)   # Linear expects shape (N, 1)

for epoch in range(EPOCHS):
    opt.zero_grad()
    loss = 0.5 * ((model(x_col) - y_col) ** 2).mean()
    loss.backward()
    opt.step()
print(f"[nn]       final: m={model[0].weight.item():.4f} b={model[0].bias.item():.4f}")


# ---------------------------------------------------------------------------
# Visualization of the manual run from Part 1
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

epochs, ms, bs, losses = (list(c) for c in zip(*history))
snapshots = [0, 100, 500, 1500, EPOCHS - 1]

fig, (ax_fit, ax_loss, ax_surf) = plt.subplots(1, 3, figsize=(17, 5))

# 1) The data and the fitted curve ReLU(m*x + b) at several points in training
xs = torch.linspace(0, 12, 200)
ax_fit.scatter(X, Y, color="black", zorder=3, label="data")
colors = plt.cm.viridis(torch.linspace(0, 1, len(snapshots)).numpy())
for e, c in zip(snapshots, colors):
    ax_fit.plot(xs, torch.relu(ms[e] * xs + bs[e]), color=c,
                label=f"epoch {e}: m={ms[e]:.2f}, b={bs[e]:.2f}")
ax_fit.set(title="Fit improving: y = ReLU(m·x + b)", xlabel="X", ylabel="Y")
ax_fit.legend(fontsize=8)

# 2) Loss per epoch
ax_loss.plot(epochs, losses)
ax_loss.set(title="Loss vs epoch", xlabel="epoch", ylabel="0.5 · mean squared error")

# 3) Loss surface over (m, b) with the path gradient descent took
mg, bg = torch.meshgrid(torch.linspace(-4, 2, 150), torch.linspace(-5, 45, 150), indexing="ij")
pred = torch.relu(mg[..., None] * X + bg[..., None])
surface = 0.5 * ((pred - Y) ** 2).mean(-1)
cs = ax_surf.contourf(mg, bg, surface, levels=30, cmap="magma_r")
fig.colorbar(cs, ax=ax_surf, label="loss")
ax_surf.plot(ms, bs, color="cyan", lw=1.5, label="descent path")
ax_surf.scatter(ms[0], bs[0], color="white", edgecolor="k", zorder=3, label="start")
ax_surf.scatter(ms[-1], bs[-1], color="red", edgecolor="k", zorder=3, label="end")
ax_surf.set(title="Gradient descent on the loss surface", xlabel="m", ylabel="b")
ax_surf.legend(fontsize=8)

fig.tight_layout()
fig.savefig("backprop.png", dpi=120)
print("saved backprop.png")
plt.show()
