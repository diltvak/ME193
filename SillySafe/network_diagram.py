"""Draw the structure of the network trained in v_fit.py (1 -> 2 ReLU -> 1)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

R = 0.42                                   # node radius
IN_C, HID_C, OUT_C = "#4C78A8", "#F58518", "#54A24B"

nodes = {
    "x":  ((0.0, 0.0), IN_C,  r"$x$"),
    "h1": ((3.0, 1.2), HID_C, r"$h_1$"),
    "h2": ((3.0, -1.2), HID_C, r"$h_2$"),
    "y":  ((6.0, 0.0), OUT_C, r"$\hat{y}$"),
}
edges = [
    ("x", "h1", r"$w1_1$", 0.25),
    ("x", "h2", r"$w1_2$", -0.25),
    ("h1", "y", r"$w2_1$", 0.25),
    ("h2", "y", r"$w2_2$", -0.25),
]

fig, ax = plt.subplots(figsize=(10, 5.5))

# Connections (weights)
for a, b, label, dy in edges:
    (xa, ya), (xb, yb) = nodes[a][0], nodes[b][0]
    ax.annotate("", xy=(xb - R, yb), xytext=(xa + R, ya),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.8,
                                shrinkA=0, shrinkB=0))
    ax.text((xa + xb) / 2, (ya + yb) / 2 + dy, label, fontsize=14,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Neurons
for (cx, cy), color, label in nodes.values():
    ax.add_patch(Circle((cx, cy), R, fc=color, ec="black", lw=1.5, zorder=3))
    ax.text(cx, cy, label, fontsize=18, color="white", ha="center",
            va="center", zorder=4)

# Biases, written next to the neuron they feed
ax.text(3.0, 1.2 + R + 0.18, r"$+\,b1_1$", fontsize=13, ha="center", va="bottom")
ax.text(3.0, -1.2 - R - 0.18, r"$+\,b1_2$", fontsize=13, ha="center", va="top")
ax.text(6.0, R + 0.18, r"$+\,b2$", fontsize=13, ha="center", va="bottom")

# Layer titles
for cx, title, sub in [(0.0, "Input layer", "1 neuron"),
                       (3.0, "Hidden layer", "2 neurons, ReLU"),
                       (6.0, "Output layer", "1 neuron, linear")]:
    ax.text(cx, 2.75, title, fontsize=14, fontweight="bold", ha="center")
    ax.text(cx, 2.42, sub, fontsize=11, color="0.3", ha="center")

# What each layer computes
ax.text(0.0, -2.55, "x  (scaled to zero mean,\nunit spread)", fontsize=11,
        ha="center", va="top")
ax.text(3.0, -2.55, r"$h_j = \mathrm{ReLU}(w1_j\,x + b1_j)$" "\n"
        r"$\mathrm{ReLU}(z) = \max(0,\,z)$", fontsize=12, ha="center", va="top")
ax.text(6.0, -2.55, r"$\hat{y} = w2_1 h_1 + w2_2 h_2 + b2$", fontsize=12,
        ha="center", va="top")

ax.set_xlim(-1.5, 7.6)
ax.set_ylim(-3.5, 3.2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Network used for the V-shaped fit  (7 trainable parameters)",
             fontsize=15, pad=6)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "network_diagram.png", dpi=150)
plt.show()
