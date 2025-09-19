import matplotlib.pyplot as plt
import numpy as np

# Set up the matplotlib parameters for publication quality
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["Times New Roman"],
        # "text.usetex": True,
        "font.size": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Plot 1: Box plot showing MSE_Validation
np.random.seed(42)  # For reproducibility

# Generate data for two distributions
dist1 = np.random.normal(loc=0.8, scale=0.2, size=5)  # Higher mean, higher variance
dist2 = np.random.normal(loc=0.75, scale=0.05, size=5)  # Lower mean, much lower variance

# Create figure for box plot
plt.figure(figsize=(3.5, 3))
boxplot = plt.boxplot([dist1, dist2], patch_artist=True, labels=["", ""], widths=0.6)

# Set colors
boxplot["boxes"][0].set_facecolor("lightgrey")
boxplot["boxes"][1].set_facecolor("grey")

# Set labels and remove ticks
plt.ylabel("MSE$_\\mathrm{Validation}$", fontsize=18)
plt.xticks([])
plt.yticks([])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)

# Save the box plot
plt.tight_layout()
plt.legend(
    [boxplot["boxes"][0], boxplot["boxes"][1]],
    ["Unoptimised", "Optimised"],
    loc="upper right",
    fontsize=12,
)
plt.savefig("validation_boxplot.png", dpi=300, bbox_inches="tight")
plt.savefig("validation_boxplot.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: Scatter plot for the two functions with noisy data
np.random.seed(42)  # Reset seed for consistency

# Generate x values
x = np.linspace(0.01, 1, 9)

# Create noisy data for x^2 and 1/x functions
noise_level = 0.2
y1 = 0.2 * x**0.1 + np.random.normal(0, 0.2 + x + noise_level * x, size=len(x))  # Noisy x^2
y2 = 0.1 * x + 2 * x**4 + np.random.normal(0, 0.25 + noise_level * 2 * x, size=len(x))  # Noisy 1/x
y3 = 0.3 * x + np.random.normal(0, 0.2 + noise_level * 0.5 * x, size=len(x))  # Noisy 1/x
# Create figure for scatter plot
plt.figure(figsize=(7, 5))
# apply dropout to the data
dropout_rate = 0.3
x1_dropout = np.random.rand(len(x)) < dropout_rate
x2_dropout = np.random.rand(len(x)) < dropout_rate
x3_dropout = np.random.rand(len(x)) < dropout_rate
x1 = x[~x1_dropout]
x2 = x[~x2_dropout]
x3 = x[~x3_dropout]
y1 = y1[~x1_dropout]
y2 = y2[~x2_dropout]
y3 = y3[~x3_dropout]


plt.scatter(x1, y1, alpha=0.7, s=200, c="limegreen", label="A")
plt.scatter(x2, y2, alpha=0.7, s=200, c="skyblue", label="B")
plt.scatter(x3, y3, alpha=0.7, s=200, c="lightcoral", label="C")

# Set labels and remove ticks
plt.ylabel("Shape Change ($\\Delta H_\\mathrm{optimisation}$)", fontsize=20)
plt.xlabel("Implied Coverage ($-T\\Delta S_\\mathrm{optimisation}$)", fontsize=20)
plt.xticks([])
plt.yticks([])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(True)
plt.gca().spines["left"].set_visible(True)
plt.gca().spines["bottom"].set_linewidth(0.5)
plt.gca().spines["left"].set_linewidth(0.5)
# Set legend
plt.legend(
    loc="upper right",
    fontsize=12,
    markerscale=1.5,
    handletextpad=0.5,
    borderpad=0.5,
    labelspacing=0.5,
)

# Save the scatter plot
plt.tight_layout()
plt.savefig("correlation_scatter.png", dpi=300, bbox_inches="tight")
plt.savefig("correlation_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Plot 3: Box plots for Shape Change and Implied Coverage
# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
labels = ["A", "B", "C"]
colors = ["limegreen", "skyblue", "lightcoral"]

# Box plot for Shape Change
bp1 = axes[0].boxplot([y1, y2, y3], patch_artist=True, labels=labels, widths=0.6)
for patch, color in zip(bp1["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0].set_ylabel("$\\Delta H_\\mathrm{optimisation}$", fontsize=18)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].spines["bottom"].set_visible(False)
axes[0].spines["left"].set_visible(False)
axes[0].set_title("Shape Change", fontsize=20)
# Add legend for Shape Change plot
axes[0].legend(bp1["boxes"], labels, loc="upper right", fontsize=10)

# Box plot for Implied Coverage
bp2 = axes[1].boxplot([x1, x2, x3], patch_artist=True, labels=labels, widths=0.6)
for patch, color in zip(bp2["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1].set_ylabel("$-T\\Delta S_\\mathrm{optimisation}$", fontsize=18)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].spines["bottom"].set_visible(False)
axes[1].spines["left"].set_visible(False)
axes[1].set_title("Implied Coverage", fontsize=20)
# Add legend for Implied Coverage plot
# axes[1].legend(bp2["boxes"], labels, loc="upper right", fontsize=10)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap
plt.savefig("shape_coverage_boxplots.png", dpi=300, bbox_inches="tight")
plt.savefig("shape_coverage_boxplots.pdf", dpi=300, bbox_inches="tight")
plt.close()
