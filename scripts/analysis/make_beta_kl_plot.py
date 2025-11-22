import matplotlib.pyplot as plt
import numpy as np
from pyprojroot import here
from scipy.stats import beta

# Set the style for better visualization
plt.style.use("default")

# Parameters for the Beta distributions
alpha1, beta1 = 3, 4  # Beta(3,4)
alpha2, beta2 = 3, 3  # Beta(3,3)

# Generate x values
x = np.linspace(0, 1, 1000)

# Calculate PDFs
pdf1 = beta.pdf(x, alpha1, beta1)
pdf2 = beta.pdf(x, alpha2, beta2)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the PDFs and fill under them
ax.plot(x, pdf1, "b-", label=f"Beta({alpha1},{beta1})", linewidth=2, color="#66c2a5")
ax.plot(x, pdf2, "r-", label=f"Beta({alpha2},{beta2})", linewidth=2, color="#fc8d62")

# Fill under each curve with transparency
ax.fill_between(x, pdf1, color="#66c2a5", alpha=0.3)
ax.fill_between(x, pdf2, color="#fc8d62", alpha=0.3)

# Add labels and title
ax.set_xlabel("x", fontsize=42, fontfamily="Charter")
ax.set_ylabel("p(x)", fontsize=42, fontfamily="Charter")

# Add grid
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# Turn off ticks and tick labels
ax.set_xticks([])
ax.set_yticks([])


plt.tight_layout()
plt.savefig(
    here("figures/beta_distributions_overlap_plot.svg"),
    bbox_inches="tight",
    transparent=True,
)
plt.show()
