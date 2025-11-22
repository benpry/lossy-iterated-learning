import matplotlib.pyplot as plt
from mpltern.datasets import get_dirichlet_pdfs

if __name__ == "__main__":
    alphas = ((2, 1, 2), (3, 2, 2), (4, 2, 3), (6, 2, 3))
    for alpha in alphas:
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1, projection="ternary")

        t, ll, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)
        cmap = "Blues"
        shading = "gouraud"
        cs = ax.tripcolor(t, ll, r, v, cmap=cmap, shading=shading, rasterized=True)
        ax.tricontour(t, ll, r, v, colors="k", linewidths=0.5)

        ax.taxis.set_ticks([])
        ax.laxis.set_ticks([])
        ax.raxis.set_ticks([])

        fig.savefig(
            f"dirichlet-{'-'.join([str(x) for x in alpha])}.svg", bbox_inches="tight"
        )
        plt.show()
