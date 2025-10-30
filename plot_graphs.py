import numpy as np
import matplotlib.pyplot as plt

### Plot the explained variance ratio vs number of PCA components


def plot_explained_variance(pca_model):
    """
    Plot the explained variance ratio vs number of PCA components.
    Args:
        pca_model: dict returned by pca_fit function
    """
    explained_variance_ratio = pca_model["explained_variance_ratio"]
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(explained_variance_ratio) + 1), cumulative_variance, marker="o"
    )
    plt.xlabel("Number of PCA Components (k)")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs Number of PCA Components")
    plt.grid()
    plt.axhline(y=0.9, color="r", linestyle="--", label="90% Variance Threshold")
    plt.legend()
    plt.show()


### Plot mean validation loss over (gamma, lambda) grid


def plot_cv_surface(cv_loss, best_gamma, best_lambda, fontsize=13):
    """
    Visualize mean validation loss over (gamma, lambda) grid.
    Args:
        cv_loss: dict {(gamma, lambda): mean_loss}
        best_gamma: float
        best_lambda: float
        fontsize: base font size for labels, legend, etc.
    """

    # Collect unique grid values in sorted order
    gammas = sorted({g for (g, l) in cv_loss.keys()})
    lambdas = sorted({l for (g, l) in cv_loss.keys()})

    # Sanity: log10 requires positive values
    if any(g <= 0 for g in gammas) or any(l <= 0 for l in lambdas):
        raise ValueError("All gamma and lambda values must be > 0 for log10 plotting.")

    # Build loss matrix in the same ordering
    M, N = len(gammas), len(lambdas)
    loss_matrix = np.empty((M, N))
    for i, g in enumerate(gammas):
        for j, l in enumerate(lambdas):
            loss_matrix[i, j] = cv_loss[(g, l)]

    # Log coordinates
    gx = np.log10(np.array(gammas))
    lx = np.log10(np.array(lambdas))

    # Avoid singular extents if one side has length 1
    eps = 1e-9
    x0, x1 = (lx[0] - eps, lx[0] + eps) if N == 1 else (lx.min(), lx.max())
    y0, y1 = (gx[0] - eps, gx[0] + eps) if M == 1 else (gx.min(), gx.max())

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        loss_matrix,
        origin="lower",
        aspect="auto",
        extent=[x0, x1, y0, y1],
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Mean validation loss", fontsize=fontsize)

    # Best point
    plt.scatter(
        np.log10(best_lambda),
        np.log10(best_gamma),
        color="red",
        marker="x",
        s=120,
        label="Best (γ, λ)",
    )

    # Labels / title
    plt.xlabel("log10(lambda)", fontsize=fontsize)
    plt.ylabel("log10(gamma)", fontsize=fontsize)
    plt.title("Cross-Validation Loss Surface", fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize)

    # Ticks as scientific notation (works for 1-value cases too)
    plt.xticks(
        lx if N > 1 else [lx[0]],
        [f"{v:.0e}" for v in lambdas],
        fontsize=fontsize - 2,
        rotation=45,
    )
    plt.yticks(
        gx if M > 1 else [gx[0]], [f"{v:.0e}" for v in gammas], fontsize=fontsize - 2
    )

    plt.tight_layout()
    plt.show()
