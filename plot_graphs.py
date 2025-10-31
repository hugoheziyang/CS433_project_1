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


def plot_cv_surface(cv_f1, fontsize=13):
    """
    Visualize mean validation F1 over (gamma, lambda) grid.
    Automatically detects and marks the best (gamma, lambda) pair.

    Args:
        cv_f1: dict {(gamma, lambda): mean_f1_score}
        fontsize: base font size for labels, legend, etc.
    """
    # Collect unique sorted grid values
    gammas = sorted({g for (g, l) in cv_f1.keys()})
    lambdas = sorted({l for (g, l) in cv_f1.keys()})

    # Sanity check for log10 plotting
    if any(g <= 0 for g in gammas) or any(l <= 0 for l in lambdas):
        raise ValueError("All gamma and lambda values must be > 0 for log10 plotting.")

    # Build matrix of F1 scores
    M, N = len(gammas), len(lambdas)
    f1_matrix = np.empty((M, N))
    for i, g in enumerate(gammas):
        for j, l in enumerate(lambdas):
            f1_matrix[i, j] = cv_f1[(g, l)]

    # Find the best (gamma, lambda)
    best_gamma, best_lambda = max(cv_f1, key=cv_f1.get)
    best_f1 = cv_f1[(best_gamma, best_lambda)]

    # Log coordinates
    gx, lx = np.log10(gammas), np.log10(lambdas)

    # Handle 1-value cases
    eps = 1e-9
    x0, x1 = (lx[0] - eps, lx[0] + eps) if N == 1 else (lx.min(), lx.max())
    y0, y1 = (gx[0] - eps, gx[0] + eps) if M == 1 else (gx.min(), gx.max())

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        f1_matrix,
        origin="lower",
        aspect="auto",
        extent=[x0, x1, y0, y1],
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Mean validation F1", fontsize=fontsize)

    # Mark best point
    plt.scatter(
        np.log10(best_lambda),
        np.log10(best_gamma),
        color="red",
        marker="x",
        s=120,
        label=f"Best (γ, λ) = ({best_gamma:.2e}, {best_lambda:.2e})\nF1 = {best_f1:.3f}",
    )

    # Labels / title
    plt.xlabel("log10(lambda)", fontsize=fontsize)
    plt.ylabel("log10(gamma)", fontsize=fontsize)
    plt.title("Cross-Validation F1 Surface", fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize - 2, loc="lower right")

    # Axis ticks
    plt.xticks(lx if N > 1 else [lx[0]], [f"{v:.0e}" for v in lambdas],
               fontsize=fontsize - 2, rotation=45)
    plt.yticks(gx if M > 1 else [gx[0]], [f"{v:.0e}" for v in gammas],
               fontsize=fontsize - 2)

    plt.tight_layout()
    plt.show()