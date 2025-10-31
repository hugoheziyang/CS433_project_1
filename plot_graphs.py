import numpy as np
import matplotlib.pyplot as plt


# Plot mean validation f1 over (gamma, lambda) grid for logistic regression
def plot_f1_logreg(cv_f1, fontsize=13, save_path=None):
    """
    Visualize mean validation F1 over (gamma, lambda) grid.
    Automatically detects and marks the best (gamma, lambda) pair.

    Args:
        cv_f1: dict {(gamma, lambda): mean_f1_score}
        fontsize: base font size for labels, legend, etc.
        save_path: optional str, if provided saves plot as .png (e.g. "f1_logreg.png")
    """
    gammas = sorted({g for (g, l) in cv_f1.keys()})
    lambdas = sorted({l for (g, l) in cv_f1.keys()})

    if any(g <= 0 for g in gammas) or any(l <= 0 for l in lambdas):
        raise ValueError("All gamma and lambda values must be > 0 for log10 plotting.")

    M, N = len(gammas), len(lambdas)
    f1_matrix = np.empty((M, N))
    for i, g in enumerate(gammas):
        for j, l in enumerate(lambdas):
            f1_matrix[i, j] = cv_f1[(g, l)]

    best_gamma, best_lambda = max(cv_f1, key=cv_f1.get)
    best_f1 = cv_f1[(best_gamma, best_lambda)]
    print(f"Best (gamma, lambda): ({best_gamma}, {best_lambda}) with F1 = {best_f1:.4f}")

    gx, lx = np.log10(gammas), np.log10(lambdas)
    eps = 1e-9
    x0, x1 = (lx[0] - eps, lx[0] + eps) if N == 1 else (lx.min(), lx.max())
    y0, y1 = (gx[0] - eps, gx[0] + eps) if M == 1 else (gx.min(), gx.max())

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

    plt.scatter(
        np.log10(best_lambda),
        np.log10(best_gamma),
        color="red",
        marker="x",
        s=120,
        label=f"Best (γ, λ) = ({best_gamma:.2e}, {best_lambda:.2e})\nF1 = {best_f1:.3f}",
    )

    plt.xlabel("log10(lambda)", fontsize=fontsize)
    plt.ylabel("log10(gamma)", fontsize=fontsize)
    plt.title("Cross-Validation F1 Surface (Logistic Regression)", fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize - 2, loc="lower right")

    plt.xticks(lx if N > 1 else [lx[0]], [f"{v:.0e}" for v in lambdas],
               fontsize=fontsize - 2, rotation=45)
    plt.yticks(gx if M > 1 else [gx[0]], [f"{v:.0e}" for v in gammas],
               fontsize=fontsize - 2)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to {save_path}")

    plt.show()


# Plot mean validation f1 over lambda for linear regression
def plot_f1_linreg(cv_f1, fontsize=13, save_path=None):
    """
    Plot mean validation F1 as a function of lambda for linear regression.
    Automatically detects and marks the best lambda.

    Args:
        cv_f1: dict {lambda: mean_f1_score}
        fontsize: base font size for labels, legend, etc.
        save_path: optional str, if provided saves plot as .png (e.g. "f1_linreg.png")
    """
    lambdas = np.array(sorted(cv_f1.keys()))
    f1_scores = np.array([cv_f1[l] for l in lambdas])

    if any(l <= 0 for l in lambdas):
        raise ValueError("All lambda values must be > 0 for log10 plotting.")

    best_lambda = lambdas[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    print(f"Best lambda: {best_lambda} with F1 = {best_f1:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(np.log10(lambdas), f1_scores, "-o", label="Mean F1 across folds")
    plt.scatter(np.log10(best_lambda), best_f1, color="red", s=100, marker="x",
                label=f"Best λ = {best_lambda:.2e}\nF1 = {best_f1:.3f}")

    plt.xlabel("log10(lambda)", fontsize=fontsize)
    plt.ylabel("Mean validation F1", fontsize=fontsize)
    plt.title("Cross-Validation F1 vs λ (Linear Regression)", fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize - 2, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to {save_path}")

    plt.show()


cv_f1_logreg = {
    (0.001, 0.001): 0.337251,
    (0.0047315125896148025, 0.001): 0.395480,
    (0.0223872113856834, 0.001): 0.308131,
    (0.10592537251772897, 0.001): 0.208748,
    (0.5011872336272722, 0.001): 0.219967,

    (0.001, 0.00262163): 0.337248,
    (0.0047315125896148025, 0.00262163): 0.395508,
    (0.0223872113856834, 0.00262163): 0.309239,
    (0.10592537251772897, 0.00262163): 0.208546,
    (0.5011872336272722, 0.00262163): 0.214082,

    (0.001, 0.00687294): 0.337241,
    (0.0047315125896148025, 0.00687294): 0.395281,
    (0.0223872113856834, 0.00687294): 0.311341,
    (0.10592537251772897, 0.00687294): 0.208814,
    (0.5011872336272722, 0.00687294): 0.207499,

    (0.001, 0.01801832): 0.337235,
    (0.0047315125896148025, 0.01801832): 0.395055,
    (0.0223872113856834, 0.01801832): 0.318089,
    (0.10592537251772897, 0.01801832): 0.217612,
    (0.5011872336272722, 0.01801832): 0.213323,

    (0.001, 0.04723737): 0.337220,
    (0.0047315125896148025, 0.04723737): 0.394583,
    (0.0223872113856834, 0.04723737): 0.333744,
    (0.10592537251772897, 0.04723737): 0.255384,
    (0.5011872336272722, 0.04723737): 0.254121,

    (0.001, 0.12383892): 0.337098,
    (0.0047315125896148025, 0.12383892): 0.392985,
    (0.0223872113856834, 0.12383892): 0.371493,
    (0.10592537251772897, 0.12383892): 0.343995,
    (0.5011872336272722, 0.12383892): 0.296637,

    (0.001, 0.32465984): 0.336820,
    (0.0047315125896148025, 0.32465984): 0.387577,
    (0.0223872113856834, 0.32465984): 0.404816,
    (0.10592537251772897, 0.32465984): 0.403952,
    (0.5011872336272722, 0.32465984): 0.066187,

    (0.001, 0.85113804): 0.336163,
    (0.0047315125896148025, 0.85113804): 0.373921,
    (0.0223872113856834, 0.85113804): 0.384660,
    (0.10592537251772897, 0.85113804): 0.384639,
    (0.5011872336272722, 0.85113804): 0.051593
}