import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


def plot_field(field, filepath: str) -> None:
    assert field.ndim == 2
    plt.imshow(field.T, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
    return None


def plot_contours(field, contours, filepath: str) -> None:
    assert field.ndim == 2
    plt.imshow(field.T, origin="lower", aspect="equal")
    for c in contours:
        plt.plot(c[:, 0], c[:, 1], 'r', linewidth=1.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    return None


def plot_material_model(replicates, x, y, df_meta, max_strain: float, filepath: str) -> None:

    for name, (x_, y_) in zip(df_meta["Specimen"].values, replicates):
        fx, fy = zip(*[(xs, ys) for xs, ys in zip(x_, y_) if (xs <= max_strain)])
        plt.plot(fx, fy, linewidth=1, alpha=0.5, label=name)

    fx, fy = zip(*[(xs, ys) for xs, ys in zip(x, y) if (xs <= max_strain)])
    plt.plot(fx, fy, linewidth=1.5, color="k", label="Model")

    name = df_meta.iloc[0, -1]
    plt.title(name)
    plt.xlabel("Strain (-)")
    plt.ylabel("Stress (MPa)")
    plt.ylim(0, 50)
    plt.grid(alpha=0.5, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return None


if __name__ == "__main__":
    exit()