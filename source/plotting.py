import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


def plot_field(field, filepath: str) -> None:
    assert field.ndim == 2
    plt.imshow(field.T)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
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


if __name__ == "__main__":
    exit()