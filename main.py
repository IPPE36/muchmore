from source.field import sample_field_srf, sample_field_grf
from source.microstructure import Microstructure
from source.plotting import plot_field


if __name__ == "__main__":
    X = sample_field_grf(dim=3, shape=128, anis=1)
    # plot_field(X, "test.png")
    # X = sample_field_srf(dim=3, shape=128)
    ms = Microstructure(X)
    ms.mesh()
