from source.field import sample_field_srf
from source.microstructure import Microstructure
from source.plotting import plot_field


if __name__ == "__main__":
    X = sample_field_srf(dim=2, shape=128)
    # plot_field(X, "field.png")
    ms = Microstructure(X)
    ms.mesh_2d()
