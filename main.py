from source.field import sample_field_srf, sample_field_grf
from source.microstructure import Microstructure
from source.plotting import plot_field


if __name__ == "__main__":
    X = sample_field_grf(dim=3, shape=128, anis=1)
    ms = Microstructure(X)
    ms.mesh()

    # TODO apply tie constraint argument
    # TODO improve mesh quality
    # TODO scale RVE
    # TODO move postprocess to MS class