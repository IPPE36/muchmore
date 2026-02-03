from source.field import sample_field_srf, sample_field_grf
from source.microstructure import Microstructure
from source.plotting import plot_field


if __name__ == "__main__":

    X = sample_field_grf(dim=3, shape=64, anis=1)
    # X = sample_field_srf(dim=2, shape=128)
    ms = Microstructure(X)
    ms.mesh()
    # https: // gmshmodel.readthedocs.io / en / latest / gmshModel / Model / GenericRVE.html
    # TODO clean field before mesh / remove min_area
    # TODO https://gmshmodel.readthedocs.io/en/latest/gmshModel/Model/GenericRVE.html
    # https://gmshmodel.readthedocs.io/en/latest/examples/helicalChain3DSphere.html

    """
    Closed surfaces do exist iff

All of the following are true:

The “1” phase is fully enclosed by 0 (or vice versa)

The connected component does not touch the domain boundary

The binary field is topologically consistent (no voxel-scale cracks)
    """