from source.plotting import plot_field
from source.field import sample_field, store_field, load_field
from source.mesh import surface_mesh

# RECONSTRUCT LAURA IMAGE
# LOAD VOXEL
# MESH ABAQUS
# PBC


if __name__ == "__main__":
    X = sample_field(dim=2, shape=64)
    surface_mesh(X)
