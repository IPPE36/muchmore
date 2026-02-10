from source.field import sample_field_grf, sample_field_srf, mesh_field, stats_field
from source.mesh_3D import mesh_3d_random_inclusion


if __name__ == "__main__":
    size = 128
    physical_size = 30 * 1e-3  # 30 microns to mm
    physical_spacing = physical_size / 128

    field = sample_field_grf(vf_inclusion=0.5, dim=3, shape=128, anis=1)
    # stats, stats_mm = stats_field(field, physical_spacing)
    # inclusion_set = stats.to_inclusion_set()

    # mesh_3d_random_inclusion(inclusion_sets=inclusion_set, show=True)

    mesh_field(
        field=field,
        element_order=1,
        h=0.04,
        name_model="RVE",
        physical_spacing=1.0,
        load_case="Tensile-X",
        strain=0.03,
    )

    # TODO add readme
    # TODO simulation automation via cmd
    # TODO apply tie constraint argument
    # TODO rework field generation
    # TODO add sphere fields (tiled generation)
    # TODO add image segmentation SAM
    # TODO add kernel fitting on reference images
    # https://gmshmodel.readthedocs.io/en/latest/examples/randomInclusions2DCircle.html