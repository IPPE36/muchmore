import os

import numpy as np

from source.field import mesh_field, stats_field, load_field_from_png, generate_inclusion_voxels
from source.plotting import plot_field
from source.materials import ElasticPlasticMM
# from source.segment import init_sam, segment_image
from source.poisson import weight_to_volume_fraction, estimate_n_particles_3d

if __name__ == "__main__":

    rho_pp = 0.90
    rho_ps = 1.07  # g/cm³

    # # Fit material models
    # models = {}
    # dir_path = os.path.join("data", "experiments")
    # for f in os.listdir(dir_path):
    #     filepath = os.path.join(dir_path, f)
    #     m = ElasticPlasticMM.from_xlsx(filepath)
    #     m.plot()
    #     models[m.name()] = m

    # # Segment literature reference images
    # sam_model = init_sam()
    # dir_path = os.path.join("data", "images", "literature", "reference_x")
    # for f in os.listdir(dir_path):
    #     if not f.startswith("Y_"):
    #         continue
    #     filepath = os.path.join(dir_path, f)
    #     segment_image(filepath, sam_model)

    # Extract features
    fields = {}
    dir_path = os.path.join("data", "images", "literature", "reference_x", "postprocessed")
    for i, f in enumerate(os.listdir(dir_path)):
        if not f.endswith("_SEG.png"):
            continue

        filepath = os.path.join(dir_path, f)
        img = load_field_from_png(filepath, crop_border=True).T

        # get weight fractions from filename
        wf_pp = float(f[2:4])
        wf_ps = float(f[5:7])
        x_span_mm = float(f.split("L")[1].split("_")[0])   # to mm
        x_size = img.shape[0]
        y_size = img.shape[1]
        spacing = x_span_mm / x_size

        stats, stats_mm = stats_field(img, spacing)

        vf_pp, vf_ps = weight_to_volume_fraction(wf_pp, wf_ps, rho_pp, rho_ps)

        out_size = x_span_mm
        out_shape = 256
        out_dim = 3

        if out_dim == 3:
            n_expected = estimate_n_particles_3d(
                img,
                physical_spacing=spacing,  # µm per pixel
                box_size=(out_size, out_size, out_size),  # µm
                inclusion_value=1,  # [0, 1]
                composition_wt=(wf_pp, wf_ps),  # (w_PP, w_PS)
                densities=(rho_pp, rho_ps),  # (rho_PP, rho_PS)
                inclusion_phase="PP",
                use_composition_VV=True,
            )["expected_N"]
        else:
            n_expected = stats.n_particles

        print(n_expected)

        field, _ = generate_inclusion_voxels(
            ndim=out_dim,
            box_size_um=(out_size, out_size, out_size),
            box_size_vox=(out_shape, out_shape, out_shape),
            mean_particle_area_um2=np.sqrt(np.median(stats_mm.area))**out_dim,
            mean_particle_aspect_ratio=np.median(stats_mm.aspect_ratio),
            n_particles=int(n_expected),
            target_vf=vf_pp,
            size_factors=np.array(stats_mm.area) ** 2,
            periodic=True,
            seed=1,
        )

        plot_field(img, f"f0{i}.png")
        try:
            plot_field(field, f"f1{i}.png")
        except AssertionError:
            plot_field(field[:, :, 0], f"f1{i}.png")
        exit()
        mesh_field(
            field=field,
            element_order=1,
            h=0.03,
            name_model="RVE",
            physical_spacing=out_size/out_shape,
            load_case="Tensile-X",
            strain=0.03,
        )
