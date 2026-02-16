import os

import numpy as np

from source.field import mesh_field, stats_field, load_field_from_png, generate_inclusion_voxels, invert_binary_field, dump_field
from source.plotting import plot_field
from source.materials import ElasticPlasticMM
# from source.segment import init_sam, segment_image
from source.poisson import weight_to_volume_fraction, estimate_n_particles_3d

if __name__ == "__main__":

    rho_pp = 0.95  # g/cm³
    rho_ps = 1.07  # g/cm³
    out_size = 50.0  # microns RVE box size
    out_shape = 128  # voxels rve size
    out_dim = 3

    # dir_path = os.path.join("data", "images", "literature", "reference_x")
    # filepath = os.path.join(dir_path, "50_50.png")
    # img = load_field_from_png(filepath, crop_border=True, invert=False).T.astype(float)
    # from scipy.ndimage import gaussian_filter
    # img_filtered = gaussian_filter(img, 1)
    # vf_pp, vf_ps = weight_to_volume_fraction(50, 50, rho_pp, rho_ps)
    # thresh = np.quantile(img_filtered, 1 - vf_ps)
    # img_filtered = np.where(img_filtered <= thresh, 0, 1)
    # plot_field(img_filtered, filepath.replace(".png", "_f.png"))

    # # Fit material models
    # models = {}
    # dir_path = os.path.join("data", "experiments")
    # for f in os.listdir(dir_path):
    #     filepath = os.path.join(dir_path, f)
    #     m = ElasticPlasticMM.from_xlsx(filepath)
    #     m.dump_json(f.replace(".xlsx", ".json"))
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
        vf_pp, vf_ps = weight_to_volume_fraction(wf_pp, wf_ps, rho_pp, rho_ps)

        # phase definition
        inclusion_phase = "PS"
        inclusion_value = 1
        vf_inclusion = vf_ps
        invert_out = False
        if wf_pp <= wf_ps:
            img = invert_binary_field(img)
            inclusion_phase = "PP"
            inclusion_value = 0
            vf_inclusion = vf_pp
            invert_out = True

        x_span = float(f.split("L")[1].split("_")[0])  # microns
        y_span = x_span * (img.shape[1] / img.shape[0])  # microns
        x_size = img.shape[0]  # voxels
        y_size = img.shape[1]  # voxels
        spacing = x_span / x_size  # microns per voxel
        stats = stats_field(img, spacing, inclusion_value)
        box_size = (out_size, out_size) if out_dim == 2 else (out_size, out_size, out_size)
        box_size_vox = (out_shape, out_shape) if out_dim == 2 else (out_shape, out_shape, out_shape)
        
        # vf_inclusion_img = np.count_nonzero(img == inclusion_value) / img.size
        # n_scaling = vf_inclusion / vf_inclusion_img

        if f.startswith("Y"):
            if out_dim == 3:
                n_expected = estimate_n_particles_3d(
                    img,
                    physical_spacing=spacing,  # µm per pixel
                    box_size=box_size,  # µm
                    inclusion_value=inclusion_value,  # [0, 1]
                    aspect_ratio_k=np.quantile(stats.aspect_ratio, 0.66),
                    composition_wt=(wf_pp, wf_ps),  # (w_PP, w_PS)
                    densities=(rho_pp, rho_ps),  # (rho_PP, rho_PS)
                    inclusion_phase=inclusion_phase,
                    use_composition_VV=True,
                )["expected_N"]
            else:
                n_expected = int(stats.n_particles * (out_size * out_size) / (x_span * y_span))

            field, stats_gen = generate_inclusion_voxels(
                ndim=out_dim,
                box_size_um=box_size,
                box_size_vox=box_size_vox,
                mean_particle_area_um2=np.median(stats.area),
                mean_particle_aspect_ratio=np.median(stats.aspect_ratio),
                n_particles=int(n_expected),
                target_vf=vf_inclusion,
                size_factors=stats.area,
                periodic=True,
                seed=1,
            )
            print(
                f"Particles Target: {stats_gen["requested"]}, Placed: {stats_gen["placed"]}\n"
                f"Volume Fraction Target: {stats_gen["target_vf"]:.3f}, Achieved: {stats_gen["achieved_vf"]:.3f}"
            )
        elif f.startswith("X"):
            pass

        if invert_out:
            field = invert_binary_field(field)
        if out_dim == 2:
            plot_field(field, os.path.join("reconstructions", f"{f[2:7]}_{out_dim}d_rec.png"))
        elif out_dim == 3:
            plot_field(field[:, :, 0], os.path.join("reconstructions", f"{f[2:7]}_{out_dim}d_rec.png"))

        dump_field(field, os.path.join("reconstructions", f"{f[2:7]}_{out_dim}d.npy"))

        # mesh_field(
        #     field=field,
        #     element_order=1,
        #     h=0.03,
        #     name_model="RVE",
        #     physical_spacing=out_size/out_shape,
        #     load_case="Tensile-X",
        #     strain=0.03,
        # )

