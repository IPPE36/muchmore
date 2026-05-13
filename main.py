import os

import numpy as np

from source.field import mesh_field, stats_field, load_field_from_png, generate_inclusion_voxels, invert_binary_field
from source.field import dump_field, load_field, field_from_stl
from source.plotting import plot_field
from source.materials import ElasticPlasticMM, MaterialModel
from source.segment import init_sam, segment_image
from source.poisson import weight_to_volume_fraction, estimate_n_particles_3d


if __name__ == "__main__":

    poisson_pp = 0.42
    poisson_ps = 0.35
    rho_pp = 0.95  # g/cm³
    rho_ps = 1.07  # g/cm³
    md_out_size = 6.73 * 1e-5  # 673 armstrong
    out_size = 50.0 * 1e-3  # microns RVE box size
    out_shape = 128  # voxels rve size
    out_dim = 3

    # Fit material models
    # dir_path = os.path.join("data", "experiments")
    # for f in os.listdir(dir_path):
    #     wf_pp = float(f[-12:-9])
    #     wf_ps = float(f[-8:-5])
    #     vf_pp, vf_ps = weight_to_volume_fraction(wf_pp, wf_ps, rho_pp, rho_ps)
    #     rho = rho_pp * vf_pp + rho_ps * vf_ps
    #     poisson = poisson_pp * vf_pp + poisson_ps * vf_ps
    #     filepath = os.path.join(dir_path, f)
    #     m = ElasticPlasticMM.from_xlsx(filepath, rho, poisson)
    #     m.dump_json(os.path.join("temp", f.replace(".xlsx", ".json")))
    #     m.plot(os.path.join("plots", f"EPM_{m.name()}.png"))

    # m_pp = ElasticPlasticMM.load_json(os.path.join("temp", "PP-PS_100-000.json"))
    # m_ps = ElasticPlasticMM.load_json(os.path.join("temp", "PP-PS_000-100.json"))

    dir_path = os.path.join("data", "microstructures", "md_zellen")
    for f in os.listdir(dir_path):
        filepath = os.path.join(dir_path, f)

        vf_pp = int(f.split("-")[1][2:4])
        vf_ps = int(f.split("-")[2][2:4])

        if vf_pp > vf_ps and f.endswith("_PP.stl"):
            continue
        if vf_ps > vf_pp and f.endswith("_PS.stl"):
            continue

        rve_name = f"MD_{vf_pp}_{vf_ps}"

        if vf_pp >= vf_ps:
            # mat_a = m_ps.to_inp_str()  # inclusion
            # mat_b = m_pp.to_inp_str()  # matrix
            name_phase_a = "PHASE-PS"
            name_phase_b = "PHASE-PP"
        else:
            # mat_a = m_pp.to_inp_str()  # inclusion
            # mat_b = m_ps.to_inp_str()  # matrix
            name_phase_a = "PHASE-PP"
            name_phase_b = "PHASE-PS"

        # mat_a = mat_a.replace("name=ELASTOPLASTIC", f"name={name_phase_a}")
        # mat_b = mat_b.replace("name=ELASTOPLASTIC", f"name={name_phase_b}")

        field = field_from_stl(filepath)

        mesh_field(
            field=field,
            element_order=1,
            h=0.02,
            name_model=rve_name,
            spacing=md_out_size / field.shape[0],
            load_case="Tensile-X",
            strain=0.15,
            # mat_a=mat_a,
            # mat_b=mat_b,
            name_phase_a=name_phase_a,
            name_phase_b=name_phase_b,
            tie_constraint=False,
            static_solver=False,
            linear_mat=False,
            show=False,
        )

    # # Segment literature reference images
    # sam_model = init_sam()
    # dir_path = os.path.join("data", "images", "literature", "reference_x")
    # for f in os.listdir(dir_path):
    #     if not f.startswith("Y_"):
    #         continue
    #     filepath = os.path.join(dir_path, f)
    #     segment_image(filepath, sam_model)

    # Reconstructions
    # dir_path = os.path.join("data", "images", "literature", "reference_x", "postprocessed")
    # for i, f in enumerate(os.listdir(dir_path)):
    #     if not f.endswith("_SEG.png"):
    #         continue
    #
    #     filepath = os.path.join(dir_path, f)
    #     img = load_field_from_png(filepath, crop_border=True).T
    #
    #     # get weight fractions from filename
    #     wf_pp = float(f[2:4])
    #     wf_ps = float(f[5:7])
    #     vf_pp, vf_ps = weight_to_volume_fraction(wf_pp, wf_ps, rho_pp, rho_ps)
    #
    #     # phase definition
    #     inclusion_phase = "PS"
    #     inclusion_value = 1
    #     vf_inclusion = vf_ps
    #     invert_out = False
    #     if wf_pp <= wf_ps:
    #         img = invert_binary_field(img)
    #         inclusion_phase = "PP"
    #         inclusion_value = 0
    #         vf_inclusion = vf_pp
    #         invert_out = True
    #
    #     # dimensions
    #     x_span = float(f.split("L")[1].split("_")[0])  # microns
    #     y_span = x_span * (img.shape[1] / img.shape[0])  # microns
    #     x_size = img.shape[0]  # voxels
    #     y_size = img.shape[1]  # voxels
    #     spacing = x_span / x_size  # microns per voxel
    #     stats = stats_field(img, spacing, inclusion_value)
    #     box_size = (out_size, out_size) if out_dim == 2 else (out_size, out_size, out_size)
    #     box_size_vox = (out_shape, out_shape) if out_dim == 2 else (out_shape, out_shape, out_shape)
    #
    #     if f.startswith("Y"):
    #         if out_dim == 3:
    #             n_expected = estimate_n_particles_3d(
    #                 img,
    #                 physical_spacing=spacing,  # µm per pixel
    #                 box_size=box_size,  # µm
    #                 inclusion_value=inclusion_value,  # [0, 1]
    #                 aspect_ratio_k=np.quantile(stats.aspect_ratio, 0.66),
    #                 composition_wt=(wf_pp, wf_ps),  # (w_PP, w_PS)
    #                 densities=(rho_pp, rho_ps),  # (rho_PP, rho_PS)
    #                 inclusion_phase=inclusion_phase,
    #                 use_composition_VV=True,
    #             )["expected_N"]
    #         else:
    #             n_expected = int(stats.n_particles * (out_size * out_size) / (x_span * y_span))
    #
    #         field, stats_gen = generate_inclusion_voxels(
    #             ndim=out_dim,
    #             box_size_um=box_size,
    #             box_size_vox=box_size_vox,
    #             mean_particle_area_um2=np.median(stats.area),
    #             mean_particle_aspect_ratio=np.median(stats.aspect_ratio),
    #             n_particles=int(n_expected),
    #             target_vf=vf_inclusion,
    #             size_factors=stats.area,
    #             periodic=True,
    #             seed=1,
    #         )
    #         print(
    #             f"Particles Target: {stats_gen["requested"]}, Placed: {stats_gen["placed"]}\n"
    #             f"Volume Fraction Target: {stats_gen["target_vf"]:.3f}, Achieved: {stats_gen["achieved_vf"]:.3f}"
    #         )
    #     elif f.startswith("X"):
    #         pass
    #
    #     if invert_out:
    #         field = invert_binary_field(field)
    #     if out_dim == 2:
    #         plot_field(field, os.path.join("plots", f"{f[2:7]}_{out_dim}d_rec.png"))
    #     elif out_dim == 3:
    #         plot_field(field[:, :, 0], os.path.join("plots", f"{f[2:7]}_{out_dim}d_rec.png"))
    #
    #     dump_field(field, os.path.join("temp", f"{f[2:7]}_{out_dim}d.npy"))

    dir_path = "temp"
    for f in os.listdir(dir_path):
        if f.endswith(".npy") and "3d" in f:
            filepath = os.path.join(dir_path, f)
            rve_name = f.replace("_3d.npy", "_PP_PS")
            field = load_field(filepath)
            wf_pp = float(f[:2])
            wf_ps = float(f[3:5])

            # invert field? inclusions should have value 1!
            vf_pp, vf_ps = weight_to_volume_fraction(wf_pp, wf_ps, rho_pp, rho_ps)
            vf = np.count_nonzero(field) / field.size
            if vf > 0.5:
                field = invert_binary_field(field)

            if vf_pp >= vf_ps:
                # mat_a = m_ps.to_inp_str()  # inclusion
                # mat_b = m_pp.to_inp_str()  # matrix
                name_phase_a = "PHASE-PS"
                name_phase_b = "PHASE-PP"
            else:
                # mat_a = m_pp.to_inp_str()  # inclusion
                # mat_b = m_ps.to_inp_str()  # matrix
                name_phase_a = "PHASE-PP"
                name_phase_b = "PHASE-PS"
            # mat_a = mat_a.replace("name=ELASTOPLASTIC", f"name={name_phase_a}")
            # mat_b = mat_b.replace("name=ELASTOPLASTIC", f"name={name_phase_b}")

            mesh_field(
                field=field,
                element_order=1,
                h=0.02,
                name_model=rve_name,
                spacing=out_size/out_shape,
                load_case="Tensile-X",
                strain=0.15,
                # mat_a=mat_a,
                # mat_b=mat_b,
                name_phase_a=name_phase_a,
                name_phase_b=name_phase_b,
                tie_constraint=False,
                static_solver=False,
                linear_mat=False,
                show=False,
            )
