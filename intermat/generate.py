"""Module to generate interface combinations."""

from jarvis.analysis.interface.zur import ZSLGenerator
from jarvis.core.atoms import fix_pbc
from jarvis.core.lattice import Lattice, lattice_coords_transformer
import os
from jarvis.db.figshare import data as j_data
import numpy as np
from jarvis.analysis.defects.surface import Surface
from jarvis.core.atoms import Atoms
from intermat.calculators import Calc
from tqdm import tqdm

# from jarvis.db.figshare import get_jid_data
# import pandas as pd
# import time
# from jarvis.db.jsonutils import dumpjson
# from jarvis.tasks.vasp.vasp import VaspJob
# from jarvis.core.atoms import add_atoms
# from jarvis.tasks.queue_jobs import Queue
# from jarvis.io.vasp.inputs import Poscar, Incar, Potcar
# from jarvis.core.kpoints import Kpoints3D
# from jarvis.analysis.structure.spacegroup import (
#    Spacegroup3D,
#    symmetrically_distinct_miller_indices,
# )

# TODO: Save InterFaceCombi.json


def write_jobpy(pyname="job.py", job_json=""):
    # job_json = os.getcwd()+'/'+'job.json'
    f = open(pyname, "w")
    f.write("from jarvis.tasks.vasp.vasp import VaspJob\n")
    f.write("from jarvis.db.jsonutils import loadjson\n")
    f.write('d=loadjson("' + str(job_json) + '")\n')
    f.write("v=VaspJob.from_dict(d)\n")
    f.write("v.runjob()\n")
    f.close()


def add_atoms(
    top, bottom, distance=[0, 0, 1], apply_strain=False, add_tags=True
):
    """
    Add top and bottom Atoms with a distance array.

    Bottom Atoms lattice-matrix is chosen as final lattice.
    """
    top = top.center_around_origin([0, 0, 0])
    bottom = bottom.center_around_origin(distance)
    strain_x = (
        top.lattice_mat[0][0] - bottom.lattice_mat[0][0]
    ) / bottom.lattice_mat[0][0]
    strain_y = (
        top.lattice_mat[1][1] - bottom.lattice_mat[1][1]
    ) / bottom.lattice_mat[1][1]
    if apply_strain:
        top.apply_strain([strain_x, strain_y, 0])
    #  print("strain_x,strain_y", strain_x, strain_y)
    elements = []
    coords = []
    props = []
    lattice_mat = bottom.lattice_mat
    for i, j in zip(bottom.elements, bottom.frac_coords):
        elements.append(i)
        coords.append(j)
        props.append("bottom")
    top_cart_coords = lattice_coords_transformer(
        new_lattice_mat=top.lattice_mat,
        old_lattice_mat=bottom.lattice_mat,
        cart_coords=top.cart_coords,
    )
    top_frac_coords = bottom.lattice.frac_coords(top_cart_coords)
    for i, j in zip(top.elements, top_frac_coords):
        elements.append(i)
        coords.append(j)
        props.append("top")

    order = np.argsort(np.array(elements))
    elements = np.array(elements)[order]
    coords = np.array(coords)[order]
    props = (np.array(props)[order]).tolist()
    determnt = np.linalg.det(np.array(lattice_mat))
    if determnt < 0.0:
        lattice_mat = -1 * np.array(lattice_mat)
    determnt = np.linalg.det(np.array(lattice_mat))
    if determnt < 0.0:
        print("Serious issue, check lattice vectors.")
        print("Many software follow right hand basis rule only.")
    combined = Atoms(
        lattice_mat=lattice_mat,
        coords=coords,
        elements=elements,
        props=props,
        cartesian=False,
    ).center_around_origin()
    # print('combined props',combined)
    return combined


class InterfaceCombi(object):
    """Module to generate interface combinations."""

    def __init__(
        self,
        film_mats=[],
        subs_mats=[],
        disp_intvl=0,
        seperations=[2.5],
        film_indices=[[0, 0, 1]],
        subs_indices=[[0, 0, 1]],
        film_ids=[],
        subs_ids=[],
        film_kplengths=[30],
        subs_kplengths=[30],
        film_thicknesses=[16],
        subs_thicknesses=[16],
        rount_digit=3,
        calculator={},
        working_dir=".",
        generated_interfaces=[],
        vacuum_interface=8,
        max_area_ratio_tol=1.00,
        max_area=300,
        ltol=0.08,
        atol=1,
        apply_strain=False,
        lowest_mismatch=True,
        rotate_xz=False,  # for transport
        lead_ratio=None,  # 0.3,
        from_conventional_structure_film=True,
        from_conventional_structure_subs=True,
        relax=False,
        wads={},
        dataset=[],
        id_tag="jid",
    ):
        """Initialize class."""
        self.film_mats = film_mats
        self.subs_mats = subs_mats
        self.film_ids = film_ids
        self.subs_ids = subs_ids
        self.film_kplengths = film_kplengths
        self.subs_kplengths = subs_kplengths
        self.dataset = dataset
        self.id_tag = id_tag
        # print("self.film_ids", self.film_ids)
        # print("self.subs_ids", self.subs_ids)
        if not self.dataset:
            self.dataset = j_data("dft_3d")
        if not film_mats:
            film_mats = []
            film_kplengths = []
            for i in self.film_ids:
                atoms = self.get_id_atoms(i)["atoms"]
                film_mats.append(atoms)
                film_kplengths.append(self.get_id_atoms(i)["kp_length"])
            self.film_kplengths = film_kplengths
            self.film_mats = film_mats
        if not subs_mats:
            subs_mats = []
            subs_kplengths = []
            for i in self.subs_ids:
                atoms = self.get_id_atoms(i)["atoms"]
                subs_mats.append(atoms)
                subs_kplengths.append(self.get_id_atoms(i)["kp_length"])
            self.subs_kplengths = subs_kplengths
            self.subs_mats = subs_mats
        # print("self.film_mats", self.film_mats)
        # print("self.subs_mats", self.subs_mats)
        self.disp_intvl = disp_intvl
        self.seperations = seperations
        self.film_indices = film_indices
        self.subs_indices = subs_indices
        self.film_thicknesses = film_thicknesses
        self.subs_thicknesses = subs_thicknesses
        self.rount_digit = rount_digit
        self.working_dir = working_dir
        self.generated_interfaces = generated_interfaces
        self.vacuum_interface = vacuum_interface
        self.max_area = max_area
        self.ltol = ltol
        self.atol = atol
        self.apply_strain = apply_strain
        self.max_area_ratio_tol = max_area_ratio_tol
        self.lowest_mismatch = lowest_mismatch
        self.calculator = calculator
        self.rotate_xz = rotate_xz
        self.lead_ratio = lead_ratio
        self.from_conventional_structure_film = (
            from_conventional_structure_film
        )
        self.from_conventional_structure_subs = (
            from_conventional_structure_subs
        )
        self.relax = relax
        self.wads = wads
        if working_dir == ".":
            working_dir = str(os.getcwd())

        if self.disp_intvl == 0:
            self.xy = [[0, 0]]
        else:
            X, Y = np.mgrid[
                -0.5 + disp_intvl : 0.5 + disp_intvl : disp_intvl,
                -0.5 + disp_intvl : 0.5 + disp_intvl : disp_intvl,
            ]
            xy = np.vstack((X.flatten(), Y.flatten())).T
            self.xy = xy
            self.X = X
            self.Y = Y
            # print("X", X.shape)
            # print("Y", Y.shape)

    def get_id_atoms(self, id=""):
        for i in self.dataset:
            if i[self.id_tag] == id:
                atoms = Atoms.from_dict(i["atoms"])
                info = {}
                info["atoms"] = atoms
                info["kp_length"] = i["kpoint_length_unit"]
                return info

    def make_interface(
        self,
        film="",
        subs="",
        seperation=3.0,
    ):
        """
        Use as main function for making interfaces/heterostructures.

        Return mismatch and other information as info dict.

        Args:
           film: top/film material.

           subs: substrate/bottom/fixed material.

           seperation: minimum seperation between two.

           vacuum: vacuum will be added on both sides.
           So 2*vacuum will be added.
        """
        vacuum = self.vacuum_interface
        ltol = self.ltol
        atol = self.atol
        z = ZSLGenerator(
            max_area_ratio_tol=self.max_area_ratio_tol,
            max_area=self.max_area,
            max_length_tol=self.ltol,
            max_angle_tol=self.atol,
        )
        film = fix_pbc(film.center_around_origin([0, 0, 0]))
        subs = fix_pbc(subs.center_around_origin([0, 0, 0]))
        matches = list(
            z(
                film.lattice_mat[:2],
                subs.lattice_mat[:2],
                lowest=self.lowest_mismatch,
            )
        )
        info = {}
        info["mismatch_u"] = "na"
        info["mismatch_v"] = "na"
        info["mismatch_angle"] = "na"
        info["area1"] = "na"
        info["area2"] = "na"
        info["film_sl"] = "na"
        info["matches"] = matches
        info["subs_sl"] = "na"
        uv1 = matches[0]["sub_sl_vecs"]
        uv2 = matches[0]["film_sl_vecs"]
        u = np.array(uv1)
        v = np.array(uv2)
        a1 = u[0]
        a2 = u[1]
        b1 = v[0]
        b2 = v[1]
        mismatch_u = np.linalg.norm(b1) / np.linalg.norm(a1) - 1
        mismatch_v = np.linalg.norm(b2) / np.linalg.norm(a2) - 1
        angle1 = (
            np.arccos(np.dot(a1, a2) / np.linalg.norm(a1) / np.linalg.norm(a2))
            * 180
            / np.pi
        )
        angle2 = (
            np.arccos(np.dot(b1, b2) / np.linalg.norm(b1) / np.linalg.norm(b2))
            * 180
            / np.pi
        )
        mismatch_angle = abs(angle1 - angle2)
        area1 = np.linalg.norm(np.cross(a1, a2))
        area2 = np.linalg.norm(np.cross(b1, b2))
        uv_substrate = uv1
        uv_film = uv2
        substrate_latt = Lattice(
            np.array(
                [
                    uv_substrate[0][:],
                    uv_substrate[1][:],
                    subs.lattice_mat[2, :],
                ]
            )
        )
        _, __, scell = subs.lattice.find_matches(
            substrate_latt, ltol=ltol, atol=atol
        )
        film_latt = Lattice(
            np.array([uv_film[0][:], uv_film[1][:], film.lattice_mat[2, :]])
        )
        scell[2] = np.array([0, 0, 1])
        scell_subs = scell
        _, __, scell = film.lattice.find_matches(
            film_latt, ltol=ltol, atol=atol
        )
        scell[2] = np.array([0, 0, 1])
        scell_film = scell
        film_scell = film.make_supercell_matrix(scell_film)
        subs_scell = subs.make_supercell_matrix(scell_subs)
        info["mismatch_u"] = mismatch_u
        info["mismatch_v"] = mismatch_v
        # print("mismatch_u,mismatch_v", mismatch_u, mismatch_v)
        info["mismatch_angle"] = mismatch_angle
        info["area1"] = area1
        info["area2"] = area2
        info["film_sl"] = film_scell.to_dict()
        info["subs_sl"] = subs_scell.to_dict()
        substrate_top_z = max(np.array(subs_scell.cart_coords)[:, 2])
        substrate_bot_z = min(np.array(subs_scell.cart_coords)[:, 2])
        film_top_z = max(np.array(film_scell.cart_coords)[:, 2])
        film_bottom_z = min(np.array(film_scell.cart_coords)[:, 2])
        thickness_sub = abs(substrate_top_z - substrate_bot_z)
        thickness_film = abs(film_top_z - film_bottom_z)
        sub_z = (
            (vacuum + substrate_top_z)
            * np.array(subs_scell.lattice_mat[2, :])
            / np.linalg.norm(subs_scell.lattice_mat[2, :])
        )
        shift_normal = (
            sub_z / np.linalg.norm(sub_z) * seperation / np.linalg.norm(sub_z)
        )
        tmp = (
            thickness_film / 2 + seperation + thickness_sub / 2
        ) / np.linalg.norm(subs_scell.lattice_mat[2, :])
        shift_normal = (
            tmp
            * np.array(subs_scell.lattice_mat[2, :])
            / np.linalg.norm(subs_scell.lattice_mat[2, :])
        )
        interface = add_atoms(
            film_scell,
            subs_scell,
            shift_normal,
            apply_strain=self.apply_strain,
        ).center_around_origin([0, 0, 0.5])
        combined = interface.center(vacuum=vacuum).center_around_origin(
            [0, 0, 0.5]
        )
        if self.rotate_xz:
            lat_mat = combined.lattice_mat
            coords = combined.frac_coords
            elements = combined.elements
            props = combined.props
            tmp = lat_mat.copy()
            indx = 2
            tmp[indx] = lat_mat[0]
            tmp[0] = lat_mat[indx]
            lat_mat = tmp
            tmp = coords.copy()
            tmp[:, indx] = coords[:, 0]
            tmp[:, 0] = coords[:, indx]
            coords = tmp
            combined = Atoms(
                lattice_mat=lat_mat,
                coords=coords,
                elements=elements,
                cartesian=False,
                props=props,
            ).center_around_origin([0.5, 0, 0])
        if self.lead_ratio is not None:
            # a = combined.lattice.abc[0]
            coords = combined.frac_coords
            lattice_mat = combined.lattice_mat
            elements = np.array(combined.elements)
            coords_left = coords[coords[:, 0] < self.lead_ratio]
            elements_left = elements[coords[:, 0] < self.lead_ratio]
            # elements_left=['Xe' for i in range(len(elements_left))]
            atoms_left = Atoms(
                lattice_mat=lattice_mat,
                elements=elements_left,
                coords=coords_left,
                cartesian=False,
            )

            coords_right = coords[coords[:, 0] > 0.5 + self.lead_ratio]
            elements_right = elements[coords[:, 0] > 0.5 + self.lead_ratio]
            # elements_right=['Ar' for i in range(len(elements_left))]
            atoms_right = Atoms(
                lattice_mat=lattice_mat,
                elements=elements_right,
                coords=coords_right,
                cartesian=False,
            )

            coords_middle = coords[
                (coords[:, 0] <= 0.5 + self.lead_ratio)
                & (coords[:, 0] >= self.lead_ratio)
            ]
            elements_middle = elements[
                (coords[:, 0] <= 0.5 + self.lead_ratio)
                & (coords[:, 0] >= self.lead_ratio)
            ]
            atoms_middle = Atoms(
                lattice_mat=lattice_mat,
                elements=elements_middle,
                coords=coords_middle,
                cartesian=False,
            )

            info["atoms_left"] = atoms_left  # .to_dict()
            info["atoms_right"] = atoms_right  # .to_dict()
            info["atoms_middle"] = atoms_middle  # .to_dict()
            # print('coords_left',coords_left)
            # print('elements_left',elements_left)
            # print("atoms_left\n", atoms_left)
            # print("atoms_right\n", atoms_right)
            # print("atoms_middle\n", atoms_middle)
            if len(elements_right) + len(elements_left) + len(
                elements_middle
            ) != len(elements):
                raise ValueError(
                    "Check fractional tolerance",
                    len(elements_right),
                    len(elements_left),
                    len(elements_middle),
                    len(elements),
                )
        info["interface"] = combined  # .to_dict()
        return info

    def get_interface(
        self,
        film_atoms=None,
        subs_atoms=None,
        film_index=[1, 1, 1],
        subs_index=[1, 1, 1],
        film_thickness=10,
        subs_thickness=10,
        seperation=2.5,
        vacuum=12.0,
    ):
        """Get interface."""
        # info = {}
        film_surf = Surface(
            film_atoms,
            indices=film_index,
            from_conventional_structure=self.from_conventional_structure_film,
            thickness=film_thickness,
            vacuum=vacuum,
        ).make_surface()
        subs_surf = Surface(
            subs_atoms,
            indices=subs_index,
            from_conventional_structure=self.from_conventional_structure_subs,
            thickness=subs_thickness,
            vacuum=vacuum,
        ).make_surface()

        het = self.make_interface(
            film=film_surf,
            subs=subs_surf,
            seperation=seperation,
        )
        # print('het2\n')
        # print(het['interface'])
        het["film_surf"] = film_surf.to_dict()
        het["subs_surf"] = subs_surf.to_dict()
        return het

    def generate(self):
        # count = 0
        # cwd = self.working_dir
        gen_intfs = []
        for ii, i in enumerate(self.film_mats):
            for jj, j in enumerate(self.subs_mats):
                for seperation in self.seperations:
                    for film_index in self.film_indices:
                        for subs_index in self.subs_indices:
                            for film_thickness in self.film_thicknesses:
                                for subs_thickness in self.subs_thicknesses:
                                    for dis in self.xy:
                                        dis_tmp = dis
                                        film_thickness = round(
                                            film_thickness, self.rount_digit
                                        )
                                        subs_thickness = round(
                                            subs_thickness, self.rount_digit
                                        )
                                        seperation = round(
                                            seperation, self.rount_digit
                                        )
                                        film_kplength = self.film_kplengths[ii]
                                        subs_kplength = self.subs_kplengths[jj]
                                        dis_tmp[0] = round(
                                            dis_tmp[0], self.rount_digit
                                        )
                                        dis_tmp[1] = round(
                                            dis_tmp[1], self.rount_digit
                                        )
                                        # print(
                                        #     "dis_tmp",
                                        #     dis_tmp,
                                        #     "_".join(map(str, dis_tmp)),
                                        #     i,
                                        #     j,
                                        #     film_index,
                                        #     subs_index,
                                        # )
                                        if not self.subs_ids:
                                            tmp_j = str(jj)
                                        else:
                                            tmp_j = self.subs_ids[jj]

                                        if not self.film_ids:
                                            tmp_i = str(ii)
                                        else:
                                            tmp_i = self.film_ids[ii]
                                        name = (
                                            "Interface-"
                                            + str(tmp_i)
                                            + "_"
                                            + str(tmp_j)
                                            + "_"
                                            + "film_miller_"
                                            + "_".join(map(str, film_index))
                                            + "_sub_miller_"
                                            + "_".join(map(str, subs_index))
                                            + "_film_thickness_"
                                            + str(film_thickness)
                                            + "_subs_thickness_"
                                            + str(subs_thickness)
                                            + "_seperation_"
                                            + str(seperation)
                                            + "_"
                                            + "disp_"
                                            + "_".join(map(str, dis_tmp))
                                        )
                                        # print("name", name)
                                        # print("dis_tmp", dis_tmp)
                                        # print(i, j)
                                        info1 = self.get_interface(
                                            film_atoms=i,
                                            subs_atoms=j,
                                            film_index=film_index,
                                            subs_index=subs_index,
                                            film_thickness=film_thickness,
                                            subs_thickness=subs_thickness,
                                            seperation=seperation,
                                        )

                                        """
                                        intf = info1["interface"]
                                        mis_u1 = info1["mismatch_u"]
                                        mis_v1 = info1["mismatch_v"]
                                        max_mis1 = max(
                                            abs(mis_u1), abs(mis_v1)
                                        )

                                        info2 = self.get_interface(
                                            film_atoms=j,
                                            subs_atoms=i,
                                            film_index=film_index,
                                            subs_index=subs_index,
                                            film_thickness=film_thickness,
                                            subs_thickness=subs_thickness,
                                            seperation=seperation,
                                        )
                                        intf = info2["interface"]
                                        mis_u2 = info2["mismatch_u"]
                                        mis_v2 = info2["mismatch_v"]
                                        max_mis2 = max(
                                            abs(mis_u2), abs(mis_v2)
                                        )
                                        if max_mis2 > max_mis1:
                                            chosen_info = info1
                                        else:
                                            chosen_info = info2
                                        """
                                        chosen_info = info1
                                        ats = chosen_info["interface"]
                                        film_surface_name = (
                                            "Surface-"
                                            + str(tmp_i)
                                            + "_miller_"
                                            # + "film_miller_"
                                            + "_".join(map(str, film_index))
                                            + "_thickness_"
                                            # + "_film_thickness_"
                                            + str(film_thickness)
                                        )
                                        subs_surface_name = (
                                            "Surface-"
                                            + str(tmp_j)
                                            + "_miller_"
                                            # + "subs_miller_"
                                            + "_".join(map(str, subs_index))
                                            + "_thickness_"
                                            # + "_subs_thickness_"
                                            + str(subs_thickness)
                                        )
                                        chosen_info["interface_name"] = name
                                        chosen_info["film_surface_name"] = (
                                            film_surface_name
                                        )
                                        chosen_info["subs_surface_name"] = (
                                            subs_surface_name
                                        )
                                        # print("interface1", ats)
                                        # film_sl = chosen_info["film_sl"]
                                        # subs_sl = chosen_info["subs_sl"]
                                        disp_coords = []
                                        coords = ats.frac_coords
                                        elements = ats.elements
                                        lattice_mat = ats.lattice_mat
                                        props = ats.props
                                        for m, n in zip(coords, props):
                                            if n == "bottom":
                                                m[0] += dis_tmp[0]
                                                m[1] += dis_tmp[1]
                                                disp_coords.append(m)
                                            else:
                                                disp_coords.append(m)
                                        new_intf = Atoms(
                                            coords=disp_coords,
                                            elements=elements,
                                            lattice_mat=lattice_mat,
                                            cartesian=False,
                                            props=props,
                                        )
                                        # print(
                                        #     "chosen_info",
                                        #     chosen_info["mismatch_u"],
                                        #     ats.num_atoms,
                                        # )
                                        chosen_info["generated_interface"] = (
                                            new_intf.to_dict()
                                        )
                                        chosen_info["interface"] = chosen_info[
                                            "interface"
                                        ].to_dict()
                                        chosen_info["film_kplength"] = (
                                            film_kplength
                                        )
                                        chosen_info["subs_kplength"] = (
                                            subs_kplength
                                        )
                                        gen_intfs.append(chosen_info)

                                        # print("interface2", new_intf)
        # return self.generated_interfaces
        self.generated_interfaces = gen_intfs
        return gen_intfs

    def calculate_wad(
        self, method="ewald", extra_params={}, do_surfaces=True, index=None
    ):
        x = self.generate()
        print("len generated", len(x))
        if index is None:
            generated_interfaces = self.generated_interfaces
        else:
            generated_interfaces = [self.generated_interfaces[index]]
        ew_wads = []
        for i in tqdm(generated_interfaces):
            # film_en = atom_to_energy(Atoms.from_dict(i["film_sl"]))
            # subs_en = atom_to_energy(Atoms.from_dict(i["subs_sl"]))
            film_atoms = Atoms.from_dict(i["film_surf"])
            subs_atoms = Atoms.from_dict(i["subs_surf"])
            film_sl = Atoms.from_dict(i["film_sl"])
            subs_sl = Atoms.from_dict(i["subs_sl"])
            film_surface_name = i["film_surface_name"] + "_" + method
            subs_surface_name = i["subs_surface_name"] + "_" + method
            intf_name = i["interface_name"] + "_" + method
            if do_surfaces:
                # TODO: Fix kp length as an option
                # Film
                try:
                    extra_params["kp_length"] = i["film_kplength"]
                except Exception:
                    # extra_params["kp_length"] = extra_params
                    pass
                calc = Calc(
                    method=method,
                    atoms=film_atoms,
                    extra_params=extra_params,
                    jobname=film_surface_name,
                )
                en = calc.predict()["energy"]
                film_en = (film_sl.num_atoms / film_atoms.num_atoms) * en

            if do_surfaces:
                # Substrate
                try:
                    extra_params["kp_length"] = i["subs_kplength"]
                except Exception:
                    # extra_params["kp_length"] = extra_params
                    pass
                calc = Calc(
                    method=method,
                    atoms=subs_atoms,
                    extra_params=extra_params,
                    jobname=subs_surface_name,
                )
                en = calc.predict()["energy"]
                subs_en = (subs_sl.num_atoms / subs_atoms.num_atoms) * en

            # Interface
            intf_kplength = max(i["film_kplength"], i["subs_kplength"])
            extra_params["kp_length"] = intf_kplength
            # intf_name = i["interface_name"]
            intf = Atoms.from_dict(i["generated_interface"])
            # print('intf',intf)
            calc = Calc(
                method=method,
                atoms=intf,
                extra_params=extra_params,
                jobname=intf_name,
            )
            interface_en = calc.predict()["energy"]
            wa = -9999
            if do_surfaces:
                m = intf.lattice.matrix
                area = np.linalg.norm(np.cross(m[0], m[1]))

                wa = 16 * (interface_en - subs_en - film_en) / area
            ew_wads.append(wa)
        self.wads["wads"] = ew_wads
        return ew_wads


def lead_mat_designer(
    lead="JVASP-813",
    mat="JVASP-1002",
    film_index=[1, 1, 1],
    subs_index=[0, 0, 1],
    disp_intvl=0.3,
    seperations=[2.5],
    fast_checker="ewald",
    dataset=[],
):
    jid_film = lead
    jid_subs = mat
    x = InterfaceCombi(
        dataset=dataset,
        film_indices=[film_index],
        subs_indices=[subs_index],
        film_ids=[jid_film],
        subs_ids=[jid_subs],
        disp_intvl=disp_intvl,
        seperations=seperations,
    )

    if fast_checker == "ewald":
        wads = x.calculate_wad(method="ewald")
        wads = np.array(x.wads["wads"])
    elif fast_checker == "alignn":
        wads = x.calculate_wad(method="alignn_ff")
        wads = np.array(x.wads["wads"])
    else:
        raise ValueError("Not implemented", fast_checker)

    index = np.argmin(wads)

    atoms = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )

    film_index = [0, 0, 1]
    x = InterfaceCombi(
        dataset=dataset,
        film_indices=[film_index],
        subs_indices=[subs_index],
        subs_ids=[jid_film],
        film_mats=[atoms],
        disp_intvl=disp_intvl,
        seperations=seperations,
    )

    if fast_checker == "ewald":
        wads = x.calculate_wad(method="ewald")
        wads = np.array(x.wads["wads"])
    elif fast_checker == "alignn":
        wads = x.calculate_wad(method="alignn_ff")
        wads = np.array(x.wads["wads"])
    else:
        raise ValueError("Not implemented", fast_checker)

    index = np.argmin(wads)

    combined = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    combined = combined.center(vacuum=1.5)
    lat_mat = combined.lattice_mat
    coords = combined.frac_coords
    elements = combined.elements
    props = combined.props
    tmp = lat_mat.copy()
    indx = 2
    tmp[indx] = lat_mat[0]
    tmp[0] = lat_mat[indx]
    lat_mat = tmp
    tmp = coords.copy()
    tmp[:, indx] = coords[:, 0]
    tmp[:, 0] = coords[:, indx]
    coords = tmp
    combined = Atoms(
        lattice_mat=lat_mat,
        coords=coords,
        elements=elements,
        cartesian=False,
        props=props,
    ).center_around_origin([0.5, 0, 0])
    return combined


"""
if __name__ == "__main__":
    # semicon_mat_interface_workflow()
    # metal_metal_interface_workflow()
    # semicon_mat_interface_workflow2()
    # quick_compare()
    # semicon_semicon_interface_workflow()
    dataset = j_data("dft_3d")
    # dataset2 = j_data("dft_2d")
    # dataset = dataset1 + dataset2
    x = InterfaceCombi(
        dataset=dataset,
        film_ids=["JVASP-816"],
        subs_ids=["JVASP-816"],
        film_indices=[[0, 0, 1]],
        subs_indices=[[0, 0, 1]],
        disp_intvl=0.0,
    )
    wads = x.calculate_wad(method="ewald", extra_params={})
    print("ewald wads", wads)
    wads = x.calculate_wad(
        method="eam_ase",
        extra_params={"potential": "Mishin-Ni-Al-Co-2013.eam.alloy"},
    )
    print("EAM wads", wads)
    wads = x.calculate_wad(method="matgl", extra_params={})
    print("Matgl wads", wads)
    wads = x.calculate_wad(method="alignn_ff", extra_params={})
    print("AFF wads", wads)
    wads = x.calculate_wad(method="emt", extra_params={})
    print("EMT wads", wads)
    # wads = x.calculate_wad(method="vasp", extra_params={})
    # print("EMT wads", wads)
"""
