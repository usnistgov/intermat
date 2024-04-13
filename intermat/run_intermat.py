#!/usr/bin/env python
"""Module to generate interface given to materials"""
import argparse
import sys
import time
import numpy as np
import os
import pprint
from matplotlib import cm
import matplotlib.pyplot as plt
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.config import IntermatConfig
from jarvis.db.jsonutils import loadjson
from jarvis.db.figshare import get_jid_data
from jarvis.db.jsonutils import dumpjson


def main(config_file_or_dict):
    if isinstance(config_file_or_dict, dict):
        config_dat = config_file_or_dict
    else:
        config_dat = loadjson(config_file_or_dict)
    # A few default setting check
    pprint.pprint(config_dat)
    if "lammps_params" in config_dat and not os.path.exists(
        config_dat["lammps_params"]["pair_coeff"]
    ):
        config_dat["lammps_params"]["pair_coeff"] = os.path.join(
            os.path.dirname(__file__),
            "tests",
            "Mishin-Ni-Al-Co-2013.eam.alloy",
        )
    if "lammps_params" in config_dat and not os.path.exists(
        config_dat["lammps_params"]["control_file"]
    ):
        config_dat["lammps_params"]["control_file"] = os.path.join(
            os.path.dirname(__file__), "tests", "relax.mod"
        )
    if "potential" in config_dat and not os.path.exists(
        config_dat["potential"]
    ):
        config_dat["potential"] = os.path.join(
            os.path.dirname(__file__),
            "tests",
            "Mishin-Ni-Al-Co-2013.eam.alloy",
        )
    # Make Pydantic configs
    config = IntermatConfig(**config_dat)
    pprint.pprint(config.dict())
    if config.film_file_path != "":
        film_mat = Atoms.from_poscar(config.film_file_path)
    elif config.film_jid != "":
        film_mat = Atoms.from_dict(
            get_jid_data(jid=config.film_jid, dataset=config.dataset)["atoms"]
        )
    else:
        raise ValueError("Enter a valid film_file_path or film_jid")
    if config.substrate_file_path != "":
        sub_mat = Atoms.from_poscar(config.substrate_file_path)
    elif config.substrate_jid != "":
        sub_mat = Atoms.from_dict(
            get_jid_data(jid=config.substrate_jid, dataset=config.dataset)[
                "atoms"
            ]
        )
    else:
        raise ValueError("Enter a valid substrate_file_path or substrate_jid")
    film_index = [int(i) for i in config.film_index.split("_")]
    subs_index = [int(i) for i in config.substrate_index.split("_")]
    rotate_xz = config.rotate_xz
    disp_intvl = config.disp_intvl
    t1 = time.time()
    x = InterfaceCombi(
        film_mats=[film_mat],
        subs_mats=[sub_mat],
        film_indices=[film_index],
        subs_indices=[subs_index],
        disp_intvl=disp_intvl,
        film_thicknesses=[float(config.film_thickness)],
        subs_thicknesses=[float(config.substrate_thickness)],
        seperations=[float(config.seperation)],
        rotate_xz=rotate_xz,
        dataset=[None],
        vacuum_interface=config.vacuum_interface,
    )

    combined_atoms = [
        Atoms.from_dict(i["generated_interface"]) for i in x.generate()
    ]
    print("Number of generated interface: ", len(combined_atoms))
    if disp_intvl == 0:
        print(
            "Quick interface generation with no scan."
            + " It might be energetically very high/less stable."
        )
        print(combined_atoms[0])
        combined_atoms = combined_atoms[0]
    else:
        if config.verbose:
            for ii, i in enumerate(combined_atoms):
                print("Structure ", ii)
                print(i)
                print()
    wads = ""
    if config.calculator_method != "":
        # print("combined_atoms", combined_atoms)
        print("config.calculator_method", config.calculator_method)
        wads = x.calculate_wad(
            method=config.calculator_method,
            do_surfaces=config.do_surfaces,
            extra_params=config.dict(),
        )
        print("w_adhesion (J/m2)", wads)
        wads = np.array(x.wads["wads"])
        index = np.argmin(wads)
        combined_atoms = Atoms.from_dict(
            x.generated_interfaces[index]["generated_interface"]
        )
        # print("Generated interface:\n", combined_atoms)
        if config.plot_wads and config.disp_intvl != 0:
            # xy = np.array(x.xy)
            # print('xy', xy)
            X = x.X
            Y = x.Y
            # X = xy[:,0]
            # Y = xy[:,1]
            wads = np.array(wads).reshape(len(X), len(Y))

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(
                X, Y, wads, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.savefig("intmat.png")
            plt.close()
            # import plotly.graph_objects as go
            # fig = go.Figure(data=[go.Surface(z=wads, x=X, y=Y)])
            # fig.show()
            wads = wads.tolist()
    t2 = time.time()
    print("Time taken:", t2 - t1)
    info = {}
    # print("combined_atoms", combined_atoms)
    info["systems"] = combined_atoms.to_dict()
    info["time_taken"] = t2 - t1
    info["wads"] = wads
    print("info", info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interface with InterMat."
    )
    parser.add_argument(
        "--config_file",
        default="config.json",
        help="Settings file for intermat.",
    )
    args = parser.parse_args(sys.argv[1:])
    results = main(config_file_or_dict=args.config_file)
    dumpjson(data=results, filename="intermat_results.json")
