#!/usr/bin/env python
"""Module to generate interface given to materials"""
import argparse
import sys
import time
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.config import IntermatConfig
import numpy as np
from jarvis.db.jsonutils import loadjson
import pprint
from jarvis.db.figshare import get_jid_data

parser = argparse.ArgumentParser(
    description="Generate interface with Intermat."
)
parser.add_argument(
    "--config_file",
    default="config.json",
    help="Settings file for intermat.",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    config_dat = loadjson(args.config_file)
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
    else:
        if config.verbose:
            for ii, i in enumerate(combined_atoms):
                print("Structure ", ii)
                print(i)
                print()

    if config.calculator_method != "":
        print("combined_atoms", combined_atoms)
        # extra_params=config.qe_params|config.lammps_params|config.gpaw_params|config.vasp_params
        # print('extra_params',pprint.pprint(extra_params))
        # if config.calculator_method=='qe':
        #       extra_params=config.qe_params
        wads = x.calculate_wad(
            method=config.calculator_method,
            do_surfaces=config.do_surfaces,
            extra_params=config.dict(),
        )
        print("w_adhesion (J/m2)", wads)
        wads = np.array(x.wads["wads"])
        index = np.argmin(wads)
        combined = Atoms.from_dict(
            x.generated_interfaces[index]["generated_interface"]
        )
        print("Generated interface:\n", combined)
    t2 = time.time()
    print("Time taken:", t2 - t1)
