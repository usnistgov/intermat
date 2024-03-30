#!/usr/bin/env python
"""Module to generate interface given to materials"""
import argparse
import sys
import time
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
import numpy as np

parser = argparse.ArgumentParser(
    description="Generate interface with Intermat."
)
parser.add_argument(
    "--film",
    default="POSCAR1",
    help="First file (film).",
)
parser.add_argument(
    "--substrate",
    default="POSCAR2",
    help="Second file (substrate).",
)
parser.add_argument(
    "--film_index",
    default="0_0_1",
    help="Film index",
)
parser.add_argument(
    "--substrate_index",
    default="0_0_1",
    help="substrate index",
)
parser.add_argument(
    "--film_thickness",
    default=16,
    help="Thickness of film in Angstrom",
)
parser.add_argument(
    "--substrate_thickness",
    default=16,
    help="Thickness of substrate in Angstrom",
)

parser.add_argument(
    "--seperation",
    default=2.5,
    help="Distance between substrate and film.",
)
parser.add_argument(
    "--vacuum_interface",
    default=2,
    help="Vacuum padding for interfacein Angstrom."
    + " Smaller values such as 2 gives ASJ interface,"
    + " Large values such as 8 gives STJ interfaces",
)
parser.add_argument(
    "--rotate_xz",
    default="False",
    help="Whether to rotate the interface",
)
parser.add_argument(
    "--disp_intvl",
    default=0,
    help="Whether to allow xy-plane scan."
    + "A smal value between -0.5 to 0.5 will generate multiple interfaces"
    + " and will try to find eneregtically stable one.",
)

parser.add_argument(
    "--fast_scan_method",
    default="ewald",
    help="Pre-selection method for interface scan:"
    + " ewald, alignn_ff, eam_ase  etc."
    + ". Only allowed if disp_intvl>0. "
    + "For more involved method such as DFT, docs provided later",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    film_mat = Atoms.from_poscar(args.film)
    sub_mat = Atoms.from_poscar(args.substrate)
    film_index = [int(i) for i in args.film_index.split("_")]
    subs_index = [int(i) for i in args.substrate_index.split("_")]
    rotate_xz = [True if args.rotate_xz.lower() == "true" else False]
    disp_intvl = float(args.disp_intvl)
    t1 = time.time()
    x = InterfaceCombi(
        film_mats=[film_mat],
        subs_mats=[sub_mat],
        film_indices=[film_index],
        subs_indices=[subs_index],
        disp_intvl=disp_intvl,
        film_thicknesses=[float(args.film_thickness)],
        subs_thicknesses=[float(args.substrate_thickness)],
        seperations=[float(args.seperation)],
        rotate_xz=rotate_xz,
        dataset=[None],
    )
    if disp_intvl == 0:
        print(
            "Quick interface generation with no scan."
            + " It might be energetically very high/less stable."
        )
        combined = Atoms.from_dict(x.generate()[0]["generated_interface"])

        print("Generated interface:\n", combined)
    else:
        # For more involved method such as VASP etc. docs provided later
        wads = x.calculate_wad(
            method=args.fast_scan_method,
        )
        print("len w_adhesion", wads)
        wads = np.array(x.wads["wads"])
        index = np.argmin(wads)
        combined = Atoms.from_dict(
            x.generated_interfaces[index]["generated_interface"]
        )
        print("Generated interface:\n", combined)
    t2 = time.time()
    print("Time taken:", t2 - t1)
