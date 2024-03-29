# from intermat.known_mats import semicons
from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.offset import offset, locpot_mean
import time

# semicons = semicons() # e.g. 1002 for silicon


def test_surface():
    semicons = [
        1002,
        890,
    ]
    for i in semicons:
        jid = "JVASP-" + str(i)
        atoms = get_jid_atoms(jid=jid)
        if atoms is not None:
            atoms = Atoms.from_dict(atoms)

            spg = Spacegroup3D(atoms=atoms)
            cvn = spg.conventional_standard_structure
            mills = symmetrically_distinct_miller_indices(
                max_index=1, cvn_atoms=cvn
            )
            for miller in mills:
                surf = Surface(
                    atoms,
                    indices=miller,
                    from_conventional_structure=True,
                    thickness=16,
                    vacuum=12,
                ).make_surface()
                # Surface-JVASP-105933_miller_1_1_0
                nm = "Surface-" + jid + "_miller_" + "_".join(map(str, miller))
                if not surf.check_polar and "-1" not in nm:
                    non_polar_semi.append(nm)
                    if len(non_polar_semi) % 100 == 0:
                        t2 = time.time()
                        print(len(non_polar_semi), t2 - t1)
                        t1 = time.time()


def test_interface():

    # Step-1: prepare and submit calculations
    combinations = [["JVASP-1002", "JVASP-1174", [1, 1, 0], [1, 1, 0]]]
    for i in combinations:
        tol = 1
        seperations = [2.5]  # can have multiple separations
        # Interface generator class
        x = InterfaceCombi(
            film_ids=[i[0]],
            subs_ids=[i[1]],
            film_indices=[i[2]],
            subs_indices=[i[3]],
            disp_intvl=0.05,
            vacuum_interface=2,
        )
        # Fast work of adhesion with Ewald/ALIGNN-FF
        wads = x.calculate_wad(method="ewald")
        wads = x.wads["wads"]
        index = np.argmin(wads)
        combined = Atoms.from_dict(
            x.generated_interfaces[index]["generated_interface"]
        )
        combined = combined.center(vacuum=seperations[0] - tol)
        print(index, combined)
