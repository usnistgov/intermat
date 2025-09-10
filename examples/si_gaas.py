from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.offset import offset, locpot_mean

#
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
    # Cluster/account specific job submission lines
    extra_lines = (
        ". ~/.bashrc\nmodule load vasp/6.3.1\n"
        + "conda activate mini_alignn\n"
    )

    info["inc"]["ISIF"] = 3
    info["inc"]["ENCUT"] = 520
    info["inc"]["NEDOS"] = 5000
    info["queue"] = "rack1"
    # VASP job submission,
    wads = x.calculate_wad(
        method="vasp",
        index=index,
        do_surfaces=False,
        extra_params=info,
    )
    # Other calculators such as QE, TB3, ALIGNN etc.
    # are also available

# Step-2: Analysis
# Once calculations are converged
# We can calculate properties such as
# bandgap, band offset, interfac energy for interfaces
# &  electron affinity, surface energy,
# ionization potential etc. for surfaces
#
# Path to LOCPOT file
fname = "Interface-JVASP-1002_JVASP-1174_film_miller_1_1_0_sub_miller_1_1_0_film_thickness_16_subs_thickness_16_seperation_2.5_disp_0.5_0.2_vasp/*/*/LOCPOT"
ofs = offset(fname=fname, left_index=-1, polar=False)
print(ofs)
# Note for interfaces we require LOCPOTs for bulk materials of the two materials as well
# For surface properties
dif, cbm, vbm, avg_max, efermi, formula, atoms, fin_en = locpot_mean(
    "PATH_TO_LOCPOT"
)
