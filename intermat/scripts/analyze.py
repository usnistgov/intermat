##!/bin/bash
# Surface-JVASP-1002_film_miller_1_0_0_film_thickness_15_VASP/Surface-JVASP-1002_film_miller_1_0_0_film_thickness_15_VASP/CONTCAR
from jarvis.io.vasp.outputs import Vasprun, Locpot
from jarvis.db.figshare import data
import glob
import numpy as np

d = data("dft_3d")

x = []
for i in glob.glob("Surf*/Surf*.json"):
    vrun_file = i.split(".json")[0] + "/vasprun.xml"
    locpot_file = i.split(".json")[0] + "/LOCPOT"
    locpot = Locpot(filename=locpot_file)
    vrun = Vasprun(vrun_file)
    efermi = vrun.efermi
    filename = i.split("/")[0] + "_Avg.png"
    avg_max, wf = locpot.vac_potential(
        Ef=efermi, direction="Z", filename=filename
    )
    atoms = vrun.all_structures[-1]
    jid = i.split("_")[0].split("Surface-")[1]
    perf_en = "na"
    for ii in d:
        if jid == ii["jid"]:
            perf_en = ii["optb88vdw_total_energy"] * atoms.num_atoms
            break
    m = atoms.lattice.matrix
    area = np.linalg.norm(np.cross(m[0], m[1]))
    surf_en = 16.022 * (vrun.final_energy - perf_en) / 2 / area
    print(i, jid, surf_en)
    x.append(i)
print(len(x))
y = []
for i in glob.glob("VASP_Interface-JVASP-*/VASP*.json"):
    vrun_file = i.split(".json")[0] + "/vasprun.xml"
    vrun = Vasprun(vrun_file)
    atoms = vrun.all_structures[-1]
    fin_en = vrun.final_energy
    m = atoms.lattice.matrix
    tmp = i.split("VASP_Interface-")[1].split("_")
    film_id = tmp[0]
    subs_id = tmp[1]
    film_miller = tmp[4] + "_" + tmp[5] + "_" + tmp[6] + "_"
    subs_miller = tmp[9] + "_" + tmp[10] + "_" + tmp[11] + "_"
    film_thickness = tmp[14]
    subs_thickness = tmp[17]
    film_name = (
        "Surface-"
        + film_id
        + "_film_miller_"
        + film_miller
        + "film_thickness_"
        + film_thickness
        + "_VASP"
    )
    subs_name = (
        "Surface-"
        + subs_id
        + "_subs_miller_"
        + subs_miller
        + "subs_thickness_"
        + subs_thickness
        + "_VASP"
    )
    film_name = film_name + "/" + film_name + "/vasprun.xml"
    subs_name = subs_name + "/" + subs_name + "/vasprun.xml"
    film_en = Vasprun(film_name).final_energy
    subs_en = Vasprun(subs_name).final_energy
    area = np.linalg.norm(np.cross(m[0], m[1]))
    intf_en = 16.022 * (fin_en - film_en - subs_en) / area
    print("intf_en", i, intf_en)
    locpot_file = i.split(".json")[0] + "/LOCPOT"
    locpot = Locpot(filename=locpot_file)
    efermi = vrun.efermi
    filename = i.split("/")[0] + "_Avg.png"
    avg_max, wf = locpot.vac_potential(
        Ef=efermi, direction="Z", filename=filename
    )
    y.append(i)
print(len(y))

# for i in glob.glob("VASP_Interface-JVASP-*/*/vasp.out"):
#    print(i)
