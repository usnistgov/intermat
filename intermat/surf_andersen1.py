# ls -altr Surf*/*/*.json |wc -l
# ls -altr Surf*/*/*/LOCPOT |wc -l
# /rk2/knc6/Surfaces/10-15/*/*.json
from numpy import ones, vstack
from numpy.linalg import lstsq
import numpy as np
import re
import matplotlib.pyplot as plt
from ase.calculators.vasp import VaspChargeDensity
from scipy.signal import find_peaks
import glob
from jarvis.io.vasp.outputs import Chgcar, Outcar, Locpot
from jarvis.db.jsonutils import loadjson
import os
from jarvis.core.atoms import ase_to_atoms
from jarvis.db.jsonutils import dumpjson

# from intermat.offset import locpot_mean
from jarvis.io.vasp.outputs import Outcar, Vasprun


def locpot_mean(
    fname="LOCPOT", axis="z", savefile="locpot.dat", outcar="OUTCAR"
):
    outcar = fname.replace("LOCPOT", "OUTCAR")
    out = Outcar(outcar)
    cbm = out.bandgap[1]  # - vac_level
    vbm = out.bandgap[2]  # - vac_level
    vrun = Vasprun(fname.replace("LOCPOT", "vasprun.xml"))
    efermi = vrun.efermi  # get_efermi(outcar)
    atoms = vrun.all_structures[-1]
    formula = atoms.composition.reduced_formula
    if atoms.check_polar:
        filename1 = (
            "Polar-"
            + fname.split("/")[0].replace("R2SCAN", "PBE")
            + "_"
            + formula
            + ".png"
        )
    else:
        filename1 = (
            fname.split("/")[0].replace("R2SCAN", "PBE")
            + "_"
            + formula
            + ".png"
        )
    dif, cbm, vbm, avg_max, efermi, formula, atoms = Locpot(
        filename=fname
    ).vac_potential(
        direction="X", Ef=efermi, cbm=cbm, vbm=vbm, filename=filename1
    )
    return dif, cbm, vbm, avg_max, float(efermi), formula, atoms


bcv = loadjson("/rk2/knc6/ALIGNN_Bands/cbm_vbm.json")


def bulk_cbm_vbm(jid):
    for i in bcv:
        if i["jid"] == jid:
            return i["cbm"], i["vbm"]


import glob

mem = []
for i in glob.glob("Surface-JVASP-*/*/*/LOCPOT"):
    # for i in glob.glob("Surface-JVASP-*/opt*/opt*.json"):
    try:
        # if "Surface-JVASP-1177_miller_1_1_0_thickness_16_VASP_PBE" in i: # and 'DP' in i:# and '1_1_0' in i:
        if "Surf" in i:  # and 'DP' in i:# and '1_1_0' in i:
            jid = i.split("_")[0].split("Surface-")[1]
            bcbm, bvbm = bulk_cbm_vbm(jid)
            dif, cbm, vbm, avg_max, efermi, formula, atoms = locpot_mean(
                i.replace(".json", "/LOCPOT")
            )
            # dif,cbm,vbm,avg_max,efermi,formula,atoms = locpot_mean(i) #.replace(".json", "/LOCPOT"))
            info = {}
            info["name"] = i.split("/")[0]
            info["formula"] = formula
            info["scf_cbm"] = bcbm  # -avg_max #cbm
            info["efermi"] = efermi  # -avg_max #cbm
            info["scf_vbm"] = bvbm  # -avg_max #vbm
            info["surf_cbm"] = cbm  # -avg_max #cbm
            info["surf_vbm"] = vbm  # -avg_max #vbm
            info["avg_max"] = avg_max  # vbm
            info["phi"] = dif
            info["atoms"] = atoms.to_dict()
            mem.append(info)
            # print(i.replace(".json", "/LOCPOT"))
            print(info)
            # out = Outcar(i.replace(".json", "/OUTCAR"))
            # vac_level = max(lc[1])
            # cbm = out.bandgap[1] - vac_level
            # vbm = out.bandgap[2] - vac_level
            # print("rep", cbm, vbm)
            print()
    except:
        pass
# dumpjson(filename='bulk_phi3a.json',data=mem)
# dumpjson(filename='bulk_phi3c.json',data=mem)
# dumpjson(filename='bulk_phi3d.json',data=mem)
# dumpjson(filename='bulk_phi3e.json',data=mem)
# dumpjson(filename='bulk_phi3f.json',data=mem)
# dumpjson(filename='bulk_phi3g.json',data=mem)
