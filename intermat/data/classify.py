from jarvis.db.jsonutils import loadjson

from jarvis.analysis.interface.zur import get_hetero_type
import alignn, os, torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from jarvis.db.jsonutils import loadjson
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.db.jsonutils import dumpjson
from jarvis.analysis.defects.surface import Surface
from sklearn.metrics import classification_report, precision_score
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph

# https://en.wikipedia.org/wiki/Anderson%27s_rule
import matplotlib
from jarvis.core.atoms import Atoms
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import glob
from jarvis.db.jsonutils import loadjson
import time

# torch.cuda.is_available = lambda : False
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
dft_3d = data("dft_3d")
d = loadjson("ALIGNN/mill_1.json")


def get_model(model_path="/wrk/knc6/CBM_VBM/Out_VBM_3D"):
    cnfg = os.path.join(model_path, "config.json")
    config = loadjson(cnfg)
    model = ALIGNN(ALIGNNConfig(**config["model"]))
    filename = os.path.join(model_path, "best_model.pt")
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model


def get_vbm():
    model_vbm = get_model()
    rel_val = 10
    for i in dft_3d:
        try:
            gap = i["optb88vdw_bandgap"]
            mbj_gap = i["mbj_bandgap"]
            if gap > 0.1 and gap < 6:
                jid = i["jid"]
                atoms = Atoms.from_dict(i["atoms"])
                spg = Spacegroup3D(atoms=atoms)
                cvn = spg.conventional_standard_structure
                mills = symmetrically_distinct_miller_indices(
                    max_index=1, cvn_atoms=cvn
                )
                for m in mills:
                    # try:
                    fname = jid + "_" + "_".join(map(str, m)) + ".json"
                    if not os.path.exists(fname):
                        surf = Surface(
                            atoms,
                            indices=m,
                            from_conventional_structure=True,
                            thickness=16,
                            vacuum=12,
                        ).make_surface()
                        g, lg = Graph.atom_dgl_multigraph(surf)
                        alignn_vbm = (
                            model_vbm([g.to(device), lg.to(device)])
                            .cpu()
                            .detach()
                            .numpy()
                            .tolist()
                        )
                        x_vbm = (
                            alignn_vbm - rel_val
                        )  # actually cbm, alignn_vbm_rel
                        x_cbm = x_vbm - gap
                        x_mbj_cbm = "na"
                        if mbj_gap != "na":
                            x_mbj_cbm = x_vbm - mbj_gap

                        info = {}
                        info["jid"] = jid
                        info["formula"] = i["formula"]
                        # info["atoms"] = surf.to_dict()
                        info["alignn_vbm_raw"] = alignn_vbm
                        info["miller"] = "_".join(map(str, m))
                        info["alignn_vbm_rel"] = x_vbm  # al_cbms
                        info["alignn_vbm_gapped"] = x_cbm  # al_vbm
                        info["alignn_vbm_mbj_gapped"] = x_mbj_cbm
                        # print(info)
                        dumpjson(data=info, filename=fname)
                # except:
                #    pass
        except:
            print("Failed", i["jid"])
            pass


# # get_vbm()

# x = []
# for i in glob.glob("*.json"):
#     x.append(i)
# count = 0
# c = 0
# t1 = time.time()
# fname = "Pred-" + str(c) + ".csv"
# f = open(fname, "w")
# f.write("name,int_type,stack\n")
# for ii, i in enumerate(x):
#     for jj, j in enumerate(x):
#         if ii > jj:  # and ii<5 :
#             d1 = loadjson(i)
#             d2 = loadjson(j)

#             A_DFT = {}
#             A_DFT["scf_vbm"] = round(d1["alignn_vbm_gapped"], 3)
#             A_DFT["scf_cbm"] = round(d1["alignn_vbm_rel"], 3)
#             A_DFT["avg_max"] = 0

#             B_DFT = {}
#             B_DFT["scf_vbm"] = round(d2["alignn_vbm_gapped"], 3)
#             B_DFT["scf_cbm"] = round(d2["alignn_vbm_rel"], 3)
#             B_DFT["avg_max"] = 0
#             int_type1, stack = get_hetero_type(A=A_DFT, B=B_DFT)
#             if int_type1 == "I":
#                 int_type1 = 0
#             if int_type1 == "II":
#                 int_type1 = 1
#             if int_type1 == "III":
#                 int_type1 = 2
#             count += 1
#             if count % 1000000 == 0:
#                 c += 1
#                 fname = "Pred-" + str(c) + ".csv"
#                 f = open(fname, "w")
#                 f.write("name,int_type,stack\n")
#                 t2 = time.time()
#                 print("count", count, t2 - t1)
#                 t1 = time.time()

#             line = (
#                 i.split(".json")[0].split("JVASP-")[1]
#                 + ";"
#                 + j.split(".json")[0].split("JVASP-")[1]
#                 + ","
#                 + str(int_type1)
#                 + ","
#                 + str(stack)
#                 + "\n"
#             )
#             # line=i.split('.json')[0]+';'+j.split('.json')[0]+','+str(A_DFT["scf_vbm"])+','+str(A_DFT["scf_cbm"])+','+str(B_DFT["scf_vbm"])+','+str(B_DFT["scf_cbm"])+','+str(int_type1)+','+str(stack)+'\n'

#             # print(line)
#             f.write(line)
# f.close()


import time
from collections import defaultdict

info = defaultdict(list)
count = 0
t1 = time.time()
for ii, d1 in enumerate(d):
    for jj, d2 in enumerate(d):
        if ii > jj:  # and ii<5 :
            # d1 = loadjson(i)
            # d2 = loadjson(j)

            A_DFT = {}
            A_DFT["scf_vbm"] = round(d1["alignn_vbm_gapped"], 3)
            A_DFT["scf_cbm"] = round(d1["alignn_vbm_rel"], 3)
            A_DFT["avg_max"] = 0

            B_DFT = {}
            B_DFT["scf_vbm"] = round(d2["alignn_vbm_gapped"], 3)
            B_DFT["scf_cbm"] = round(d2["alignn_vbm_rel"], 3)
            B_DFT["avg_max"] = 0
            int_type1, stack = get_hetero_type(A=A_DFT, B=B_DFT)
            if int_type1 == "I":
                int_type1 = 0
            if int_type1 == "II":
                int_type1 = 1
            if int_type1 == "III":
                int_type1 = 2
            # print ('int_type1',int_type1)
            name = (
                d1["jid"] + "_" + d1["miller"] + d2["jid"] + "_" + d2["miller"]
            )
            info[int_type1].append(name)
            count += 1
            if count % 100000000 == 0:
                t2 = time.time()
                print(
                    "Num, 0,1,2,t",
                    count,
                    len(info[0]),
                    len(info[1]),
                    len(info[2]),
                    round(t2 - t1, 3),
                )
                t1 = time.time()

                # import sys
                # sys.exit()
