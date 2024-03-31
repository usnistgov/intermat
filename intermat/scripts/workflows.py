"""Module to generate interface combinations."""

from jarvis.analysis.interface.zur import ZSLGenerator
from jarvis.core.atoms import fix_pbc
from jarvis.core.lattice import Lattice, lattice_coords_transformer
import os
from jarvis.db.figshare import data as j_data
import numpy as np
from jarvis.analysis.defects.surface import Surface
from jarvis.core.atoms import Atoms
import pandas as pd
import time
from intermat.calculators import Calc
from intermat.calculators import Calc
from tqdm import tqdm
from intermat.generate import InterfaceCombi



def metal_metal_interface_workflow():
    # df = pd.read_csv("Interface.csv")
    df = pd.read_csv("Interface_metals.csv")
    dataset1 = j_data("dft_3d")
    dataset2 = j_data("dft_2d")
    dataset = dataset1 + dataset2
    for i, ii in df.iterrows():
        film_ids = []
        subs_ids = []
        film_indices = []
        subs_indices = []
        # try:
        film_ids.append("JVASP-" + str(ii["JARVISID-Film"]))
        subs_ids.append("JVASP-" + str(ii["JARVISID-Subs"]))
        film_indices.append(
            [
                int(ii["Film-miller"][1]),
                int(ii["Film-miller"][2]),
                int(ii["Film-miller"][3]),
            ]
        )
        subs_indices.append(
            [
                int(ii["Subs-miller"][1]),
                int(ii["Subs-miller"][2]),
                int(ii["Subs-miller"][3]),
            ]
        )
        print(film_indices[-1], subs_indices[-1], film_ids[-1], subs_ids[-1])
        x = InterfaceCombi(
            dataset=dataset,
            film_indices=film_indices,
            subs_indices=subs_indices,
            film_ids=film_ids,
            subs_ids=subs_ids,
            disp_intvl=0.1,
        )
        wads = x.calculate_wad(method="ewald")
        wads = np.array(x.wads["wads"])
        index = np.argmin(wads)
        wads = x.calculate_wad(method="vasp", index=index)
        # wads = x.calculate_wad_vasp(sub_job=True)
    # except:
    #  pass


def semicon_mat_interface_workflow():
    dataset = j_data("dft_3d")
    # Cu(867),Al(816),Ni(943),Pt(972),Cu(816),Ti(1029),
    # Pd(963),Au(825),Ag(813),Hf(802), Nb(934)
    combinations = [
        ["JVASP-1002", "JVASP-867", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-867", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-867", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-867", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-867", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-867", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-867", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-816", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-816", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-816", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-816", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-816", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-816", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-816", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-943", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-943", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-943", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-943", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-943", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-943", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-943", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-972", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-972", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-972", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-972", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-972", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-972", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-972", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-867", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-867", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-867", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-867", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-867", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-867", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-867", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1029", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1029", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1029", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1029", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1029", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1029", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1029", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-963", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-963", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-963", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-867", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-963", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-963", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-963", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-825", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-825", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-825", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-825", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-825", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-825", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-825", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-813", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-813", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-813", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-813", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-813", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-813", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-813", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-802", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-802", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-802", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-802", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-802", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-802", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-802", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-41", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-41", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-41", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-41", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-41", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-41", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-41", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-8158", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8158", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-8158", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-8158", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8158", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8158", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-8158", [1, 1, 0], [1, 1, 0]],
    ]

    for i in combinations:
        x = InterfaceCombi(
            dataset=dataset,
            film_indices=[i[2]],
            subs_indices=[i[3]],
            film_ids=[i[0]],
            subs_ids=[i[1]],
            disp_intvl=0.1,
        )
        wads = x.calculate_wad(method="ewald")
        wads = np.array(x.wads["wads"])
        index = np.argmin(wads)
        wads = x.calculate_wad(method="vasp", index=index)
        # wads = x.calculate_wad_vasp(sub_job=True)


def semicon_mat_interface_workflow2():
    dataset = j_data("dft_3d")
    # Cu(867),Al(816),Ni(943),Pt(972),Cu(816),Ti(1029),Pd(963),
    # Au(825),Ag(813),Hf(802), Nb(934)
    # Ge(890), AlN(39), GaN(30), BN(62940), CdO(20092), CdS(8003),
    # CdSe(1192), CdTe(23), ZnO(1195), ZnS(96), ZnSe(10591),
    # ZnTe(1198), BP(1312), BAs(133719),
    # BSb(36873), AlP(1327), AlAs(1372), AlSb(1408), GaP(8184),
    # GaAs(1174), GaSb(1177), InN(1180), InP(1183), InAs(1186),
    # InSb(1189), C(91), SiC(8158,8118,107), GeC(36018),
    # SnC(36408), SiGe(105410), SiSn(36403), , Sn(1008)
    combinations = [
        ["JVASP-1002", "JVASP-890", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-890", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-890", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-39", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-39", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-39", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-39", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-62940", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-62940", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-62940", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-20092", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-20092", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-20092", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-8003", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-8003", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8003", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1192", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1192", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1192", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-23", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-23", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-23", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1195", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1195", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1195", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-96", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-96", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-96", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-10591", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-10591", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-10591", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1198", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1198", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1198", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1312", [1, 1, 0], [1, 1, 0]],
        ######################################################################
        ["JVASP-1002", "JVASP-1312", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1312", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-133719", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-133719", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-133719", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36873", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-36873", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-36873", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1327", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1327", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1327", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1372", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1372", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1372", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1372", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1408", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1408", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1408", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8184", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8184", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-8184", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1174", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1174", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1174", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1177", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1177", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1177", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1180", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1180", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1180", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1183", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1183", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1183", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1186", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1186", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1186", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1189", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1189", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1189", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-91", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-91", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-91", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-8118", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8118", [0, 0, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-8118", [0, 0, 1], [1, 1, 0]],
        ["JVASP-1002", "JVASP-107", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-107", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-107", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-36018", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-36018", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36018", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-36408", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-36408", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36408", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-105410", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-105410", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-105410", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-36403", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-36403", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36403", [1, 1, 1], [1, 1, 1]],
        ["JVASP-1002", "JVASP-1008", [1, 1, 0], [1, 1, 0]],
        ["JVASP-1002", "JVASP-1008", [1, 1, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1008", [1, 1, 1], [1, 1, 1]],
    ]

    for i in combinations:
        try:
            x = InterfaceCombi(
                dataset=dataset,
                film_indices=[i[2]],
                subs_indices=[i[3]],
                film_ids=[i[0]],
                subs_ids=[i[1]],
                disp_intvl=0.1,
            )
            wads = x.calculate_wad(method="ewald")
            wads = np.array(x.wads["wads"])
            index = np.argmin(wads)
            wads = x.calculate_wad(method="vasp", index=index)
            # wads = x.calculate_wad_vasp(sub_job=True)
        except Exception as exp:
            print("exp", exp)
            pass


def quick_test():
    box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
    coords = [[0, 0, 0], [0.25, 0.2, 0.25]]
    elements = ["Si", "Si"]
    atoms_si = Atoms(lattice_mat=box, coords=coords, elements=elements)

    box = [[1.7985, 1.7985, 0], [0, 1.7985, 1.7985], [1.7985, 0, 1.7985]]
    coords = [[0, 0, 0]]
    elements = ["Ag"]
    atoms_cu = Atoms(lattice_mat=box, coords=coords, elements=elements)
    x = InterfaceCombi(
        film_mats=[atoms_cu],
        subs_mats=[atoms_si],
        film_indices=[[0, 0, 1]],
        subs_indices=[[0, 0, 1]],
        vacuum_interface=2,
        film_ids=["JVASP-867"],
        subs_ids=["JVASP-816"],
        # disp_intvl=0.1,
    ).generate()
    print("x", len(x))


def semicon_semicon_interface_workflow():
    dataset = j_data("dft_3d")
    # Cu(867),Al(816),Ni(943),Pt(972),Cu(816),Ti(1029),Pd(963),
    # Au(825),Ag(813),Hf(802), Nb(934)
    # Ge(890), AlN(39), GaN(30), BN(62940), CdO(20092), CdS(8003),
    # CdSe(1192), CdTe(23), ZnO(1195), ZnS(96), ZnSe(10591), ZnTe(1198),
    # BP(1312), BAs(133719),
    # BSb(36873), AlP(1327), AlAs(1372), AlSb(1408), GaP(8184), GaAs(1174),
    # GaSb(1177), InN(1180), InP(1183), InAs(1186), InSb(1189), C(91),
    # SiC(8158,8118,107), GeC(36018), SnC(36408),
    # SiGe(105410), SiSn(36403), , Sn(1008)
    combinations = [
        ["JVASP-1372", "JVASP-1174", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-39", "JVASP-30", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-1327", "JVASP-8184", [0, 0, 1], [1, 1, 0]],
        # ["JVASP-39", "JVASP-8184", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-890", [0, 0, 1], [0, 0, 1]],
    ]

    for i in combinations:
        try:
            x = InterfaceCombi(
                dataset=dataset,
                film_indices=[i[2]],
                subs_indices=[i[3]],
                film_ids=[i[0]],
                subs_ids=[i[1]],
                disp_intvl=0.1,
            )
            wads = x.calculate_wad(method="ewald")
            wads = np.array(x.wads["wads"])
            index = np.argmin(wads)
            wads = x.calculate_wad(method="vasp", index=index)
        except Exception as exp:
            print("exp", exp)
            pass


def quick_compare(
    jid_film="JVASP-1002",
    jid_subs="JVASP-1002",
    film_index=[0, 0, 1],
    subs_index=[0, 0, 1],
    disp_intvl=0.1,
    seperations=[1.5, 2.5, 3.5],
):
    dataset = j_data("dft_3d")
    info = {}
    x = InterfaceCombi(
        dataset=dataset,
        film_indices=[film_index],
        subs_indices=[subs_index],
        film_ids=[jid_film],
        subs_ids=[jid_subs],
        disp_intvl=disp_intvl,
        seperations=seperations,
    )
    t1 = time.time()
    # model_path = "temp1"
    wads = x.calculate_wad(method="alignn")
    # wads = x.calculate_wad_alignn(model_path=model_path)
    wads = np.array(x.wads["alignn_wads"])
    index = np.argmin(wads)
    index_al = index
    atoms = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    print(atoms)
    print("wads", wads)
    print(wads[index])
    t2 = time.time()
    t_alignn = t2 - t1
    info["alignn_wads"] = wads
    info["t_alignn"] = t_alignn
    info["index_alignn"] = index_al
    info["alignn_min_wads"] = wads[index]

    t1 = time.time()
    wads = x.calculate_wad(method="matgl")  # model_path=model_path)
    wads = np.array(x.wads["wads"])
    atoms = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    print(atoms)
    print("wads", wads)
    print(wads[index])
    index_matg = index
    t2 = time.time()
    t_matgl = t2 - t1
    info["matgl_wads"] = wads
    info["t_matgl"] = t_matgl
    info["index_matg"] = index_matg
    info["matgl_min_wads"] = wads[index]

    t1 = time.time()
    wads = x.calculate_wad(method="ewald")  # model_path=model_path)
    wads = np.array(x.wads["wads"])
    atoms = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    print(atoms)
    print("wads", wads)
    print(wads[index])
    index_ew = index
    t2 = time.time()
    t_ew = t2 - t1
    info["ew_wads"] = wads
    info["t_ew"] = t_ew
    info["index_ew"] = index_ew
    info["ewald_min_wads"] = wads[index]

    # wads = x.calculate_wad_vasp(sub_job=False)

    print(
        "index_al,index_matg",
        index_al,
        index_matg,
        index_ew,
        t_alignn,
        t_matgl,
        t_ew,
    )
    return info


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
