from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.job_manager import compile_jobs, get_json_data

# 34
# Si(JVASP-1002),Ge(JVASP-890), AlN(JVASP-39), GaN(JVASP-30), BN(JVASP-62940), CdO(JVASP-20092)
# CdS(JVASP-8003), CdSe(JVASP-1192), CdTe(JVASP-23), ZnO(JVASP-1195), ZnS(JVASP-96),
# ZnSe(JVASP-10591), ZnTe(JVASP-1198), BP(JVASP-1312), BAs(JVASP-133719),BSb(JVASP-36873),
# AlP(JVASP-1327), AlAs(JVASP-1372), AlSb(JVASP-1408), GaP(JVASP-8184), GaAs(JVASP-1174),
# GaSb(JVASP-1177), InN(JVASP-1180), InP(JVASP-1183), InAs(JVASP-1186), InSb(JVASP-1189),
# C(JVASP-91), SiC(8158),SiC(JVASP-8118),SiC(JVASP-107), GeC(JVASP-36018), SnC(JVASP-36408),
# SiGe(JVASP-105410), SiSn(JVASP-36403), Sn(JVASP-1008)
# x1=[1002, 890, 39, 30, 62940, 20092, 8003, 1192, 23, 1195, 96, 10591, 1198, 1312, 133719, 36873, 1327, 1372, 1408, 8184, 1174, 1177, 1180, 1183, 1186, 1189, 91, 8158, 8118, 107, 36018, 36408, 105410, 36403, 1008]

# 70 others from wiki
# S(JVASP-95268	),Se(JVASP-21211),Te(JVASP-1023),BN(JVASP-7836)
# B6As(JVASP-9166),CuCl(JVASP-1201),Cu2S(JVASP-85478),PbSe(JVASP-1115)
# PbS(JVASP-1112),PbTe(JVASP-1103),SnS(JVASP-1109),SnS2(JVASP-131)
# SnTe(JVASP-149916),PbSnTe(JVASP-111005),Bi2Te3(JVASP-25),Bi2Se3(JVASP-1067)
# Bi2S3(JVASP-154954),Cd3P2(JVASP-59712),Cd3As2(JVASP-10703),
# Zn3P2(JVASP-1213), ZnP2(JVASP-19007), ZnAs(JVASP-10114), ZnSb(JVASP-9175)
# TiO2(JVASP-104),TiO2(JVASP-10036),TiO2(JVASP-18983),Cu2O(JVASP-1216)
# CuO(JVASP-79522),UO2(JVASP-1222),SnO2(JVASP-10037), BaTiO3(JVASP-110)
# SrTiO3(JVASP-8082),LiNbO3(JVASP-1240),VO2(JVASP-51480),PbI2(JVASP-29539)
# MoS2(JVASP-54),GaSe(JVASP-29556),InSe(JVASP-1915),GaMn2As(JVASP-75662),
# FeO(JVASP-101764),NiO(JVASP-22694),CrBr3(JVASP-4282),CrI3(JVASP-76195)
# CuInSe2(JVASP-8554),AgGaS2(JVASP-149871),ZnSiP2(JVASP-2376),As2S3(JVASP-14163)
# AsS(JVASP-26248),PtSi(JVASP-18942),BiI3(JVASP-3510),HgI2(JVASP-5224),TlBr(JVASP-8559)
# Ag2S(JVASP-85416),FeS(JVASP-9117), Cu2ZnSnS4(JVASP-90668),Cu2SnS3(JVASP-10689)
# AlGaAs2(JVASP-106381), InGaAs(JVASP-108773), InGaP(JVASP-101184),AlInSb2(JVASP-103127)
# Al3GaN4(JVASP-104764), InGa3N4(JVASP-102336), InGaN2(JVASP-110231),InGaSb2(JVASP-108770)
# CdZnTe(JVASP-101074),CdZnTe(JVASP-149906),HgCdTe(JVASP-99732),HgZnTe2(JVASP-106686)
# HgSnSe2(JVASP-110952),Cu2InGaSe4(JVASP-106363),

# semicons=[1002, 890, 39, 30, 62940, 20092, 8003, 1192, 23, 1195, 96, 10591, 1198, 1312, 133719, 36873, 1327, 1372, 1408, 8184, 1174, 1177, 1180, 1183, 1186, 1189, 91, 8158, 8118, 107, 36018, 36408, 105410, 36403, 1008, 95268, 21211, 1023, 7836,  9166, 1201,  85478, 1115, 1112, 1103, 1109,  131, 149916, 111005,  25,  1067,  154954,  59712,  10703,  1213,  19007, 10114, 9175,  104,  10036,  18983,  1216, 79522,  1222,  10037,  110,  8082,  1240,  51480,  29539,  54, 29556, 1915,  75662, 101764, 22694,  4282,  76195,  8554,  149871,  2376,  14163, 26248, 18942,  3510, 5224, 8559,  85416, 9117,  90668,  10689,  106381, 108773, 101184,  103127,  104764,  102336,  110231,  108770, 101074, 149906, 99732,  106686,  110952, 106363]

# Metals
# Pt(JVASP-972),Au(JVASP-825),Ag(JVASP-813),Al(JVASP-816),
# Hf(JVASP-802),Ti(JVASP-1029),Cr(JVASP-861),Ni(JVASP-943),
# Pd(JVASP-963),Li(JVASP-14616),Cu(JVASO-867),TiSi2(JVASP-14968)
# MoSi2(JVASP-14970),WSi2(JVASP-19780)
# metals=[972, 825, 813, 816, 802, 1029, 861, 943, 963, 14616, 867, 2, 14968,  14970,  19780]


# Insulators
# HfO2(JVASP-9147), HfO2(JVASP-34249),HfO2(JVASP-43367),
# ZrO2(JVASP-113), SiO2(JVASP-41),SiO2(JVASP-58349),SiO2(JVASP-34674)
# SiO2(JVASP-34656), HfO2(34249), Al2O3(JVASP-32)
# insul=[9147,  34249,  43367, 113, 41,  58349,  34674,  34656, 34249,  32]

# To Add:CuO,Zn3As2,Zn3Sb2,PbSnTe,Bi2S3,GaMnAs,Ag2S
# PbMnTe,La0.7Ca0.3MnO3,EuO,GaAsN, GaAsP, GaAsSb, AlGaP,InAsSb,
# AlGaInP, AlGaAsP,InGaAsP,InGaAsSb,AlInAsP,AlGaAsSn,InGaAsSn,In2O3
info = template_extra_params(method="vasp")


def semi_flow(sub=None):
    combinations = [
        # ["JVASP-1372", "JVASP-1174", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-39", "JVASP-30", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-39", "JVASP-8184", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-1002", "JVASP-890", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-1327", "JVASP-8184", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-825", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-813", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-816", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-802", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1029", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-972", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-861", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-943", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-963", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-39", [1, 0, 0], [0, 0, 1]],
        ["JVASP-1002", "JVASP-62940", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-20092", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8003", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1192", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-23", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1195", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-96", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-10591", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1198", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1312", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-133719", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36873", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1372", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1408", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8184", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1174", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1177", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1180", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1183", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1186", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1189", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-91", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8118", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-107", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36408", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-105410", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36403", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-1008", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8158", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36018", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-8158", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-36018", [0, 0, 1], [0, 0, 1]],
    ]

    jids1 = [
        "JVASP-1002",
        "JVASP-890",
        "JVASP-1327",
        "JVASP-1372",
        "JVASP-1408",
        "JVASP-8184",
        "JVASP-1174",
        "JVASP-1177",
        "JVASP-1183",
        "JVASP-1186",
        "JVASP-1189",
        "JVASP-96",
        "JVASP-10591",
        "JVASP-1198",
        "JVASP-8003",
        "JVASP-1192",
        "JVASP-23",
    ]

    x = []
    for i in list(itertools.product(jids1, jids1)):
        if i[0] != i[1]:
            tmp = [i[0], i[1], [1, 1, 0], [1, 1, 0]]
            tmp2 = [i[1], i[0], [1, 1, 0], [1, 1, 0]]
            if tmp not in x and tmp2 not in x:
                x.append(tmp)
    combinations = x
    combinations = [["JVASP-1372", "JVASP-1174", [0, 0, 1], [0, 0, 1]]]
    already_submitted_jobs = compile_jobs()
    already_submitted_jobs = [
        aa.split("_film_thickness")[0] for aa in already_submitted_jobs
    ]
    for i in combinations:
        if sub is not None:
            sjid = sub
        else:
            sjid = i[0]
        name1 = (
            "Interface-"
            + i[1]
            + "_"
            + sjid
            + "_film_miller_"
            + "_".join(map(str, i[3]))
            + "_sub_miller_"
            + "_".join(map(str, i[2]))
        )
        name2 = (
            "Interface-"
            + sjid
            + "_"
            + i[1]
            + "_film_miller_"
            + "_".join(map(str, i[2]))
            + "_sub_miller_"
            + "_".join(map(str, i[3]))
        )
        if (
            name2 not in already_submitted_jobs
            and name1 not in already_submitted_jobs
        ):
            try:
                # if sub is not None:
                #    sjid = sub
                # else:
                #    sjid = i[0]

                tol = 1
                seperations = [2.5]

                x = InterfaceCombi(
                    film_ids=[sjid],
                    subs_ids=[i[1]],
                    # subs_ids=[i[1]],
                    film_indices=[i[2]],
                    subs_indices=[i[3]],
                    disp_intvl=0.05,
                    vacuum_interface=2,
                )

                # InterfaceCombi(
                # film_mats=[],
                # subs_mats=[],
                # disp_intvl=0.05,
                # seperations=[2.5],
                # film_indices=[[0, 0, 1]],
                # subs_indices=[[0, 0, 1]],
                # film_ids=[],
                # subs_ids=[],
                # film_kplengths=[30],
                # subs_kplengths=[30],
                # film_thicknesses=[16],
                # subs_thicknesses=[16],
                # rount_digit=3,
                # calculator={},
                # working_dir=".",
                # generated_interfaces=[],
                # vacuum_interface=2,
                # max_area_ratio_tol=1.00,
                # max_area=300,
                # ltol=0.08,
                # atol=1,
                # apply_strain=False,
                # lowest_mismatch=True,
                # rotate_xz=False,  # for transport
                # lead_ratio=None,  # 0.3,
                # from_conventional_structure_film=True,
                # from_conventional_structure_subs=True,
                # relax=False,
                # wads={},
                # dataset=[],
                # id_tag="jid",)

                wads = x.calculate_wad(method="ewald")
                wads = x.wads["wads"]
                index = np.argmin(wads)
                combined = Atoms.from_dict(
                    x.generated_interfaces[index]["generated_interface"]
                )
                combined = combined.center(vacuum=seperations[0] - tol)
                print(index, combined)
                info["inc"]["ISIF"] = 3
                info["inc"]["ENCUT"] = 520
                info["inc"]["NEDOS"] = 5000
                # info["queue"] = "rack1,rack2,rack2e,rack3,rack4,rack4e,rack5,rack6"
                info["queue"] = "rack2,rack2e,rack3,rack4,rack4e,rack5,rack6"
                wads = x.calculate_wad(
                    method="vasp",
                    index=index,
                    do_surfaces=False,
                    extra_params=info,
                )
                # break
                # import sys
                # sys.exit()
            except:
                pass


semi_flow()

# semi_flow()
# semi_flow(sub='JVASP-1109')
# semi_flow(sub='JVASP-30')
# jids=['JVASP-39','JVASP-1174','JVASP-890']
# jids=['JVASP-1327']
# TODO: jids=['JVASP-8184','JVASP-1372']
# for i in jids:
#     semi_flow(sub=i)
