from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np

# AlN(39),GaN(30),Si(1002),Ge(890),AlP(1327),GaP(8184),AlAs(1372),GaAs(1174)
# Ge(890), AlN(39), GaN(30), BN(62940), CdO(20092), CdS(8003), CdSe(1192), CdTe(23), ZnO(1195), ZnS(96), ZnSe(10591), ZnTe(1198), BP(1312), BAs(133719),
# BSb(36873), AlP(1327), AlAs(1372), AlSb(1408), GaP(8184), GaAs(1174), GaSb(1177), InN(1180), InP(1183), InAs(1186), InSb(1189), C(91), SiC(8158,8118,107), GeC(36018), SnC(36408), SiGe(105410), SiSn(36403), , Sn(1008)

info = template_extra_params(method="vasp")


def semi_flow(sub="JVASP-1109"):
    combinations = [
        # ["JVASP-1372", "JVASP-1174", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-39", "JVASP-30", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-39", "JVASP-8184", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-1002", "JVASP-890", [0, 0, 1], [0, 0, 1]],
        # ["JVASP-1327", "JVASP-8184", [0, 0, 1], [0, 0, 1]],
        ["JVASP-1002", "JVASP-972", [0, 0, 1], [0, 0, 1]],
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
    for i in combinations:
        try:
            if sub is not None:
                sjid = sub
            else:
                sjid = i[0]

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
            wads = x.calculate_wad(method="ewald")
            wads = x.wads["wads"]
            index = np.argmin(wads)
            combined = Atoms.from_dict(
                x.generated_interfaces[index]["generated_interface"]
            )
            combined = combined.center(vacuum=seperations[0] - tol)
            print(index, combined)
            info["inc"]["ISIF"] = 3
            info["inc"]["ENCUT"] = 500
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


# semi_flow(sub='JVASP-1109')
# semi_flow(sub='JVASP-30')
jids = [
    "JVASP-39",
    "JVASP-1174",
    "JVASP-890",
    "JVASP-1327",
    "JVASP-8184",
    "JVASP-1372",
]
for i in jids:
    semi_flow(sub=i)
