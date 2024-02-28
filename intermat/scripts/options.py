from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.job_manager import compile_jobs, get_json_data
from jarvis.db.jsonutils import loadjson, dumpjson

metals = [
    972,
    825,
    813,
    816,
    802,
    1029,
    861,
    943,
    963,
    14616,
    867,
    2,
    14968,
    14970,
    19780,
]

semicons = [
    1002,
    890,
    39,
    30,
    62940,
    20092,
    8003,
    1192,
    23,
    1195,
    96,
    10591,
    1198,
    1312,
    133719,
    36873,
    1327,
    1372,
    1408,
    8184,
    1174,
    1177,
    1180,
    1183,
    1186,
    1189,
    91,
    8158,
    8118,
    107,
    36018,
    36408,
    105410,
    36403,
    1008,
    95268,
    21211,
    1023,
    7836,
    9166,
    1201,
    85478,
    1115,
    1112,
    1103,
    1109,
    131,
    149916,
    111005,
    25,
    1067,
    154954,
    59712,
    10703,
    1213,
    19007,
    10114,
    9175,
    104,
    10036,
    18983,
    1216,
    79522,
    1222,
    10037,
    110,
    8082,
    1240,
    51480,
    29539,
    54,
    29556,
    1915,
    75662,
    101764,
    22694,
    4282,
    76195,
    8554,
    149871,
    2376,
    14163,
    26248,
    18942,
    3510,
    5224,
    8559,
    85416,
    9117,
    90668,
    10689,
    106381,
    108773,
    101184,
    103127,
    104764,
    102336,
    110231,
    108770,
    101074,
    149906,
    99732,
    106686,
    110952,
    106363,
]


insul = [9147, 34249, 43367, 113, 41, 58349, 34674, 34656, 34249, 32]

metals = ["JVASP-" + str(i) for i in metals]
semicons = ["JVASP-" + str(i) for i in semicons]
insul = ["JVASP-" + str(i) for i in insul]


"""
x = []
for i in list(itertools.product(metals, semicons)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)

#        tmp = [i[0], i[1], [1, 1, 0], [1, 1, 0]]
#        tmp2 = [i[1], i[0], [1, 1, 0], [1, 1, 0]]
#        if tmp not in x and tmp2 not in x:
#            x.append(tmp)
#        tmp = [i[0], i[1], [1, 1, 1], [1, 1, 1]]
#        tmp2 = [i[1], i[0], [1, 1, 1], [1, 1, 1]]
#        if tmp not in x and tmp2 not in x:
#            x.append(tmp)
print(len(x))
for i in list(itertools.product(semicons, semicons)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)
print(len(x))

for i in list(itertools.product(metals, metals)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)
print(len(x))
for i in list(itertools.product(metals, insul)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)


print(len(x))
for i in list(itertools.product(semicons, insul)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)

print(len(x))
for i in list(itertools.product(metals, semicons)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)
print(len(x))
for i in list(itertools.product(metals, semicons)):
    if i[0] != i[1]:
        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        if tmp not in x and tmp2 not in x:
            x.append(tmp)

already_submitted_jobs = compile_jobs()
already_submitted_jobs = [
    aa.split("_film_thickness")[0] for aa in already_submitted_jobs
]
y = []
for i in x:
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
        y.append(i)
dumpjson(data=y, filename="options.json")
print(len(x), len(y))
"""
d = loadjson("options.json")
# print(d[0:100])  # dobby
print(d[100:200])  # raritan
import sys

sys.exit()

# To Add:CuO,Zn3As2,Zn3Sb2,PbSnTe,Bi2S3,GaMnAs,Ag2S
# PbMnTe,La0.7Ca0.3MnO3,EuO,GaAsN, GaAsP, GaAsSb, AlGaP,InAsSb,
# AlGaInP, AlGaAsP,InGaAsP,InGaAsSb,AlInAsP,AlGaAsSn,InGaAsSn,In2O3
