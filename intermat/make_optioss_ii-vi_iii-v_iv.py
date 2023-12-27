import itertools
from jarvis.db.figshare import data
from intermat.job_manager import compile_jobs, get_json_data

dft_3d = data("dft_3d")
from jarvis.core.composition import Composition
from jarvis.db.jsonutils import dumpjson

already_submitted_jobs = compile_jobs()
already_submitted_jobs = [
    aa.split("_film_thickness")[0] for aa in already_submitted_jobs
]

combs = [
    "BN",
    "BP",
    "BAs",
    "BSb",
    "AlN",
    "AlP",
    "AlAs",
    "AlSb",
    "GaN",
    "GaP",
    "GaAs",
    "GaSb",
    "InN",
    "InP",
    "InAs",
    "InSb",
    "ZnS",
    "ZnSe",
    "ZnTe",
    "CdS",
    "CdSe",
    "CdTe",
    "ZnO",
    "SiC",
    "SiGe",
]
combs = [Composition.from_string(i).reduced_formula for i in combs]
x = []
y = []
z = []
for i in dft_3d:
    spg = int(i["spg_number"])
    if spg == 186 or spg == 216:
        formula = Composition.from_string(i["formula"]).reduced_formula
        name = formula + "_" + str(spg)
        if name not in y and formula in combs:
            y.append(name)
            x.append(i["jid"])
            print(name, i["jid"])
print(x, len(x))
print(y, len(y))
opts = []
for i in list(itertools.product(x, x)):
    if i[0] != i[1]:
        #        tmp = [i[0], i[1], [0, 0, 1], [0, 0, 1]]
        #        tmp2 = [i[1], i[0], [0, 0, 1], [0, 0, 1]]
        #        if tmp not in x and tmp2 not in x:
        #            x.append(tmp)

        tmp = [i[0], i[1], [1, 1, 0], [1, 1, 0]]
        tmp2 = [i[1], i[0], [1, 1, 0], [1, 1, 0]]
        if tmp not in opts and tmp2 not in opts:
            opts.append(tmp)

print(opts, len(opts))

for i in opts:
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
        # if i not in already_submitted_jobs:
        z.append(i)

# 1-100 raritan
# 100:200 dobby
print(len(opts), len(z))
dumpjson(data=z, filename="grp_ii-vi_iii-v_iv.json")
