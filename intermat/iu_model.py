import alignn, os, torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig

torch.cuda.is_available = lambda: False
from jarvis.analysis.defects.surface import Surface
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph

device = "cpu"
dft_3d = data("dft_3d")


def get_model(
    model_path="/mnt/c/Users/knc6/OneDrive - NIST/KamalLaptop/JARVIS-CHIPS/ALIGNN/Out_CBM",
):
    cnfg = os.path.join(model_path, "config.json")
    config = loadjson(cnfg)
    model = ALIGNN(ALIGNNConfig(**config["model"]))
    filename = os.path.join(model_path, "best_model.pt")
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model


def get_surface(jid="", index=[]):
    for i in dft_3d:
        if i["jid"] == jid:
            atoms = Atoms.from_dict(i["atoms"])
            break

    surf = Surface(
        atoms,
        indices=index,
        from_conventional_structure=True,
        thickness=16,
        vacuum=12,
    ).make_surface()
    return surf


model_cbm = get_model()
model_vbm = get_model(
    "/mnt/c/Users/knc6/OneDrive - NIST/KamalLaptop/JARVIS-CHIPS/ALIGNN/Out_VBM"
)
model_evac = get_model(
    "/mnt/c/Users/knc6/OneDrive - NIST/KamalLaptop/JARVIS-CHIPS/ALIGNN/Out_phi_opt2"
)
# def get_val
#     s=get_surface(jid=jid,index=mill)
#     g,lg=Graph.atom_dgl_multigraph(s)
#     c=cbm([g,lg]).cpu().detach().numpy().tolist()
#     v=vbm([g,lg]).cpu().detach().numpy().tolist()


# https://en.wikipedia.org/wiki/Anderson%27s_rule
import matplotlib

##%matplotlib inline
from jarvis.core.atoms import Atoms
import numpy as np
from math import floor
import matplotlib.pyplot as plt

# With and height in pixels
ppi = 100
figw = 850
figh = 500
plt.close()
fig = plt.figure(figsize=(figw / ppi, figh / ppi), dpi=ppi)
ax = fig.add_subplot(1, 1, 1)
plt.rcParams.update({"font.size": 12})
#'JVASP-91','JVASP-39',
include_jids = [
    "JVASP-1180",
    "JVASP-30",
    "JVASP-1408",
    "JVASP-8184",
    "JVASP-1183",
    "JVASP-62940",
    "JVASP-1002",
    "JVASP-1174",
    "JVASP-8158",
    "JVASP-1195",
    "JVASP-8003",
    "JVASP-1192",
    "JVASP-1327",
    "JVASP-1372",
]
from jarvis.db.figshare import data

dft_3d = data("dft_3d")


def get_gap(jid):
    for i in dft_3d:
        if i["jid"] == jid:
            # return i['optb88vdw_bandgap']#mbj_bandgap
            return i["mbj_bandgap"], i["formula"]


labels = []
vbms = []
cbms = []

al_vbms = []
al_cbms = []

rel_val = 10  # 10
water_splitters = []
count = 5  # 10
from jarvis.db.jsonutils import loadjson

dat = loadjson("../../WF/bulk_phi2.json")

for ii in dat:
    # print (i['phi'])
    # if i['phi']['avg_max']!='na' and i['phi']['scf_gap']!='na' and i['phi']['scf_gap']>=0.1 and ef_i<200:
    # if i['l']['jid'] in ll:
    label = "MX2"
    count = count + 1
    jid = ii["name"].split("_")[0].split("Surface-")[1]
    miller = (
        ii["name"].split("miller_")[1].split("_thickness")[0].replace("_", "")
    )
    atoms = Atoms.from_dict(ii["atoms"])
    ano = jid

    if (
        jid in include_jids
        and "JVASP-91_miller_1_1_0" not in ii["name"]
        and "JVASP-91_miller_1_1_1" not in ii["name"]
        and "JVASP-1002_1_0_0" not in ii["name"]
        and "x" not in ii["name"]
        and "1180" not in ii["name"]
        and "Surface-JVASP-1195_miller_0_0_1_" not in ii["name"]
        and "Surface-JVASP-1002_miller_1_0_0_thickness_1" not in ii["name"]
    ):
        gap, formula = get_gap(jid)
        # if ano==proto:
        label = (
            jid.replace("JVASP", formula) + " (" + miller + ")"
        )  # ii['name'].split('_thickness')[0].split('Surface-')[1].split('JVASP-')[1].replace('_miller','')
        labels.append(label)
        cbm = ii["surf_vbm"] + gap - ii["phi"]  # ii['scf_vbm']-ii['phi']
        vbm = ii["surf_vbm"] - ii["phi"]
        vbms.append(vbm)
        cbms.append(cbm)
        # gap=(cbm-vbm)

        g, lg = Graph.atom_dgl_multigraph(atoms)
        alignn_cbm = model_cbm([g, lg]).cpu().detach().numpy().tolist()
        alignn_vbm = model_vbm([g, lg]).cpu().detach().numpy().tolist()
        alignn_evac = model_evac([g, lg]).cpu().detach().numpy().tolist()
        x_vbm = alignn_vbm - rel_val  # -alignn_evac
        x_cbm = x_vbm - gap  # alignn_cbm-10#-alignn_evac-5

        al_cbms.append(x_vbm)
        al_vbms.append(x_cbm)
        print(
            ii["name"],
            gap,
            vbm,
            cbm,
            atoms.composition.reduced_formula,
            x_vbm,
            x_cbm,
        )
        # print (vbm,cbm,gap)
        if cbm >= -4.5 and vbm <= -5.73:  # and gap>=1.23:
            water_splitters.append(label)
        # print (vbm,cbm,i['l']['jid'],label)
        # print (i['l']['jid'],'vbm=',i['l']['phi']['scf_vbm']-i['l']['phi']['avg_max'],'fermi=',i['l']['phi']['Ef']-i['l']['phi']['avg_max'],'cbm=',i['l']['phi']['scf_cbm']-i['l']['phi']['avg_max'],'max=',i['l']['phi']['avg_max']-i['l']['phi']['avg_max'],'wf=',i['l']['phi']['phi'])
        # cbm.append(i['l']['phi']['scf_cbm']-i['l']['phi']['avg_max'])
        # vbm.append(-i['l']['phi']['scf_vbm']+i['l']['phi']['avg_max'])


x = np.arange(len(vbms)) + 0.5
emin = floor(min(vbms)) - 1.0


y = np.array(vbms) - emin
width = 0.9
# ax.bar(x, y, bottom=emin,color='lightskyblue',width=width,align='edge')
ax.bar(x, y, bottom=emin, color="skyblue", width=width, align="edge")
y = -np.array(cbms)
# ax.bar(x, y, bottom=cbms,color='lightgreen',width=width,align='edge')
ax.bar(x, y, bottom=cbms, color="lightgreen", width=width, align="edge")
ax.set_xlim(0.2, len(labels) + 0.4)
ax.set_ylim(emin, 0)
# ax.set_xticks(x)
ax.set_xticks(np.arange(len(labels)) + 1, labels)
ax.set_xticklabels(labels, rotation=90)
ax.axhline(y=-4.5, linestyle="-.", color="black")
ax.axhline(y=-5.73, linestyle="-.", color="black")
ax.text(max(x) + 1, -4, "${H^+}/{H_2}$")
ax.text(max(x) + 1, -6.2, "${O_2}/{H_2O}$")
# plt.title("2Positions of VBM and CBM ")
ax.set_ylabel("Energy wrt vacuum (eV)")
# ax.set_xlabel(r'$\leftarrow$2D materials$\rightarrow $')
# ax.set_ylabel(r'$\Delta \Theta / \omega $ \Huge{$\longleftarrow$}')
count = 0
for i, j in zip(cbms, vbms):
    count += 1
    plt.text(count - 0.5, emin + 0.1, round(-1 * j, 1), fontsize=8)
    plt.text(count - 0.5, -0.5, round(-1 * i, 1), fontsize=8)
# plt.tight_layout()
plt.show()
# C:\Users\knc6\OneDrive - NIST\KamalLaptop\JARVIS-CHIPS\WF\Interface-JVASP-30_JVASP-1195_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_0.35_0.3_vasp_R2SCAN

########################################################################################################################################################################
import matplotlib

# %matplotlib inline
from jarvis.core.atoms import Atoms
import numpy as np
from math import floor
import matplotlib.pyplot as plt

# With and height in pixels
ppi = 100
figw = 850
figh = 500
plt.close()
fig = plt.figure(figsize=(figw / ppi, figh / ppi), dpi=ppi)
ax = fig.add_subplot(1, 1, 1)
plt.rcParams.update({"font.size": 12})
x = np.arange(len(al_vbms)) + 0.5
emin = floor(min(al_vbms)) - 1.0


y = np.array(al_vbms) - emin
width = 0.9
# ax.bar(x, y, bottom=emin,color='lightskyblue',width=width,align='edge')
ax.bar(x, y, bottom=emin, color="skyblue", width=width, align="edge")
y = -np.array(al_cbms)
# ax.bar(x, y, bottom=cbms,color='lightgreen',width=width,align='edge')
ax.bar(x, y, bottom=al_cbms, color="lightgreen", width=width, align="edge")
ax.set_xlim(0.2, len(labels) + 0.4)
ax.set_ylim(emin, 0)
# ax.set_xticks(x)
ax.set_xticks(np.arange(len(labels)) + 1, labels)
ax.set_xticklabels(labels, rotation=90)
ax.axhline(y=-4.5, linestyle="-.", color="black")
ax.axhline(y=-5.73, linestyle="-.", color="black")
ax.text(max(x) + 1, -4, "${H^+}/{H_2}$")
ax.text(max(x) + 1, -6.2, "${O_2}/{H_2O}$")
# plt.title("2Positions of VBM and CBM ")
ax.set_ylabel("Energy wrt vacuum (eV)")
# ax.set_xlabel(r'$\leftarrow$2D materials$\rightarrow $')
# ax.set_ylabel(r'$\Delta \Theta / \omega $ \Huge{$\longleftarrow$}')
count = 0
for i, j in zip(al_cbms, al_vbms):
    count += 1
    plt.text(count - 0.5, emin + 0.1, round(-1 * j, 1), fontsize=8)
    plt.text(count - 0.5, -0.5, round(-1 * i, 1), fontsize=8)
# plt.tight_layout()
plt.show()
