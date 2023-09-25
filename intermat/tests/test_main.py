import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from intermat.main import InterfaceCombi

# import plotly.graph_objects as go
# import plotly.express as px


# atoms_al = Atoms.from_dict(
#    get_jid_data(jid="JVASP-816", dataset="dft_3d")["atoms"]
# )
# atoms_ni = Atoms.from_dict(
#    get_jid_data(jid="JVASP-943", dataset="dft_3d")["atoms"]
# )

pot = os.path.join(os.path.dirname(__file__), "Mishin-Ni-Al-Co-2013.eam.alloy")


def test_eam():
    x = InterfaceCombi(
        # film_mats=[atoms_al],
        # subs_mats=[atoms_ni],
        film_indices=[[1, 1, 1]],
        subs_indices=[[1, 1, 1]],
        vacuum_interface=2,
        film_ids=["JVASP-816"],
        subs_ids=["JVASP-943"],
        disp_intvl=0.2,
    )
    wads = x.calculate_wad_eam(potential=pot)
    print(len(wads))

    X = x.X
    Y = x.Y

    wads = np.array(wads).reshape(len(X), len(Y))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        X, Y, wads, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("intmat.png")
    plt.close()


test_eam()
# plt.show()


# fig = go.Figure(data=[go.Surface(z=wads, x=X, y=Y)])
# fig.show()
