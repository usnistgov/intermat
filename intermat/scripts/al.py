from intermat.main import InterfaceCombi
from alignn.ff.ff import get_figshare_model_ff,default_path,wt10_path
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_jid_data
import numpy as np
import plotly.graph_objects as go 
import plotly.express as px
atoms_al = Atoms.from_dict(
    get_jid_data(jid='JVASP-816', dataset="dft_3d")["atoms"]
)
x = InterfaceCombi(
    film_mats=[atoms_al],
    subs_mats=[atoms_al],
    film_indices=[[1,1,1]],
    subs_indices=[[1,1,1]],
    vacuum_interface=2,
    film_ids=['JVASP-816'],
    subs_ids=['JVASP-816'],
    disp_intvl=0.1,

)
#wads = x.calculate_wad_eam()
path='/wrk/knc6/ALINN_FC/aff25_lg_6/temp1/'
wads = x.calculate_wad_alignn(model_path=wt10_path())
X=x.X
Y=x.Y
tmp_wads=wads
print('wads',len(wads))
#wads=wads[0:len(X)*len(Y)]
wads=np.array(wads).reshape(len(X),len(Y))

fig = go.Figure(data=[go.Surface(z=wads, x=X, y=Y)])
fig.write_image("al_image.pdf", engine="kaleido")
#fig.show()
