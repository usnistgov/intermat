import numpy as np
from ase.transport.calculators import TransportCalculator
hkl=np.load('Si4Ag_10-19-2023_heavy_tb3_left/hk.npz')
skl=np.load('Si4Ag_10-19-2023_heavy_tb3_left/sk.npz')
skr=np.load('Si4Ag_10-20-2023_heavy_left_tot_tb3_rightv2/sk.npz')
hkr=np.load('Si4Ag_10-20-2023_heavy_left_tot_tb3_rightv2/hk.npz')
#skr=np.load('Si4Ag_10-19-2023_heavy_tot20_tb3_right/sk.npz')
#hkr=np.load('Si4Ag_10-19-2023_heavy_tot20_tb3_right/hk.npz')
hk=np.load('AlSi4_nightly_tb3_all/hk.npz')
sk=np.load('AlSi4_nightly_tb3_all/sk.npz')
ef=-0.20623105303074457
energies=np.arange(-1.0, 1.0, 0.01)
tcalc = TransportCalculator(h=hk, h1=hkl, h2=hkr, s=sk, s1=skl, s2=skr, align_bf=10, eta=0.4,energies=energies)
T_e = tcalc.get_transmission()
current = tcalc.get_current(energies, T=1.0)
#current = tcalc.get_current(energies+ef, T=0.0)
from ase import units
import matplotlib.pyplot as plt
#plt.axhline(y=0)
plt.plot(energies, units._e**2 / units._hplanck * current)
plt.savefig('tmp.png')
plt.close()

