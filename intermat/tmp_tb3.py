import numpy as np
hkl=np.load('AlSi4_nightly_tb3_left/hk.npz')
skl=np.load('AlSi4_nightly_tb3_left/sk.npz')
skr=np.load('AlSi4_nightly_tb3_right/sk.npz')
hkr=np.load('AlSi4_nightly_tb3_right/hk.npz')
hk=np.load('AlSi4_nightly_tb3_all/hk.npz')
sk=np.load('AlSi4_nightly_tb3_all/sk.npz')
from ase.transport.calculators import TransportCalculator
ef=-0.20623105303074457
energies=np.arange(-0.45, 0.45, 0.01)
tcalc = TransportCalculator(h=hk, h1=hkl, h2=hkr, s=sk, s1=skl, s2=skr, align_bf=0, eta=0.4,energies=energies)
T_e = tcalc.get_transmission()
current = tcalc.get_current(energies, T=1.0)
#current = tcalc.get_current(energies+ef, T=0.0)
from ase import units
import matplotlib.pyplot as plt
#plt.axhline(y=0)
plt.plot(energies, units._e**2 / units._hplanck * current)
plt.savefig('tmp.png')
plt.close()

