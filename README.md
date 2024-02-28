[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/c00ee48c524f5000e7f80a974bc6dc71)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/debf9cbefa9a290502d73fd3cbc4fd69)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/7492b51b371a8e9dbaa01d76bb438467)

# InterMat

## Introduction

Interface materials design (InterMat) package allows: 

 1) generation of an atomistic interface geometry,
 2) integrating multi-scale methods,
 3) determining interface properties such as equilibrium geometries, energetics, work functions, ionization potentials, electron affinities, band offsets, carrier effective masses, mobilities, and thermal conductivities,
 4) classification of heterojunction using various models such as ASJ, STJ and IU models,
 5) benchmarking calculated properties with experiments,
 6) training machine learning models especially for interface design.




## Installation

-   We recommend installing miniconda environment from
    <https://conda.io/miniconda.html> :

        bash Miniconda3-latest-Linux-x86_64.sh (for linux)
        bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
        Download 32/64 bit python 3.9 miniconda exe and install (for windows)
        Now, let's make a conda environment just for JARVIS::
        conda create --name my_intermat python=3.9
        source activate my_intermat


        git clone https://github.com/usnistgov/intermat.git
        cd inermat
        python setup.py develop

## Functionalities

### Getting bulk structures -starting structures

We can get bulk structures of a system for JARVIS-DFT or other databases as listed [here](https://pages.nist.gov/jarvis/databases/)

Example for Silicon from the [JARVIS-DFT](https://jarvis.nist.gov/jarvisdft/)

   ```
   from jarvis.db.fighshare import get_jid_data
   from jarvis.core.atoms import Atoms
   jid = 'JVASP-1002'
   atoms_si = Atoms.from_dict(get_jid_data(jid=jid,dataset='dft_3d')['atmoms'])
   print(atoms_si)
   ```
### Surfaces

Example of generating non-polar surfaces of semiconductors

```
from intermat.known_mats import semicons
from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
import time
semicons = semicons() # e.g. 1002 for silicon
for i in semicons:
    jid='JVASP-'+str(i)
    atoms=get_jid_atoms(jid=jid)
    if atoms is not None:
        atoms=Atoms.from_dict(atoms)

        spg = Spacegroup3D(atoms=atoms)
        cvn = spg.conventional_standard_structure
        mills = symmetrically_distinct_miller_indices(
            max_index=1, cvn_atoms=cvn
        )
        for miller in mills:
            surf = Surface(
                atoms,
                indices=miller,
                from_conventional_structure=True,
                thickness=16,
                vacuum=12,
            ).make_surface()
            # Surface-JVASP-105933_miller_1_1_0
            nm='Surface-'+jid+'_miller_'+'_'.join(map(str,miller))
            if not surf.check_polar and '-1' not in nm:
                non_polar_semi.append(nm)
                if len(non_polar_semi)%100==0:
                    t2=time.time()
                    print(len(non_polar_semi),t2-t1)
                    t1=time.time()

```

### Generating interface structures and calculations
Zur algorithm based interface (& terminations) ASJ vs STJ, etc. models


```
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.offset import offset, locpot_mean

# Step-1: prepare and submit calculations
combinations = [["JVASP-1002", "JVASP-1174", [1, 1, 0], [1, 1, 0]]]
for i in combinations:
    tol = 1
    seperations = [2.5]  # can have multiple separations
    # Interface generator class
    x = InterfaceCombi(
        film_ids=[i[0]],
        subs_ids=[i[1]],
        film_indices=[i[2]],
        subs_indices=[i[3]],
        disp_intvl=0.05,
        vacuum_interface=2,
    )
    # Fast work of adhesion with Ewald/ALIGNN-FF
    wads = x.calculate_wad(method="ewald")
    wads = x.wads["wads"]
    index = np.argmin(wads)
    combined = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    combined = combined.center(vacuum=seperations[0] - tol)
    print(index, combined)
    # Cluster/account specific job submission lines
    extra_lines = (
        ". ~/.bashrc\nmodule load vasp/6.3.1\n"
        + "conda activate mini_alignn\n"
    )

    info["inc"]["ISIF"] = 3
    info["inc"]["ENCUT"] = 520
    info["inc"]["NEDOS"] = 5000
    info["queue"] = "rack1"
    # VASP job submission,
    wads = x.calculate_wad(
        method="vasp",
        index=index,
        do_surfaces=False,
        extra_params=info,
    )
    # Other calculators such as QE, TB3, ALIGNN etc.
    # are also available

    # Once calculations are converged
    # We can calculate properties such as
    # bandgap, band offset, interfac energy for interfaces
    # &  electron affinity, surface energy,
    # ionization potential etc. for surfaces
    #
    # Path to LOCPOT file
fname = "Interface-JVASP-1002_JVASP-1174_film_miller_1_1_0_sub_miller_1_1_0_film_thickness_16_subs_thickness_16_seperation_2.5_disp_0.5_0.2_vasp/*/*/LOCPOT"
ofs = offset(fname=fname, left_index=-1, polar=False)
print(ofs)
    # Note for interfaces we require LOCPOTs for bulk materials of the two materials as well
    # For surface properties
dif, cbm, vbm, avg_max, efermi, formula, atoms, fin_en = locpot_mean(
    "PATH_TO_LOCPOT"
)
```
### Rapid structure screening / relaxation
ALIGNN-FF, Ewald, TB?

### Computational Engines
VASP, QE, LAMMPS.

### Properties using post-processing (band alignment/offset)
IU vs ASJ/STJ
Band alignment, band gap, work function, ionization potential, electron affinity, adhesion energy, surface energy

### AI/ML
Property prediction

## Curated dataset
Experimental validation


