<!-- [![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/c00ee48c524f5000e7f80a974bc6dc71)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/debf9cbefa9a290502d73fd3cbc4fd69)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/7492b51b371a8e9dbaa01d76bb438467)  -->
![InterMat Schematic](https://github.com/usnistgov/intermat/blob/intermat/intermat/Schematic.png)


## Introduction

Interfaces are critical for a variety of technological applications including semiconductor transistors and diodes, solid-state lighting devices, solar-cells, data-storage and battery applications. While interfaces are ubiquitous, predicting even basic interface properties from bulk data or chemical models remains challenging. Furthermore, the continued scaling of devices towards the atomic limit makes interface properties even more important. There have been numerous scientific efforts to model interfaces with a variety of techniques including density functional theory (DFT), force-field (FF), tight-binding, TCAD and machine learning (ML) techniques. However, to the best of our knowledge, there is no systematic investigation of interfaces for a large class of structural variety and chemical compositions. Most of the previous efforts focus on a limited number of interfaces, and hence there is a need for a dedicated infrastructure for data-driven interface materials design.

The Interface materials design (InterMat) package ([https://arxiv.org/abs/2401.02021](https://arxiv.org/abs/2401.02021)) introduces a multi-scale and data-driven approach for material interface/heterostructure design. This package allows: 

 1) Generation of an atomistic interface geometry given two similar or different materials,
 2) Performing calculations using multi-scale methods such as DFT, MD/FF, ML, TB, QMC, TCAD etc.,
 3) analyzing properties such as equilibrium geometries, energetics, work functions, ionization potentials, electron affinities, band offsets, carrier effective masses, mobilities, and thermal conductivities, classification of heterojunctions, benchmarking calculated properties with experiments,
 4) training machine learning models especially to accelerate interface design.




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

## Generation

### Bulk structures from scratch
An atomic structure can consist of atomic element types, corresponding
xyz coordinates in space (either in real or reciprocal space) and
lattice matrix used in setting periodic boundary conditions.

An example of constructing an atomic structure class using
`jarvis.core.Atoms` is given below. After creating the Atoms class, we
can simply print it and visualize the POSCAR format file in a software
such as VESTA. While the examples below use Silicon elemental crystal
creation and analysis, it can be used for multi-component systems as
well.

``` python
from jarvis.core.atoms import Atoms
box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
coords = [[0, 0, 0], [0.25, 0.25, 0.25]]
elements = ["Si", "Si"]
Si = Atoms(lattice_mat=box, coords=coords, elements=elements, cartesian=False)
print (Si) # To visualize 
Si.write_poscar('POSCAR.vasp')
Si.write_cif('POSCAR.vasp')
```

The <span class="title-ref">Atoms</span> class here is created from the
raw data, but it can also be read from different file formats such as:
<span class="title-ref">'.cif', 'POSCAR', '.xyz', '.pdb', '.sdf',
'.mol2'</span> etc. The Atoms class can also be written to files in
formats such as POSCAR/.cif etc.

Note that for molecular systems, we use a large vaccum padding (say 50
Angstrom in each direction) and set lattice_mat accordingly, e.g.
lattice_mat = \[\[50,0,0\],\[0,50,0\],\[0,0,50\]\]. Similarly, for free
surfaces we set high vaccum in one of the crystallographic directions
(say z) by giving a large z-comonent in the lattice matrix while keeping
the x, y comonents intact.

``` python
my_atoms = Atoms.from_poscar('POSCAR')
my_atoms.write_poscar('MyPOSCAR')
```

Once this Atoms class is created, several imprtant information can be
obtained such as:

``` python
print ('volume',Si.volume)
print ('density in g/cm3', Si.density)
print ('composition as dictionary', Si.composition)
print ('Chemical formula', Si.composition.reduced_formula)
print ('Spacegroup info', Si.spacegroup())
print ('lattice-parameters', Si.lattice.abc, Si.lattice.angles)
print ('packing fraction',Si.packing_fraction)
print ('number of atoms',Si.num_atoms)
print ('Center of mass', Si.get_center_of_mass())
print ('Atomic number list', Si.atomic_numbers)
```

For creating/accessing dataset(s), we use `Atoms.from_dict()` and
`Atoms.to_dict()` methods:

``` python
d = Si.to_dict()
new_atoms = Atoms.from_dict(d)
```

The <span class="title-ref">jarvis.core.Atoms</span> object can be
converted back and forth to other simulation toolsets such as Pymatgen
and ASE if insyalled, as follows

``` python
pmg_struct = Si.pymatgen_converter()
ase_atoms = Si.ase_converter()
```

In order to make supercell, the following example can be used:

``` python
supercell_1 = Si.make_supercell([2,2,2])
supercell_2 = Si.make_supercell_matrix([[2,0,0],[0,2,0],[0,0,2]])
supercell_1.density == supercell_2.density
```

### Bulk structures from existing database

There are more than [50 databases available in the JARVIS-Tools](https://pages.nist.gov/jarvis/databases/). These can be used to easily obtain a structure, e.g. for Silicon (JVASP-1002):

``` python
from jarvis.tasks.lammps.lammps import LammpsJob, JobFactory
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.structure.spacegroup import Spacegroup3D


# atoms = Atoms.from_poscar('POSCAR')
# Get Silicon diamond structure from JARVIS-DFT database
dataset = "dft_3d"
jid = "JVASP-1002"
tmp_dict = get_jid_data(jid=jid, dataset=dataset)["atoms"]
atoms = Atoms.from_dict(tmp_dict)
```

The JARVIS-OPTIMADE and similar OPTIMADE tools can also be used to obtain structures.e.g.

``` python
from jarvis.db.restapi import jarvisdft_optimade
response_data = jarvisdft_optimade(query = "elements HAS  ALL C,Si")
response_data = jarvisdft_optimade(query = "id=1002")
```



### Surface/slab structures

An example of creating, free surfaces is shown below:

``` python
from jarvis.analysis.defects.surface import wulff_normals, Surface

# Let's create (1,1,1) surface with three layers, and vacuum=18.0 Angstrom
# We center it around origin so that it looks good during visualization
surface_111 = (
    Surface(atoms=Si, indices=[1, 1, 1], layers=3, vacuum=18)
        .make_surface()
        .center_around_origin()
)
print(surface_111)
```

While the above example makes only one surface (111), we can ask
jarvis-tools to provide all symmetrically distinct surfaces as follows:

``` python
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
spg = Spacegroup3D(atoms=Si)
cvn = spg.conventional_standard_structure
mills = symmetrically_distinct_miller_indices(max_index=3, cvn_atoms=cvn)
for i in mills:
    surf = Surface(atoms=Si, indices=i, layers=3, vacuum=18).make_surface()
    print ('Index:', i)
    print (surf)
```

We can streamline surface generation for numerous structures. e.g.,

Example of generating non-polar surfaces of semiconductors

``` python
from jarvis.analysis.defects.surface import Surface
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
import time
dataset = 'dft_3d'
semicons = ['1002', '1174', '30'] 
for i in semicons:
    jid='JVASP-'+str(i)
    atoms=get_jid_atoms(jid=jid, dataset=dataset)
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
            nm='Surface-'+jid+'_miller_'+'_'.join(map(str,miller))
            print(surf)
            print()
            if not surf.check_polar and '-1' not in nm:
                non_polar_semi.append(nm)
                if len(non_polar_semi)%100==0:
                    t2=time.time()
                    print(len(non_polar_semi),t2-t1)
                    t1=time.time()

```

### Interface structures

We generate the interfaces following the Zur et. al. algorithm. The Zur algorithm generates a number of superlattice transformations within a specified maximum surface area and also evaluates the length and angle between film and substrate superlattice vectors to determine if they can match within a tolerance. This algorithm is applicable to different crystal structures and their surface orientations. 

For generating interface/heterostructures (combination of film and substrate), we can use the `run_intermat.py` command. It requies a config file for generation settings such as, `film_file_path` e.g., `POSCAR-1`, `substrate_file_path` e.g., `POSCAR-2`, or `film_jid` (e.g. [JVASP-1002](https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/JVASP-1002.xml) for Si)/`substrate_jid` (e.g. [JVASP-1174](https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/JVASP-1174.xml) for GaAs) if you want use JARVIS-DFT structure instead, `film_index` for film's Miller index such as `0_0_1`, `substrate_index` for substrate's Miller index such as `0_0_1`, `film_thickness`, `substrate_thickness`, maximum allowed area in Angstrom^2 (`max_area`), maximum mismath in latice lengths (`ltol`), separation between the slabs (`seperation`), whether to generate interface which are peridoic (low `vacuum_interface`) or aperidoic (high `vacuum_interface`)  etc. 

InterMat default settings are available [here](https://github.com/usnistgov/intermat/blob/intermat/intermat/config.py#L136).

Thus far, we have determined a candidate unit cell, but the relative alignment of the structures in the in-plane, as well as the terminations still need to be decided. We can perform a grid search of possible in-plane alignments with a space of 0.05 (`disp_intvl`) fractional coordinates to determine the initial structure for further relaxation. Doing such a large number of calculation with DFT would be prohibitive, so we use faster checks using ALIGNN-FF/ Ewald summation/ classical FFs etc. A typical value of non-dimensional `disp_intvl` could be 0.1. The method is set using `calculator_method` tag. 

An example [JSON](https://www.w3schools.com/js/js_json_intro.asp) `config.json` file is available [here](https://github.com/usnistgov/intermat/blob/intermat/intermat/tests/config.json). We start with a config.json file:

``` python

{ film_jid:"JVASP-1002", substrate_jid:"JVASP-1174"}
```
keeping all the default parametrs intact:

``` python
run_intermat.py --config_file "config.json"
```

The `run_intermat.py` is a helper script and is based on a broader [`InterfaceCombi`](https://github.com/usnistgov/intermat/blob/main/intermat/generate.py#L99) class.

Similar to the surface generation workflow, a similar one for interfaces high-throughput workflows can be as follows:


``` python
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.offset import offset, locpot_mean

# Step-1: prepare and submit calculations
# Si/GaAs, Si/GaP example
combinations = [["JVASP-1002", "JVASP-1174", [1, 1, 0], [1, 1, 0]], ["JVASP-1002", "JVASP-1327", [1, 1, 0], [1, 1, 0]]]
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


## Calculation

### Rapid structure screening / relaxation
ALIGNN-FF, Ewald, Tight-binding etc.

### Computational Engines
VASP, QE, GPAW, LAMMPS, ASE,...

Details [here](https://github.com/usnistgov/intermat/blob/main/intermat/calculators.py#L199)

### Properties using post-processing (band alignment/offset)
IU vs ASJ/STJ
Band alignment, band gap, work function, ionization potential, electron affinity, adhesion energy, surface energy

### AI/ML
Property prediction

## Curated dataset
Experimental validation

# More detailed documentation coming soon!

