![alt text](https://github.com/usnistgov/intermat/actions/workflows/main.yml/badge.svg)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/knc6/0d9aa89f671687c6e925eea2be9b824a/intermat_gettingstarted.ipynb)
<!-- [![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/c00ee48c524f5000e7f80a974bc6dc71)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/debf9cbefa9a290502d73fd3cbc4fd69)
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/knc6/7492b51b371a8e9dbaa01d76bb438467)  -->
![InterMat Schematic](https://github.com/usnistgov/intermat/blob/intermat/intermat/Schematic.png)

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Generation](#generation)
  * [Bulk structures from scratch](#bulk)
  * [Bulk structures from databases](#databases)
  * [Surface/slab structures](#slabs)
  * [Interface structures](#interfaces)
* [Calculators and analyzers](#calc)
  * [Available calculators](#calcs)
  * [Surface energy](#surfen)
  * [Vacuum level, ionization potential, electron affinity](#ipeavc)
  * [Andersen model based band offset](#andersen)
  * [Alternate Slab Junction (ASJ) model based band offset](#asj)
* [Benchmarking](#benchmarking)
   * [Bandgaps](#gaps)
   * [Surface energy, ionization potential, electron affinity](#sen)
   * [Band offsets](#boffs)
* [Datasets](#data)
  * [Surface datasets](#sdata)
  * [Interface datasets](#idata)
* [AI/ML](#aiml)
  * [Training a new model](#newai)
  * [Using pretrained models](#preai)
* [Webapp](#webapp)
* [References](#refs)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)



<a name="intro"></a>
## Introduction

Interfaces are critical for a variety of technological applications including semiconductor transistors and diodes, solid-state lighting devices, solar-cells, data-storage and battery applications. While interfaces are ubiquitous, predicting even basic interface properties from bulk data or chemical models remains challenging. Furthermore, the continued scaling of devices towards the atomic limit makes interface properties even more important. There have been numerous scientific efforts to model interfaces with a variety of techniques including density functional theory (DFT), force-field (FF), tight-binding, TCAD and machine learning (ML) techniques. However, to the best of our knowledge, there is no systematic investigation of interfaces for a large class of structural variety and chemical compositions. Most of the previous efforts focus on a limited number of interfaces, and hence there is a need for a dedicated infrastructure for data-driven interface materials design.

The Interface materials design (InterMat) package ([https://arxiv.org/abs/2401.02021](https://arxiv.org/abs/2401.02021)) introduces a multi-scale and data-driven approach for material interface/heterostructure design. This package allows: 

 1) Generation of an atomistic interface geometry given two similar or different materials,
 2) Performing calculations using multi-scale methods such as DFT, MD/FF, ML, TB, QMC, TCAD etc.,
 3) analyzing properties such as equilibrium geometries, energetics, work functions, ionization potentials, electron affinities, band offsets, carrier effective masses, mobilities, and thermal conductivities, classification of heterojunctions, benchmarking calculated properties with experiments,
 4) training machine learning models especially to accelerate interface design.



<a name="install"></a>
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

<a name="generation"></a>
## Generation

<a name="bulk"></a>
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

<a name="databases"></a>
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


<a name="slabs"></a>
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
<a name="interfaces"></a>
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


An example of application of alignn_ff for xy scan is shown below.


<a name="calc"></a>
## Calculators and analyzers

<a name="calcs"></a>
### Available Calculators

There are more than 10 multi-scale methods available with InterMat. Most of them are open-access such as [QE](https://www.quantum-espresso.org/), [GPAW](https://wiki.fysik.dtu.dk/gpaw/), [LAMMPS](https://www.lammps.org/#gsc.tab=0), [ALIGNN-FF](https://github.com/usnistgov/alignn?tab=readme-ov-file#alignnff), [ASE](https://wiki.fysik.dtu.dk/ase/index.html), [EMT](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html) but some could be proprietary such as [VASP](https://www.vasp.at/).

<a name="surfen"></a>
### Surface energy
One of the most common quantities to calculate for bulk materials, surfaces and interfaces is its energy. All the methods mentioned above allow calculation of energies and have their strength and limitations.

An example to calulate energy of FCC aluminum with default (tutorial purposes) settings with QE is as follows:

Here we define an atomic structure of Aluminum and then use [`Calc`](https://github.com/usnistgov/intermat/blob/intermat/intermat/calculators.py#L170) class which can be used for several methods.
In the method we can switch to `alignn_ff`, 'eam_ase`, `lammps`, `vasp`, 'tb3`, `emt`, `gpaw` etc. Respecive setting parameters are defined in `IntermatConfig` as mentioned above.

``` python
from intermat.config import IntermatConfig
from jarvis.io.vasp.inputs import Poscar
from intermat.calculators import Calc
Al = """Al
1.0
2.4907700981617955 -1.4394159e-09 1.4380466239515413
0.8302566980301707 2.348320626706396 1.438046623951541
-4.0712845e-09 -2.878833e-09 2.8760942620064256
Al
1
Cartesian
0.0 0.0 0.0
"""
params = IntermatConfig().dict()
atoms = Poscar.from_string(Al).atoms
pprint.pprint(params)

method = "qe"
calc = Calc(
    method=method,
    atoms=atoms,
    extra_params=params,
    jobname="FCC_Aluminum_JVASP-816",
)
en = calc.predict()["total_energy"]
print(en)
```

In the config file, the default value of `sub_job` is `False` which runs calculations on the head node, turning it to `True`, submis the job in the HPC queue with respective settings such as queue name, walltime etc. One of the important quantities for surfaces is surface energy ($\gamma$). The calculation of surface energy using atomistic simulations typically involves the following steps:

1. **Model Construction**: Build a slab model of the material with a surface of interest. The slab should be thick enough to ensure that the atoms in the middle of the slab have bulk-like properties.

2. **Energy Minimization**: Perform energy minimization or geometry optimization to relax the atomic positions in the slab. This step is important to remove any artificial stresses or strains that might be present due to the initial construction of the slab.

3. **Total Energy Calculation**: Calculate the total energy of the relaxed slab model.

4. **Surface Energy Calculation**: The surface energy ($\gamma$)  can be calculated using the formula:


```math
    \gamma = \frac{E_{\text{slab}} - N_{\text{bulk}} \cdot E_{\text{bulk}}}{2A} 
```
   where:
   - $\(E_{\text{slab}}\)$ is the total energy of the relaxed slab model.
   - $\(N_{\text{bulk}}\)$ is the number of bulk-like atoms in the slab model.
   - $\(E_{\text{bulk}}\)$ is the energy per atom in the bulk material, which can be obtained from a separate calculation of a bulk model.
   - $\(A\)$ is the surface area of the slab model.
   - The factor of 2 accounts for the fact that there are two surfaces in the slab model (top and bottom).

It's important to ensure that the slab model is sufficiently large in the surface plane to minimize the interaction between periodic images and sufficiently thick to separate the two surfaces. Additionally, the choice of boundary conditions, potential energy function (force field for classical simulations or exchange-correlation functional for quantum simulations), and convergence criteria can significantly affect the accuracy of the surface energy calculation.

<a name="ipeavc"></a>
### Vacuum level, ionization potential, electron affinity

In addition to energetics based quantities such as surface energies , electronic properties of surfaces such as ionization potentials, electron affinities, and independent unit (IU)-based band offsets can be calculated from the electronic structure calculations. It requires electrostatic local potential (such as `LOCPOT`) file. An example for VASP can be given as follows:

``` python
from intermat.offset import offset, locpot_mean
phi, cbm, vbm, avg_max, efermi, formula, atoms, fin_en = locpot_mean("LOCPOT")
```

We can obtain the DFT VBM and vacuum level (from the maximum value of average electrostatic potential, here `phi`) of surface slabs using DFT. Subtracting the vacuum level from the VBM provides ionization potential (IP) information. Then, we add the bandgap ($E_g$) of the material to the ionization potential to get the electron affinity (EA, $\chi$).


<a name="andersen"></a>
### Andersen model based band offset

IU band alignment, also known as Anderson's rule, predicts semicondcutor band offsets at interfaces using only the IP and EA data from independent surface calculations. For a semiconductor heterojunction between A and B, the conduction band offset can be given by: 

```math
  \Delta E_c = \chi_B -  \chi_A
```

Similarly, the valence band offset is given by:

```math
  \Delta E_v = (\chi_A+ E_{gA}) -  (\chi_B+ E_{gB})
```


<a name="asj"></a>
### Alternate Slab Junction (ASJ) model based band offset
Similarly, for inerface band offset calculations, its important to have local potentials of each constituent slabs/bulk materials (depending on STJ/ASJ models) as well as the interface.

After determining the optimized geometric structure for the interface using DFT, we can obtain band offset data. As an example, we show a detailed analysis of Si(110)/GaAs(110) and AlN(001)/GaN(001) in Fig. \ref{fig:band_alignn}. In Fig. \ref{fig:band_alignn}a, we show the atomic structure of the ASJ based heterostructure of Si(110)/GaAs(110). The left side (with blue atoms) represents the Si and the right side is the GaAs region. In Fig. \ref{fig:band_alignn}c, we show the electrostatic potential profile, averaged in-plane, of the interface. The approximately sinusoidal profile on both regions represents the presence of atomic layers. The cyan lines show the region used to define the repeat distance, $L$, used for averaging in each material (see below). The red and green lines show the average potential profiles for the left and right parts using the repeat distance. The valence band offset ($\Delta E_v$) of an interface between semiconductor A and B, $\Delta E_v$ is obtained using eq. 4. The difference in the averages for the left and right parts gives the $\Delta V$ term. Now the bulk VBMs of the left and right parts are also calculated to determine the $\Delta E$. The sum of these two quantities gives the band offset that can be compared to experiments. 

```math
 \Delta E_v (A/B)= (E_v^B-E_v^A) + \Delta V
```

```math
  \Delta V = \bar{\bar{V}}_A - \bar{\bar{V}}_B
```



``` python
from intermat.analyze import offset
fname = "Interface-JVASP-1002_JVASP-1174_film_miller_1_1_0_sub_miller_1_1_0_film_thickness_16_subs_thickness_16_seperation_2.5_disp_0.5_0.2_vasp/*/*/LOCPOT"
ofs = offset(fname=fname, left_index=-1, polar=False)
```


An example, for high-throughput workflow with surfaces could be as follows:

``` python
from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from intermat.offset import offset, locpot_mean

# Step-1: prepare and submit calculations
# Si/GaAs, Si/GaP example
combinations = [
    ["JVASP-1002", "JVASP-1174", [1, 1, 0], [1, 1, 0]],
    ["JVASP-1002", "JVASP-1327", [1, 1, 0], [1, 1, 0]],
]
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

```

In the above example, we calculate, work of adhesion (`wad`) as well as band offset.
Calculations of band offsets and band-alignment at semiconductor heterojunctions are of special interest for device design. Semiconductor device transport and performance depend critically on valence band offsets and conduction band offsets  as well as interfacial roughness and defects.


<a name="benchmarking"></a>
## Benchmarking

||||||||||
|--- |--- |--- |--- |--- |--- |--- |--- |--- |
|System|IDs|Miller|$\phi$ (OPT)|$\phi$ (Exp)|$\chi (OPT)$|$\chi $(Exp)|$\gamma$ (OPT)|$\gamma$ (Exp)|
|Si|1002|111|5.00|4.77|4.10|4.05|1.60|1.14|
|Si|1002|110|5.30|4.89|4.10|-|1.66|1.9|
|Si|1002|001|5.64|4.92|3.60|-|2.22|2.13|
|C|91|111|4.67|5.0|-2.9|-|5.27|5.50|
|Ge|890|111|4.87|4.80|5.2|4.13|0.99|1.30|
|SiGe|105410|111|4.93|4.08|4.5|-|1.36|-|
|SiC|8118|001|5.26|4.85|1.3|-|3.51|-|
|GaAs|1174|110|4.89|4.71|4.40|4.07|0.67|0.86|
|InAs|1186|110|4.85|4.90|4.9|4.9|0.57|-|
|AlSb|1408|110|5.11|4.86|3.70|3.65|0.77|-|
|GaSb|1177|110|4.48|4.76|3.70|4.06|0.71|-|
|AlN|39|100|5.56|5.35|1.3|2.1|2.27|-|
|GaN|30|100|5.74|5.90|2.8|3.3|1.67|-|
|BN|79204|110|6.84|7.0|1.4|-|2.41|-|
|GaP|1393|110|5.31|6.0|4.0|4.3|0.88|1.9|
|BP|1312|110|5.61|5.05|2.8|-|2.08|-|
|InP|1183|110|5.17|4.65|4.10|4.35|0.73|-|
|CdSe|1192|110|5.70|5.35|6.4|-|0.38|-|
|ZnSe|96|110|5.67|6.00|5.4|-|0.44|-|
|ZnTe|1198|110|5.17|5.30|4.10|3.5|0.36|-|
|Al|816|111|4.36|4.26|-|-|0.82|-|
|Au|825|111|5.5|5.31|-|-|0.90|-|
|Ni|943|111|5.35|5.34|-|-|2.02|2.34|
|Ag|813|001|4.5|4.2|-|-|0.99|-|
|Cu|867|001|4.7|5.1|-|-|1.47|-|
|Pd|963|111|5.54|5.6|-|-|1.57|-|
|Pt|972|001|5.97|5.93|-|-|1.94|-|
|Ti|1029|100|3.84|4.33|-|-|2.27|-|
|Mg|919|100|3.76|3.66|-|-|0.35|-|
|Na|931|001|2.97|2.36|-|-|0.10|-|
|Hf|802|111|3.7|3.9|-|-|2.02|-|
|Co|858|001|5.22|5.0|-|-|3.49|-|
|Rh|984|001|5.4|4.98|-|-|2.46|-|
|Ir|901|100|5.85|5.67|-|-|2.77|-|
|Nb|934|100|3.87|4.02|-|-|2.41|-|
|Re|981|100|4.96|4.72|-|-|2.87|-|
|Mo|21195|100|4.17|4.53|-|-|3.30||
|Zn|1056|001|4.27|4.24|-|-|0.36|-|
|Bi|837|001|4.31|4.34|-|-|0.65|0.43|
|Cr|861|110|5.04|4.5|-|-|3.31|-|
|Sb|993|001|4.64|4.7|-|-|0.67|-|
|Sn|1008|110|4.82|4.42|-|-|0.91|-|
|MAE|-|-|0.29|-|0.39|-|0.34|-|


<table>
    <tr>
        <td>System</td>
        <td>ID</td>
        <td>Miller</td>
        <td>IU (OPT)</td>
        <td>ASJ (OPT)</td>
        <td>ASJ (R2SCAN)</td>
        <td>Exp</td>
    </tr>
    <tr>
        <td>AlP/Si</td>
        <td>1327/1002</td>
        <td>110/110</td>
        <td>1.24</td>
        <td>0.88</td>
        <td>1.04</td>
        <td>1.35 </td>
    </tr>
    <tr>
        <td>GaAs/Si</td>
        <td>1174/1002</td>
        <td>110/110</td>
        <td>0.30</td>
        <td>0.31</td>
        <td>0.39</td>
        <td>0.23 </td>
    </tr>
    <tr>
        <td>CdS/Si</td>
        <td>8003/1002</td>
        <td>110/110</td>
        <td>3.22</td>
        <td>1.48</td>
        <td>1.70</td>
        <td>1.6 </td>
    </tr>
    <tr>
        <td>AlAs/GaAs</td>
        <td>1372/1174</td>
        <td>110/110</td>
        <td>0.60</td>
        <td>0.48</td>
        <td>0.50</td>
        <td>0.55 </td>
    </tr>
    <tr>
        <td>CdS/CdSe</td>
        <td>8003/1192</td>
        <td>110/110</td>
        <td>0.35</td>
        <td>0.10</td>
        <td>0.11</td>
        <td>0.55 </td>
    </tr>
    <tr>
        <td>InP/GaAs</td>
        <td>1183/1174</td>
        <td>110/110</td>
        <td>0.25</td>
        <td>0.72</td>
        <td>0.75</td>
        <td>0.19</td>
    </tr>
    <tr>
        <td>ZnTe/AlSb</td>
        <td>1198/1408</td>
        <td>110/110</td>
        <td>0.8</td>
        <td>0.25</td>
        <td>0.33</td>
        <td>0.35 </td>
    </tr>
    <tr>
        <td>CdSe/ZnTe</td>
        <td>1192/1198</td>
        <td>110/110</td>
        <td>1.8</td>
        <td>0.58</td>
        <td>0.67</td>
        <td>0.64</td>
    </tr>
    <tr>
        <td>InAs/AlAs</td>
        <td>1186/1372</td>
        <td>110/110</td>
        <td>-</td>
        <td>0.46</td>
        <td>0.39</td>
        <td>0.5 </td>
    </tr>
    <tr>
        <td>InAs/AlSb</td>
        <td>1186/1408</td>
        <td>110/110</td>
        <td>-</td>
        <td>0.05</td>
        <td>0.16</td>
        <td>0.09</td>
    </tr>
    <tr>
        <td>ZnSe/InP</td>
        <td>96/1183</td>
        <td>110/110</td>
        <td>-</td>
        <td>0.13</td>
        <td>0.18</td>
        <td>0.41 </td>
    </tr>
    <tr>
        <td>InAs/InP</td>
        <td>1186/1183</td>
        <td>110/110</td>
        <td>-</td>
        <td>0.11</td>
        <td>0.09</td>
        <td>0.31</td>
    </tr>
    <tr>
        <td>ZnSe/AlAs</td>
        <td>96/1372</td>
        <td>110/110</td>
        <td>-</td>
        <td>0.38</td>
        <td>0.45</td>
        <td>0.4 </td>
    </tr>
    <tr>
        <td>GaAs/ZnSe</td>
        <td>1174/96</td>
        <td>110/110</td>
        <td>-</td>
        <td>0.72</td>
        <td>0.80</td>
        <td>0.98 </td>
    </tr>
    <tr>
        <td>ZnS/Si</td>
        <td>10591/1002</td>
        <td>001/001</td>
        <td>-</td>
        <td>0.92</td>
        <td>1.16</td>
        <td>1.52 </td>
    </tr>
    <tr>
        <td>Si/SiC</td>
        <td>1002/8118</td>
        <td>001/001</td>
        <td>-</td>
        <td>0.51</td>
        <td>0.47</td>
        <td>0.5 </td>
    </tr>
    <tr>
        <td>GaN/SiC (P)</td>
        <td>30/8118</td>
        <td>001/001</td>
        <td>-</td>
        <td>1.12</td>
        <td>1.37</td>
        <td>0.70 </td>
    </tr>
    <tr>
        <td>Si/AlN (P)</td>
        <td>1002/30</td>
        <td>001/001</td>
        <td>-</td>
        <td>3.51</td>
        <td>3.60</td>
        <td>3.5</td>
    </tr>
    <tr>
        <td>GaN/AlN (P)</td>
        <td>30/39</td>
        <td>001/001</td>
        <td>-</td>
        <td>0.80</td>
        <td>0.86</td>
        <td>0.73 </td>
    </tr>
    <tr>
        <td>AlN/InN (P)</td>
        <td>39/1180</td>
        <td>001/001</td>
        <td>-</td>
        <td>1.24</td>
        <td>1.07</td>
        <td>1.81 </td>
    </tr>
    <tr>
        <td>GaN/ZnO (P)</td>
        <td>30/1195</td>
        <td>001/001</td>
        <td>-</td>
        <td>0.51</td>
        <td>0.46</td>
        <td>0.7 </td>
    </tr>
    <tr>
        <td>MAE</td>
        <td>-</td>
        <td>-</td>
        <td>0.45</td>
        <td>0.22</td>
        <td>0.23</td>
        <td>-</td>
    </tr>

</table>

<a name="webapp"></a>
## Webapp

A preliminary web-app for making heterostructure is available at  [https://jarvis.nist.gov/jarvish/](https://jarvis.nist.gov/jarvish/)

<a name="refs"></a>
### References

1) [InterMat: Accelerated Band Offset Prediction in Semiconductor Interfaces with DFT and Deep Learning](https://arxiv.org/abs/2401.02021)
2) [Efficient computational design of two-dimensional van der Waals heterostructures: Band alignment, lattice mismatch, and machine learning](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.014009)

For more references see: [https://pages.nist.gov/jarvis/publications/](https://pages.nist.gov/jarvis/publications/)


<a name="contrib"></a>
## How to contribute


For detailed instructions, please see [Contribution instructions](https://github.com/usnistgov/jarvis/blob/master/Contribution.rst)

<a name="corres"></a>
## Correspondence


Please report bugs as Github issues (https://github.com/usnistgov/alignn/issues) or email to kamal.choudhary@nist.gov.

<a name="fund"></a>
## Funding support


[NIST-MGI](https://www.nist.gov/mgi) and [NIST-CHIPS](https://www.nist.gov/chips).

## Code of conduct


Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)
