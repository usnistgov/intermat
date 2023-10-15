# from ase import Atoms
from gpaw import GPAW, Mixer, FermiDirac
from gpaw.lcao.tools import (
    remove_pbc,
    get_lcao_hamiltonian,
    get_lead_lcao_hamiltonian,
)
import pickle as pickle
from ase.transport.calculators import TransportCalculator
import pickle
from jarvis.io.vasp.inputs import Poscar
import numpy as np
import pylab
import pylab
import numpy as np
import matplotlib.pyplot as plt
from jarvis.core.atoms import Atoms
from main import InterfaceCombi
from calculators import Calc


def lead_mat_designer(
    lead="JVASP-813",
    mat="JVASP-1002",
    film_index=[0, 0, 1],
    subs_index=[0, 0, 1],
    disp_intvl=0.1,
    seperations=[2.5],
    fast_checker="ewald",
    dataset=[],
    film_thickness=12,
    subs_thickness=12,
    tol=1,
    lead_ratio=0.15,
    iv_tb3=True,
):
    jid_film = lead
    jid_subs = mat
    x = InterfaceCombi(
        dataset=dataset,
        film_indices=[film_index],
        subs_indices=[subs_index],
        film_ids=[jid_film],
        subs_ids=[jid_subs],
        disp_intvl=disp_intvl,
        film_thicknesses=[film_thickness],
        subs_thicknesses=[subs_thickness],
        seperations=seperations,
    )

    if fast_checker == "ewald":
        wads = x.calculate_wad_ewald()
        wads = np.array(x.wads["ew_wads"])
    elif fast_checker == "alignn":
        wads = x.calculate_wad_alignn()
        wads = np.array(x.wads["alignn_wads"])
    else:
        raise ValueError("Not implemented", fast_checker)

    index = np.argmin(wads)

    atoms = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )

    film_index = [0, 0, 1]
    seperations = np.array(seperations) + tol
    x = InterfaceCombi(
        dataset=dataset,
        film_indices=[film_index],
        subs_indices=[subs_index],
        subs_ids=[jid_film],
        film_mats=[atoms],
        disp_intvl=disp_intvl,
        seperations=seperations,
        # film_thicknesses=[film_thickness],
        subs_thicknesses=[subs_thickness],
    )

    if fast_checker == "ewald":
        wads = x.calculate_wad_ewald()
        wads = np.array(x.wads["ew_wads"])
    elif fast_checker == "alignn":
        wads = x.calculate_wad_alignn()
        wads = np.array(x.wads["alignn_wads"])
    else:
        raise ValueError("Not implemented", fast_checker)

    index = np.argmin(wads)

    combined = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    combined = combined.center(vacuum=tol)
    # combined = combined.center(vacuum=seperations[0]-tol)
    lat_mat = combined.lattice_mat
    coords = combined.frac_coords
    elements = combined.elements
    props = combined.props
    tmp = lat_mat.copy()
    indx = 2
    tmp[indx] = lat_mat[0]
    tmp[0] = lat_mat[indx]
    lat_mat = tmp
    tmp = coords.copy()
    tmp[:, indx] = coords[:, 0]
    tmp[:, 0] = coords[:, indx]
    coords = tmp
    combined = Atoms(
        lattice_mat=lat_mat,
        coords=coords,
        elements=elements,
        cartesian=False,
        props=props,
    ).center_around_origin([0.5, 0, 0])
    # calc=Calc(atoms=combined,method='matgl',relax_cell=True)
    # info=calc.predict()
    # combined=info['atoms']
    print("combined", combined)
    a = combined.lattice.abc[0]
    coords = combined.frac_coords
    lattice_mat = combined.lattice_mat
    elements = np.array(combined.elements)
    coords_left = coords[coords[:, 0] < lead_ratio]
    elements_left = elements[coords[:, 0] < lead_ratio]
    # elements_left=['Xe' for i in range(len(elements_left))]
    atoms_left = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_left,
        coords=coords_left,
        cartesian=False,
    )

    coords_right = coords[coords[:, 0] > 1 - lead_ratio]
    elements_right = elements[coords[:, 0] > 1 - lead_ratio]
    # elements_right=['Ar' for i in range(len(elements_left))]
    atoms_right = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_right,
        coords=coords_right,
        cartesian=False,
    )

    coords_middle = coords[
        (coords[:, 0] <= 1 - lead_ratio) & (coords[:, 0] >= lead_ratio)
    ]
    elements_middle = elements[
        (coords[:, 0] <= 1 - lead_ratio) & (coords[:, 0] >= lead_ratio)
    ]
    atoms_middle = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_middle,
        coords=coords_middle,
        cartesian=False,
    )
    atoms_left = atoms_left.center(axis=0, vacuum=1.0)
    atoms_right = atoms_right.center(axis=0, vacuum=1.0)
    info = {}
    info["atoms_left"] = atoms_left
    info["atoms_right"] = atoms_right
    info["atoms_middle"] = atoms_middle
    info["combined"] = combined
    if iv_tb3:
        calc = Calc(atoms=atoms_left, method="tb3", jobname="tb3_left")
        en = calc.predict()
        calc = Calc(atoms=atoms_right, method="tb3", jobname="tb3_right")
        en = calc.predict()
        calc = Calc(atoms=combined, method="tb3", jobname="tb3_all")
        en = calc.predict()

        h = np.load("tb3_all/hk.npz")
        s = np.load("tb3_all/sk.npz")
        h1 = np.load("tb3_left/hk.npz")
        s1 = np.load("tb3_left/sk.npz")
        h2 = np.load("tb3_right/hk.npz")
        s2 = np.load("tb3_right/sk.npz")
        energies = np.arange(-0.5, 0.5, 0.001)
        tcalc = TransportCalculator(
            h=h, h1=h1, h2=h2, s=s, s1=s1, s2=s2, align_bf=1, energies=energies
        )
        T_e = tcalc.get_transmission()
        current = tcalc.get_current(bias, T=0.0)
        plt.plot(bias, 2000 * units._e**2 / units._hplanck * current)
        plt.ylabel("uI [A]")
        plt.xlabel("U [V]")
        plt.savefig("ii.png")
        plt.close()
        x = np.arange(-0.5, 0.5, 0.001)
        y = []
        for i in x:
            c = tcalc.get_current(bias=i)
            y.append(c)

        plt.plot(x, y)
        plt.savefig("i.png")
        plt.close()

    return info


def gpaw_iv(combined=[], lead_ratio=0.15, energies=np.arange(-3, 3, 0.1)):
    a = combined.lattice.abc[0]
    coords = combined.frac_coords
    lattice_mat = combined.lattice_mat
    elements = np.array(combined.elements)
    coords_left = coords[coords[:, 0] < lead_ratio]
    elements_left = elements[coords[:, 0] < lead_ratio]
    # elements_left=['Xe' for i in range(len(elements_left))]
    atoms_left = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_left,
        coords=coords_left,
        cartesian=False,
    )

    coords_right = coords[coords[:, 0] > 1 - lead_ratio]
    elements_right = elements[coords[:, 0] > 1 - lead_ratio]
    # elements_right=['Ar' for i in range(len(elements_left))]
    atoms_right = JAtoms(
        lattice_mat=lattice_mat,
        elements=elements_right,
        coords=coords_right,
        cartesian=False,
    )

    coords_middle = coords[
        (coords[:, 0] <= 1 - lead_ratio) & (coords[:, 0] >= lead_ratio)
    ]
    elements_middle = elements[
        (coords[:, 0] <= 1 - lead_ratio) & (coords[:, 0] >= lead_ratio)
    ]
    atoms_middle = JAtoms(
        lattice_mat=lattice_mat,
        elements=elements_middle,
        coords=coords_middle,
        cartesian=False,
    )
    atoms_left = atoms_left.center(axis=0, vacuum=1.0)
    atoms_right = atoms_right.center(axis=0, vacuum=1.0)

    # Attach a GPAW calculator
    calc = GPAW(
        h=0.3,
        xc="PBE",
        basis="szp(dzp)",
        occupations=FermiDirac(width=0.1),
        kpts=(1, 1, 1),
        mode="lcao",
        txt="pt_h2_lcao_scat.txt",
        mixer=Mixer(0.1, 5, weight=100.0),
        symmetry={"point_group": False, "time_reversal": False},
    )
    atoms = combined.ase_converter()
    # atoms.set_pbc=(1, 0, 0)
    atoms.calc = calc

    atoms.get_potential_energy()  # Converge everything!
    Ef = atoms.calc.get_fermi_level()

    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    # Only use first kpt, spin, as there are no more
    H, S = H_skMM[0, 0], S_kMM[0]
    H -= Ef * S
    # remove_pbc(atoms, H, S, 0)

    # Dump the Hamiltonian and Scattering matrix to a pickle file
    pickle.dump(
        (H.astype(complex), S.astype(complex)), open("scat_hs.pickle", "wb"), 2
    )

    ########################
    # Left principal layer #
    ########################

    # Use four Pt atoms in the lead, so only take those from before
    atoms = atoms_left.ase_converter()

    # Attach a GPAW calculator
    calc = GPAW(
        h=0.3,
        xc="PBE",
        basis="szp(dzp)",
        occupations=FermiDirac(width=0.1),
        kpts=(4, 1, 1),  # More kpts needed as the x-direction is shorter
        mode="lcao",
        txt="pt_h2_lcao_llead.txt",
        mixer=Mixer(0.1, 5, weight=100.0),
        symmetry={"point_group": False, "time_reversal": False},
    )
    atoms.calc = calc

    atoms.get_potential_energy()  # Converge everything!
    Ef = atoms.calc.get_fermi_level()

    ibz2d_k, weight2d_k, H_skMM, S_kMM = get_lead_lcao_hamiltonian(calc)
    # Only use first kpt, spin, as there are no more
    H, S = H_skMM[0, 0], S_kMM[0]
    H -= Ef * S

    # Dump the Hamiltonian and Scattering matrix to a pickle file
    pickle.dump((H, S), open("lead1_hs.pickle", "wb"), 2)

    #########################
    # Right principal layer #
    #########################
    # This is identical to the left prinicpal layer so we don't have to do anything
    # Just dump the same Hamiltonian and Scattering matrix to a pickle file

    atoms = atoms_right.ase_converter()

    # Attach a GPAW calculator
    calc = GPAW(
        h=0.3,
        xc="PBE",
        basis="szp(dzp)",
        occupations=FermiDirac(width=0.1),
        kpts=(4, 1, 1),  # More kpts needed as the x-direction is shorter
        mode="lcao",
        txt="pt_h2_lcao_llead.txt",
        mixer=Mixer(0.1, 5, weight=100.0),
        symmetry={"point_group": False, "time_reversal": False},
    )
    atoms.calc = calc

    atoms.get_potential_energy()  # Converge everything!
    Ef = atoms.calc.get_fermi_level()

    ibz2d_k, weight2d_k, H_skMM, S_kMM = get_lead_lcao_hamiltonian(calc)
    # Only use first kpt, spin, as there are no more
    H, S = H_skMM[0, 0], S_kMM[0]
    H -= Ef * S

    # Dump the Hamiltonian and Scattering matrix to a pickle file
    pickle.dump((H, S), open("lead2_hs.pickle", "wb"), 2)

    # Read in the hamiltonians
    h, s = pickle.load(open("scat_hs.pickle", "rb"))
    h1, s1 = pickle.load(open("lead1_hs.pickle", "rb"))
    h2, s2 = pickle.load(open("lead2_hs.pickle", "rb"))

    tcalc = TransportCalculator(
        h=h, h1=h1, h2=h2, s=s, s1=s1, s2=s2, align_bf=1, energies=energies
    )

    T_e = tcalc.get_transmission()
    pylab.plot(tcalc.energies, T_e)
    pylab.title("Transmission function")
    pylab.savefig("t.png")
    pylab.close()

    x = np.arange(-0.5, 0.5, 0.01)
    y = []
    for i in x:
        c = tcalc.get_current(bias=i)
        y.append(c)

    plt.plot(x, y)
    pylab.savefig("i.png")
    pylab.close()


x = lead_mat_designer()
print(x)
# iv_calculator(combined=combined,)
