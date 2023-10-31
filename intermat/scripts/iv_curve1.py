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
import os
from ase import units


def rotate_atoms(atoms=[], index_1=0, index_2=2):
    """Rotate structure."""
    # x,y,z : 0,1,2
    lat_mat = atoms.lattice_mat
    coords = atoms.frac_coords
    elements = atoms.elements
    props = atoms.props
    tmp = lat_mat.copy()
    tmp[index_2] = lat_mat[index_1]
    tmp[index_1] = lat_mat[index_2]
    lat_mat = tmp
    tmp = coords.copy()
    tmp[:, index_2] = coords[:, index_1]
    tmp[:, index_1] = coords[:, index_2]
    coords = tmp
    atoms = Atoms(
        lattice_mat=lat_mat,
        coords=coords,
        elements=elements,
        cartesian=False,
        props=props,
    )
    return atoms


def calc_iv_tb3(
    atoms_left=[],
    atoms_right=[],
    combined=[],
    energies=np.arange(-1.0, 1.0, 0.01),
    prefix=None,
):
    tb3_params = [
        "using ThreeBodyTB\nusing NPZ\n",
        'crys = makecrys("POSCAR")\n',
        "energy, tbc, flag = scf_energy(crys,mixing_mode=:simple,mix=0.05,iters=500);\n",
        # "cfinal, tbc, energy, force, stress = relax_structure(crys,mixing_mode=:simple,mix=0.05);\n",
        # "println(cfinal)\n",
        "vects, vals, hk, sk, vals0 = ThreeBodyTB.TB.Hk(tbc,[0,0,0])\n",
        # ThreeBodyTB.TB.write_tb_crys("tbc.xml.gz",tbc)\n',
        'npzwrite("hk.npz",hk)\n',
        'npzwrite("sk.npz",sk)\n',
        'open("energy","w") do file\n',
        "write(file,string(energy))\n",
        "end\n",
        'open("fermi_energy","w") do file\n',
        'println("efermi",string(tbc.efermi))\n',
        "write(file,string(tbc.efermi))\n",
        "end\n",
        'open("dq","w") do file\n',
        'println("dq",(ThreeBodyTB.TB.get_dq(tbc)))\n',
        "write(file,string(ThreeBodyTB.TB.get_dq(tbc)))\n",
        "end\n",
    ]

    if not prefix:
        prefix = combined.composition.reduced_formula + "_10-20-2023_heavy_left_tot"
    extra_params = {}
    extra_params["tb3_params"] = tb3_params
    jobname_left = prefix + "_tb3_left"
    calc = Calc(
        atoms=atoms_left,
        method="tb3",
        jobname=jobname_left,
        extra_params=extra_params,
    )
    en = calc.predict()




    jobname_right = prefix + "_tb3_right"
    extra_params["tb3_params"] = [
        "using ThreeBodyTB\nusing NPZ\n",
        'crys = makecrys("POSCAR")\n',
        #"energy, tbc, flag = scf_energy(crys,mixing_mode=:simple,mix=0.05,iters=500);\n",
        "energy, tbc, flag = scf_energy(crys,mixing_mode=:simple,mix=0.05,tot_charge=1.0,iters=500);\n",
        # "cfinal, tbc, energy, force, stress = relax_structure(crys,mixing_mode=:simple,mix=0.05);\n",
        # "println(cfinal)\n",
        "vects, vals, hk, sk, vals0 = ThreeBodyTB.TB.Hk(tbc,[0,0,0])\n",
        # ThreeBodyTB.TB.write_tb_crys("tbc.xml.gz",tbc)\n',
        'npzwrite("hk.npz",hk)\n',
        'npzwrite("sk.npz",sk)\n',
        'open("energy","w") do file\n',
        "write(file,string(energy))\n",
        "end\n",
        'open("fermi_energy","w") do file\n',
        'println("efermi",string(tbc.efermi))\n',
        "write(file,string(tbc.efermi))\n",
        "end\n",
        'open("dq","w") do file\n',
        'println("dq",(ThreeBodyTB.TB.get_dq(tbc)))\n',
        "write(file,string(ThreeBodyTB.TB.get_dq(tbc)))\n",
        "end\n",
    ]
    calc = Calc(
        atoms=atoms_right,
        method="tb3",
        jobname=jobname_right,
        extra_params=extra_params,
    )
    en = calc.predict()






    extra_params = {}
    extra_params["tb3_params"] = tb3_params
    jobname_left = prefix + "_tb3_left"
    jobname_all = prefix + "_tb3_all"
    calc = Calc(
        atoms=combined,
        method="tb3",
        jobname=jobname_all,
        extra_params=extra_params,
    )
    en = calc.predict()




    fname = os.path.join(jobname_all, "hk.npz")
    h = np.load(fname)
    fname = os.path.join(jobname_all, "sk.npz")
    s = np.load(fname)
    fname = os.path.join(jobname_left, "hk.npz")
    h1 = np.load(fname)
    fname = os.path.join(jobname_left, "sk.npz")
    s1 = np.load(fname)
    fname = os.path.join(jobname_right, "hk.npz")
    h2 = np.load(fname)
    fname = os.path.join(jobname_right, "sk.npz")
    s2 = np.load(fname)
    tcalc = TransportCalculator(
        h=h, h1=h1, h2=h2, s=s, s1=s1, s2=s2, align_bf=1, energies=energies
    )
    T_e = tcalc.get_transmission()
    current = tcalc.get_current(energies, T=0.0)
    plt.plot(energies, units._e**2 / units._hplanck * current)
    plt.ylabel("uI [A]")
    plt.xlabel("U [V]")
    fname = combined.composition.reduced_formula + "_iv_tb3.png"
    plt.savefig(fname)
    plt.close()
    y = []
    for i in energies:
        c = tcalc.get_current(bias=i)
        y.append(c)

    plt.plot(energies, y)
    fname = combined.composition.reduced_formula + "_iv_tb3v3.png"
    plt.savefig(fname)
    plt.close()


def calc_iv_gpaw(
    atoms_left=[],
    atoms_right=[],
    combined=[],
    energies=np.arange(-0.5, 0.5, 0.001),
    prefix=None,
):
    if not prefix:
        prefix = combined.composition.reduced_formula
    from gpaw import GPAW, Mixer, FermiDirac
    from gpaw.lcao.tools import (
        remove_pbc,
        get_lcao_hamiltonian,
        get_lead_lcao_hamiltonian,
    )

    # Attach a GPAW calculator
    calc = GPAW(
        h=0.3,
        xc="PBE",
        basis="szp(dzp)",
        occupations=FermiDirac(width=0.1),
        kpts=(1, 2, 2),
        mode="lcao",
        nbands="200%",
        txt="pt_h2_lcao_scat.txt",
        mixer=Mixer(0.1, 5, weight=100.0),
        symmetry={"point_group": False, "time_reversal": False},
    )
    atoms = combined.ase_converter()
    atoms.calc = calc
    atoms.get_potential_energy()  # Converge everything!
    Ef = atoms.calc.get_fermi_level()
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    # Only use first kpt, spin, as there are no more
    H, S = H_skMM[0, 0], S_kMM[0]
    H -= Ef * S
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
        nbands="200%",
        basis="szp(dzp)",
        occupations=FermiDirac(width=0.1),
        kpts=(2, 2, 2),  # More kpts needed as the x-direction is shorter
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
        kpts=(2, 2, 2),  # More kpts needed as the x-direction is shorter
        mode="lcao",
        nbands="200%",
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
    current = tcalc.get_current(energies, T=0.0)
    plt.plot(energies, units._e**2 / units._hplanck * current)
    plt.ylabel("uI [A]")
    plt.xlabel("U [V]")
    fname = combined.composition.reduced_formula + "_iv_gpaw.png"
    plt.savefig(fname)
    plt.close()
    pylab.plot(tcalc.energies, T_e)
    pylab.title("Transmission function")
    pylab.savefig("tranmission_gpaw.png")
    pylab.close()

    y = []
    for i in energies:
        c = tcalc.get_current(bias=i)
        y.append(c)

    plt.plot(energies, y)
    pylab.savefig("iv_gpaw2.png")
    pylab.close()


def check_frac(n, sanitize_tol=2e-4):
    """Check fractional coordinates or lattice parameters."""
    items = [
        0.0,
        0.3333333333333333,
        0.25,
        0.5,
        0.75,
        0.6666666666666667,
        1.0,
        1.5,
        2.0,
        -0.5,
        -2.0,
        -1.5,
        -1.0,
        1.0 / 2.0**0.5,
        -1.0 / 2.0**0.5,
        3.0**0.5 / 2.0,
        -(3.0**0.5) / 2.0,
        1.0 / 3.0**0.5,
        -1.0 / 3.0**0.5,
        1.0 / 2.0 / 3**0.5,
        -1.0 / 2.0 / 3**0.5,
        1 / 6,
        5 / 6,
    ]
    items = items + [(-1) * i for i in items]
    for f in items:
        if abs(f - n) < sanitize_tol:
            return f % 1
    return n


def sanitize_atoms(atoms=[]):
    coords = np.array(atoms.frac_coords)
    ntot = atoms.num_atoms
    for i in range(ntot):
        for j in range(3):  # neatin
            coords[i, j] = check_frac(n=coords[i, j])
    lattice_mat = atoms.lattice_mat
    elements = atoms.elements
    props = atoms.props
    atoms = Atoms(
        lattice_mat=lattice_mat,
        elements=elements,
        props=props,
        coords=coords,
        cartesian=False,
    )
    return atoms


def divide_atoms_left_right(combined=[], indx=0, lead_ratio=0.15):
    # combined=sanitize_atoms(combined)
    a = combined.lattice.abc[0]
    coords = combined.frac_coords % 1
    # print('coords2',coords)
    lattice_mat = combined.lattice_mat
    elements = np.array(combined.elements)
    props = np.array(combined.props)
    coords_left = coords[coords[:, indx] < lead_ratio]
    elements_left = elements[coords[:, indx] < lead_ratio]
    props_left = props[coords[:, indx] < lead_ratio]
    # elements_left=['Xe' for i in range(len(elements_left))]
    atoms_left = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_left,
        coords=coords_left,
        cartesian=False,
        # props=props_left
    )

    coords_right = coords[coords[:, indx] > 1 - lead_ratio]
    elements_right = elements[coords[:, indx] > 1 - lead_ratio]
    # elements_right=['Ar' for i in range(len(elements_left))]
    props_right = props[coords[:, indx] > 1 - lead_ratio]
    atoms_right = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_right,
        coords=coords_right,
        cartesian=False,
        # props=props_right
    )

    coords_middle = coords[
        (coords[:, indx] <= 1 - lead_ratio) & (coords[:, indx] >= lead_ratio)
    ]
    elements_middle = elements[
        (coords[:, indx] <= 1 - lead_ratio) & (coords[:, indx] >= lead_ratio)
    ]
    props_middle = props[
        (coords[:, indx] <= 1 - lead_ratio) & (coords[:, indx] >= lead_ratio)
    ]
    atoms_middle = Atoms(
        lattice_mat=lattice_mat,
        elements=elements_middle,
        coords=coords_middle,
        cartesian=False,
        # props=props_middle
    )
    atoms_left = atoms_left.center(axis=0, vacuum=1.0)
    atoms_right = atoms_right.center(axis=0, vacuum=1.0)
    info = {}
    info["atoms_left"] = atoms_left
    info["atoms_right"] = atoms_right
    info["atoms_middle"] = atoms_middle
    info["combined"] = combined
    print("Submit calc")
    return info


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
    iv_tb3=False,
    iv_gpaw=False,
    rotate_xz=True,
    try_center=True,
    center_value=None,
    # center_value=0.5,
    vasp_job=False,
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

    """
    atoms = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    subs_index = [0, 0, 1]
    seperations = np.array(seperations) + tol
    x = InterfaceCombi(
        dataset=dataset,
        film_indices=[film_index],
        subs_indices=[subs_index],
        film_ids=[jid_film],
        subs_mats=[atoms],
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
    """

    combined = Atoms.from_dict(
        x.generated_interfaces[index]["generated_interface"]
    )
    print("Initial combined", combined)
    combined = combined.center(vacuum=tol)
    if center_value is None:
        film_sl_element = Atoms.from_dict(
            x.generated_interfaces[index]["film_sl"]
        ).elements[0]
        print("film_sl_element", film_sl_element)
        channel_coords = []
        for i, j in zip(combined.frac_coords, combined.elements):
            if j == film_sl_element:
                # if j!=film_sl_element:
                channel_coords.append(i)
        channel_coords = np.array(channel_coords)
        mean_val = np.mean(channel_coords, axis=0)
        mean_val = np.min(channel_coords, axis=0) - 0.02
        if rotate_xz:
            center_value = mean_val[0]
        else:
            center_value = mean_val[2]

    combined = combined.center(vacuum=seperations[0] - tol)
    center_point = [0, 0, center_value]
    if rotate_xz:
        combined = rotate_atoms(atoms=combined)
        center_point = [center_value, 0, 0]
    print("center_value used !!!", center_value)
    if try_center:
        combined = combined.center_around_origin(center_point)
    # calc=Calc(atoms=combined,method='matgl',relax_cell=True)
    # info=calc.predict()
    # combined=info['atoms']
    print("combined", combined)
    indx = 2
    if rotate_xz:
        indx = 0

    info = divide_atoms_left_right(
        combined=combined, indx=0, lead_ratio=lead_ratio
    )
    atoms_left = info["atoms_left"]
    atoms_right = info["atoms_right"]
    print("atoms_left", atoms_left)
    print("atoms_right", atoms_right)
    v_jobname = (
        "vasp_" + lead + "_" + mat + "_" + combined.composition.reduced_formula
    )
    if vasp_job:
        calc = Calc(
            atoms=combined, method="vasp", relax_cell=True, jobname=v_jobname
        ).predict()
    if iv_tb3:
        calc_iv_tb3(
            atoms_left=atoms_left,
            atoms_right=atoms_right,
            combined=combined,
            energies=np.arange(-1.0, 1.0, 0.01),
        )
    if iv_gpaw:
        calc_iv_gpaw(
            atoms_left=atoms_left,
            atoms_right=atoms_right,
            combined=combined,
            energies=np.arange(-1.0, 1.0, 0.01),
        )

    return combined


if __name__ == "__main__":
    # q=Atoms.from_poscar('q')
    # info = divide_atoms_left_right(combined=q, indx=0, lead_ratio=0.15)
    # print(info)
    #atoms_left = Atoms.from_poscar("atk_right/POSCAR")
    #atoms_right = Atoms.from_poscar("atk_right/POSCAR")
    #atoms_middle = Atoms.from_poscar("atk_middle/POSCAR")
    #calc_iv_tb3(
    #    atoms_left=atoms_left, atoms_right=atoms_right, combined=atoms_middle
    #)
    import sys

    x = lead_mat_designer(
        rotate_xz=True,
        # lead="JVASP-972",
        lead="JVASP-813",
        mat="JVASP-1002",
        iv_tb3=True,
        subs_thickness=150,
        film_thickness=25,
        # subs_thickness=50,
        disp_intvl=0.0,
        # center_value=0.6,
        center_value=None,
        seperations=[2.5],
        # lead_ratio=0.15,
        lead_ratio=0.05,
        vasp_job=False,
    )
    sys.exit()

    x = lead_mat_designer(
        rotate_xz=False,
        lead="JVASP-1029",
        mat="JVASP-1109",
        iv_tb3=False,
        disp_intvl=0.1,
        center_value=0.82,
        seperations=[2.0],
        lead_ratio=0.15,
    )
    # print(x)
    combined = Atoms.from_poscar(
        "/rk2/knc6/Interfaces/InterMat2/intermat/intermat/tmp/vasp_job_ag_si/vasp_job_ag_si/CONTCAR"
    )

    combined = rotate_atoms(atoms=combined)
    center_point = [0.5, 0, 0]
    combined = combined.center_around_origin(center_point)
    print("combined", combined)

    info = divide_atoms_left_right(combined=combined, indx=0, lead_ratio=0.15)
    atoms_left = info["atoms_left"]
    atoms_right = info["atoms_right"]

    # calc_iv_gpaw(
    #    atoms_left=atoms_left,
    #    atoms_right=atoms_right,
    #    combined=combined,
    #    energies=np.arange(-0.5, 0.5, 0.001),
    # )
    # calc_iv_tb3(
    #    atoms_left=atoms_left,
    #    atoms_right=atoms_right,
    #    combined=combined,
    #    energies=np.arange(-0.5, 0.5, 0.001),
    # )
