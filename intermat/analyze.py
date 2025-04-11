"""Module for surface and interface analysis."""

import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
from numpy import ones, vstack
from numpy.linalg import lstsq
import numpy as np
import re
from ase.calculators.vasp import VaspChargeDensity
from scipy.signal import find_peaks
from jarvis.io.vasp.outputs import Outcar
from jarvis.io.vasp.outputs import Locpot
from jarvis.io.vasp.outputs import recast_array_on_uniq_array_elements
from matplotlib.gridspec import GridSpec
from jarvis.io.vasp.outputs import Vasprun
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.analysis.defects.surface import Surface

dft_3d = data("dft_3d")
step_size = 10


def get_dir(jid="JVASP-1002"):
    """Get bulk materials calc directory."""
    # pth = jid + "_R2SCAN/r2scan_" + jid + "/r2scan_" + jid + "/OUTCAR"
    pth = jid + "_OPT/opt_" + jid + "/opt_" + jid + "/OUTCAR"
    if os.path.exists(pth):
        bandg = Outcar(pth)
    else:
        pth = jid + "_R2SCAN/opt_" + jid + "/opt_" + jid + "/OUTCAR"
        bandg = Outcar(pth)
    return bandg.bandgap


def locpot_mean_jarvis(
    fname="LOCPOT", axis="X", savefile="locpot.dat", outcar="OUTCAR"
):
    """Get avg. electro. potential with jarvis-tools."""
    outcar = fname.replace("LOCPOT", "OUTCAR")
    out = Outcar(outcar)
    cbm = out.bandgap[1]  # - vac_level
    vbm = out.bandgap[2]  # - vac_level
    vrun = Vasprun(fname.replace("LOCPOT", "vasprun.xml"))
    efermi = vrun.efermi  # get_efermi(outcar)
    atoms = vrun.all_structures[-1]
    formula = atoms.composition.reduced_formula
    fin_en = vrun.final_energy
    if atoms.check_polar:
        filename1 = (
            "Polar-"
            + fname.split("/")[0].replace("R2SCAN", "PBE")
            + "_"
            + formula
            + ".png"
        )
    else:
        filename1 = (
            fname.split("/")[0].replace("R2SCAN", "PBE")
            + "_"
            + formula
            + ".png"
        )
    dif, cbm, vbm, avg_max, efermi, formula, atoms = Locpot(
        filename=fname
    ).vac_potential(
        direction="X", Ef=efermi, cbm=cbm, vbm=vbm, filename=filename1
    )
    return dif, cbm, vbm, avg_max, float(efermi), formula, atoms, fin_en


def locpot_mean(
    fname="LOCPOT", axis="z", savefile="locpot.dat", outcar="OUTCAR"
):
    """Get avg. electro. potential with ase."""
    outcar = fname.replace("LOCPOT", "OUTCAR")
    out = Outcar(outcar)

    def get_efermi(outcar="OUTCAR"):
        if not os.path.isfile(outcar):
            print("OUTCAR file not found. E-fermi set to 0.0eV")
            return None
        txt = open(outcar).read()
        efermi = re.findall(
            r"E-fermi :\s*([-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?)", txt
        )[-1][0]

        return float(efermi)

    efermi = get_efermi(outcar)

    locd = VaspChargeDensity(fname)
    atoms = locd.atoms[0]
    print(atoms)
    # print(ase_to_atoms(atoms)).composition.reduced_formula
    cell = atoms.cell
    latlens = np.linalg.norm(cell, axis=1)
    vol = np.linalg.det(cell)

    iaxis = ["x", "y", "z"].index(axis.lower())
    axes = [0, 1, 2]
    axes.remove(iaxis)
    axes = tuple(axes)

    locpot = locd.chg[0]
    # must multiply with cell volume, similar to CHGCAR

    mean = np.mean(locpot, axes) * vol

    print("gap", outcar, out.bandgap, float(efermi))
    xvals = np.linspace(0, latlens[iaxis], locpot.shape[iaxis])

    # save to 'locpot.dat'
    mean -= efermi
    avg_max = max(mean)
    print("avg_mx", avg_max)
    return xvals, mean, out.bandgap[-1]


def get_best_L(start_L, end_L, S, x_target):
    """Find best L on grid of 10 points."""
    L = 0
    best = 1000000000.0
    for L_guess in np.arange(
        start_L, end_L + 1e-5, (end_L - start_L) / (step_size + 0.55)
    ):
        current = 0.0
        for xx in x_target:
            current += abs(S(xx) - S(xx + L_guess))
        if current < best:
            best = current
            L = L_guess

    return L


def best_L_recursive(start_L, end_L, S, x_target):
    """Find recursively best L."""
    L_best = 0.0
    L_range = end_L - start_L
    for iter in range(step_size):
        L_best = get_best_L(start_L, end_L, S, x_target)
        L_range = L_range / step_size
        start_L = L_best - L_range
        end_L = L_best + L_range

    return L_best


def do_average(L, x, S):
    """Do the actual averaging."""
    AVG = []
    XX = []
    for xx in x:
        if xx - L / 2.0 < x[0]:
            continue
        if xx + L / 2.0 > x[-1]:
            continue
        XX.append(xx)

        # tmp_x = np.arange(xx - L / 2.0, xx + L / 2.0, 0.1)
        # x=np.arange(xx-L/2.0, xx+L/2.0,0.1)
        # tmp_y = S(tmp_x)
        # intg2=np.trapz(tmp_y,tmp_x,tmp_x[1]-tmp_x[0])/L
        intg2 = integrate.quad(S, xx - L / 2.0, xx + L / 2.0)[0] / L
        # print('tmp_x[1]-tmp_x[0]',tmp_x[1]-tmp_x[0])
        # intg2=integrate.simpson(tmp_y,tmp_x,tmp_x[1]-tmp_x[0]) / L
        # print('intg',int)
        AVG.append(intg2)  # integration
    #        print("int ", integrate.quad(S, xx-L/2.0, xx+L/2.0))
    return XX, AVG


def get_mean_val(x_target, XX, AVG):
    """Get man val."""
    x_target = np.array(x_target)
    XX = np.array(XX)
    AVG = np.array(AVG)
    new_x = XX.searchsorted(x_target)
    new_mean = np.mean(AVG[new_x])
    # print('new_mean',new_mean)
    m, c = get_m_c(x=XX[new_x], y=AVG[new_x])
    # print('new_x',new_x)
    # print('AVG[new_x]',AVG[new_x])
    # print('m,c',m,c)
    return new_mean, m, c


def delta_E(fname=""):
    """Get deltaE for bulk mats."""
    jid1 = fname.split("_")[0].split("Interface-")[1]
    jid2 = fname.split("_")[1]
    print("jid1, jid2", jid1, jid2)
    x1 = get_dir(jid1)
    print(jid1, x1)
    x2 = get_dir(jid2)
    print(jid2, x2)
    delta_E = x2[2] - x1[2]
    return delta_E


def get_m_c(x=[], y=[]):
    """Get least sq. fit."""
    A = vstack([x, ones(len(x))]).T
    m, c = lstsq(A, y)[0]
    return m, c


def check_inerface_polar(fname=""):
    """Check if interface is polar."""
    tmp = fname.split("/")[0].split("Interface-")[1].split("_")
    jid1 = tmp[0]
    jid2 = tmp[0]
    film_index = [int(tmp[4]), int(tmp[5]), int(tmp[6])]
    subs_index = [int(tmp[9]), int(tmp[10]), int(tmp[11])]
    film_thickness = float(tmp[14])
    subs_thickness = float(tmp[17])
    for i in dft_3d:
        if i["jid"] == jid1:
            film_atoms = Atoms.from_dict(i["atoms"])
        if i["jid"] == jid2:
            subs_atoms = Atoms.from_dict(i["atoms"])

    film_surf = Surface(
        film_atoms,
        indices=film_index,
        from_conventional_structure=True,
        thickness=film_thickness,
        # vacuum=vacuum,
    ).make_surface()
    subs_surf = Surface(
        subs_atoms,
        indices=subs_index,
        from_conventional_structure=True,
        thickness=subs_thickness,
        # vacuum=vacuum,
    ).make_surface()
    polar = False
    if film_surf.check_polar:
        polar = True
    if subs_surf.check_polar:
        polar = True
    return polar


def offset(
    fname="",
    x=[],
    s=[],
    width=5,
    left_index=-1,
    polar=None,
    deltaE=None,
    filename="offset.png",
):
    """Get valence band offset."""
    if len(x) == 0:
        x, s, _ = locpot_mean(fname)
    if polar is None:
        polar = check_inerface_polar(fname)
    print("Check polar", polar)
    if deltaE is None:
        deltaE = delta_E(fname)
    S = CubicSpline(x, s)

    max_peaks, properties = find_peaks(s, prominence=1, width=width)
    max_peaks = max_peaks[:-1]

    print("Number of peaks ", len(max_peaks))
    if left_index == -1:  # automatically pick left_index from max_peaks
        print("auto detect left index")
        if len(max_peaks) <= 8:
            print("WARNING, not many peaks found")
            left_index = 1
        elif len(max_peaks) <= 12:
            left_index = 2
        else:
            left_index = 3
    else:
        print("use input left index")
    print("left index ", left_index)
    right_index = left_index * -1 + 1

    plt.plot(x[max_peaks], s[max_peaks], "x")
    # tmp=int((max_peaks[left_index]-max_peaks[left_index+2]))
    x_target1 = x[
        np.arange(max_peaks[left_index], max_peaks[left_index + 1], 2)
    ]

    # initial guess left
    L_guess_peaks_left = x_target1[-1] - x_target1[0]

    # points in left cell
    # x_target1 = x[ range(50, 100,2) ]
    print("Initial guess left ", L_guess_peaks_left)
    L = best_L_recursive(1.0, L_guess_peaks_left * 1.5, S, x_target1)
    print("Lleft ", L)

    plt.plot(x, s, c="k")
    XX, AVG = do_average(L, x, S)

    meanval1, m1, c1 = get_mean_val(x_target1, XX, AVG)
    if polar:
        plt.plot(XX, np.array(XX) * m1 + c1, c="purple", linestyle="-.")
    # print('x_target1',x_target1)
    # print('XX',XX)
    # print('AVG',AVG)
    plt.plot(XX, AVG, c="r")

    # x_target2 = x[-200:-150]
    # tmp=int((max_peaks[right_index-1]-max_peaks[right_index]))
    x_target2 = x[
        np.arange(max_peaks[right_index - 1], max_peaks[right_index], 2)
    ]
    # x_target2 = x[ range(-100, ,2) ]

    # initial guess right
    L_guess_peaks_right = x_target2[-1] - x_target2[0]
    L = best_L_recursive(1.0, L_guess_peaks_right * 1.5, S, x_target2)
    print("Initial guess right ", L_guess_peaks_right)
    print("Lright ", L)

    plt.plot(x, s, c="k")
    XX, AVG = do_average(L, x, S)
    meanval2, m2, c2 = get_mean_val(x_target2, XX, AVG)
    if polar:
        plt.plot(XX, np.array(XX) * m2 + c2, c="orange", linestyle="-.")

    if polar:
        mid_point = int((len(XX) - 1) / 2)
        polar_del_V = (np.array(XX) * m2 + c2)[mid_point] - (
            np.array(XX) * m1 + c1
        )[mid_point]
        plt.plot(mid_point, -7, "*")
        plt.axvline(x=XX[mid_point], linestyle="-.", c="blue")
        print(
            "polar delV,mid_point",
            polar_del_V,
            mid_point,
            len(XX),
            XX[mid_point],
        )

    plt.plot(XX, AVG, c="g")
    plt.plot(x_target1, S(x_target1), "c")
    plt.plot(x_target2, S(x_target2), "c")
    deltaV = meanval2 - meanval1
    phi = deltaV + deltaE
    if polar:
        phi = polar_del_V + deltaE
        deltaV = polar_del_V
    plt.grid(color="gray", ls="-.")
    plt.minorticks_on()
    plt.ylabel("Potential (eV)")
    plt.xlim(0, max(x))
    plt.xlabel(r"Distance ($\AA$)")
    # plt.plot(x,[meanval1 for i in range(len(x))],linestyle='-.',color='blue')
    # plt.plot(x,[meanval2 for i in range(len(x))],linestyle='-.',color='blue')
    print("meanval ", [meanval1, meanval2], meanval2 - meanval1, phi)
    plt.title("Offset (eV): " + str(round(phi, 2)))
    # if filename is None:
    #    filename = "offset_max-" + fname.split("/")[0] + ".png"
    print("deltaE", deltaE)
    # plt.show()
    plt.savefig(filename)
    plt.close()
    info = {}
    info["phi"] = phi
    info["polar"] = polar
    info["deltaV"] = deltaV
    info["deltaE"] = deltaE
    info["L_guess_peaks_right"] = L_guess_peaks_right
    info["L"] = L
    info["left_index"] = left_index

    return info  # phi


def atomdos(
    vrun_file="",
    uniq_colors=["r", "g", "b", "orange", "cyan", "pink"],
    num_atoms_include=None,
):
    """Get atom projected DOS."""
    vrun = Vasprun(vrun_file)
    # spin_pol = self.is_spin_polarized
    atoms = vrun.all_structures[-1]
    num_atoms = atoms.num_atoms
    if num_atoms_include is None:
        num_atoms_include = num_atoms
    elements = atoms.elements
    unique_elements = atoms.uniq_species
    pdos = vrun.partial_dos_spdf  # spin,atom,spdf
    energy = pdos[0][0]["energy"] - vrun.efermi
    element_dict = recast_array_on_uniq_array_elements(
        unique_elements, elements
    )
    valid_keys = []
    info = {}
    info["spin_up_info"] = {}
    info["spin_down_info"] = {}
    info["energy"] = energy
    for i in pdos[0][0].keys():
        if "energ" not in i:
            valid_keys.append(i)
    # print (valid_keys)
    spin_up_info = {}
    for i in range(num_atoms):
        spin_up_info[i] = np.zeros(len(energy))
    for atom in range(num_atoms):
        for k in valid_keys:
            spin_up_info[atom] += pdos[0][atom][k]

    info["spin_up_info"] = spin_up_info
    if vrun.is_spin_polarized:
        spin_down_info = {}
        for i in range(num_atoms):
            spin_down_info[i] = np.zeros(len(energy))
        for atom in range(num_atoms):
            for k in valid_keys:
                spin_up_info[atom] += -1 * pdos[0][atom][k]

        info["spin_down_info"] = spin_down_info

    index = np.argsort(atoms.cart_coords[:, 2])
    the_grid = GridSpec(1, num_atoms + 1)
    plt.rcParams.update({"font.size": 18})
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(wspace=0.0)
    count = 0
    mx = 3
    for i in range(num_atoms):
        if i < num_atoms_include:
            count += 1
            plt.subplot(the_grid[0, count])
            c = uniq_colors[unique_elements.index(elements[index[i]])]
            tmp = np.abs(info["spin_up_info"][index[i]])
            # plt.plot(tmp,info["energy"],c=c,alpha=0.8)
            # plt.hist(tmp,bins=500)
            plt.barh(info["energy"], tmp, color=c)
            print(c, (elements[index[i]]))
            plt.ylim([-4, 4])
            mval = np.max(tmp) / mx
            # plt.xlim([0,mval])
            if i != 0:
                plt.xticks([])
                plt.yticks([])
                plt.axis("off")
    plt.savefig("atomdos.png")
    plt.close()


"""
import glob
if __name__ == "__main__":
    for i in glob.glob("Int*/opt_*/opt*/LOCPOT"):
        offset(fname=i, left_index=2)
"""
