import numpy as np
import re
import matplotlib.pyplot as plt
from ase.calculators.vasp import VaspChargeDensity
from scipy.signal import find_peaks
import os
from jarvis.io.vasp.outputs import Outcar


def locpot_mean(
    fname="LOCPOT", axis="z", savefile="locpot.dat", outcar="OUTCAR"
):
    outcar = fname.replace("LOCPOT", "OUTCAR")
    out = Outcar(outcar)

    def get_efermi(outcar="OUTCAR"):
        if not os.path.isfile(outcar):
            logger.warning("OUTCAR file not found. E-fermi set to 0.0eV")
            return None
        txt = open(outcar).read()
        efermi = re.findall(
            r"E-fermi :\s*([-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?)", txt
        )[-1][0]

        return float(efermi)

    efermi = get_efermi(outcar)
    print("gap", outcar, out.bandgap, float(efermi))

    locd = VaspChargeDensity(fname)
    cell = locd.atoms[0].cell
    latlens = np.linalg.norm(cell, axis=1)
    vol = np.linalg.det(cell)

    iaxis = ["x", "y", "z"].index(axis.lower())
    axes = [0, 1, 2]
    axes.remove(iaxis)
    axes = tuple(axes)

    locpot = locd.chg[0]
    # must multiply with cell volume, similar to CHGCAR

    mean = np.mean(locpot, axes) * vol

    xvals = np.linspace(0, latlens[iaxis], locpot.shape[iaxis])

    # save to 'locpot.dat'
    mean -= efermi
    avg_max = max(mean)
    return xvals, mean, out.bandgap[-1]


def get_offset(
    fname="LOCPOT",
    width=5,
    left_index=2,
    window=2,
    step=100,
    filename="offset.png",
):
    xvals, mean, _ = locpot_mean(fname)
    right_index = -1 * left_index

    max_peaks, properties = find_peaks(mean, prominence=1, width=width)
    plt.plot(mean)
    max_peaks = max_peaks[:-1]
    plt.plot(max_peaks, mean[max_peaks], "x")
    tmp1 = mean[max_peaks]

    # print(max_peaks)
    min_peaks, properties = find_peaks(-1 * mean, prominence=1, width=width)
    min_peaks = min_peaks[1:]
    plt.plot(min_peaks, mean[min_peaks], "x")
    # print(min_peaks)

    tmp2 = mean[min_peaks]
    avg = []
    avg_val = []
    for i, j, k in zip(tmp1, tmp2, min_peaks):
        # print(k,i,j,(i+j)/2)
        avg.append((i + j) / 2)
        avg_val.append(k)
    # list(tmp1)+list(tmp2)
    plt.plot(avg_val, avg)
    # plt.plot(avg_val, avg, ".")

    avg = np.array(avg)
    orig_indx = np.arange(len(mean))
    avg = np.convolve(avg, np.ones(window) / window, mode="valid")
    plt.plot(avg_val[:-1], avg, "o", c="r")
    indx = np.arange(len(avg))
    plt.plot(orig_indx, [avg[left_index] for i in orig_indx], "-.")
    plt.plot(orig_indx, [avg[-left_index] for i in orig_indx], "-.")
    # plt.plot(avg_val,[avg[left_index] for i in avg_val],'-.')
    # plt.plot(avg_val,[avg[-left_index] for i in avg_val],'-.')
    delta_V = avg[-left_index] - avg[left_index]

    plt.title("Band offset (eV): " + str(round(delta_V, 2)))
    print("Band offset (eV): ", delta_V, avg[-left_index], avg[left_index])
    plt.grid(color="gray", ls="-.")
    plt.xlabel("Distance ($\AA$)")
    plt.xlim(0, np.max(orig_indx))
    plt.minorticks_on()
    steps = np.arange(len(xvals), step=step)

    plt.xticks(steps, np.round(xvals[steps], 1))
    plt.ylabel("Potential (eV)")
    plt.savefig(filename)
    plt.close()
    return delta_V


if __name__ == "__main__":
    # fname='VASP_Interface-JVASP-39_JVASP-30_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_0.3_0.4_VASP/VASP_Interface-JVASP-39_JVASP-30_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_0.3_0.4_VASP/LOCPOT'
    fname_interface = "VASP_Interface-JVASP-1002_JVASP-816_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/VASP_Interface-JVASP-1002_JVASP-816_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/LOCPOT"
    fname_metal = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-134_bulk_PBEBO/MAIN-RELAX-bulk@mp-134/LOCPOT"
    fname_semi = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-149_bulk_PBEBO/MAIN-RELAX-bulk@mp-149/LOCPOT"
    delV = get_offset(fname_interface)
    m, m, xv_al = locpot_mean(fname_metal)
    m, m, xv_si = locpot_mean(fname_semi)
    if xv_si > xv_al:
        tmp = xv_al
        xv_al = xv_si
        xv_si = tmp

    print("PhiBH-p", fname_interface, delV - (xv_al - xv_si))
    print()
    print()

    fname_interface = "VASP_Interface-JVASP-1002_JVASP-825_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/VASP_Interface-JVASP-1002_JVASP-825_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/LOCPOT"
    fname_metal = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-81_bulk_PBEBO/MAIN-RELAX-bulk@mp-81/LOCPOT"
    fname_semi = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-149_bulk_PBEBO/MAIN-RELAX-bulk@mp-149/LOCPOT"
    delV = get_offset(fname_interface)
    m, m, xv_al = locpot_mean(fname_metal)
    m, m, xv_si = locpot_mean(fname_semi)
    if xv_si > xv_al:
        tmp = xv_al
        xv_al = xv_si
        xv_si = tmp
    print("PhiBH-p", fname_interface, delV - (xv_al - xv_si))
    print()
    print()

    fname_interface = "VASP_Interface-JVASP-1002_JVASP-972_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/VASP_Interface-JVASP-1002_JVASP-972_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/LOCPOT"
    fname_metal = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-126_bulk_PBEBO/MAIN-RELAX-bulk@mp-126/LOCPOT"
    delV = get_offset(fname_interface)
    m, m, xv_al = locpot_mean(fname_metal)
    m, m, xv_si = locpot_mean(fname_semi)
    if xv_si > xv_al:
        tmp = xv_al
        xv_al = xv_si
        xv_si = tmp
    print("PhiBH-p", fname_interface, delV - (xv_al - xv_si))
    print()
    print()

    fname_interface = "VASP_Interface-JVASP-1002_JVASP-813_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/VASP_Interface-JVASP-1002_JVASP-813_film_miller_0_0_1_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.1_-0.4_VASP/LOCPOT"
    fname_metal = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-124_bulk_PBEBO/MAIN-RELAX-bulk@mp-124/LOCPOT"
    delV = get_offset(fname_interface)
    m, m, xv_al = locpot_mean(fname_metal)
    m, m, xv_si = locpot_mean(fname_semi)
    if xv_si > xv_al:
        tmp = xv_al
        xv_al = xv_si
        xv_si = tmp
    print("PhiBH-p", fname_interface, delV - (xv_al - xv_si))
    print()
    print()

    fname_interface = "VASP_Interface-JVASP-1002_JVASP-802_film_miller_1_0_0_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.4_-0.2_VASP/VASP_Interface-JVASP-1002_JVASP-802_film_miller_1_0_0_sub_miller_0_0_1_film_thickness_16_subs_thickness_16_seperation_2.5_disp_-0.4_-0.2_VASP/LOCPOT"
    fname_metal = "/rk2/knc6/JARVIS-DFT/Elements-bulkk/mp-103_bulk_PBEBO/MAIN-RELAX-bulk@mp-103/LOCPOT"
    delV = get_offset(fname_interface)
    m, m, xv_al = locpot_mean(fname_metal)
    m, m, xv_si = locpot_mean(fname_semi)
    if xv_si > xv_al:
        tmp = xv_al
        xv_al = xv_si
        xv_si = tmp
    print("PhiBH-p", fname_interface, delV - (xv_al - xv_si))
    print()
    print()
