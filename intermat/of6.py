# 12-7-2023 simpson
##############################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (
    CubicSpline,
    interp1d,
    CubicHermiteSpline,
    Akima1DInterpolator,
)
import scipy.integrate as integrate
from numpy import ones, vstack
from numpy.linalg import lstsq
import numpy as np
import re
import matplotlib.pyplot as plt
from ase.calculators.vasp import VaspChargeDensity
from scipy.signal import find_peaks
import glob
from jarvis.io.vasp.outputs import Outcar
from jarvis.db.jsonutils import loadjson
import os
from jarvis.core.atoms import ase_to_atoms
from jarvis.core.atoms import Atoms


def get_m_c(x=[], y=[]):
    A = vstack([x, ones(len(x))]).T
    m, c = lstsq(A, y)[0]
    return m, c


def get_dir(jid="JVASP-1002"):
    pth = jid + "_R2SCAN/opt_" + jid + "/opt_" + jid + "/OUTCAR"
    # pth = jid + "_R2SCAN/r2scan_" + jid + "/r2scan_" + jid + "/OUTCAR"
    bandg = Outcar(pth)
    return bandg.bandgap


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
    # mean-=avg_max
    return xvals, mean, out.bandgap[-1], atoms


step_size = 10


# find best L on grid of 10 points
def get_best_L(start_L, end_L, S, x_target):
    L = 0
    best = 1000000000.0
    for L_guess in np.arange(
        start_L, end_L + 1e-5, (end_L - start_L) / (step_size + 0.55)
    ):
        current = 0.0
        for xx in x_target:
            # current += abs(S(xx) - S(xx-L_guess))
            current += abs(S(xx) - S(xx + L_guess))
        if current < best:
            best = current
            L = L_guess

    return L


# recursively find best L
def best_L_recursive(start_L, end_L, S, x_target):
    L_best = 0.0
    L_range = end_L - start_L
    for iter in range(step_size):
        L_best = get_best_L(start_L, end_L, S, x_target)
        L_range = L_range / step_size
        start_L = L_best - L_range
        end_L = L_best + L_range

    return L_best


# do the actual averaging
def do_average(L, x, S):
    AVG = []
    XX = []
    for xx in x:
        if xx - L / 2.0 < x[0]:
            continue
        if xx + L / 2.0 > x[-1]:
            continue
        XX.append(xx)
        # print('step',xx)
        tmp_x = np.arange(xx - L / 2.0, xx + L / 2.0, 0.001)
        # x=np.arange(xx-L/2.0, xx+L/2.0,0.1)
        tmp_y = S(tmp_x)
        # intg2=np.trapz(tmp_y,tmp_x,tmp_x[1]-tmp_x[0])/L
        # intg2=integrate.quad(S, xx-L/2.0, xx+L/2.0)[0] / L
        # intg2=integrate.quad(S, xx-L/2.0, xx+L/2.0)[0] / L
        # print('tmp_x[1]-tmp_x[0]',tmp_x[1]-tmp_x[0])
        intg2 = integrate.simpson(tmp_y, tmp_x, tmp_x[1] - tmp_x[0]) / L
        # print('intg',int)
        AVG.append(intg2)  # integration
    #        print("int ", integrate.quad(S, xx-L/2.0, xx+L/2.0))
    return XX, AVG


def get_mean_val(x_target, XX, AVG):
    # tot = 0.0
    # N = 0
    # for x_t in x_target:
    #     for (xx, avg) in zip(XX, AVG):
    #         if abs(x_t - xx) < 1e-8:
    #             tot += avg
    #             N += 1
    # mean = tot / N
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
    jid1 = fname.split("_")[0].split("Interface-")[1]
    jid2 = fname.split("_")[1]
    x1 = get_dir(jid1)
    print(jid1, x1)
    x2 = get_dir(jid2)
    print(jid2, x2)
    delta_E = x2[2] - x1[2]
    return delta_E


def offset(fname="", x=[], s=[], width=5, left_index=-1, polar=False):
    if len(x) == 0:
        x, s, _, _ = locpot_mean(fname)

    # x = np.load("x.npy")
    # s = np.load("mean.npy")

    deltaE = delta_E(fname)

    # deltaE = 0.0

    # plt.plot(s,s1)
    # plt.plot(s,s)
    # import sys
    # sys.exit()

    # fake data
    # x = np.arange(0, 20.123*np.pi + 0.0000001*np.pi, np.pi/59.123)
    # s = np.sin(x) + np.cos(2*x)

    # x = np.load("x.npy")
    # s = np.load("mean.npy")

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
    print("L ", L)

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
    print("Initial guess right ", L_guess_peaks_right)
    L = best_L_recursive(1.0, L_guess_peaks_right * 1.5, S, x_target2)
    print("L ", L)

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
    plt.grid(color="gray", ls="-.")
    plt.minorticks_on()
    plt.ylabel("Potential (eV)")
    plt.xlim(0, max(x))
    plt.xlabel("Distance ($\AA$)")
    # plt.plot(x,[meanval1 for i in range(len(x))],linestyle='-.',color='blue')
    # plt.plot(x,[meanval2 for i in range(len(x))],linestyle='-.',color='blue')
    print("meanval ", [meanval1, meanval2], meanval2 - meanval1, phi)
    plt.title("Offset (eV): " + str(round(phi, 2)))
    filename = "opt_offset_max-" + fname.split("/")[0] + ".png"
    # filename = "r2scan_offset_max-" + fname.split("/")[0] + ".png"
    print("deltaE", deltaE)
    # plt.show()
    plt.savefig(filename)
    plt.close()


for i in glob.glob("Int*/opt_*/opt*/LOCPOT"):
    # try:
    if "Interface-JVASP-1002_JVASP-8184_film_miller_1_1_0" in i:  # 3
        # if 'JVASP-1174_JVASP-96' in i: #3
        # if 'JVASP-30_JVASP-1195' in i: #3
        # if 'JVASP-30_JVASP-8118' in i: #3
        # if 'JVASP-39_JVASP-1180' in i: #3
        # if 'JVASP-39_JVASP-8118' in i: #3
        # if 'JVASP-1002_JVASP-10591' in i: #3
        # if 'JVASP-1002_JVASP-8118' in i: #3
        # if 'JVASP-39_JVASP-30' in i: #3
        # if 'Interface-JVASP-1372_JVASP-1174_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1372_JVASP-96_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1372_JVASP-1186_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1002_JVASP-8184_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1327_JVASP-8184_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1174_JVASP-1183_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1198_JVASP-1192_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-8003_JVASP-1192_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-1183_JVASP-1186_film_miller_1_1_0' in i: #3
        # if 'Interface-JVASP-96_JVASP-1198_film_miller_1_1_0' in i: #3
        # if 'JVASP-1183_JVASP-96' in i: #3
        # if 'JVASP-1408_JVASP-1198' in i: #3
        # if 'JVASP-1408_JVASP-1186' in i: #3
        # if 'Interface-JVASP-1002_JVASP-8003_film_miller_1_1_0_sub_miller_1_1_' in i: #3
        # if 'Interface-JVASP-1002_JVASP-1174_film_miller_1_1_0_sub_miller_1_1_' in i: #3
        # if 'Interface-JVASP-1002_JVASP-1327_film_miller_1_1_0_sub_miller_1_1_' in i: #3
        # if 'Interface-JVASP-1174_JVASP-825_film_miller_0_0_1_sub_miller_0_0_' in i: #4
        # if 'Interface-JVASP-1002_JVASP-816_film_miller_0_0_1_sub_miller_0_0_' in i: #3
        # if 'Interface-JVASP-1002_JVASP-825_film_miller_0_0_1_sub_miller_0_0_' in i: #4
        # for j in arr:

        offset(fname=i, left_index=-1, polar=False)
        # offset(fname=i,left_index=4,polar=True)
# except:
#  pass
