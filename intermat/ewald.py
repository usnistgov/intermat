# This Ewald summation module is developed by Qijing Zheng (http://staff.ustc.edu.cn/~zqj)
# Forked from https://github.com/QijingZheng/VaspBandUnfolding/blob/master/ewald.py
# Constants are from https://github.com/QijingZheng/VaspBandUnfolding/blob/master/vasp_constant.py
# https://github.com/Tack-Tau/fplib3/blob/8b16a53580eb6df290f67111496874cf5b8582ba/Ewald.py#L63

import numpy as np
from scipy.special import erf, erfc
from jarvis.core.specie import Specie

"""
Physical constants used in VASP
"""

#  Some important Parameters, to convert to a.u.
#  - AUTOA     =  1. a.u. in Angstroem
#  - RYTOEV    =  1 Ry in Ev
#  - EVTOJ     =  1 eV in Joule
#  - AMTOKG    =  1 atomic mass unit ("proton mass") in kg
#  - BOLKEV    =  Boltzmanns constant in eV/K
#  - BOLK      =  Boltzmanns constant in Joule/K

AUTOA = 0.529177249
RYTOEV = 13.605826
CLIGHT = 137.037  # speed of light in a.u.
EVTOJ = 1.60217733e-19
AMTOKG = 1.6605402e-27
BOLKEV = 8.6173857e-5
BOLK = BOLKEV * EVTOJ
EVTOKCAL = 23.06

# FELECT    =  (the electronic charge)/(4*pi*the permittivity of free space)
#         in atomic units this is just e^2
# EDEPS    =  electron charge divided by the permittivity of free space
#         in atomic units this is just 4 pi e^2
# HSQDTM    =  (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
#
PI = 3.141592653589793238
TPI = 2 * PI
CITPI = 1j * TPI
FELECT = 2 * AUTOA * RYTOEV
EDEPS = 4 * PI * 2 * RYTOEV * AUTOA
HSQDTM = RYTOEV * AUTOA * AUTOA

# vector field A times momentum times e/ (2 m_e c) is an energy
# magnetic moments are supplied in Bohr magnetons
# e / (2 m_e c) A(r) p(r)    =  energy
# e / (2 m_e c) m_s x ( r - r_s) / (r-r_s)^3 hbar nabla    =
# e^2 hbar^2 / (2 m_e^2 c^2) 1/ lenght^3    =  energy
# conversion factor from magnetic moment to energy
# checked independently in SI by Gilles de Wijs

MAGMOMTOENERGY = 1 / CLIGHT**2 * AUTOA**3 * RYTOEV

# dimensionless number connecting input and output magnetic moments
# AUTOA e^2 (2 m_e c^2)
MOMTOMOM = AUTOA / CLIGHT / CLIGHT / 2
AUTOA2 = AUTOA * AUTOA
AUTOA3 = AUTOA2 * AUTOA
AUTOA4 = AUTOA2 * AUTOA2
AUTOA5 = AUTOA3 * AUTOA2

# dipole moment in atomic units to Debye
AUTDEBYE = 2.541746


class ewaldsum(object):
    """
    Ewald summation.
    """

    def __init__(
        self,
        atoms,
        Z={},
        eta: float = None,
        Rcut: float = 4.0,
        Gcut: float = 4.0,
    ):
        """ """
        # if not Z:
        for i in atoms.elements:
            Z[i] = Specie(i).Z

        # atoms=atoms.ase_converter()
        # assert Z, "'Z' can not be empty!\n\
        #        It is a dictionary containing charges for each element,\
        #        e.g. {'Na':1.0, 'Cl':-1.0}."
        # print("Z", Z)
        # the poscar storing the atoms information
        self._atoms = atoms
        self._na = self._atoms.num_atoms
        # factional coordinates in range [0,1]
        self._scapos = self._atoms.frac_coords  # get_scaled_positions()

        elements = np.unique(self._atoms.elements)
        # elements = np.unique(self._atoms.get_chemical_symbols())
        for elem in elements:
            if elem not in Z:
                raise ValueError(f"Charge for {elem} missing!")

        self._ZZ = np.array([Z[x] for x in self._atoms.elements])
        # self._ZZ = np.array([Z[x] for x in self._atoms.get_chemical_symbols()])
        # z_i * z_j
        self._Zij = np.prod(
            np.meshgrid(self._ZZ, self._ZZ, indexing="ij"), axis=0
        )

        # FELECT = (the electronic charge)/(4*pi*the permittivity of free space)
        #          in atomic units this is just e^2
        self._inv_4pi_epsilon0 = FELECT

        self._Acell = np.array(self._atoms.lattice_mat)  # real-space cell
        # self._Acell = np.array(self._atoms.cell)        # real-space cell
        self._Bcell = np.linalg.inv(self._Acell).T  # reciprocal-space cell
        self._omega = np.linalg.det(self._Acell)  # Volume of real-space cell

        # the decaying parameter
        if eta is None:
            self._eta = np.sqrt(np.pi) / (self._omega) ** (1.0 / 3)
            # self._eta = np.sqrt(np.pi) / np.linalg.norm(self._Acell, axis=1).min()
        else:
            self._eta = eta

        self._Rcut = Rcut
        self._Gcut = Gcut

    def get_sum_real(self):
        """
        Real-space contribution to the Ewald sum.
                 1                              erfc(eta | r_ij + R_N |)
            U = --- \sum_{ij} \sum'_N Z_i Z_j -----------------------------
                 2                                    | r_ij + R_N |
        where the prime in \sum_N means i != j when N = 0.
        """
        ii, jj = np.mgrid[0 : self._na, 0 : self._na]

        # r_i - r_j, rij of shape (natoms, natoms, 3)
        rij = self._scapos[ii, :] - self._scapos[jj, :]
        # move rij to the range [-0.5,0.5]
        rij[rij >= 0.5] -= 1.0
        rij[rij < -0.5] += 1.0

        ############################################################
        # contribution from N = 0 cell
        ############################################################
        rij0 = np.linalg.norm(
            np.tensordot(self._Acell, rij.T, axes=(0, 0)), axis=0
        )
        dd = range(self._na)
        # make diagonal term non-zero to avoid divide-by-zero error
        rij0[dd, dd] = 0.1
        Uij = erfc(rij0 * self._eta) / rij0
        # set diagonal term zero
        Uij[dd, dd] = 0

        ############################################################
        # contribution from N != 0 cells
        ############################################################
        rij = rij.reshape((-1, 3)).T

        nx, ny, nz = (
            np.array(
                self._Rcut / self._eta / np.linalg.norm(self._Acell, axis=1),
                dtype=int,
            )
            + 1
        )
        Rn = np.mgrid[-nx : nx + 1, -ny : ny + 1, -nz : nz + 1].reshape(
            (3, -1)
        )
        # remove N = 0 term
        cut = np.sum(np.abs(Rn), axis=0) != 0
        Rn = Rn[:, cut]

        # R_N + rij
        Rr = np.linalg.norm(
            np.tensordot(
                self._Acell, Rn[:, None, :] + rij[:, :, None], axes=(0, 0)
            ),
            axis=0,
        )
        Uij += np.sum(erfc(self._eta * Rr) / Rr, axis=1).reshape(
            (self._na, self._na)
        )

        return 0.5 * Uij

    def get_sum_recp(self):
        """
        Reciprocal-space contribution to the Ewald sum.
                  1            4pi
            U = ----- \sum'_G ----- exp(-G^2/(4 eta^2)) \sum_{ij} Z_i Z_j exp(-i G r_ij)
                 2 V           G^2
        where the prime in \sum_G means G != 0.
        """
        nx, ny, nz = (
            np.array(
                self._Gcut
                * self._eta
                / np.pi
                / np.linalg.norm(self._Bcell, axis=1),
                dtype=int,
            )
            + 1
        )
        Gn = np.mgrid[-nx : nx + 1, -ny : ny + 1, -nz : nz + 1].reshape(
            (3, -1)
        )
        # remove G = 0 term
        cut = np.sum(np.abs(Gn), axis=0) != 0
        Gn = Gn[:, cut]

        G2 = (
            np.linalg.norm(
                np.tensordot(self._Bcell * 2 * np.pi, Gn, axes=(0, 0)), axis=0
            )
            ** 2
        )
        expG2_invG2 = 4 * np.pi * np.exp(-G2 / 4 / self._eta**2) / G2

        # r_i - r_j, rij of shape (natoms, natoms, 3)
        # no need to move rij from [0,1] to [-0.5,0.5], which will not affect
        # the phase G*rij
        ii, jj = np.mgrid[0 : self._na, 0 : self._na]
        rij = self._scapos[ii, :] - self._scapos[jj, :]

        sfac = np.exp(-2j * np.pi * (rij @ Gn))

        Uij = 0.5 * np.sum(expG2_invG2 * sfac, axis=-1) / self._omega

        return Uij.real

    def get_ewaldsum(self):
        """
        Total Coulomb energy from Ewald summation.
        """

        # real-space contribution
        Ur = np.sum(self.get_sum_real() * self._Zij)
        # reciprocal--space contribution
        Ug = np.sum(self.get_sum_recp() * self._Zij)

        # interaction between charges
        Us = -self._eta / np.sqrt(np.pi) * np.sum(self._ZZ**2)
        # interaction with the neutralizing background
        Un = (
            -(2 * np.pi / self._eta**2 / self._omega) * self._ZZ.sum() ** 2 / 4
        )

        # total coulomb energy
        Ut = (Ur + Ug + Us + Un) * self._inv_4pi_epsilon0

        return Ut

    def get_madelung(self, iref: int = 0):
        """ """
        assert iref < self._na
        # index for reference atom
        ii = iref
        # nearest-neighbour of ref atom
        rij = self._scapos - self._scapos[ii]
        rij[rij >= 0.5] -= 1.0
        rij[rij < -0.5] += 1.0
        rij0 = np.linalg.norm(rij @ self._Acell, axis=1)
        dd = np.arange(self._na)
        jj = dd[np.argsort(rij0)[1]]
        r0 = rij0[jj]

        Ur = self.get_sum_real() * self._Zij
        Ug = self.get_sum_recp() * self._Zij
        Ui = (Ur[ii] + Ug[ii]).sum() - self._eta / np.sqrt(np.pi) * self._ZZ[
            ii
        ] ** 2
        M = 2 * Ui * r0 / self._ZZ[ii] / self._ZZ[jj]

        return M


if __name__ == "__main__":
    from jarvis.core.atoms import Atoms

    box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
    coords = [[0, 0, 0], [0.25, 0.2, 0.25]]
    elements = ["Si", "Si"]
    atoms_si = Atoms(lattice_mat=box, coords=coords, elements=elements)
    atoms = atoms_si
    eww = ewaldsum(atoms=atoms)
    en = eww.get_ewaldsum()
    print(en)
