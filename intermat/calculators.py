"""Module to calculate properties."""
from jarvis.tasks.queue_jobs import Queue
from jarvis.io.vasp.inputs import Poscar, Incar, Potcar
from jarvis.core.kpoints import Kpoints3D
import os
from jarvis.db.jsonutils import dumpjson
from jarvis.tasks.vasp.vasp import VaspJob
from jarvis.db.figshare import data as j_data
import numpy as np
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.analysis.defects.surface import Surface
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
import pandas as pd
import time
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter


# TB
# LAMMPS
# QE
class Calc(object):
    def __init__(
        self,
        atoms=[],
        energy_only=True,
        relax_atoms=False,
        relax_cell=False,
        method_name="",
        ase_based=["eam_ase", "alignn_ff", "matgl", "emt", "gpaw", "other"],
        extra_params={},
        fmax=0.01,
        steps=100,
    ):
        self.atoms = atoms
        self.energy_only = energy_only
        self.relax_atoms = relax_atoms
        self.relax_cell = relax_cell
        self.method = method
        self.ase_based = ase_based
        self.extra_params
        self.fmax = fmax
        self.steps

    def predict(
        self,
    ):
        if self.method in self.ase_based:
            atoms = self.atoms.ase_converter()
            if self.method == "eam_ase":
                if "potential" not in self.extra_params:
                    # Download from https://doi.org/10.6084/m9.figshare.24187602
                    self.extra_params[
                        "potential"
                    ] = "Mishin-Ni-Al-Co-2013.eam.alloy"
                from ase.calculators.eam import EAM

                calculator = EAM(potential=extra_params["potential"])
            elif self.method == "alignn":
                from alignn.ff.ff import (
                    AlignnAtomwiseCalculator,
                    default_path,
                    wt01_path,
                    wt10_path,
                )

                if self.extra_params["model_path"] == "":
                    model_path = wt10_path()  # wt01_path()
                calculator = AlignnAtomwiseCalculator(
                    path=model_path, stress_wt=0.3
                )

            elif self.method == "matgl":
                from matgl.ext.ase import M3GNetCalculator
                import matgl

                pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                calculator = M3GNetCalculator(pot)
            elif self.method == "gpaw":
                from gpaw import GPAW, PW, FermiDirac, Davidson

                if "gpaw_params" not in self.extra_params:
                    self.extra_params["gpaw_params"] = dict(
                        smearing=0.01,
                        cutoff=None,
                        kp_length=20,
                        xc="PBE",
                        basis="szp(dzp)",
                        mode="lcao",
                        spinpol=True,
                        nbands="120%",
                        symmetry="on",
                        parallel={
                            "sl_auto": True,
                            "domain": 2,
                            "augment_grids": True,
                        },
                        maxiter=1000,
                        convergence={"density": 1e-12, "energy": 1e-1},
                        eigensolver=Davidson(niter=2),
                        kp_length=30,
                        cutoff=None,
                        out_file="gs.out",
                        out_gpw="out.gpw",
                    )

                kp = Kpoints3D().automatic_length_mesh(
                    lattice_mat=atoms.lattice_mat,
                    length=self.extra_params["kp_length"],
                )
                if "occupations" not in self.extra_params["gpaw_params"]:
                    occupations = FermiDirac(self.extra_params["smearing"])
                kpts = kp._kpoints[0]
                kpts = {"size": (kpts[0], kpts[1], kpts[2]), "gamma": True}
                if gpaw_params["cutoff"] is not None:
                    mode = PW(cutoff)
                else:
                    mode = "lcao"

                calculator = GPAW(
                    mode=mode,
                    basis=self.extra_params["basis"],
                    xc=self.extra_params["xc"],
                    kpts=self.extra_params["kpts"],
                    occupations=occupations,
                    txt=self.extra_params["out_file"],
                    spinpol=self.extra_params["spinpol"],
                    nbands=self.extra_params["nbands"],
                    # symmetry=symmetry,
                    # parallel=parallel,
                    # convergence=convergence,
                    # eigensolver=eigensolver,
                )
            elif self.method == "other":
                calculator = self.extra_params["calculator"]
            else:
                print("ASE Calc not implemented:", self.method)
            atoms = atoms.ase_converter()
            atoms.calc = calc
            optimizer = FIRE
            dyn = optimizer(atoms)
            dyn.run(fmax=fmax, steps=steps)

            en = atoms.get_potential_energy()
            homo, lumo = calc.get_homo_lumo()
            efermi = calc.get_fermi_level()
            bandgap = lumo - homo
            print("Band gap", bandgap)
            print("Fermi level", efermi)
            calc.write(out_gpw)
            atoms = ase_to_atoms(atoms)
            return en

            info = {}
            if (
                self.energy_only
                and not self.relax_atoms
                and not self.relax_cell
            ):
                atoms.calc = calculator
                forces = atoms.get_forces()
                energy = atoms.get_potential_energy()
                # stress = atoms.get_stress()
                info["energy"] = energy
                info["atoms"] = atoms
                return info  # ,forces,stress
            elif relax_atoms and not relax_cell:
                optimizer = FIRE
                dyn = optimizer(atoms)
                dyn.run(fmax=self.fmax, steps=self.steps)
                energy = atoms.get_potential_energy()
                atoms = ase_to_atoms(atoms)
                info["energy"] = energy
                info["atoms"] = atoms
                return info  # ,forces,stress
            elif relax_atoms and not relax_cell:
                atoms = ExpCellFilter(atoms)
                optimizer = FIRE
                dyn = optimizer(atoms)
                dyn.run(fmax=fmax, steps=steps)
                energy = atoms.get_potential_energy()
                atoms = ase_to_atoms(atoms)
                info["energy"] = energy
                info["atoms"] = atoms
                return info  # ,forces,stress
            else:
                print("Not implemeneted")
        elif self.method == "ewald":
            from ewald import ewaldsum

            ew = ewaldsum(self.atoms)
            energy = ew.get_ewaldsum()
            info["energy"] = energy
            info["atoms"] = self.atoms
            return info
        else:
            raise ValueError("Not implemented:", self.method)
        return energy  # ,forces,stress

    def vasp(
        self,
        jobname="",
        kp_length=30,
        extra_lines="\n module load vasp/6.3.1\n"
        + "source ~/anaconda2/envs/my_jarvis/bin/activate my_jarvis\n",
        copy_files=["/users/knc6/bin/vdw_kernel.bindat"],
        vasp_cmd="mpirun vasp_std",
        inc="",
        film_kp_length=30,
        subs_kp_length=30,
        sub_job=True,
    ):
        def write_jobpy(pyname="job.py", job_json=""):
            # job_json = os.getcwd()+'/'+'job.json'
            f = open(pyname, "w")
            f.write("from jarvis.tasks.vasp.vasp import VaspJob\n")
            f.write("from jarvis.db.jsonutils import loadjson\n")
            f.write('d=loadjson("' + str(job_json) + '")\n')
            f.write("v=VaspJob.from_dict(d)\n")
            f.write("v.runjob()\n")
            f.close()

        if inc == "":
            data = dict(
                PREC="Accurate",
                ISMEAR=0,
                SIGMA=0.05,
                IBRION=2,
                LORBIT=11,
                GGA="BO",
                PARAM1=0.1833333333,
                PARAM2=0.2200000000,
                LUSE_VDW=".TRUE.",
                AGGAC=0.0000,
                EDIFF="1E-6",
                NSW=500,
                NELM=500,
                ISIF=2,
                ISPIN=2,
                LCHARG=".TRUE.",
                LVTOT=".TRUE.",
                LVHAR=".TRUE.",
                LWAVE=".FALSE.",
                LREAL="Auto",
            )

            inc = Incar(data)

        cwd = os.getcwd()
        name_dir = os.path.join(cwd, jobname)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        os.chdir(name_dir)
        pos = Poscar(atoms)
        pos_name = "POSCAR-" + jobname + ".vasp"
        atoms.write_poscar(filename=pos_name)
        pos.comment = jobname

        new_symb = []
        for ee in atoms.elements:
            if ee not in new_symb:
                new_symb.append(ee)
        pot = Potcar(elements=new_symb)
        # if leng - 25 > 0:
        #    leng = leng - 25
        # print("leng", k1, k2, leng)
        kp = Kpoints3D().automatic_length_mesh(
            lattice_mat=atoms.lattice_mat,
            length=kp_length,
        )
        [a, b, c] = kp.kpts[0]
        # if "Surf" in jobname:
        #    kp = Kpoints3D(kpoints=[[a, b, 1]])
        # else:
        #    kp = Kpoints3D(kpoints=[[a, b, c]])
        kp = Kpoints3D(kpoints=[[a, b, 1]])
        # Step-1 Make VaspJob
        v = VaspJob(
            poscar=pos,
            incar=inc,
            potcar=pot,
            kpoints=kp,
            copy_files=copy_files,
            jobname=jobname,
            vasp_cmd=vasp_cmd,
        )

        # Step-2 Save on a dict
        jname = os.getcwd() + "/" + "VaspJob_" + jobname + "_job.json"
        dumpjson(
            data=v.to_dict(),
            filename=jname,
        )

        # Step-3 Write jobpy
        write_jobpy(job_json=jname)
        path = (
            # "\n module load vasp/6.3.1\n"
            # + "source ~/anaconda2/envs/my_jarvis/bin/activate my_jarvis\n"
            extra_lines
            + "python "
            + os.getcwd()
            + "/job.py"
        )
        # Step-4 QSUB
        # jobid=os.getcwd() + "/" + jobname + "/jobid"
        jobid = os.getcwd() + "/jobid"
        print("jobid", jobid)
        if sub_job and not os.path.exists(jobid):
            # if sub_job:# and not os.path.exists(jobid):
            Queue.slurm(
                job_line=path,
                jobname=jobname,
                walltime="70-00:00:00",
                directory=os.getcwd(),
                submit_cmd=["sbatch", "submit_job"],
                queue="coin,epyc,highmem",
            )
        else:
            print("jobid exists", jobid)
        out = os.getcwd() + "/" + jobname + "/Outcar"
        print("out", out)
        energy = -999
        if os.path.exists(out):
            if Outcar(out).converged:
                vrun_file = os.getcwd() + "/" + jobname + "/vasprun.xml"
                vrun = Vasprun(vrun_file)
                energy = float(vrun.final_energy)

        os.chdir(cwd)

        return energy  # ,forces,stress

        def gpaw(
            smearing=0.01,
            xc="PBE",
            basis="szp(dzp)",
            mode="lcao",
            spinpol=True,
            nbands="120%",
            symmetry="on",
            parallel={"sl_auto": True, "domain": 2, "augment_grids": True},
            maxiter=1000,
            convergence={"density": 1e-12, "energy": 1e-1},
            eigensolver=Davidson(niter=2),
            kp_length=30,
            cutoff=None,
            out_file="gs.out",
            out_gpw="out.gpw",
            fmax=0.05,
        ):
            from gpaw import GPAW, PW, FermiDirac, Davidson

            kp = Kpoints3D().automatic_length_mesh(
                lattice_mat=atoms.lattice_mat,
                length=kp_length,
            )
            kpts = kp._kpoints[0]
            kpts = {"size": (kpts[0], kpts[1], kpts[2]), "gamma": True}
            if cutoff is not None:
                mode = PW(cutoff)
            else:
                mode = "lcao"
            calc = GPAW(
                mode=mode,
                basis=basis,
                xc=xc,
                kpts=kpts,
                occupations=FermiDirac(smearing),
                txt=out_file,
                spinpol=spinpol,
                nbands=nbands,
                # symmetry=symmetry,
                # parallel=parallel,
                # convergence=convergence,
                # eigensolver=eigensolver,
            )
            atoms = atoms.ase_converter()
            atoms.calc = calc
            optimizer = FIRE
            dyn = optimizer(atoms)
            dyn.run(fmax=fmax, steps=steps)

            en = atoms.get_potential_energy()
            homo, lumo = calc.get_homo_lumo()
            efermi = calc.get_fermi_level()
            bandgap = lumo - homo
            print("Band gap", bandgap)
            print("Fermi level", efermi)
            calc.write(out_gpw)
            atoms = ase_to_atoms(atoms)
            return en


if __name__ == "__main__":
    # semicon_mat_interface_workflow()
    # metal_metal_interface_workflow()
    # semicon_mat_interface_workflow2()
    # quick_compare()
    semicon_semicon_interface_workflow()
