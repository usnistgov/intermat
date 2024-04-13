"""Module to calculate properties."""

from jarvis.tasks.queue_jobs import Queue
from jarvis.io.vasp.inputs import Poscar, Incar, Potcar
from jarvis.io.vasp.outputs import Vasprun, Outcar
from jarvis.core.kpoints import Kpoints3D
import os
from jarvis.db.jsonutils import dumpjson
from jarvis.tasks.vasp.vasp import VaspJob
from jarvis.tasks.lammps.lammps import LammpsJob
from jarvis.tasks.qe.qe import QEjob
import numpy as np
from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.db.jsonutils import loadjson


# TODO: Use pydantic
def template_extra_params(method="vasp"):
    info = {}
    if method == "vasp":
        incar = dict(
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
            NPAR=4,
            LCHARG=".TRUE.",
            LVTOT=".FALSE.",
            LVHAR=".TRUE.",
            LWAVE=".FALSE.",
            LREAL="Auto",
        )
        vasp_cmd = "mpirun vasp_std"
        copy_files = ["/users/knc6/bin/vdw_kernel.bindat"]
        extra_lines = (
            "\nmodule load vasp/6.3.1\n" + "conda activate mini_alignn\n"
        )
        info["inc"] = incar
        info["vasp_cmd"] = vasp_cmd
        info["copy_files"] = copy_files
        info["extra_lines"] = extra_lines
        info["kp_length"] = 30
        info["sub_job"] = True
        info["walltime"] = "30-00:00:00"
        info["queue"] = "rack1,rack2,rack2e,rack3,rack4,rack4e,rack5,rack6"
    elif method == "tb3":
        lines = [
            "using ThreeBodyTB\nusing NPZ\n",
            'crys = makecrys("POSCAR")\n',
            # "energy, tbc, flag =
            # scf_energy(crys,mixing_mode=:simple,mix=0.05);\n",
            "cfinal, tbc, energy, force, stress = relax_structure(crys);\n",
            # "println(cfinal)\n",
            "vects, vals, hk, sk, vals0 = ThreeBodyTB.TB.Hk(tbc,[0,0,0])\n",
            'ThreeBodyTB.TB.write_tb_crys("tbc.xml.gz",tbc)\n',
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
            'open("band_summary","w") do file\n',
            'println("bandst.",(ThreeBodyTB.BandStruct.band_summary(tbc)))\n',
            "write(file,string(ThreeBodyTB.BandStruct.band_summary(tbc)))\n",
            "end\n",
        ]
        info["tb3_params"] = lines
    elif method == "qe":
        qe_params = {
            "control": {
                "calculation": "'scf'",
                # "calculation":  "'vc-relax'",
                "restart_mode": "'from_scratch'",
                "prefix": "'RELAX'",
                "outdir": "'./'",
                "tstress": ".true.",
                "tprnfor": ".true.",
                "disk_io": "'nowf'",
                "wf_collect": ".true.",
                "pseudo_dir": None,
                "verbosity": "'high'",
                "nstep": 100,
            },
            "system": {
                "ibrav": 0,
                "nat": None,
                "ntyp": None,
                "ecutwfc": 45,
                "ecutrho": 250,
                "q2sigma": 1,
                "ecfixed": 44.5,
                "qcutz": 800,
                "occupations": "'smearing'",
                "degauss": 0.01,
                "lda_plus_u": ".false.",
            },
            "electrons": {
                "diagonalization": "'david'",
                "mixing_mode": "'local-TF'",
                "mixing_beta": 0.3,
                "conv_thr": "1d-9",
            },
            "ions": {"ion_dynamics": "'bfgs'"},
            "cell": {"cell_dynamics": "'bfgs'", "cell_dofree": "'all'"},
        }
        info["qe_params"] = qe_params
        info["qe_cmd"] = "pw.x"
        extra_lines = "conda activate mini_alignn\n"
        info["extra_lines"] = extra_lines
        info["walltime"] = "70-00:00:00"
        info["queue"] = "epyc"
    elif method == "lammps":
        lammps_params = dict(
            cmd="lmp_serial<in.main>out",
            lammps_cmd="lmp_serial<in.main>out",
            pair_style="eam/alloy",
            pair_coeff="lammps/potentials/Al_zhou.eam.alloy",
            atom_style="charge",
            control_file="jarvis/jarvis/tasks/lammps/templates/inelast.mod",
        )
        info["lammps_params"] = lammps_params
    elif method == "gpaw":
        from gpaw import Davidson

        info["gpaw_params"] = dict(
            smearing=0.01,
            cutoff=400,
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
            out_file="gs.out",
            out_gpw="out.gpw",
            kp_length=30,
        )

    return info


class Calc(object):
    def __init__(
        self,
        atoms=[],
        energy_only=True,
        relax_atoms=False,
        relax_cell=False,
        method="",
        ase_based=["eam_ase", "alignn_ff", "matgl", "emt", "gpaw", "other"],
        extra_params={},
        fmax=0.01,
        steps=100,
        jobname="temp_job",
    ):
        self.atoms = atoms
        self.energy_only = energy_only
        self.relax_atoms = relax_atoms
        self.relax_cell = relax_cell
        self.method = method
        self.ase_based = ase_based
        self.extra_params = extra_params
        self.fmax = fmax
        self.steps = steps
        self.jobname = jobname
        implemented_methds = [
            "eam_ase",
            "ewald",
            "alignn_ff",
            "matgl",
            "emt",
            "gpaw",
            "other",
            "vasp",
            "qe",
            "tb3",
            "lammps",
        ]
        if method not in implemented_methds:
            raise ValueError("Not implemented", method)

    def predict(
        self,
    ):
        if self.method in self.ase_based:
            atoms = self.atoms.ase_converter()
            if self.method == "eam_ase":
                if "potential" not in self.extra_params:
                    # Download from
                    # https://doi.org/10.6084/m9.figshare.24187602
                    self.extra_params[
                        "potential"
                    ] = "Mishin-Ni-Al-Co-2013.eam.alloy"

                from ase.calculators.eam import EAM

                calculator = EAM(potential=self.extra_params["potential"])
            elif self.method == "alignn_ff":
                from alignn.ff.ff import (
                    AlignnAtomwiseCalculator,
                    # default_path,
                    # wt01_path,
                    wt10_path,
                )

                if "model_path" not in self.extra_params:
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
                from gpaw import GPAW, PW, FermiDirac

                if "gpaw_params" not in self.extra_params:
                    self.extra_params["gpaw_params"] = template_extra_params(
                        method="gpaw"
                    )["gpaw_params"]

                kp = Kpoints3D().automatic_length_mesh(
                    lattice_mat=self.atoms.lattice_mat,
                    length=self.extra_params["gpaw_params"]["kp_length"],
                )
                if "occupations" not in self.extra_params["gpaw_params"]:
                    occupations = FermiDirac(
                        self.extra_params["gpaw_params"]["smearing"]
                    )
                kpts = kp._kpoints[0]
                kpts = {"size": (kpts[0], kpts[1], kpts[2]), "gamma": True}
                if self.extra_params["gpaw_params"]["cutoff"] is not None:
                    mode = PW(self.extra_params["gpaw_params"]["cutoff"])
                else:
                    mode = "lcao"

                calculator = GPAW(
                    mode=mode,
                    basis=self.extra_params["gpaw_params"]["basis"],
                    xc=self.extra_params["gpaw_params"]["xc"],
                    kpts=kpts,
                    occupations=occupations,
                    txt=self.extra_params["gpaw_params"]["out_file"],
                    spinpol=self.extra_params["gpaw_params"]["spinpol"],
                    nbands=self.extra_params["gpaw_params"]["nbands"],
                    maxiter=self.extra_params["gpaw_params"]["maxiter"],
                    # symmetry=self.extra_params["gpaw_params"]["symmetry"],
                    # parallel=parallel,
                    convergence=self.extra_params["gpaw_params"][
                        "convergence"
                    ],
                    # eigensolver=eigensolver,
                )
                print("calculator", calculator)
            elif self.method == "emt":
                from ase.calculators.emt import EMT

                calculator = EMT()
            elif self.method == "other":
                calculator = self.extra_params["calculator"]
            else:
                print("ASE Calc not implemented:", self.method)
            atoms.calc = calculator
            # print("calculator", self.method)
            info = {}
            if (
                self.energy_only
                and not self.relax_atoms
                and not self.relax_cell
            ):
                atoms.calc = calculator
                # forces = atoms.get_forces()
                energy = atoms.get_potential_energy()
                # stress = atoms.get_stress()
                info["energy"] = energy
                info["atoms"] = ase_to_atoms(atoms)
                return info  # ,forces,stress
            elif self.relax_atoms and not self.relax_cell:
                from ase.optimize.fire import FIRE

                optimizer = FIRE
                dyn = optimizer(atoms)
                dyn.run(fmax=self.fmax, steps=self.steps)
                energy = atoms.get_potential_energy()
                atoms = ase_to_atoms(atoms)
                info["energy"] = energy
                info["atoms"] = atoms
                return info  # ,forces,stress
            elif self.relax_cell:
                from ase.optimize.fire import FIRE
                from ase.constraints import ExpCellFilter

                atoms = ExpCellFilter(atoms)
                optimizer = FIRE
                dyn = optimizer(atoms)
                dyn.run(fmax=self.fmax, steps=self.steps)
                energy = atoms.get_potential_energy(force_consistent=False)
                info["energy"] = energy
                jatoms = ase_to_atoms(atoms.atoms)
                info["atoms"] = jatoms
                return info  # ,forces,stress
            else:
                print(
                    "Not implemeneted",
                )
        elif self.method == "ewald":
            from intermat.ewald import ewaldsum

            info = {}
            ew = ewaldsum(self.atoms)
            energy = ew.get_ewaldsum()
            info["energy"] = energy
            info["atoms"] = self.atoms
            return info
        elif self.method == "vasp":
            info = self.vasp()
            return info
        elif self.method == "qe":
            info = self.qe()
            return info
        elif self.method == "lammps":
            info = self.lammps()
            return info
        elif self.method == "tb3":
            info = self.tb3()
            return info
        else:
            raise ValueError("Not implemented:", self.method)

    def vasp(self):
        jobname = self.jobname
        if "vasp_cmd" not in self.extra_params:
            self.extra_params = template_extra_params(method="vasp")
        isif = 2
        if self.relax_cell:
            isif = 3
            self.extra_params["inc"]["ISIF"] = isif

        inc = Incar(self.extra_params["inc"])

        def write_jobpy(pyname="job.py", job_json=""):
            f = open(pyname, "w")
            f.write("from jarvis.tasks.vasp.vasp import VaspJob\n")
            f.write("from jarvis.db.jsonutils import loadjson\n")
            f.write('d=loadjson("' + str(job_json) + '")\n')
            f.write("v=VaspJob.from_dict(d)\n")
            f.write("v.runjob()\n")
            f.close()

        atoms = self.atoms
        pos = Poscar(self.atoms)
        pos_name = "POSCAR-" + jobname + ".vasp"
        cwd = os.getcwd()
        name_dir = os.path.join(cwd, jobname)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        os.chdir(name_dir)
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
            length=self.extra_params["kp_length"],
        )
        [a, b, c] = kp.kpts[0]
        if "Surf" in jobname:
            # kp = Kpoints3D(kpoints=[[a, b, c]])
            kp = Kpoints3D(kpoints=[[a, b, 1]])
        else:
            kp = Kpoints3D(kpoints=[[a, b, c]])
        # kp = Kpoints3D(kpoints=[[a, b, 1]])
        # Step-1 Make VaspJob
        v = VaspJob(
            poscar=pos,
            incar=inc,
            potcar=pot,
            kpoints=kp,
            copy_files=self.extra_params["copy_files"],
            jobname=jobname,
            vasp_cmd=self.extra_params["vasp_cmd"],
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
            self.extra_params["extra_lines"]
            + "python "
            + os.getcwd()
            + "/job.py"
        )
        # Step-4 QSUB
        # jobid=os.getcwd() + "/" + jobname + "/jobid"
        jobid = os.getcwd() + "/jobid"
        print("jobid", jobid)

        if self.extra_params["sub_job"] and not os.path.exists(jobid):
            # if sub_job:# and not os.path.exists(jobid):
            if os.path.exists(jobid):
                print("Job might already be submitted!!!")
            Queue.slurm(
                job_line=path,
                jobname=jobname,
                walltime=self.extra_params["walltime"],
                directory=os.getcwd(),
                submit_cmd=["sbatch", "submit_job"],
                queue=self.extra_params["queue"],
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
                atoms = vrun.all_structures[-1]

        os.chdir(cwd)
        info = {}
        info["energy"] = energy
        info["atoms"] = atoms
        return info

    def tb3(
        self,
    ):
        # TODO: Use pydantic
        if "tb3_params" not in self.extra_params:
            self.extra_params = template_extra_params(method="tb3")
        lines = self.extra_params["tb3_params"]
        # print("lines", lines)
        # print("")
        # print("")
        # print("")
        atoms = self.atoms
        # pos = Poscar(self.atoms)
        pos_name = "POSCAR"
        cwd = os.getcwd()
        name_dir = os.path.join(cwd, self.jobname)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        os.chdir(name_dir)
        atoms.write_poscar(filename=pos_name)
        f = open("job.jl", "w")
        for i in lines:
            # print("line", i)
            f.write(i)
        f.close()

        cmd = "julia job.jl"
        os.system(cmd)
        info = {}
        # f = open("energy", "r")
        # en = f.read().splitlines()[0]
        # f.close()
        dq = loadjson("dq")
        en = float(np.loadtxt("energy"))
        efermi = float(np.loadtxt("fermi_energy"))
        # jobname = self.jobname
        # atoms=self.atoms
        # from tb3py.main import get_energy
        # en = get_energy(atoms=atoms)
        info["energy"] = en
        info["dq"] = dq
        info["efermi"] = efermi
        os.chdir(cwd)
        return info

    def qe(self):
        def write_qejob(pyname="job.py", job_json=""):
            """Write template job.py with VaspJob.to_dict() job.json."""
            f = open(pyname, "w")
            f.write("from jarvis.tasks.qe.qe import QEjob\n")
            f.write("from jarvis.db.jsonutils import loadjson\n")
            f.write('d=loadjson("' + str(job_json) + '")\n')
            f.write("v=QEjob.from_dict(d)\n")
            f.write("v.runjob()\n")
            f.close()

        atoms = self.atoms
        # pos = Poscar(self.atoms)
        pos_name = "POSCAR-" + self.jobname + ".vasp"
        cwd = os.getcwd()
        name_dir = os.path.join(cwd, self.jobname)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        os.chdir(name_dir)
        atoms.write_poscar(filename=pos_name)
        if "kp_length" not in self.extra_params:
            kp_length = 30
            print("Setting kp_length", kp_length)
        else:
            kp_length = self.extra_params["kp_length"]
        kp = Kpoints3D().automatic_length_mesh(
            lattice_mat=atoms.lattice_mat,
            length=kp_length,
        )
        [a, b, c] = kp.kpts[0]
        if "Surf" in self.jobname:
            kp = Kpoints3D(kpoints=[[a, b, 1]])
        else:
            kp = Kpoints3D(kpoints=[[a, b, c]])

        # if "qe_params" not in self.extra_params:
        #    self.extra_params = template_extra_params(method="qe")
        # else:
        #    qe_params = self.extra_params["qe_params"]
        #    qe_cmd = self.extra_params["qe_cmd"]
        qe_params = self.extra_params["qe_params"]["qe_params"]
        qe_cmd = self.extra_params["qe_params"]["qe_cmd"]
        qejob = QEjob(
            atoms=atoms,
            input_params=qe_params,
            output_file="relax.out",
            qe_cmd=qe_cmd,
            jobname=self.jobname,
            kpoints=kp,
            input_file="arelax.in",
            url=None,
            psp_dir=None,
            psp_temp_name=None,
        )
        dumpjson(data=qejob.to_dict(), filename="sup.json")
        write_qejob(job_json=os.path.abspath("sup.json"))
        path = (
            "echo hello"
            + " \n. ~/.bashrc\nconda activate mini_alignn\npython "
            + os.getcwd()
            + "/job.py"
        )
        info = {}
        submit_job = self.extra_params["sub_job"]
        if submit_job:
            directory = os.getcwd()
            Queue.slurm(
                job_line=path,
                submit_cmd=["sbatch", "submit_job"],
                jobname=self.jobname,
                queue=self.extra_params["queue"],
                directory=directory,
                walltime=self.extra_params["walltime"],
            )
        else:
            info = qejob.runjob()
        os.chdir(cwd)

        return info

    def lammps(self):
        atoms = self.atoms
        # pos = Poscar(self.atoms)
        pos_name = "POSCAR-" + self.jobname + ".vasp"
        # print('self.jobname',self.jobname)
        cwd = os.getcwd()
        name_dir = os.path.join(cwd, self.jobname)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        os.chdir(name_dir)
        atoms.write_poscar(filename=pos_name)

        lammps_params = self.extra_params["lammps_params"]
        parameters = {
            "pair_style": (lammps_params["pair_style"]),
            "pair_coeff": lammps_params["pair_coeff"],
            "atom_style": lammps_params["atom_style"],
            "control_file": (lammps_params["control_file"]),
        }
        # print('parameters',parameters)
        cmd = lammps_params["lammps_cmd"]
        # Test LammpsJob
        en, final_str, forces = LammpsJob(
            atoms=atoms,
            parameters=parameters,
            lammps_cmd=cmd,
            jobname=self.jobname,
        ).runjob()
        info = {}
        info["energy"] = en
        info["atoms"] = final_str
        os.chdir(cwd)
        return info


cu_pos = """System
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Ni
4
direct
0.0 0.0 0.0 Cu
0.0 0.5 0.5 Cu
0.5 0.0 0.5 Cu
0.5 0.5 0.0 Cu
"""

cu_pos = """System
1.0
3.62621 0.0 0.0
0.0 3.62621 0.0
0.0 0.0 3.62621
Al
4
direct
0.0 0.0 0.0 Cu
0.0 0.5 0.5 Cu
0.5 0.0 0.5 Cu
0.5 0.5 0.0 Cu
"""

# """
if __name__ == "__main__":
    box = [[1.7985, 1.7985, 0], [0, 1.7985, 1.7985], [1.7985, 0, 1.7985]]
    coords = [[0, 0, 0]]
    elements = ["Cu"]
    atoms = Atoms(lattice_mat=box, coords=coords, elements=elements)
    # atoms = Poscar.from_string(cu_pos).atoms
    # calc = Calc(
    #    atoms=atoms, method="tb3", relax_cell=False, jobname="tbtest_job"
    # )
    calc = Calc(
        atoms=atoms, method="vasp", relax_cell=True, jobname="vvqe_job"
    )
    en = calc.predict()
    print(en)
    # semicon_mat_interface_workflow()
    # metal_metal_interface_workflow()
    # semicon_mat_interface_workflow2()
    # quick_compare()
    # semicon_semicon_interface_workflow()
# """
