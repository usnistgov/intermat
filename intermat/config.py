from typing import List
from typing import Literal
from typing import Dict
from pydantic_settings import BaseSettings

try:
    from gpaw import Davidson
except Exception:
    pass
template_vasp_incar_dict = dict(
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


template_qe_params = {
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


template_tb3_lines = [
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


template_lammps_params = dict(
    cmd="lmp_serial<in.main>out",
    lammps_cmd="lmp_serial<in.main>out",
    pair_style="eam/alloy",
    pair_coeff="tests/Mishin-Ni-Al-Co-2013.eam.alloy",
    atom_style="charge",
    control_file="tests/inelast.mod",
)

template_gpaw_params = dict(
    smearing=0.01,
    cutoff=400,
    xc="PBE",
    basis="szp(dzp)",
    mode="lcao",
    spinpol=False,
    nbands="120%",
    symmetry="on",
    parallel={
        "sl_auto": True,
        "domain": 2,
        "augment_grids": True,
    },
    maxiter=100,
    convergence={"energy": 1},
    # convergence={"density": 1e-12, "energy": 1e-1},
    # eigensolver=Davidson(niter=2),
    out_file="gs.out",
    out_gpw="out.gpw",
    kp_length=10,
)


class IntermatConfig(BaseSettings):
    # Generator config
    film_file_path: str = ""
    substrate_file_path: str = ""
    film_jid: str = ""
    substrate_jid: str = ""
    film_index: str = "0_0_1"
    substrate_index: str = "0_0_1"
    film_thickness: float = 16
    substrate_thickness: float = 16
    seperation: float = 2.5
    vacuum_interface: float = 2.0
    rotate_xz: bool = False
    disp_intvl: float = 0.0
    calculator_method: Literal[
        "",
        "ewald",
        "alignn_ff",
        "emt",
        "matgl",
        "eam_ase",
        "vasp",
        "tb3",
        "qe",
        "lammps",
        "gpaw",
    ] = ""
    max_area: float = 300.0
    ltol: float = 0.08
    atol: float = 1
    from_conventional_structure_film: bool = True
    from_conventional_structure_subs: bool = True
    dataset: str = "dft_3d"
    verbose: bool = True
    plot_wads: bool = True
    # Calculator generic config
    extra_lines: str = (
        "\nmodule load vasp/6.3.1\n" + "conda activate my_intermat\n"
    )
    sub_job: bool = False
    walltime: str = "30-00:00:00"
    queue: str = "rack1,rack2"
    do_surfaces: bool = True
    copy_files: List = []
    # Calculator vasp config
    vasp_params: Dict = {
        "vasp_cmd": "mpirun vasp_std",
        "inc": template_vasp_incar_dict,
    }
    kp_length: int = 0
    # Calculator tb3 config
    tb3_lines: List = template_tb3_lines
    # Calculator qe config
    qe_params: Dict = {"qe_cmd": "pw.x", "qe_params": template_qe_params}
    # Calculator lammps config
    lammps_params: Dict = template_lammps_params
    # Calculator gpaw config
    gpaw_params: Dict = template_gpaw_params
    # Calculator eam_ase
    potential: str = ""
