from jarvis.core.atoms import Atoms
from intermat.generate import InterfaceCombi
from intermat.calculators import template_extra_params
import numpy as np
import itertools
from jarvis.db.figshare import data
from alignn.ff.ff import default_path
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from intermat.calculators import template_extra_params
from intermat.calculators import Calc

dft_3d = data("dft_3d")
info = template_extra_params(method="vasp")


from jarvis.analysis.structure.spacegroup import Spacegroup3D
def get_primitive(atoms, symprec=1e-4):
    spg = Spacegroup3D(atoms, symprec=symprec)
    return spg.primitive_atoms


combinations = [
    ["JVASP-1002", "JVASP-816", [1, 1, 1], [1, 1, 1]],
]
seperations = np.arange(0.5, 3.5, 0.1)
atoms_arr = []
for i in combinations:
    # try:

    x = InterfaceCombi(
        film_ids=[i[0]],
        subs_ids=[i[1]],
        film_indices=[i[2]],
        subs_indices=[i[3]],
        disp_intvl=0.1,
        vacuum_interface=2,
        dataset=dft_3d,
        max_area=300,
        ltol=0.08,
        seperations=[2.5],
        from_conventional_structure_film=True,
        from_conventional_structure_subs=True,
    )

    extra_params = {}
    extra_params["alignn_params"] = {}
    extra_params["alignn_params"]["model_path"] = default_path()
    structs = x.generate()
    wads = x.calculate_wad(method="alignn_ff", extra_params=extra_params)
    index = np.argmin(wads)
    disp = x.xy[index]

    x = InterfaceCombi(
        film_ids=[i[0]],
        subs_ids=[i[1]],
        film_indices=[i[2]],
        subs_indices=[i[3]],
        disp_intvl=[disp],
        vacuum_interface=2,
        seperations=seperations,
        dataset=dft_3d,
        max_area=300,
        ltol=0.08,
        from_conventional_structure_film=True,
        from_conventional_structure_subs=True,
    )
    extra_params = {}
    extra_params["alignn_params"] = {}
    extra_params["alignn_params"]["model_path"] = default_path()
    structs = x.generate()
    wads = x.calculate_wad(method="alignn_ff", extra_params=extra_params)
    index = np.argmin(wads)
    sep = seperations[index]

    x = InterfaceCombi(
        film_ids=[i[0]],
        subs_ids=[i[1]],
        film_indices=[i[2]],
        subs_indices=[i[3]],
        disp_intvl=[disp],
        vacuum_interface=sep / 2,
        seperations=[sep],
        dataset=dft_3d,
        max_area=300,
        ltol=0.08,
        from_conventional_structure_film=True,
        from_conventional_structure_subs=True,
    )
    extra_params = {}
    extra_params["alignn_params"] = {}
    extra_params["alignn_params"]["model_path"] = default_path()
    structs = x.generate()
    wads = x.calculate_wad(method="alignn_ff", extra_params=extra_params)
    index = np.argmin(wads)
    final_dat = structs[np.argmin(wads)]
    final_interface = Atoms.from_dict(final_dat["interface"])
    print("final_interface", final_interface)
    # import sys
    # sys.exit()
    final_name = final_dat["interface_name"]
    extra_lines = (
        ". ~/.bashrc\nmodule load vasp/6.3.1\n"
        + "conda activate mini_alignn\n"
    )

    info["inc"]["ISIF"] = 7
    info["inc"]["ENCUT"] = 520
    info["inc"]["NEDOS"] = 5000
    info["queue"] = "debug"
    vasp_cmd = "mpirun vasp_std"
    job_line = (
        "conda activate /wrk/knc6/Software/intermat310 \n" + "python job.py"
    )
    pre_job_lines = (
        "#SBATCH -n 32\n#SBATCH --hint=nomultithread\n#SBATCH --exclusive\nmodule purge\nmodule load ucx/1.13.1\nmodule load oneapi/2023.0.0\nmodule load imp
i/oneapi-2023.0.0\nmodule load vasp/6.3.1/impi-oneapi-2023.0.0\nexport OMP_NUM_THREADS=1\n"
        + job_line
    )
    info["extra_lines"] = pre_job_lines
    info["vasp_cmd"] = vasp_cmd
    copy_files = ["/home/knc6/bin/vdw_kernel.bindat"]
    info["copy_files"] = copy_files
    info["extra_lines"] = job_line
    info["pre_job_lines"] = pre_job_lines
    info["queue"] = "debug"
    info["kp_length"] = "2_2_1"
    info["walltime"] = "2:00:00"
    info["cores"] = None

    calc = Calc(
        method="vasp",
        atoms=final_interface,
        extra_params=info,
        jobname=final_name,
    )
    en = calc.predict()["energy"]
    opt_atoms = calc.predict()["atoms"]
    """
    final_name = final_name + "_mbj"
    info["inc"]["METAGGA"] = "MBJ"
    info["inc"]["ISYM"] = 0
    # info["inc"]["LOPTICS"] = '.TRUE.'
    calc = Calc(
        method="vasp",
        atoms=opt_atoms,
        extra_params=info,
        jobname=final_name,
    )
    en = calc.predict()["energy"]
    wads = x.calculate_wad(
        method="vasp",
        index=index,
        do_surfaces=False,
        extra_params=info,
    )
    # break
    import sys
    sys.exit()
    """

    #    pass
