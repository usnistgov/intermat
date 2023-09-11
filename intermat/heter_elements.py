# qselect -u knc6 | xargs qdel
from jarvis.analysis.interface.zur import (
    ZSLGenerator,
    get_hetero_type,
    make_interface,
    add_atoms,
)
from jarvis.tasks.queue_jobs import Queue
from jarvis.io.vasp.inputs import Poscar, Incar, Potcar
from jarvis.core.kpoints import Kpoints3D
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
import sys, os
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import get_jid_data
from jarvis.tasks.vasp.vasp import VaspJob
from jarvis.db.figshare import data as j_data

data = dict(
    PREC="Accurate",
    ISMEAR=0,
    SIGMA=0.01,
    IBRION=2,
    LORBIT=11,
    GGA="BO",
    PARAM1=0.1833333333,
    PARAM2=0.2200000000,
    LUSE_VDW=".TRUE.",
    AGGAC=0.0000,
    EDIFF="1E-7",
    NSW=500,
    NELM=500,
    ISIF=2,
    ISPIN=2,
    LCHARG=".TRUE.",
    LVTOT=".TRUE.",
    LVHAR=".TRUE.",
    LWAVE=".FALSE.",
)

inc = Incar(data)

import numpy as np
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    symmetrically_distinct_miller_indices,
)
from jarvis.analysis.interface.zur import make_interface
from jarvis.analysis.defects.surface import Surface
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms


# TODO:
# 1) find all possible interfaces i.e. with strain/rotation/tranlation/interlayer distance/terminations/passivation
# 2) Phase diagram package


def get_interface(
    film_atoms=None,
    subs_atoms=None,
    film_index=[1, 1, 1],
    subs_index=[1, 1, 1],
    film_thickness=10,
    subs_thickness=10,
    model_path="",
    seperation=2.5,
    vacuum=15.0,
    max_area_ratio_tol=1.00,
    max_area=200,
    ltol=0.04,
    atol=1,
    apply_strain=False,
    from_conventional_structure=True,
    disp=[0, 0],
):
    """Get work of adhesion."""
    info = {}
    film_surf = Surface(
        film_atoms,
        indices=film_index,
        from_conventional_structure=from_conventional_structure,
        thickness=film_thickness,
        vacuum=vacuum,
    ).make_surface()
    subs_surf = Surface(
        subs_atoms,
        indices=subs_index,
        from_conventional_structure=from_conventional_structure,
        thickness=subs_thickness,
        vacuum=vacuum,
    ).make_surface()
    tmp = subs_surf
    coords = subs_surf.frac_coords
    # print('disp',disp)
    # print('coords1\n')
    # print(coords)
    coords[:, 0] += disp[0]
    coords[:, 1] += disp[1]
    # print('coords2\n')
    # print(coords)
    elements = subs_surf.elements
    lattice_mat = subs_surf.lattice_mat
    new_subs_surf = Atoms(
        coords=coords,
        elements=elements,
        lattice_mat=lattice_mat,
        cartesian=False,
    )
    subs_surf = new_subs_surf

    het = make_interface(
        film=film_surf,
        subs=subs_surf,
        seperation=seperation,
        vacuum=vacuum,
        max_area_ratio_tol=max_area_ratio_tol,
        max_area=max_area,
        ltol=ltol,
        atol=atol,
        apply_strain=apply_strain,
    )
    return het


dat = j_data("dft_3d")


def get_atoms(jid=""):
    for i in dat:
        if i["jid"] == jid:
            return i


def get_hetero_jids(
    jid1="JVASP-664",
    jid2="JVASP-52",
    film_index="",
    subs_index="",
    thickness="",
    disp="",
):
    from jarvis.db.figshare import get_jid_data

    d1 = get_atoms(jid1)  # get_jid_data(jid1, dataset="dft_3d")
    m1 = d1["atoms"]
    k1 = d1["kpoint_length_unit"]

    d2 = get_atoms(jid2)  # get_jid_data(jid2, dataset="dft_3d")
    m2 = d2["atoms"]
    k2 = d2["kpoint_length_unit"]

    mat1 = Atoms.from_dict(m1)
    mat2 = Atoms.from_dict(m2)

    info = get_interface(
        film_atoms=mat1,
        subs_atoms=mat2,
        film_index=film_index,
        subs_index=subs_index,
        film_thickness=thickness,
        subs_thickness=thickness,
        disp=disp,
    )

    return info, max(k1, k2), min(k1, k2)


def write_jobpy(pyname="job.py", job_json=""):
    # job_json = os.getcwd()+'/'+'job.json'
    f = open(pyname, "w")
    f.write("from jarvis.tasks.vasp.vasp import VaspJob\n")
    f.write("from jarvis.db.jsonutils import loadjson\n")
    f.write('d=loadjson("' + str(job_json) + '")\n')
    f.write("v=VaspJob.from_dict(d)\n")
    f.write("v.runjob()\n")
    f.close()


jids = [
    "JVASP-816",
    "JVASP-943",
    # "JVASP-867",
    # "JVASP-963",
    # "JVASP-14606",
    # "JVASP-972",
    # "JVASP-825",
    # "JVASP-1002",
]

film_index = [1, 1, 1]
subs_index = [1, 1, 1]
thickness = 15
disp = [0, 0]

disp_intvl = 0.1
X, Y = np.mgrid[
    -0.5 + disp_intvl : 0.5 + disp_intvl : disp_intvl,
    -0.5 + disp_intvl : 0.5 + disp_intvl : disp_intvl,
]
xy = np.vstack((X.flatten(), Y.flatten())).T


def test_hetero():
    count = 0
    cwd = str(os.getcwd())
    for ii in range(len(jids)):
        for jj in range(len(jids)):
            i = jids[ii]
            j = jids[jj]
            if count < 20000 and i != j and ii > jj:
                # try:
                for dis in xy:
                    dis_tmp = dis
                    dis_tmp[0] = round(dis_tmp[0], 3)
                    dis_tmp[1] = round(dis_tmp[1], 3)
                    print(i, j)
                    info1, k1, k2 = get_hetero_jids(
                        jid1=i,
                        jid2=j,
                        film_index=film_index,
                        subs_index=subs_index,
                        thickness=thickness,
                        disp=dis_tmp,
                    )
                    intf = info1["interface"]
                    mis_u1 = info1["mismatch_u"]
                    mis_v1 = info1["mismatch_v"]
                    max_mis1 = max(abs(mis_u1), abs(mis_v1))

                    info2, k1, k2 = get_hetero_jids(
                        jid1=j,
                        jid2=i,
                        film_index=film_index,
                        subs_index=subs_index,
                        thickness=thickness,
                        disp=dis_tmp,
                    )
                    intf = info2["interface"]
                    mis_u2 = info2["mismatch_u"]
                    mis_v2 = info2["mismatch_v"]
                    max_mis2 = max(abs(mis_u2), abs(mis_v2))
                    if max_mis2 > max_mis1:
                        chosen_info = info1
                    else:
                        chosen_info = info2
                    ats = chosen_info["interface"]  # .get_string(cart=False))
                    print(
                        "chosen_info", chosen_info["mismatch_u"], ats.num_atoms
                    )
                    if ats.num_atoms < 500:
                        pos = Poscar(ats)
                        ####print(pos)
                        name = (
                            "Interface-"
                            + i
                            + "_"
                            + j
                            + "_"
                            + "film_miller_"
                            + "_".join(map(str, film_index))
                            + "_sub_miller_"
                            + "_".join(map(str, subs_index))
                            + "_thickness_"
                            + str(thickness)
                            + "_"
                            + "disp_"
                            + "_".join(map(str, dis))
                        )
                        print("name", name)
                        pos_name = "POSCAR-" + name + ".vasp"
                        ats.write_poscar(filename=pos_name)
                        name_dir = os.path.join(cwd, name)
                        if not os.path.exists(name_dir):
                            os.mkdir(name_dir)
                        os.chdir(name_dir)
                        pos.comment = name

                        new_symb = []
                        for ee in ats.elements:
                            if ee not in new_symb:
                                new_symb.append(ee)
                        pot = Potcar(elements=new_symb)
                        leng = min([k1, k2])
                        if leng - 25 > 0:
                            leng = leng - 25
                        print("leng", k1, k2, leng)
                        kp = Kpoints3D().automatic_length_mesh(
                            lattice_mat=ats.lattice_mat, length=leng
                        )
                        [a, b, c] = kp.kpts[0]
                        kp = Kpoints3D(kpoints=[[a, b, 1]])
                        # Step-1 Make VaspJob
                        v = VaspJob(
                            poscar=pos,
                            incar=inc,
                            potcar=pot,
                            kpoints=kp,
                            copy_files=["/users/knc6/bin/vdw_kernel.bindat"],
                            jobname=name,
                            vasp_cmd="mpirun vasp_std",
                            # vasp_cmd="mpirun /users/knc6/VASP/vasp54/src/vasp.5.4.1Dobby/bin/vasp_std",
                        )

                        count = count + 1

                        # Step-2 Save on a dict
                        jname = (
                            os.getcwd() + "/" + "VaspJob_" + name + "_job.json"
                        )
                        dumpjson(data=v.to_dict(), filename=jname)

                        # Step-3 Write jobpy
                        write_jobpy(job_json=jname)
                        path = (
                            "\nmodule load vasp/6.3.1 \nsource ~/anaconda2/envs/my_jarvis/bin/activate my_jarvis \npython "
                            + os.getcwd()
                            + "/job.py"
                        )

                        # Step-4 QSUB
                        # Queue.slurm(
                        #   job_line=path,
                        #   jobname=name,
                        #   walltime="7-00:00:00",
                        #   directory=os.getcwd(),
                        #   submit_cmd=["sbatch", "submit_job"],
                        # )
                        os.chdir(cwd)
            # except:
            #    pass


test_hetero()
