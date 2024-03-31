from intermat.run_intermat import main
import os
from jarvis.db.jsonutils import loadjson
from intermat.generate import InterfaceCombi, lead_mat_designer
from intermat.analyze import offset
import glob

config_file = os.path.join(os.path.dirname(__file__), "config.json")
config_dat = loadjson(config_file)

offse_example = os.path.join(os.path.dirname(__file__), "forpytest.zip")


def test_gen():
    combinations = [["JVASP-1002", "JVASP-816", [1, 1, 0], [1, 1, 0]]]
    for i in combinations:
        x = InterfaceCombi(
            film_ids=[i[0]],
            subs_ids=[i[1]],
            film_indices=[i[2]],
            subs_indices=[i[3]],
            disp_intvl=0.05,
            vacuum_interface=2,
            lead_ratio=0.3,
        )


def test_leadmat():
    x = lead_mat_designer()


def test_alignn_ff():

    main(config_file)


def test_eam_ase():
    config_dat["calculator_method"] = "eam_ase"
    main(config_dat)


def test_ewald():
    config_dat["calculator_method"] = "ewald"
    main(config_dat)


def test_eam_lammps():
    config_dat["calculator_method"] = "lammps"
    main(config_dat)


def test_offset():
    cmd = "unzip " + offse_example
    os.system(cmd)
    os.chdir("forpytest")
    for i in glob.glob("Int*/opt_*/opt*/LOCPOT"):
        phi = offset(fname=i, left_index=2)
        print("phi", phi)
        assert phi < -0.2
    os.chdir("..")
    cmd = "rm -r forpytest"
    os.system(cmd)


# test_offset()
# def test_qe():
#    config_dat["calculator_method"] = "qe"
#    config_dat["kp_length"] = 0
#    config_dat["disp_intvl"] = 0
#    main(config_dat)

# test_qe()
