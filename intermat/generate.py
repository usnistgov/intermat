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
# 0) automate stuff in this paper https://link.springer.com/article/10.1007/s10853-012-6425-z   
# 1) find all possible interfaces i.e. with strain/rotation/tranlation/interlayer distance/terminations/passivation
# 2) Phase diagram package
# 3) SrTiO3/Si


def get_interface(
    film_atoms=None,
    subs_atoms=None,
    film_index=[1, 1, 1],
    subs_index=[0, 0, 1],
    film_thickness=25,
    subs_thickness=25,
    model_path="",
    seperation=3.0,
    vacuum=8.0,
    max_area_ratio_tol=1.00,
    max_area=500,
    ltol=0.05,
    atol=1,
    apply_strain=False,
    from_conventional_structure=True,
    gpaw_verify=False,
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
    return het["interface"]

if __name__ == "__main__":
    jid1 = "JVASP-1002"
    jid2 = "JVASP-86503"
    film_index = [0, 0, 1]
    subs_index = [0, 0, 1]
    film_thickness = 5
    subs_thickness = 5
    ltol = 0.1
    m1 = get_jid_data(jid=jid1, dataset="dft_3d")["atoms"]
    m2 = get_jid_data(jid=jid2, dataset="dft_3d")["atoms"]
    mat1 = Atoms.from_dict(m1)
    mat2 = Atoms.from_dict(m2)
    h = get_interface(
        film_atoms=mat1,
        subs_atoms=mat2,
        film_index=film_index,
        subs_index=subs_index,
        film_thickness=film_thickness,
        ltol=ltol,
        subs_thickness=subs_thickness,
    )
    print(h)
