def angstrom2bohr(a: float):
    # convert the unit from angstrom to bohr
    return 1.8897259886 * a

def energy2hartree(a: float, unit0: str):
    # convert the energy in some units to hartree
    unit = unit0.lower().strip()
    if unit.startswith("kj"):  # kJ/mol
        return a * 0.0003808798
    elif unit.startswith("kcal"):  # kcal/mol
        return a * 0.001593601
    elif unit.startswith("hartree"):
        return a
    elif unit.startswith("ev"):
        return a * 0.036749304951
    elif unit.startswith("mev"):
        return a * 0.000036749304951
    elif unit.startswith("cm"):
        return a * 0.0000045563352812
    else:
        raise RuntimeError("Unknown energy unit: %s" % unit0)

def kcalmol2hartree(a: float):
    # convert kcal/mol to hartree
    return a * 0.0015936011

# enthalpy of formation of a single atom at 0K (Hartree)
# ref: CCCBDB
_ATOM_DHF0 = {
    "H": energy2hartree(216.04, "kj/mol"),
    "He": energy2hartree(0.0, "kj/mol"),
    "Li": energy2hartree(157.74, "kj/mol"),
    "Be": energy2hartree(319.75, "kj/mol"),
    "B": energy2hartree(559.91, "kj/mol"),
    "C": energy2hartree(711.19, "kj/mol"),
    "N": energy2hartree(470.82, "kj/mol"),
    "O": energy2hartree(246.84, "kj/mol"),
    "F": energy2hartree(77.27, "kj/mol"),
    "Ne": energy2hartree(0.0, "kj/mol"),
    "Na": energy2hartree(107.76, "kj/mol"),
    "Mg": energy2hartree(145.90, "kj/mol"),
    "Al": energy2hartree(327.62, "kj/mol"),
    "Si": energy2hartree(445.67, "kj/mol"),
    "P": energy2hartree(315.66, "kj/mol"),
    "S": energy2hartree(274.92, "kj/mol"),
    "Cl": energy2hartree(119.63, "kj/mol"),
    "Ar": energy2hartree(0.0, "kj/mol"),
}
def get_atom_dHf0(atom: str):
    # returns the enthalpy of formation of a single atom at 0K
    # the unit is in Hartree
    return _ATOM_DHF0[atom]
