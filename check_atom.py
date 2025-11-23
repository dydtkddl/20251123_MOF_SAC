############ check_atom.py ############
from ase.io import read
from ase.data import chemical_symbols
from mace.calculators import MACECalculator


# -------------------
# Init calculator
# -------------------
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cpu",
    default_dtype="float64"
)

# -------------------
# extract supported atomic numbers
# -------------------

try:
    SUPPORTED_Z = list(calc.model.atomic_numbers)
except:
    SUPPORTED_Z = list(calc.models[0].atomic_numbers)

SUPPORTED_Z = sorted(SUPPORTED_Z)
SUPPORTED_ELEMENTS = set(SUPPORTED_Z)


print("==============================================")
print(" Supported atomic numbers:")
print(" ", SUPPORTED_Z)
print("")
print(" Supported symbols:")
print(" ", [chemical_symbols[z] for z in SUPPORTED_Z])
print("==============================================\n")


def cif_supported(cif_path):
    try:
        atoms = read(cif_path)
    except:
        return False

    zs = atoms.get_atomic_numbers()

    for z in zs:
        if z not in SUPPORTED_ELEMENTS:
            return False

    return True

