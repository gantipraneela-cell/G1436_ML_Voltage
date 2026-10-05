from mp_api.client import MPRester
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd
import numpy as np


# ============================================================
# MATERIALS PROJECT API KEY
# ============================================================

API_KEY = "Lw72xn9mQ0pOCvtIZot0X5wryFOYqO4c"


# ============================================================
# DOWNLOAD LI-ONLY DATA
# ============================================================

print("Connecting to Materials Project...")

with MPRester(API_KEY) as mpr:

    data = mpr.materials.insertion_electrodes.search(
        working_ion="Li",
        fields=[
            "battery_formula",
            "working_ion",
            "host_structure",
            "average_voltage",
            "capacity_grav",
            "capacity_vol",
            "energy_grav",
            "energy_vol",
            "max_voltage_step",
            "num_steps",
            "max_delta_volume"
        ]
    )

print("Li records received:", len(data))


# ============================================================
# FEATURE CREATION FUNCTION
# ============================================================

def create_features(doc):

    structure = doc.host_structure

    if structure is None:
        return None

    composition = structure.composition
    elements = list(composition.elements)

    total_atoms = composition.num_atoms

    row = {}


    # ========================================================
    # 1. BASIC COMPOSITION
    # ========================================================

    row["num_elements"] = len(elements)

    row["num_sites"] = structure.num_sites

    row["total_atoms"] = total_atoms


    # ========================================================
    # 2. ELEMENTAL FRACTIONS
    # ========================================================

    fractions = composition.get_el_amt_dict()

    # Fixed 118 element columns
    # This guarantees Li and mixed datasets have
    # exactly the same columns.

    for z in range(1, 119):

        element = Element.from_Z(z)

        row[f"fraction_{element.symbol}"] = (
            fractions.get(element.symbol, 0)
            / total_atoms
        )


    # ========================================================
    # 3. ELEMENTAL PROPERTY STATISTICS
    # ========================================================

    properties = {
        "atomic_number": [],
        "atomic_mass": [],
        "electronegativity": [],
        "atomic_radius": [],
        "mendeleev_number": []
    }

    weights = []


    for element in elements:

        fraction = (
            fractions[element.symbol]
            / total_atoms
        )

        weights.append(fraction)


        # Atomic number
        properties["atomic_number"].append(
            float(element.Z)
        )


        # Atomic mass
        properties["atomic_mass"].append(
            float(element.atomic_mass)
            if element.atomic_mass is not None
            else np.nan
        )


        # Electronegativity
        properties["electronegativity"].append(
            float(element.X)
            if element.X is not None
            else np.nan
        )


        # Atomic radius
        properties["atomic_radius"].append(
            float(element.atomic_radius)
            if element.atomic_radius is not None
            else np.nan
        )


        # Mendeleev number
        properties["mendeleev_number"].append(
            float(element.mendeleev_no)
            if element.mendeleev_no is not None
            else np.nan
        )


    weights = np.array(weights)


    for name, values in properties.items():

        values = np.array(values)

        valid = ~np.isnan(values)

        if valid.sum() == 0:

            row[f"{name}_mean"] = np.nan
            row[f"{name}_min"] = np.nan
            row[f"{name}_max"] = np.nan
            row[f"{name}_std"] = np.nan

            continue


        v = values[valid]

        w = weights[valid]

        w = w / w.sum()


        mean = np.average(v, weights=w)

        std = np.sqrt(
            np.average(
                (v - mean) ** 2,
                weights=w
            )
        )


        row[f"{name}_mean"] = mean
        row[f"{name}_min"] = np.min(v)
        row[f"{name}_max"] = np.max(v)
        row[f"{name}_std"] = std


    # ========================================================
    # 4. OXIDATION STATE DESCRIPTORS
    # ========================================================

    oxidation_states = []


    for element in elements:

        try:

            states = element.common_oxidation_states

            if states:
                oxidation_states.extend(states)

        except Exception:
            pass


    if oxidation_states:

        oxidation_states = np.array(
            oxidation_states,
            dtype=float
        )

        row["oxidation_mean"] = np.mean(
            oxidation_states
        )

        row["oxidation_min"] = np.min(
            oxidation_states
        )

        row["oxidation_max"] = np.max(
            oxidation_states
        )

        row["oxidation_std"] = np.std(
            oxidation_states
        )

        row["oxidation_range"] = (
            np.max(oxidation_states)
            - np.min(oxidation_states)
        )

    else:

        row["oxidation_mean"] = np.nan
        row["oxidation_min"] = np.nan
        row["oxidation_max"] = np.nan
        row["oxidation_std"] = np.nan
        row["oxidation_range"] = np.nan


    # ========================================================
    # 5. LATTICE PARAMETERS
    # ========================================================

    lattice = structure.lattice

    row["lattice_a"] = lattice.a
    row["lattice_b"] = lattice.b
    row["lattice_c"] = lattice.c

    row["lattice_alpha"] = lattice.alpha
    row["lattice_beta"] = lattice.beta
    row["lattice_gamma"] = lattice.gamma

    row["lattice_volume"] = lattice.volume


    # ========================================================
    # 6. STRUCTURAL FEATURES
    # ========================================================

    row["density"] = structure.density

    row["volume_per_atom"] = (
        structure.volume /
        structure.num_sites
    )


    # ========================================================
    # 7. SPACE GROUP
    # ========================================================

    try:

        analyzer = SpacegroupAnalyzer(structure)

        row["spacegroup_number"] = (
            analyzer.get_space_group_number()
        )

    except Exception:

        row["spacegroup_number"] = np.nan


    # ========================================================
    # 8. BATTERY FEATURES
    # ========================================================

    row["capacity_grav"] = doc.capacity_grav

    row["capacity_vol"] = doc.capacity_vol

    row["energy_grav"] = doc.energy_grav

    row["energy_vol"] = doc.energy_vol

    row["max_voltage_step"] = doc.max_voltage_step

    row["num_steps"] = doc.num_steps

    row["max_delta_volume"] = doc.max_delta_volume


    # ========================================================
    # 9. WORKING ION
    # ========================================================

    # For Li dataset this is always Li.
    # Keep it because the mixed dataset also needs
    # the exact same column.

    row["working_ion"] = doc.working_ion


    # ========================================================
    # 10. TARGET
    # ========================================================

    row["average_voltage"] = doc.average_voltage


    return row


# ============================================================
# PROCESS LI DATA
# ============================================================

rows = []

print()
print("Creating Li features...")


for i, doc in enumerate(data):

    try:

        result = create_features(doc)

        if result is not None:
            rows.append(result)

    except Exception as e:

        print(
            f"Skipped record {i}: {e}"
        )


    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{len(data)}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

li_df = pd.DataFrame(rows)


# ============================================================
# FIX COLUMN ORDER
# ============================================================

li_df = li_df.sort_index(axis=1)


# ============================================================
# SAVE
# ============================================================

li_df.to_csv(
    "battery_Li_features.csv",
    index=False
)


# ============================================================
# INFORMATION
# ============================================================

print()
print("======================================")
print("LI DATASET COMPLETED")
print("======================================")

print()

print("Shape:")
print(li_df.shape)

print()

print("Number of features/columns:")
print(len(li_df.columns))

print()

print("Saved as:")
print("battery_Li_features.csv")