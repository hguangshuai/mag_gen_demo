"""Streamlit app for magnetic VAE generator.

Deploy to Streamlit Cloud:
1. Push this file to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Select this file
5. Deploy!
"""

import sys
from pathlib import Path
from io import StringIO

import streamlit as st
import numpy as np

# Add current directory to path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import simple_generator (should be in same directory)
from simple_generator import (
    SimpleGenerator,
    format_composition,
    lattice_to_params,
)


@st.cache_resource
def load_generator():
    """Load generator (cached for performance)."""
    # Weights should be in same directory as app
    weights_path = CURRENT_DIR / "weights.npz"
    
    if not weights_path.exists():
        st.error(f"❌ Weights file not found at {weights_path}")
        st.error("Please ensure weights.npz is in the same directory as streamlit_app.py")
        st.stop()
    
    return SimpleGenerator(weights_path)


def bsat_to_magmom_per_atom(bsat_tesla: float, unit_cell_volume_m3: float = None, num_atoms: int = None) -> float:
    """
    Convert B_sat (Tesla) to magnetic moment per atom (μB).
    
    Formula: B_s = μ_0 × M_s
    M_s = (Total moment (µ_B) × 9.274 × 10^-24) / (Unit cell volume)
    
    If unit_cell_volume and num_atoms are not provided, uses typical values for estimation.
    """
    mu_0 = 4 * np.pi * 1e-7  # T·m/A
    bohr_magneton = 9.274e-24  # A·m^2
    
    # Calculate M_s from B_s
    M_s = bsat_tesla / mu_0  # A/m
    
    # Use typical values if not provided
    if unit_cell_volume_m3 is None:
        # Typical unit cell volume: ~100 Å³ = 1e-28 m³
        unit_cell_volume_m3 = 1e-28
    
    if num_atoms is None:
        # Typical number of atoms per unit cell
        num_atoms = 4
    
    # Calculate total moment in μB
    total_moment_mub = (M_s * unit_cell_volume_m3) / bohr_magneton
    
    # Calculate per atom
    magmom_per_atom = total_moment_mub / num_atoms
    
    return magmom_per_atom


def calculate_unit_cell_volume(lattice: np.ndarray) -> float:
    """
    Calculate unit cell volume from lattice matrix.
    Volume = |det(lattice)| in m^3 (assuming lattice is in Angstroms, convert to meters)
    """
    # Lattice is typically in Angstroms, convert to meters
    lattice_m = lattice * 1e-10  # Convert Å to m
    volume_m3 = abs(np.linalg.det(lattice_m))  # m^3
    return volume_m3


def magmom_per_atom_to_bsat(magmom_per_atom: float, unit_cell_volume_m3: float, num_atoms: int) -> float:
    """
    Convert magnetic moment per atom (μB) to B_sat (Tesla).
    
    Formula: B_s = μ_0 × M_s
    M_s = (Total moment (µ_B) × 9.274 × 10^-24) / (Unit cell volume)
    """
    mu_0 = 4 * np.pi * 1e-7  # T·m/A
    bohr_magneton = 9.274e-24  # A·m^2
    
    # Calculate total moment
    total_moment_mub = magmom_per_atom * num_atoms
    
    # Calculate M_s
    M_s = (total_moment_mub * bohr_magneton) / unit_cell_volume_m3  # A/m
    
    # Calculate B_s
    bsat = mu_0 * M_s  # Tesla
    
    return bsat


def params_to_lattice(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert lattice parameters to 3x3 lattice matrix."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Calculate volume
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    
    volume = a * b * c * np.sqrt(
        1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    )
    
    # Build lattice matrix
    lattice = np.zeros((3, 3))
    lattice[0, 0] = a
    lattice[0, 1] = 0.0
    lattice[0, 2] = 0.0
    
    lattice[1, 0] = b * cos_gamma
    lattice[1, 1] = b * sin_gamma
    lattice[1, 2] = 0.0
    
    lattice[2, 0] = c * cos_beta
    if abs(sin_gamma) > 1e-10:
        lattice[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        lattice[2, 2] = volume / (a * b * sin_gamma)
    else:
        # Handle edge case when sin_gamma ≈ 0
        lattice[2, 1] = 0.0
        lattice[2, 2] = c
    
    return lattice


def apply_symmetry_constraints(
    a: float, b: float, c: float, 
    alpha: float, beta: float, gamma: float,
    uniaxial_symmetry: bool,
    crystal_system: str
) -> tuple[float, float, float, float, float, float, str]:
    """
    Apply symmetry constraints with small random perturbations.
    
    Returns: (a, b, c, alpha, beta, gamma, crystal_system_name)
    """
    # Small random perturbation (±1-3%)
    perturbation = np.random.uniform(-0.03, 0.03)
    
    if not uniaxial_symmetry:
        # Cubic: a = b = c, α = β = γ = 90°
        base_a = (a + b + c) / 3.0
        a_new = base_a * (1 + perturbation)
        b_new = a_new  # Enforce a = b = c
        c_new = a_new
        alpha_new = 90.0
        beta_new = 90.0
        gamma_new = 90.0
        crystal_system_name = "Cubic"
        
    elif crystal_system == "Tetragonal":
        # Tetragonal: a = b ≠ c, α = β = γ = 90°
        base_a = (a + b) / 2.0
        a_new = base_a * (1 + perturbation)
        b_new = a_new  # Enforce a = b
        c_new = c * (1 + np.random.uniform(-0.03, 0.03))
        alpha_new = 90.0
        beta_new = 90.0
        gamma_new = 90.0
        crystal_system_name = "Tetragonal"
        
    elif crystal_system == "Trigonal":
        # Trigonal (Rhombohedral): a = b = c, α = β = γ ≠ 90°
        base_a = (a + b + c) / 3.0
        base_alpha = (alpha + beta + gamma) / 3.0
        # Ensure alpha is not 90° and is reasonable (60-120°)
        if abs(base_alpha - 90.0) < 5.0:
            base_alpha = 75.0 if base_alpha < 90.0 else 105.0
        
        a_new = base_a * (1 + perturbation)
        b_new = a_new  # Enforce a = b = c
        c_new = a_new
        alpha_new = base_alpha * (1 + np.random.uniform(-0.02, 0.02))
        alpha_new = np.clip(alpha_new, 60.0, 120.0)
        beta_new = alpha_new  # Enforce α = β = γ
        gamma_new = alpha_new
        crystal_system_name = "Trigonal"
        
    elif crystal_system == "Hexagonal":
        # Hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
        base_a = (a + b) / 2.0
        a_new = base_a * (1 + perturbation)
        b_new = a_new  # Enforce a = b
        c_new = c * (1 + np.random.uniform(-0.03, 0.03))
        alpha_new = 90.0
        beta_new = 90.0
        gamma_new = 120.0  # Enforce γ = 120°
        crystal_system_name = "Hexagonal"
        
    else:
        # Fallback: no constraints
        a_new = a * (1 + perturbation)
        b_new = b * (1 + np.random.uniform(-0.03, 0.03))
        c_new = c * (1 + np.random.uniform(-0.03, 0.03))
        alpha_new = alpha
        beta_new = beta
        gamma_new = gamma
        crystal_system_name = "Triclinic"
    
    return a_new, b_new, c_new, alpha_new, beta_new, gamma_new, crystal_system_name


def compute_shape_analysis(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> dict:
    """
    Compute geometric quantities based on lattice parameters.
    
    Returns:
        dict with keys: metric_tensor, eigenvalues, asphericity, uniaxiality_index
    """
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Compute metric tensor G
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    
    G = np.array([
        [a**2, a * b * cos_gamma, a * c * cos_beta],
        [a * b * cos_gamma, b**2, b * c * cos_alpha],
        [a * c * cos_beta, b * c * cos_alpha, c**2]
    ])
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(G)
    eigenvalues = np.real(eigenvalues)  # Ensure real values
    eigenvalues_sorted = np.sort(eigenvalues)
    
    lambda_min = eigenvalues_sorted[0]
    lambda_mid = eigenvalues_sorted[1]
    lambda_max = eigenvalues_sorted[2]
    
    # Compute asphericity: max of sqrt of lambdas divided by min
    # When max and min are the same, asphericity = 1
    if abs(lambda_min) > 1e-10:
        sqrt_lambda_max = np.sqrt(lambda_max)
        sqrt_lambda_min = np.sqrt(lambda_min)
        asphericity = sqrt_lambda_max / sqrt_lambda_min
    else:
        asphericity = 1.0  # Default when min is too small
    
    # Compute uniaxiality index
    if abs(lambda_mid) > 1e-10:
        uniaxiality_index = (lambda_max - lambda_min) / lambda_mid
    else:
        uniaxiality_index = 0.0
    
    return {
        'metric_tensor': G,
        'eigenvalues': eigenvalues_sorted,
        'asphericity': asphericity,
        'uniaxiality_index': uniaxiality_index
    }


def structure_to_cif(
    lattice: np.ndarray,
    species: list[str],
    frac_coords: np.ndarray,
    pocc: np.ndarray,
    ordered: int,
    name: str = "generated",
    uniaxial_symmetry: bool = False,
    crystal_system: str = "Cubic",
) -> tuple[str, str, dict]:
    """Write CIF file with optional symmetry constraints.
    
    Returns: (cif_string, crystal_system_name, shape_analysis_dict)
    """
    a, b, c, alpha, beta, gamma = lattice_to_params(lattice)
    
    # Apply symmetry constraints
    a, b, c, alpha, beta, gamma, crystal_system_name = apply_symmetry_constraints(
        a, b, c, alpha, beta, gamma, uniaxial_symmetry, crystal_system
    )
    
    # Compute shape analysis after symmetry adjustments
    shape_analysis = compute_shape_analysis(a, b, c, alpha, beta, gamma)
    
    buffer = StringIO()
    buffer.write(f"data_{name}\n")
    buffer.write(f"_cell_length_a    {a:.6f}\n")
    buffer.write(f"_cell_length_b    {b:.6f}\n")
    buffer.write(f"_cell_length_c    {c:.6f}\n")
    buffer.write(f"_cell_angle_alpha {alpha:.6f}\n")
    buffer.write(f"_cell_angle_beta  {beta:.6f}\n")
    buffer.write(f"_cell_angle_gamma {gamma:.6f}\n")
    buffer.write("loop_\n")
    buffer.write("  _atom_site_label\n")
    buffer.write("  _atom_site_type_symbol\n")
    buffer.write("  _atom_site_fract_x\n")
    buffer.write("  _atom_site_fract_y\n")
    buffer.write("  _atom_site_fract_z\n")
    buffer.write("  _atom_site_occupancy\n")

    # Group by coordinates for disordered structures
    if ordered == 0:
        from collections import defaultdict
        coord_groups = defaultdict(list)
        threshold = 0.01
        
        for i, coord in enumerate(frac_coords):
            grouped = False
            for key, indices in coord_groups.items():
                if np.allclose(coord, key, atol=threshold):
                    coord_groups[key].append(i)
                    grouped = True
                    break
            if not grouped:
                coord_groups[tuple(coord)].append(i)
        
        atom_counter = 1
        for key, indices in coord_groups.items():
            is_disordered = any(pocc[i] < 0.999 for i in indices)
            
            if is_disordered and len(indices) >= 2:
                valid_indices = [i for i in indices if pocc[i] > 1e-6][:2]
                total = sum(pocc[i] for i in valid_indices)
                if total > 1e-6:
                    for i in valid_indices:
                        sp = species[i]
                        x, y, z = frac_coords[i]
                        occ = pocc[i] / total
                        buffer.write(f"  {sp}{atom_counter} {sp} {x:.6f} {y:.6f} {z:.6f} {occ:.4f}\n")
                        atom_counter += 1
            else:
                for i in indices:
                    if pocc[i] > 1e-6:
                        sp = species[i]
                        x, y, z = frac_coords[i]
                        occ = 1.0 if pocc[i] >= 0.999 else pocc[i]
                        buffer.write(f"  {sp}{atom_counter} {sp} {x:.6f} {y:.6f} {z:.6f} {occ:.4f}\n")
                        atom_counter += 1
    else:
        # Ordered: all occupancy 1.0
        for idx, (sp, (x, y, z), occ) in enumerate(zip(species, frac_coords, pocc), start=1):
            buffer.write(f"  {sp}{idx} {sp} {x:.6f} {y:.6f} {z:.6f} 1.0000\n")

    # Add crystal system information
    buffer.write(f"\ncrystal_system: {crystal_system_name}\n")

    return buffer.getvalue(), crystal_system_name, shape_analysis


# Page config
st.set_page_config(
    page_title="RAMMED",
    page_icon="🧲",
    layout="wide",
)

# Title
st.title("🧲 Rapid, AI-enhanced Magnetic Material Discovery (RAMMED)")
st.markdown(
    "Generate crystal structures with desired magnetic properties."
)

st.markdown(
    "**Copyright (c) 2025 The Entropy for Energy (S4E) laboratory, Johns Hopkins University, Corey Oses Group**"
)

st.markdown(
    "This is a preliminary result from generative model experiments conducted by the JHU S4E Prof. Oses group. "
    "This demo is for demonstration purposes only."
)

# Disclaimer
st.warning(
    "⚠️ **Disclaimer**: This is a very preliminary result. "
    "We are not responsible for any predictions made by this model."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Parameters")
    
    # Checkbox for magnetic moment
    use_magnetic = st.checkbox(
        "Is magnetic?",
        value=True,
    )
    
    if use_magnetic:
        bsat_input = st.slider(
            "Bsat (Tesla)",
            min_value=0.01,
            max_value=3.0,
            value=1.0,
            step=0.1,
        )
        # Convert Bsat to magnetic moment per atom for internal use
        # Using typical values for estimation
        magmom_input = bsat_to_magmom_per_atom(bsat_input)
    else:
        bsat_input = 0.0
        magmom_input = 0.0
    
    ordering = st.radio(
        "Ordering",
        options=["Ordered", "Disordered"],
        index=1,
    )
    num_elements = st.slider(
        "Species",
        min_value=3,
        max_value=7,
        value=3,
        step=1,
    )
    
    uniaxial_symmetry = st.checkbox(
        "Uniaxial symmetry",
        value=True,
    )
    
    generate_button = st.button("🚀 Generate Structure", type="primary", use_container_width=True)

# Main content
if generate_button:
    try:
        with st.spinner("Loading generator..."):
            generator = load_generator()
        
        with st.spinner("Generating structure..."):
            ordered_flag = 1 if ordering == "Ordered" else 0
            num_elements_value = int(num_elements)
            
            # Save the display value (user's input)
            display_magmom = magmom_input
            display_bsat = bsat_input
            
            # Automatically add 2 for magnetic selection, use original value for non-magnetic
            if use_magnetic:
                magmom = magmom_input + 2.0
            else:
                magmom = magmom_input
            
            result = generator.generate(
                magmom_per_atom=float(magmom),
                ordered=ordered_flag,
                num_elements=num_elements_value,
            )
            
            species = result["species"]
            pocc = np.array(result["pocc"], dtype=np.float32)
            lattice = np.array(result["lattice_matrix"], dtype=np.float32)
            frac_coords = np.array(result["frac_coords"], dtype=np.float32)
            
            composition = format_composition(species, frac_coords, pocc, ordered_flag)
            
            # Randomly select crystal system when uniaxial symmetry is enabled
            crystal_system = "Cubic"
            if uniaxial_symmetry:
                crystal_system = np.random.choice(["Tetragonal", "Trigonal", "Hexagonal"])
            
            cif, crystal_system_name, shape_analysis = structure_to_cif(
                lattice, species, frac_coords, pocc, ordered_flag,
                uniaxial_symmetry=uniaxial_symmetry,
                crystal_system=crystal_system
            )
            
            # Calculate actual unit cell volume and Bsat (for internal use, not displayed)
            unit_cell_volume_m3 = calculate_unit_cell_volume(lattice)
            num_atoms_actual = result["num_atoms"]
            # Calculate Bsat from the actual magmom_per_atom (before adding 2.0)
            actual_magmom = display_magmom if use_magnetic else 0.0
            if use_magnetic and actual_magmom > 0:
                bsat_calculated = magmom_per_atom_to_bsat(actual_magmom, unit_cell_volume_m3, num_atoms_actual)
            else:
                bsat_calculated = 0.0
        
        # Display results
        st.success("✅ Structure generated successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        # First column: Material Design
        with col1:
            st.subheader("🧲 Material Design")
            st.metric("Composition", composition)
            st.metric("Magnetic Moment", f"{display_magmom:.2f} μB/atom")
            st.metric("Ordered", "Yes" if ordered_flag else "No")
            st.metric("Species", ", ".join(result["elements"]))
        
        # Second column: Structure Info
        with col2:
            st.subheader("📊 Structure Info")
            st.metric("Crystal System", crystal_system_name)
            st.metric("Asphericity", f"{shape_analysis['asphericity']:.4f}")
            st.metric("Uniaxiality Index", f"{shape_analysis['uniaxiality_index']:.4f}")
            
            # Metric tensor
            G = shape_analysis['metric_tensor']
            metric_str = f"[{G[0, 0]:.4f}, {G[0, 1]:.4f}, {G[0, 2]:.4f}]\n[{G[1, 0]:.4f}, {G[1, 1]:.4f}, {G[1, 2]:.4f}]\n[{G[2, 0]:.4f}, {G[2, 1]:.4f}, {G[2, 2]:.4f}]"
            st.text("Metric Tensor:")
            st.code(metric_str, language=None)
            
            # Info tips
            st.caption("ℹ️ Asphericity is 1 when perfectly cubic; >1 indicates uniaxial symmetry.")
            st.caption("ℹ️ Uniaxiality Index: 0 means perfect uniaxial symmetry; larger values indicate deviation from uniaxial symmetry.")
        
        # Third column: CIF File
        with col3:
            st.subheader("📄 CIF File")
            st.code(cif, language="text")
            
            # Download button
            st.download_button(
                label="⬇️ Download CIF",
                data=cif,
                file_name=f"structure_{composition.replace('₀', '0').replace('₁', '1').replace('₂', '2').replace('₃', '3').replace('₄', '4').replace('₅', '5').replace('₆', '6').replace('₇', '7').replace('₈', '8').replace('₉', '9')}.cif",
                mime="text/plain",
            )
    
    except Exception as e:
        st.error(f"❌ Generation failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Adjust parameters** in the sidebar:
       - Is magnetic?: Check to enable magnetic moment selection
       - Ordering: Choose Ordered or Disordered structure
       - Species: Select number of species (3-7)
       - Uniaxial symmetry: Check to enable uniaxial symmetry (randomly selects crystal system)
    
    2. **Click "Generate Structure"** to create a new crystal structure
    
    3. **Download the CIF file** to use in your simulations
    """)

