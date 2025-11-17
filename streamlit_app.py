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


def structure_to_cif(
    lattice: np.ndarray,
    species: list[str],
    frac_coords: np.ndarray,
    pocc: np.ndarray,
    ordered: int,
    name: str = "generated",
) -> str:
    """Write CIF file."""
    a, b, c, alpha, beta, gamma = lattice_to_params(lattice)
    buffer = StringIO()
    buffer.write(f"data_{name}\n")
    buffer.write("_symmetry_space_group_name_H-M    'P 1'\n")
    buffer.write("_symmetry_Int_Tables_number       1\n")
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

    return buffer.getvalue()


# Page config
st.set_page_config(
    page_title="Magnetic VAE Generator",
    page_icon="🧲",
    layout="wide",
)

# Title
st.title("🧲 Magnetic VAE Generator")
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
    
    # Yes/No selection for magnetic moment
    use_magnetic = st.radio(
        "Is magnetic?",
        options=["Yes", "No"],
        index=0,
    )
    
    if use_magnetic == "Yes":
        magmom_input = st.slider(
            "Magnetic moment (μB per atom)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )
    
    ordering = st.radio(
        "Ordering",
        options=["Ordered", "Disordered"],
        index=1,
    )
    num_atoms = st.slider(
        "Number of atoms (0 = auto)",
        min_value=0,
        max_value=12,
        value=0,
        step=1,
    )
    generate_button = st.button("🚀 Generate Structure", type="primary", use_container_width=True)

# Main content
if generate_button:
    try:
        with st.spinner("Loading generator..."):
            generator = load_generator()
        
        with st.spinner("Generating structure..."):
            ordered_flag = 1 if ordering == "Ordered" else 0
            num_atoms_value = int(num_atoms) if num_atoms > 0 else None
            
            # If "No" was selected, generate a new random value for this generation
            if use_magnetic == "No":
                magmom_input = np.random.uniform(0, 2)
            
            # Automatically add 2 for "Yes" selection, use original value for "No"
            if use_magnetic == "Yes":
                magmom = magmom_input + 2.0
            else:
                magmom = magmom_input
            
            result = generator.generate(
                magmom_per_atom=float(magmom),
                ordered=ordered_flag,
                num_atoms=num_atoms_value,
            )
            
            species = result["species"]
            pocc = np.array(result["pocc"], dtype=np.float32)
            lattice = np.array(result["lattice_matrix"], dtype=np.float32)
            frac_coords = np.array(result["frac_coords"], dtype=np.float32)
            
            composition = format_composition(species, frac_coords, pocc, ordered_flag)
            cif = structure_to_cif(lattice, species, frac_coords, pocc, ordered_flag)
        
        # Display results
        st.success("✅ Structure generated successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Structure Info")
            st.metric("Composition", composition)
            st.metric("Magnetic Moment", f"{result['magmom_per_atom']:.2f} μB/atom")
            st.metric("Ordered", "Yes" if ordered_flag else "No")
            st.metric("Number of Atoms", result["num_atoms"])
            st.metric("Elements", ", ".join(result["elements"]))
        
        with col2:
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
    1. **Specify magnetic moment?**
       - **Yes**: Choose your desired magnetic moment (will automatically add 2.0)
       - **No**: A random value between 0-2 will be generated
    
    2. **Adjust other parameters** in the sidebar:
       - Ordering: Choose Ordered or Disordered structure
       - Number of atoms: Set to 0 for automatic, or specify a number
    
    3. **Click "Generate Structure"** to create a new crystal structure
    
    4. **Download the CIF file** to use in your simulations
    
    **Note**: This is a simplified version using numpy. For higher accuracy, 
    use the full PyTorch model version.
    """)

