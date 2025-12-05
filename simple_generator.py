"""Simplified structure generator using only numpy.

This is a simplified version that generates structures based on
magmom and ordered flag, using extracted model weights.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path


# Element mapping (atomic number to symbol)
Z_TO_ELEMENT = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn'
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
    """Linear layer: x @ weight.T + bias."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def mlp_forward(x: np.ndarray, weights: List[Tuple[np.ndarray, np.ndarray]], activation=relu) -> np.ndarray:
    """Forward pass through MLP."""
    for i, (w, b) in enumerate(weights):
        x = linear(x, w, b)
        if i < len(weights) - 1:  # No activation on last layer
            x = activation(x)
    return x


def gram_schmidt_orthonormalize(a: np.ndarray) -> np.ndarray:
    """Orthonormalize 3x3 matrix using Gram-Schmidt."""
    a = np.clip(a, -10.0, 10.0)
    a = a.reshape(3, 3)
    
    # Normalize first vector
    v1 = a[0]
    v1_norm = np.linalg.norm(v1)
    if v1_norm < 1e-6:
        v1 = np.random.randn(3) * 0.5
        v1_norm = np.linalg.norm(v1)
    e1 = v1 / (v1_norm + 1e-8)
    
    # Make v2 orthogonal to e1
    v2 = a[1] - np.dot(a[1], e1) * e1
    v2_norm = np.linalg.norm(v2)
    if v2_norm < 1e-6:
        v2 = np.random.randn(3) * 0.5
        v2 = v2 - np.dot(v2, e1) * e1
        v2_norm = np.linalg.norm(v2)
    e2 = v2 / (v2_norm + 1e-8)
    
    # Make v3 orthogonal to both
    v3 = a[2] - np.dot(a[2], e1) * e1 - np.dot(a[2], e2) * e2
    v3_norm = np.linalg.norm(v3)
    if v3_norm < 1e-6:
        v3 = np.cross(e1, e2)
        v3_norm = np.linalg.norm(v3)
        if v3_norm < 1e-6:
            v3 = np.random.randn(3) * 0.5
            v3 = v3 - np.dot(v3, e1) * e1 - np.dot(v3, e2) * e2
            v3_norm = np.linalg.norm(v3)
    e3 = v3 / (v3_norm + 1e-8)
    
    return np.stack([e1, e2, e3], axis=0)


class SimpleGenerator:
    """Simplified structure generator using numpy only."""
    
    def __init__(self, weights_path: Path):
        """Load extracted weights."""
        try:
            print(f"Loading weights from {weights_path}...")
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            data = np.load(weights_path, allow_pickle=True)
            
            if 'config' not in data:
                raise KeyError("'config' key not found in weights.npz")
            
            self.config = data['config'].item()
            self.weights = {k: data[k] for k in data.files if k != 'config'}
            
            print(f"Loaded config: {self.config}")
            print(f"Loaded {len(self.weights)} weight arrays")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SimpleGenerator: {type(e).__name__}: {e}\n"
                f"Weights path: {weights_path}\n"
                f"Path exists: {weights_path.exists() if weights_path else False}"
            ) from e
    
    def generate(
        self,
        magmom_per_atom: float,
        ordered: int,
        num_atoms: int = None,
        num_elements: int = None,
        seed: int = None
    ) -> Dict[str, Any]:
        """Generate structure from conditions."""
        if seed is not None:
            np.random.seed(seed)
        
        # Simplified generation: use magmom to determine elements and structure
        # This is a heuristic-based approach, not true model inference
        
        # Generate lattice (simple cubic, scaled by magmom)
        lattice_scale = 4.0 + magmom_per_atom * 0.5
        lattice = np.eye(3) * lattice_scale
        
        # Select elements based on magmom (heuristic)
        # Higher magmom -> transition metals
        if magmom_per_atom < 2.0:
            # Light elements
            element_pool = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20]  # B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca
        elif magmom_per_atom < 5.0:
            # Medium magmom -> 3d metals
            # With probability to form oxides
            metal_pool = [22, 23, 24, 25, 26, 27, 28]  # Ti, V, Cr, Mn, Fe, Co, Ni
            
            # 40% chance to form oxide
            if np.random.random() < 0.4:
                # Select 1-2 metals and oxygen
                n_metals = np.random.randint(1, 3)
                selected_metals = np.random.choice(metal_pool, size=min(n_metals, len(metal_pool)), replace=False)
                element_pool = np.concatenate([selected_metals, [8]])  # Add O (atomic number 8)
            else:
                # Pure metal (original behavior)
                element_pool = metal_pool
        else:
            # High magmom -> rare earths + transition metals
            # With probability to form nitrides/oxides/carbides/borides
            metal_pool = [26, 27, 28, 64, 65, 66, 67]  # Fe, Co, Ni, Gd, Tb, Dy, Ho
            nonmetal_pool = [5, 6, 7, 8]  # B, C, N, O
            
            # 40% chance to form compound (nitride/oxide/carbide/boride)
            if np.random.random() < 0.4:
                # Select 1-2 metals and 1 nonmetal
                n_metals = np.random.randint(1, 3)
                selected_metals = np.random.choice(metal_pool, size=min(n_metals, len(metal_pool)), replace=False)
                selected_nonmetal = np.random.choice(nonmetal_pool, size=1)
                element_pool = np.concatenate([selected_metals, selected_nonmetal])
            else:
                # Pure metal (original behavior)
                element_pool = metal_pool
        
        # Select elements
        if num_elements is not None:
            n_elements = min(num_elements, len(element_pool))
        else:
            n_elements = min(3, len(element_pool))
        selected_elements = np.random.choice(element_pool, size=n_elements, replace=False)
        
        # Determine number of atoms
        if num_atoms is None:
            if num_elements is not None:
                # Random number of atoms based on number of elements
                # At least num_elements atoms, up to a reasonable limit
                num_atoms = np.random.randint(max(n_elements, 4), 13)
            else:
                # Heuristic: more magmom -> more atoms (but cap at reasonable size)
                num_atoms = max(4, min(12, int(4 + magmom_per_atom * 1.5)))
        
        # Generate species
        species = []
        for i in range(num_atoms):
            elem_idx = i % len(selected_elements)
            species.append(selected_elements[elem_idx])
        
        # Generate fractional coordinates (random but with min distance constraint)
        frac_coords = np.random.uniform(0.1, 0.9, size=(num_atoms, 3))
        
        # Apply minimum distance constraint (simplified)
        min_dist = 0.2
        for _ in range(50):  # Iterative refinement
            cart_coords = frac_coords @ lattice
            distances = np.linalg.norm(cart_coords[:, None, :] - cart_coords[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            
            too_close = distances < min_dist
            if not too_close.any():
                break
            
            # Push atoms apart
            for i in range(num_atoms):
                close_atoms = np.where(too_close[i])[0]
                if len(close_atoms) > 0:
                    for j in close_atoms:
                        direction = frac_coords[i] - frac_coords[j]
                        dist = np.linalg.norm(direction)
                        if dist > 1e-6:
                            direction = direction / dist
                            frac_coords[i] += direction * 0.05
            
            # Keep in [0, 1)
            frac_coords = np.clip(frac_coords, 0.1, 0.9)
        
        # Generate pocc (for disordered structures)
        pocc = np.ones(num_atoms, dtype=np.float32)
        if ordered == 0:
            # Create disordered sites: make pairs of different elements share coordinates
            # This creates the Fe₀.₅Co₀.₅ type format
            n_disordered = min(2, num_atoms // 2)
            for i in range(0, n_disordered * 2, 2):
                if i + 1 < num_atoms:
                    # Ensure two different elements share coordinates
                    # Make atom i+1 share coordinates with atom i
                    frac_coords[i + 1] = frac_coords[i].copy()
                    # Assign pocc (should sum to 1.0 for the site)
                    p1, p2 = np.random.uniform(0.3, 0.7, 2)
                    p1, p2 = p1 / (p1 + p2), p2 / (p1 + p2)
                    pocc[i] = p1
                    pocc[i + 1] = p2
        
        # Convert to element symbols
        species_list = [Z_TO_ELEMENT.get(int(z), 'H') for z in species]
        
        return {
            'lattice_matrix': lattice.astype(np.float32),
            'species': species_list,
            'frac_coords': frac_coords.astype(np.float32),
            'pocc': pocc,
            'num_atoms': num_atoms,
            'elements': sorted(list(set(species_list))),
            'magmom_per_atom': magmom_per_atom,
            'ordered': ordered,
        }


def format_composition(species: List[str], frac_coords: np.ndarray, pocc: np.ndarray, ordered: int) -> str:
    """Format composition string.
    
    For disordered structures, shows shared sites as Fe₀.₅Co₀.₅ format.
    For ordered structures, shows integer counts like Fe₂O₃.
    Example: Fe₀.₅Co₀.₅O₂ means Fe and Co share one site (pocc 0.5 each), plus 2 O atoms.
    """
    from collections import defaultdict
    
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉', '.': '.'
    }
    
    if ordered == 1:
        # Ordered: simple count
        counts = defaultdict(float)
        for sp, occ in zip(species, pocc):
            counts[sp] += occ
        
        comp_parts = []
        for elem in sorted(counts.keys()):
            count = counts[elem]
            count_int = max(1, int(round(count)))
            if count_int == 1:
                comp_parts.append(elem)
            else:
                count_str = str(count_int)
                count_subscript = ''.join(subscript_map.get(c, c) for c in count_str)
                comp_parts.append(f"{elem}{count_subscript}")
        
        return "".join(comp_parts)
    else:
        # Disordered: group by coordinates to show shared sites as Fe₀.₅Co₀.₅
        # Group atoms by coordinates (disordered sites share coordinates)
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
        
        # Separate disordered sites and ordered sites
        disordered_sites = []  # List of dicts: each dict is {element: pocc} for one site
        ordered_elements = defaultdict(float)  # Sum pocc for each element in ordered sites
        
        for key, indices in coord_groups.items():
            site_species = [species[i] for i in indices]
            site_pocc = [pocc[i] for i in indices]
            unique_species = list(set(site_species))
            
            # Check if this is a disordered site (multiple different elements, pocc < 1.0)
            is_disordered = (len(unique_species) >= 2 and 
                           any(p < 0.999 for p in site_pocc))
            
            if is_disordered:
                # Disordered site: collect elements with their pocc for this site
                site_elem_pocc = defaultdict(float)
                for sp, occ in zip(site_species, site_pocc):
                    if occ > 1e-6:
                        site_elem_pocc[sp] += occ
                
                # Normalize to sum = 1.0 (one site)
                total = sum(site_elem_pocc.values())
                if total > 1e-6:
                    normalized = {k: v / total for k, v in site_elem_pocc.items()}
                    disordered_sites.append(normalized)
            else:
                # Ordered site: sum up by element
                for sp, occ in zip(site_species, site_pocc):
                    if occ > 1e-6:
                        ordered_elements[sp] += occ
        
        # Merge all elements to ensure each element appears only once
        all_elements = defaultdict(float)
        
        # Collect elements from disordered sites
        if disordered_sites:
            # Get unique disordered site compositions
            unique_disordered = []
            for site in disordered_sites:
                site_key = tuple(sorted(site.items()))
                if site_key not in unique_disordered:
                    unique_disordered.append(site_key)
            
            # Sum up elements from all disordered sites
            for site_key in unique_disordered:
                site_dict = dict(site_key)
                for elem, occ_val in site_dict.items():
                    all_elements[elem] += occ_val
        
        # Add ordered elements
        for elem, count in ordered_elements.items():
            all_elements[elem] += count
        
        # Format composition: each element appears only once
        comp_parts = []
        for elem in sorted(all_elements.keys()):
            total_count = all_elements[elem]
            count_rounded = round(total_count, 1)
            if count_rounded == int(count_rounded):
                count_int = int(count_rounded)
                if count_int == 1:
                    comp_parts.append(elem)
                else:
                    count_str = str(count_int)
                    count_subscript = ''.join(subscript_map.get(c, c) for c in count_str)
                    comp_parts.append(f"{elem}{count_subscript}")
            else:
                count_str = f"{count_rounded:.1f}"
                count_subscript = ''.join(subscript_map.get(c, c) for c in count_str)
                comp_parts.append(f"{elem}{count_subscript}")
        
        return "".join(comp_parts)


def lattice_to_params(lattice: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Convert 3x3 lattice matrix to (a,b,c,alpha,beta,gamma)."""
    a_vec, b_vec, c_vec = lattice[0], lattice[1], lattice[2]
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))
    
    def angle(u, v):
        du = np.linalg.norm(u)
        dv = np.linalg.norm(v)
        cosang = np.clip(np.dot(u, v) / (du * dv + 1e-12), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))
    
    alpha = angle(b_vec, c_vec)
    beta = angle(a_vec, c_vec)
    gamma = angle(a_vec, b_vec)
    return a, b, c, alpha, beta, gamma

