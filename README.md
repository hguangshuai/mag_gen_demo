# Rapid, AI-enhanced Magnetic Material Discovery (RAMMED)

## ⚠️ Disclaimer

**This is a very preliminary result. We are not responsible for any predictions made by this model.**

---

A Streamlit web application for generating crystal structures with desired magnetic properties.

## Overview

This is a VAE (Variational Autoencoder) model for generating crystal structures with desired magnetic properties. This is a preliminary result from generative model experiments conducted by the JHU S4E Prof. Oses group. This is a test demo.

Preliminary results: Generate CIF files from given magnetic moments. This model was trained on 900+ magnetic structures and 2000+ non-magnetic structures.

## Files

- `streamlit_app.py` - Main Streamlit application
- `simple_generator.py` - Structure generator (pure numpy)
- `weights.npz` - Model weights (extracted from trained VAE)
- `requirements.txt` - Python dependencies

## Model Information

- **Training data**: 900+ magnetic structures + 2000+ non-magnetic structures
- **Output**: CIF files with crystal structures
- **Input**: Magnetic moment (μB per atom) and ordering preference
- **Implementation**: Simplified numpy-based generator (no PyTorch dependency)

## Ownership

This code is the exclusive property of **The Entropy for Energy (S4E) laboratory, Johns Hopkins University, Corey Oses Group**.

**Any use of this code is not permitted.**

## License

MIT License

Copyright (c) 2024 The Entropy for Energy (S4E) laboratory, Johns Hopkins University, Corey Oses Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
