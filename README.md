# Rapid, AI-enhanced Magnetic Material Discovery (RAMMED)

A Streamlit web application for generating crystal structures with desired magnetic properties.

## 🧲 Overview

**Preliminary results**: Generate CIF files from given magnetic moments. This model was trained on 900+ magnetic structures and 2000+ non-magnetic structures.

**No PyTorch required** - uses pure numpy for fast deployment!

## 🚀 Quick Start

### Deploy to Streamlit Cloud

1. **Fork or clone this repository**
   ```bash
   git clone git@github.com:hguangshuai/mag_gen_demo.git
   cd mag_gen_demo
   ```

2. **Deploy to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub account
   - Select this repository
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"
   - Wait 1-2 minutes for deployment

3. **Share the link** - Your app will be live at a URL like:
   ```
   https://mag-gen-demo.streamlit.app
   ```

## 📁 Files

- `streamlit_app.py` - Main Streamlit application
- `simple_generator.py` - Structure generator (pure numpy)
- `weights.npz` - Model weights (extracted from trained VAE)
- `requirements.txt` - Python dependencies

## ⚙️ Usage

1. **Adjust parameters** in the sidebar:
   - **Magnetic moment**: Desired magnetic moment per atom (μB)
   - **Ordering**: Choose Ordered or Disordered structure
   - **Number of atoms**: Set to 0 for automatic (minimum 2 atoms), or specify a number (2-12)

2. **Click "Generate Structure"** to create a new crystal structure

3. **Download the CIF file** for use in your simulations

## 📊 Model Information

- **Training data**: 900+ magnetic structures + 2000+ non-magnetic structures
- **Output**: CIF files with crystal structures
- **Input**: Magnetic moment (μB per atom) and ordering preference
- **Implementation**: Simplified numpy-based generator (no PyTorch dependency)

## 🔄 Updates

This repository is automatically synced with Streamlit Cloud. Simply push changes to GitHub and the app will automatically redeploy.

```bash
git add .
git commit -m "Your update message"
git push
```

## ⚠️ Note

This is a **preliminary version** using a simplified numpy-based generator. For higher accuracy, consider using the full PyTorch model version.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

Trained on magnetic and non-magnetic crystal structure datasets.
