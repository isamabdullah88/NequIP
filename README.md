# NequIP: From-Scratch Implementation & Molecular Dynamics Study

This repository contains a from-scratch implementation of the **NequIP** (Neural network Quantum Interatomic Potentials) architecture, developed as part of my MS Physics thesis research. The project focuses on verifying E(3)-equivariant neural networks for predicting interatomic forces and energies, specifically tested on the Aspirin MD17 dataset.

## 🚀 Project Overview

The goal of this project was to move beyond high-level abstractions and implement the core equivariant layers and message-passing mechanics from the ground up. By training on a **DigitalOcean GPU droplet**, I've established a full pipeline from data preprocessing to production-level Molecular Dynamics (MD) simulations.

### Key Features
* **From-Scratch Architecture**: Full implementation of equivariant layers, ensuring rotationally and translationally invariant energy predictions.
* **Equivariant Mechanics**: Heavy utilization of the **e3nn** library to handle irreducible representations and spherical harmonics.
* **Cloud-Based Training**: Optimized training workflows deployed on **DigitalOcean GPU Droplets**.
* **Production MD**: Implementation of NVT ensemble simulations using a Langevin thermostat.
* **Physical Post-Processing**: Automated scripts for computing Radial Distribution Functions (RDF), Vibrational Power Spectra (VDOS), and Torsional Angle analysis.

---

## 📊 Physical Verification Results

To validate the model, I performed several benchmarks comparing the learned potential against ground-truth physics:

1. **Structural Fidelity**: RDF analysis confirmed that the carbon skeleton (aromatic and single bonds) matches the expected equilibrium distances.
2. **Numerical Stability**: The production MD maintained high energy conservation, with a final energy drift of $1.94 \times 10^{-7}$ eV/fs.
3. **Dynamic Calibration**: Comparison of the MD-derived Power Spectrum against static **Normal Mode Analysis (Hessian)**. This helped identify specific force-constant nuances associated with the original MD17 dataset.



---

## 🛠️ Tech Stack
* **Deep Learning**: **PyTorch**, **e3nn** (Euclidean Neural Networks)
* **Molecular Tools**: ASE (Atomic Simulation Environment)
* **Infrastructure**: DigitalOcean (GPU Droplets)
* **Visualization**: Matplotlib, Seaborn

---

## 🔬 Thesis Context: Towards DSpinGNN

This repository serves as the foundational verification phase of my MS Thesis. The established equivariant framework is currently being extended into **DSpinGNN** (Spin-Disentangled Equivariant Graph Neural Network). 

The next stage of the project involves modeling non-collinear magnetic systems in **Janus Manganese Dichalcogenides** to study the stability and mobility of **Skyrmions**.



---
**Author**: Isam Balghari  
**Degree**: MS Physics  
**Expected Completion**: May 2026
