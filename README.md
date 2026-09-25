# ✈️ Airfoil CFD Studio (2D Aerodynamics & Flow Visualizer)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Aerodynamics-8CAAE6.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, production-ready 2D Airfoil Aerodynamics & CFD Flow Field Visualizer built with Streamlit, NumPy, SciPy, and Matplotlib. 

The application is completely self-contained with no external CFD solver binaries (no OpenFOAM or XFOIL installation required), allowing instant deployment to **Streamlit Cloud**, **Hugging Face Spaces**, or local machines.

---

## 🌟 Key Features

- **NACA 4-Digit Airfoil Geometry Engine:**
  - Full parametric control over maximum camber ($m$), camber position ($p$), and maximum thickness ($t$).
  - Cosine clustering along chord for high leading-edge and trailing-edge resolution.
  - Closed trailing-edge formulation.

- **Real-Time 2D Flow Field CFD Simulation:**
  - High-speed 2D **Hess-Smith Boundary Element Panel Method** with exact Kutta condition enforcement.
  - Vectorized Cartesian grid velocity evaluation with instantaneous (~50ms) rendering.
  - Normalized velocity magnitude contours ($|\vec{V}| / U_\infty$) with customizable colormaps (`plasma`, `viridis`, `turbo`, `coolwarm`, `inferno`).
  - Streamlines with automatic downwash deflection and physical stagnation point marking.
  - Downstream viscous boundary layer wake momentum deficit modeling.

- **Surface Pressure Distribution ($-C_p$):**
  - Upper and lower surface pressure coefficient line plots.
  - Standard aeronautical convention with inverted Y-axis (suction upwards).
  - Shaded aerodynamic normal force area ($\oint \Delta C_p \, d(x/c)$).
  - Live computation of Center of Pressure ($x_{cp}$) and Quarter-Chord Pitching Moment ($C_{m, c/4}$).

- **Aerodynamic Polars & Non-Linear Stall Modeling:**
  - Viscous lift curve ($C_l$ vs $\alpha$) from $-10^\circ$ to $+20^\circ$.
  - Schlichting turbulent flat-plate boundary layer skin friction + Hoerner form factor drag.
  - Kirchhoff-Helmholtz non-linear sigmoid stall onset model.
  - Interactive operating point marker on lift curve and drag polar ($C_l$ vs $C_d$).

- **Data Export:**
  - One-click CSV export of airfoil geometry coordinates ($x, y_{\text{upper}}, y_{\text{lower}}, y_{\text{camber}}$).
  - One-click CSV export of surface pressure distribution ($x, C_{p,\text{upper}}, C_{p,\text{lower}}$).

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/NB6RULES/airfoil-cfd-studio.git
cd airfoil-cfd-studio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```

---

## 🧮 Mathematical & Aerodynamic Formulation

### 1. Hess-Smith 2D Panel Method
The airfoil boundary is discretized into $N$ planar panels, each carrying a constant-strength source distribution $q_j$ and a global uniform vortex circulation $\gamma$:

- **Flow Tangency Condition** at panel midpoints:
  $$\vec{V}_i \cdot \vec{n}_i = 0 \implies \sum_{j=1}^N A_{ij} q_j + \gamma \sum_{j=1}^N B_{ij} = -\vec{V}_\infty \cdot \vec{n}_i$$

- **Kutta Condition** ensuring finite, tangential flow departure at the sharp trailing edge:
  $$V_{t, 1} + V_{t, N} = 0$$

- **Surface Pressure Distribution**:
  $$C_p = 1 - \left(\frac{V_t}{U_\infty}\right)^2$$

### 2. Viscous Drag & Stall Modeling
- **Turbulent Skin Friction (Schlichting Formula):**
  $$C_f = \frac{0.455}{(\log_{10} Re)^{2.58}}$$

- **Airfoil Form Factor (Hoerner):**
  $$k_f = 1 + 2\left(\frac{t}{c}\right) + 60\left(\frac{t}{c}\right)^4, \quad C_{d0} = 2 C_f k_f$$

- **Stall Transition:**
  $$\sigma(\alpha) = \frac{1}{1 + \exp(-(\alpha - \alpha_{\text{stall}})/\Delta)}$$
  $$C_l(\alpha) = (1 - \sigma(\alpha)) C_{l, \text{linear}} + \sigma(\alpha) [1.8 \sin\alpha \cos\alpha]$$

---

## 📄 License
This project is licensed under the MIT License.
