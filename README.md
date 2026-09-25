# 🏎️ Airfoil CFD Studio (2D Aerodynamics, Motorsport & CAD)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Aerodynamics-8CAAE6.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, high-speed 2D Airfoil Aerodynamics & CFD Flow Field Visualizer built with Streamlit, NumPy, SciPy, and Matplotlib. Engineered for high rendering responsiveness on **Streamlit Community Cloud**, with dedicated features for aeronautics, motorsport aerodynamics, and CAD / 3D printing.

The app is completely self-contained with no external CFD solver binaries (no OpenFOAM or XFOIL installation required).

---

## 🌟 Key Features

### ⚡ High Rendering Speed & Cloud Performance
- **Performance / Quality Modes:**
  - **⚡ Fast (Interactive):** 70×50 Cartesian grid with streamline density 0.6 and optimized 85 DPI rendering for instantaneous, lag-free slider adjustments.
  - **💎 High Def:** 140×100 grid with streamline density 1.1 and 115 DPI for publication-quality flow field plots.
- **Lazy-Loaded & Cached Polars:** Aerodynamic polars and Cartesian flow evaluations are cached with `@st.cache_data`.
- **Zero Memory Leaks:** Systematic `plt.close('all')` execution after every render to keep browser and server memory light.

### 🛩️ High-Lift Plain Flap Engine
- **Real-Time Camber Deflection:** Articulates the trailing-edge plain flap downward ($-15^\circ$ to $+35^\circ$) around a customizable hinge point ($x_f/c \in [0.60, 0.85]$).
- **Thin Airfoil Flap Effectiveness:** Real-time computation of flap zero-lift shift $\Delta \alpha_{0} = -\frac{\theta_f - \sin\theta_f}{\pi} \delta_f$, boosting lift coefficient ($C_l$) and profile drag ($C_d$).

### 🏎️ Motorsport Downforce & Ground Effect
- **Inverted Wing Mode:** Inverts the airfoil geometry ($y \to -y$), shifting the suction surface underneath the wing to simulate automotive rear/front wings (Formula 1, GT3, LMP, FSAE).
- **Metric Cards for Racing:** Displays **Downforce Coefficient ($C_{df}$)**, Downforce in $\text{N/m}$, Drag in $\text{N/m}$, and Aerodynamic Downforce-to-Drag Efficiency.
- **Ground Effect Venturi Suction:**
  - Realistic road surface boundary plane ($y = y_{\text{road}}$) with asphalt styling and yellow track marking.
  - Squeezes flow through the under-wing channel, accelerating flow via Venturi suction and calculating ground effect downforce amplification and ground stall limits.

### 📐 CAD & 3D Printing Export Suite
- **Native 2D DXF (AutoCAD Release 12):** Clean ASCII DXF with a closed 2D polyline on layer `AIRFOIL_PROFILE`. Directly compatible with **SolidWorks**, **Fusion 360**, **AutoCAD**, **Rhino**, **FreeCAD**, **Bambu Studio / OrcaSlicer**, and laser cutters.
- **SolidWorks & Fusion 360 XYZ CSV:** Closed-loop 3D spline CSV format (`X, Y, Z` with $Z = 0$) for SolidWorks "Curve Through XYZ Points" and Fusion 360 "Import Spline CSV".
- **Scale Selector:** Export normalized ($c = 1.0\text{ m}$), Full Scale ($1000\text{ mm}$), Wind Tunnel Model ($200\text{ mm}$), or 3D Printing Test ($150\text{ mm}$).
- **Surface Pressure Distribution CSV:** Download surface coordinates and local pressure coefficients ($C_p$).

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

- **Stall Transition (Kirchhoff-Helmholtz Sigmoid Model):**
  $$\sigma(\alpha) = \frac{1}{1 + \exp(-(\alpha - \alpha_{\text{stall}})/\Delta)}$$
  $$C_l(\alpha) = (1 - \sigma(\alpha)) C_{l, \text{linear}} + \sigma(\alpha) [1.8 \sin\alpha \cos\alpha]$$

---

## 📄 License
This project is licensed under the MIT License.
