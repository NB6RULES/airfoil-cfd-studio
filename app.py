"""
================================================================================
Interactive 2D Airfoil Aerodynamics & CFD Flow Field Visualizer
Built by Senior Aerodynamicist & Streamlit Engineer
================================================================================
Features:
- NACA 4-digit airfoil parametric geometry generator (cosine spacing)
- 2D Boundary Element / Hess-Smith Panel Method for potential flow
- Viscous wake deficit & boundary layer modeling
- Normalized velocity field contours and streamlines
- Surface pressure coefficient distribution (-Cp) with suction inverted
- Viscous aerodynamic polars (Cl vs alpha, Cl vs Cd, L/D vs alpha) with stall modeling
- Pure NumPy & SciPy self-contained implementation (no external C/Fortran solvers required)
"""

import streamlit as st
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import io

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="2D Airfoil Aerodynamics & CFD Visualizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom aerodynamic dashboard CSS styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.6);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 0.82rem !important;
    }

    /* Tab bar aesthetic */
    button[data-baseweb="tab"] {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
    /* Code/Formula styling */
    .stCodeBlock {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. AERODYNAMIC GEOMETRY GENERATOR (NACA 4-DIGIT)
# ------------------------------------------------------------------------------
@st.cache_data
def generate_naca4_airfoil(m_pct: float, p_pct: float, t_pct: float, chord: float = 1.0, num_points: int = 80):
    """
    Generate coordinates for a standard NACA 4-digit airfoil using cosine spacing.
    Returns:
        x_body, y_body: closed boundary polygon ordered clockwise (TE -> Upper -> LE -> Lower -> TE)
        xu, yu: upper surface coordinates (LE to TE)
        xl, yl: lower surface coordinates (LE to TE)
        xc_line, yc_line: mean camber line coordinates
        designation: canonical NACA string (e.g., 'NACA 2412')
    """
    m = m_pct / 100.0   # Max camber fraction
    p = p_pct / 100.0   # Camber position fraction
    t = t_pct / 100.0   # Max thickness fraction
    c = chord
    
    # Cosine spacing along chord to concentrate points at LE and TE
    beta = np.linspace(0.0, np.pi, num_points)
    x = c * 0.5 * (1.0 - np.cos(beta))
    
    # Thickness distribution (closed trailing edge coefficient -0.1036)
    yt = 5.0 * t * c * (
        0.2969 * np.sqrt(np.maximum(x / c, 1e-10))
        - 0.1260 * (x / c)
        - 0.3516 * (x / c)**2
        + 0.2843 * (x / c)**3
        - 0.1036 * (x / c)**4
    )
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if m > 0.0 and p > 0.0:
        idx_fwd = x < p * c
        idx_aft = ~idx_fwd
        
        # Forward of maximum camber
        yc[idx_fwd] = (m / (p**2)) * (2.0 * p * (x[idx_fwd] / c) - (x[idx_fwd] / c)**2) * c
        dyc_dx[idx_fwd] = (2.0 * m / (p**2)) * (p - (x[idx_fwd] / c))
        
        # Aft of maximum camber
        yc[idx_aft] = (m / ((1.0 - p)**2)) * ((1.0 - 2.0 * p) + 2.0 * p * (x[idx_aft] / c) - (x[idx_aft] / c)**2) * c
        dyc_dx[idx_aft] = (2.0 * m / ((1.0 - p)**2)) * (p - (x[idx_aft] / c))
    
    theta = np.arctan(dyc_dx)
    
    # Upper and Lower surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Form closed polygon (Clockwise: TE upper -> LE -> TE lower)
    # xu, yu are ordered LE -> TE. Reverse upper so it starts at TE and ends at LE.
    xb = np.concatenate([xu[::-1], xl[1:]])
    yb = np.concatenate([yu[::-1], yl[1:]])
    
    # Format canonical NACA 4-digit name
    m_int = int(round(m_pct))
    p_int = int(round(p_pct / 10.0))
    t_int = int(round(t_pct))
    if m_int == 0:
        designation = f"NACA 00{t_int:02d}"
    else:
        designation = f"NACA {m_int}{p_int}{t_int:02d}"
        
    return xb, yb, xu, yu, xl, yl, x, yc, designation


# ------------------------------------------------------------------------------
# 3. 2D HESS-SMITH PANEL METHOD (POTENTIAL FLOW SOLVER)
# ------------------------------------------------------------------------------
@st.cache_data
def solve_panel_method(xb: np.ndarray, yb: np.ndarray, alpha_deg: float, U_inf: float):
    """
    Exact 2D Hess-Smith Panel Method with Kutta Condition.
    Panels consist of constant-strength source distributions + constant uniform vortex sheet.
    Boundary nodes are ordered clockwise (TE -> Upper -> LE -> Lower -> TE).
    """
    alpha = np.radians(alpha_deg)
    num_panels = len(xb) - 1
    
    # Panel vectors & geometry
    dx = xb[1:] - xb[:-1]
    dy = yb[1:] - yb[:-1]
    L = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    
    # Control points at panel midpoints
    xc = 0.5 * (xb[:-1] + xb[1:])
    yc = 0.5 * (yb[:-1] + yb[1:])
    
    # Tangent and Normal vectors (Clockwise: normal points outward)
    tx = np.cos(phi)
    ty = np.sin(phi)
    nx = np.sin(phi)
    ny = -np.cos(phi)
    
    # Freestream velocity components
    u_inf = U_inf * np.cos(alpha)
    v_inf = U_inf * np.sin(alpha)
    
    V_inf_n = u_inf * nx + v_inf * ny
    V_inf_t = u_inf * tx + v_inf * ty
    
    # Vectorized influence matrix assembly
    Xj = xb[:-1][np.newaxis, :]   # Shape: (1, N)
    Yj = yb[:-1][np.newaxis, :]
    Lj = L[np.newaxis, :]
    phij = phi[np.newaxis, :]
    
    Xi = xc[:, np.newaxis]         # Shape: (N, 1)
    Yi = yc[:, np.newaxis]
    
    cos_p = np.cos(phij)
    sin_p = np.sin(phij)
    
    dx_loc = Xi - Xj
    dy_loc = Yi - Yj
    
    xi = dx_loc * cos_p + dy_loc * sin_p
    eta = -dx_loc * sin_p + dy_loc * cos_p
    
    r1_sq = np.maximum(xi**2 + eta**2, 1e-12)
    r2_sq = np.maximum((xi - Lj)**2 + eta**2, 1e-12)
    
    theta1 = np.arctan2(eta, xi)
    theta2 = np.arctan2(eta, xi - Lj)
    dtheta = (theta2 - theta1 + np.pi) % (2.0 * np.pi) - np.pi
    
    log_ratio = 0.5 * np.log(r1_sq / r2_sq)
    
    # Local induced velocities
    u_s_loc = (1.0 / (2.0 * np.pi)) * log_ratio
    v_s_loc = (1.0 / (2.0 * np.pi)) * dtheta
    
    u_v_loc = (1.0 / (2.0 * np.pi)) * dtheta
    v_v_loc = -(1.0 / (2.0 * np.pi)) * log_ratio
    
    # Global transformation
    u_s = u_s_loc * cos_p - v_s_loc * sin_p
    v_s = u_s_loc * sin_p + v_s_loc * cos_p
    
    u_v = u_v_loc * cos_p - v_v_loc * sin_p
    v_v = u_v_loc * sin_p + v_v_loc * cos_p
    
    # Self-induction handling on diagonal
    np.fill_diagonal(u_s, 0.0)
    np.fill_diagonal(v_s, 0.0)
    np.fill_diagonal(u_v, 0.0)
    np.fill_diagonal(v_v, 0.0)
    
    n_xi = nx[:, np.newaxis]
    n_yi = ny[:, np.newaxis]
    t_xi = tx[:, np.newaxis]
    t_yi = ty[:, np.newaxis]
    
    # A: Normal source influence, C: Tangential source influence
    A = u_s * n_xi + v_s * n_yi
    C = u_s * t_xi + v_s * t_yi
    
    # B: Normal vortex influence, D: Tangential vortex influence
    B = u_v * n_xi + v_v * n_yi
    D = u_v * t_xi + v_v * t_yi
    
    # Analytical self-induction boundary limits
    for i in range(num_panels):
        A[i, i] = 0.5
        B[i, i] = 0.0
        C[i, i] = 0.0
        D[i, i] = 0.5
        
    # Assemble linear system: (N+1) x (N+1)
    M = np.zeros((num_panels + 1, num_panels + 1))
    RHS = np.zeros(num_panels + 1)
    
    # Equation 1..N: Flow tangency at control points: A @ q + gamma * sum(B) = -V_inf_n
    M[:num_panels, :num_panels] = A
    M[:num_panels, num_panels] = np.sum(B, axis=1)
    RHS[:num_panels] = -V_inf_n
    
    # Equation N+1: Kutta condition at trailing edge (V_t,0 + V_t,N-1 = 0)
    M[num_panels, :num_panels] = C[0, :] + C[-1, :]
    M[num_panels, num_panels] = np.sum(D[0, :]) + np.sum(D[-1, :])
    RHS[num_panels] = -(V_inf_t[0] + V_inf_t[-1])
    
    # Solve linear system
    solution = np.linalg.solve(M, RHS)
    q = solution[:num_panels]
    gamma = solution[num_panels]
    
    # Surface tangential velocity and Cp
    Vt = V_inf_t + C @ q + gamma * np.sum(D, axis=1)
    Cp = 1.0 - (Vt / U_inf)**2
    
    # Inviscid lift coefficient via Kutta-Joukowski theorem
    Gamma_total = gamma * np.sum(L)
    Cl_inviscid = (2.0 * Gamma_total) / (U_inf * 1.0)
    
    return {
        'q': q,
        'gamma': gamma,
        'xc': xc,
        'yc': yc,
        'Vt': Vt,
        'Cp': Cp,
        'Cl_inviscid': Cl_inviscid,
        'L': L,
        'phi': phi,
        'xb': xb,
        'yb': yb
    }


# ------------------------------------------------------------------------------
# 4. 2D CARTESIAN FLOW FIELD & VISCOUS WAKE EVALUATOR
# ------------------------------------------------------------------------------
@st.cache_data
def evaluate_flow_field(xb_rot: np.ndarray, yb_rot: np.ndarray, q: np.ndarray, gamma: float,
                        L: np.ndarray, phi: np.ndarray, U_inf: float, alpha_deg: float,
                        Cd_val: float, nx: int = 120, ny: int = 80, enable_wake: bool = True):
    """
    Evaluate the total 2D velocity vector field (u, v) and normalized speed on a Cartesian grid.
    Flow enters horizontally (freestream along +x) with airfoil rotated by angle of attack.
    Includes viscous wake momentum deficit downstream of trailing edge.
    """
    x_lin = np.linspace(-0.5, 1.8, nx)
    y_lin = np.linspace(-0.8, 0.8, ny)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    
    X_flat = Xg.ravel()
    Y_flat = Yg.ravel()
    M = len(X_flat)
    num_panels = len(q)
    
    # Inside-body mask using polygon path
    poly_verts = np.column_stack([xb_rot, yb_rot])
    body_path = Path(poly_verts)
    inside_mask = body_path.contains_points(np.column_stack([X_flat, Y_flat]))
    
    u_ind = np.zeros(M)
    v_ind = np.zeros(M)
    
    # Vectorized accumulation across panels
    for j in range(num_panels):
        xj = xb_rot[j]
        yj = yb_rot[j]
        lj = L[j]
        phij = phi[j]
        cos_p = np.cos(phij)
        sin_p = np.sin(phij)
        
        dx = X_flat - xj
        dy = Y_flat - yj
        
        xi = dx * cos_p + dy * sin_p
        eta = -dx * sin_p + dy * cos_p
        
        r1_sq = np.maximum(xi**2 + eta**2, 1e-7)
        r2_sq = np.maximum((xi - lj)**2 + eta**2, 1e-7)
        
        theta1 = np.arctan2(eta, xi)
        theta2 = np.arctan2(eta, xi - lj)
        dtheta = (theta2 - theta1 + np.pi) % (2.0 * np.pi) - np.pi
        
        log_ratio = 0.5 * np.log(r1_sq / r2_sq)
        
        u_s_loc = (1.0 / (2.0 * np.pi)) * log_ratio
        v_s_loc = (1.0 / (2.0 * np.pi)) * dtheta
        
        u_v_loc = (1.0 / (2.0 * np.pi)) * dtheta
        v_v_loc = -(1.0 / (2.0 * np.pi)) * log_ratio
        
        q_j = q[j]
        u_loc = q_j * u_s_loc + gamma * u_v_loc
        v_loc = q_j * v_s_loc + gamma * v_v_loc
        
        u_ind += u_loc * cos_p - v_loc * sin_p
        v_ind += u_loc * sin_p + v_loc * cos_p
        
    u_total = U_inf + u_ind
    v_total = 0.0 + v_ind
    
    # Calculate base velocity magnitude
    speed = np.sqrt(u_total**2 + v_total**2)
    speed_wake = speed.copy()
    
    # Downstream Viscous Wake Deficit Modeling
    x_te = xb_rot[0]
    y_te = yb_rot[0]
    downwash_angle = np.radians(alpha_deg * 0.7)
    
    if enable_wake:
        wake_mask = (X_flat > x_te)
        dx_w = X_flat[wake_mask] - x_te
        y_wake_center = y_te - dx_w * np.tan(downwash_angle)
        dy_w = Y_flat[wake_mask] - y_wake_center
        
        # Turbulent wake spreading: b(x) ~ x^0.5
        wake_width = 0.035 + 0.12 * np.sqrt(dx_w)
        # Velocity defect scaling with Cd
        defect_strength = np.clip(0.40 * (Cd_val / 0.012), 0.15, 0.85)
        deficit = defect_strength * np.exp(-0.5 * (dy_w / wake_width)**2)
        speed_wake[wake_mask] *= (1.0 - deficit)
    
    speed_ratio = speed_wake / U_inf
    
    speed_ratio_2d = speed_ratio.reshape(ny, nx)
    u_2d = u_total.reshape(ny, nx)
    v_2d = v_total.reshape(ny, nx)
    inside_mask_2d = inside_mask.reshape(ny, nx)
    
    return Xg, Yg, speed_ratio_2d, u_2d, v_2d, inside_mask_2d, x_te, y_te, downwash_angle


# ------------------------------------------------------------------------------
# 5. VISCOUS AERODYNAMIC PERFORMANCE MODEL (Cl, Cd, L/D, STALL)
# ------------------------------------------------------------------------------
@st.cache_data
def calculate_aero_performance(alpha_deg: float, m_pct: float, p_pct: float, t_pct: float,
                               U_inf: float = 25.0, rho: float = 1.225, nu: float = 1.5e-5):
    """
    Combines Thin Airfoil Theory, Schlichting Flat-Plate Turbulent Boundary Layer Drag,
    empirical thickness/camber corrections, and a smooth Kirchhoff-Helmholtz stall model.
    """
    m = m_pct / 100.0
    p = p_pct / 100.0
    t = t_pct / 100.0
    c = 1.0
    
    # 1. Reynolds Number
    Re = max(U_inf * c / nu, 1e4)
    
    # 2. Skin Friction & Profile Drag (Schlichting formula + Hoerner form factor)
    Cf = 0.455 / (np.log10(Re)**2.58)
    kf = 1.0 + 2.0 * t + 60.0 * (t**4)
    Cd0 = 2.0 * Cf * kf
    
    # 3. Lift Characteristics
    alpha_0_deg = -1.15 * m * (180.0 / np.pi)
    cla_rad = 2.0 * np.pi * (1.0 + 0.77 * t) * 0.95
    cla_deg = cla_rad * (np.pi / 180.0)
    
    # Stall angles
    alpha_stall_pos = 12.5 + 25.0 * t + 10.0 * m
    alpha_stall_neg = -(12.5 + 25.0 * t - 10.0 * m)
    
    alpha = np.asarray(alpha_deg, dtype=float)
    a_rad = np.radians(alpha)
    
    # Attached linear lift
    Cl_lin = cla_deg * (alpha - alpha_0_deg)
    
    # Separation blending functions (sigmoid stall onset)
    x_pos = (alpha - alpha_stall_pos) / 1.8
    sig_pos = 1.0 / (1.0 + np.exp(-np.clip(x_pos, -25, 25)))
    
    x_neg = -(alpha - alpha_stall_neg) / 1.8
    sig_neg = 1.0 / (1.0 + np.exp(-np.clip(x_neg, -25, 25)))
    
    sigma = np.maximum(sig_pos, sig_neg)
    
    # Separated cross-flow lift
    Cl_sep = 1.8 * np.sin(a_rad) * np.cos(a_rad)
    Cl = (1.0 - sigma) * Cl_lin + sigma * Cl_sep
    
    # 4. Drag Characteristics
    Cl_ideal = cla_deg * (-alpha_0_deg)
    kp = 0.006 + 0.012 * t
    Cd_prof = Cd0 + kp * (Cl - Cl_ideal)**2
    Cd_sep = sigma * (1.8 * (np.sin(a_rad)**2) + 0.035)
    
    Cd = np.maximum(Cd_prof + Cd_sep, 0.0035)
    L_D = np.where(Cd > 0, Cl / Cd, 0.0)
    
    # Dimensional Forces per meter of span (N/m)
    q_dyn = 0.5 * rho * (U_inf**2)
    lift_force = q_dyn * c * Cl
    drag_force = q_dyn * c * Cd
    
    return {
        'Cl': float(Cl) if np.ndim(Cl) == 0 else Cl,
        'Cd': float(Cd) if np.ndim(Cd) == 0 else Cd,
        'L_D': float(L_D) if np.ndim(L_D) == 0 else L_D,
        'Re': Re,
        'Cd0': Cd0,
        'alpha_0_deg': alpha_0_deg,
        'alpha_stall_pos': alpha_stall_pos,
        'alpha_stall_neg': alpha_stall_neg,
        'lift_force': float(lift_force) if np.ndim(lift_force) == 0 else lift_force,
        'drag_force': float(drag_force) if np.ndim(drag_force) == 0 else drag_force,
        'cla_deg': cla_deg
    }


# ------------------------------------------------------------------------------
# 6. SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.markdown("## ✈️ Airfoil Geometry (NACA 4-Digit)")
st.sidebar.caption("Define the 4-digit airfoil parametric parameters:")

m_pct = st.sidebar.slider(
    "Max Camber (m)",
    min_value=0.0,
    max_value=9.0,
    value=2.0,
    step=0.5,
    format="%.1f%% chord",
    help="Maximum height of the camber line as a percentage of the chord length."
)

p_pct = st.sidebar.slider(
    "Camber Position (p)",
    min_value=10.0,
    max_value=90.0,
    value=40.0,
    step=10.0,
    format="%.0f%% chord",
    help="Location of maximum camber along the chord line from the leading edge."
)

t_pct = st.sidebar.slider(
    "Max Thickness (t)",
    min_value=5.0,
    max_value=30.0,
    value=12.0,
    step=1.0,
    format="%.0f%% chord",
    help="Maximum thickness of the airfoil profile as a percentage of chord length."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 💨 Operating Flow Conditions")

alpha_deg = st.sidebar.slider(
    "Angle of Attack (AoA, α)",
    min_value=-10.0,
    max_value=20.0,
    value=4.0,
    step=0.5,
    format="%.1f°",
    help="Geometric angle between the chord line and the oncoming freestream flow."
)

U_inf = st.sidebar.slider(
    "Freestream Velocity ($U_\\infty$)",
    min_value=5.0,
    max_value=60.0,
    value=25.0,
    step=1.0,
    format="%.0f m/s",
    help="Magnitude of the oncoming freestream air velocity."
)

# Fluid constants
rho_air = 1.225     # kg/m^3
nu_air = 1.5e-5     # m^2/s
chord_len = 1.0     # m fixed

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ CFD Visualization Settings", expanded=False):
    cmap_choice = st.selectbox(
        "Velocity Colormap",
        options=["plasma", "viridis", "turbo", "inferno", "coolwarm", "cividis"],
        index=0,
        help="Colormap used to render the normalized flow velocity magnitude contours."
    )
    streamline_density = st.slider(
        "Streamline Density",
        min_value=0.8,
        max_value=2.2,
        value=1.3,
        step=0.1,
        help="Density of flow trajectories in streamplot."
    )
    show_wake_deficit = st.checkbox(
        "Include Boundary Layer Wake Deficit",
        value=True,
        help="Simulate the turbulent viscous momentum deficit trailing behind the trailing edge."
    )
    grid_res_choice = st.radio(
        "CFD Grid Resolution",
        options=["Fast (110 × 70)", "High Quality (150 × 95)"],
        index=0,
        help="Control the resolution of the 2D Cartesian flow field."
    )
    nx_grid, ny_grid = (110, 70) if "Fast" in grid_res_choice else (150, 95)

st.sidebar.caption("Fluid: Air ($\\\\rho = 1.225\\\\ \\mathrm{kg/m^3},\\\\ \\nu = 1.5\\\\times10^{-5}\\\\ \\mathrm{m^2/s}$), Chord = 1.0 m.")


# ------------------------------------------------------------------------------
# 7. COMPUTATION EXECUTION
# ------------------------------------------------------------------------------
# 1. Airfoil geometry (unrotated)
xb, yb, xu, yu, xl, yl, xc_line, yc_line, naca_designation = generate_naca4_airfoil(
    m_pct, p_pct, t_pct, chord=chord_len, num_points=70
)

# 2. Viscous aerodynamic performance metrics
aero = calculate_aero_performance(
    alpha_deg, m_pct, p_pct, t_pct, U_inf=U_inf, rho=rho_air, nu=nu_air
)

# 3. Rotate airfoil by AoA around quarter-chord (0.25, 0)
# Clockwise rotation pitches leading edge UP (+y) relative to oncoming horizontal flow
alpha_rad = np.radians(alpha_deg)
cos_a = np.cos(alpha_rad)
sin_a = np.sin(alpha_rad)

xb_rot = 0.25 + (xb - 0.25) * cos_a + yb * sin_a
yb_rot = -(xb - 0.25) * sin_a + yb * cos_a

# 4. Solve panel method flow on rotated body with horizontal freestream (U_inf, 0)
panel_res = solve_panel_method(xb_rot, yb_rot, alpha_deg=0.0, U_inf=U_inf)

# 5. Evaluate 2D Cartesian field
Xg, Yg, speed_ratio_2d, u_2d, v_2d, inside_mask_2d, x_te, y_te, downwash_ang = evaluate_flow_field(
    xb_rot, yb_rot, panel_res['q'], panel_res['gamma'], panel_res['L'], panel_res['phi'],
    U_inf=U_inf, alpha_deg=alpha_deg, Cd_val=aero['Cd'],
    nx=nx_grid, ny=ny_grid, enable_wake=show_wake_deficit
)

# 6. Identify Stagnation Point
xc_rot = panel_res['xc']
yc_rot = panel_res['yc']
stag_idx = np.argmax(panel_res['Cp'])
x_stag = xc_rot[stag_idx]
y_stag = yc_rot[stag_idx]
cp_stag = panel_res['Cp'][stag_idx]


# ------------------------------------------------------------------------------
# 8. MAIN DASHBOARD: HEADER & METRIC CARDS
# ------------------------------------------------------------------------------
st.title(f"🚀 {naca_designation} Aerodynamics & 2D Flow Field Visualizer")

camber_str = "Symmetric (0% camber)" if m_pct == 0 else f"{m_pct:.1f}% chord at {p_pct:.0f}% chord"
st.markdown(
    f"**Profile:** {naca_designation} &nbsp;|&nbsp; "
    f"**Camber:** {camber_str} &nbsp;|&nbsp; "
    f"**Thickness:** {t_pct:.1f}% &nbsp;|&nbsp; "
    f"**Operating Point:** $\\alpha = {alpha_deg:+.1f}^\\circ$, $U_\\infty = {U_inf:.0f}\\ \\mathrm{{m/s}}$"
)

# Top Row Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Lift Coefficient (Cl)",
        value=f"{aero['Cl']:+.3f}",
        delta=f"Lift: {aero['lift_force']:.1f} N/m",
        help="Dimensionless lift coefficient per unit span. Positive indicates upward aerodynamic force."
    )

with col2:
    st.metric(
        label="Drag Coefficient (Cd)",
        value=f"{aero['Cd']:.4f}",
        delta=f"Drag: {aero['drag_force']:.2f} N/m",
        delta_color="inverse",
        help="Dimensionless drag coefficient accounting for skin friction, form drag, and pressure separation."
    )

with col3:
    efficiency_label = "High Efficiency" if aero['L_D'] > 40 else ("Moderate" if aero['L_D'] > 15 else "High Drag / Stalled")
    st.metric(
        label="Lift-to-Drag Ratio (L/D)",
        value=f"{aero['L_D']:.1f}",
        delta=efficiency_label,
        help="Aerodynamic efficiency metric. Measures the glide ratio and lifting effectiveness."
    )

with col4:
    re_exp = int(np.floor(np.log10(aero['Re'])))
    re_base = aero['Re'] / (10**re_exp)
    st.metric(
        label="Reynolds Number (Re)",
        value=f"{re_base:.2f} × 10⁶",
        delta="Subsonic Flow",
        help=f"Exact Re = {aero['Re']:,.0f}. Characterizes the ratio of inertial to viscous forces."
    )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 9. TABS: CFD FLOW FIELD, PRESSURE (-Cp), AERODYNAMIC POLARS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 CFD Flow Field Visualization",
    "📊 Surface Pressure Distribution (-Cp)",
    "📈 Aerodynamic Polars & Stall",
    "📚 Physics & Mathematical Formulation"
])


# ==============================================================================
# TAB 1: CFD FLOW FIELD VISUALIZATION
# ==============================================================================
with tab1:
    st.markdown("### 2D Velocity Magnitude Contours & Streamline Flow Field")
    st.caption(
        "Interactive 2D potential flow solution with Kutta circulation, solid body masking, "
        "and downstream viscous boundary layer wake defect."
    )
    
    plt.style.use('dark_background')
    fig1, ax1 = plt.subplots(figsize=(11.5, 6.2), dpi=130, facecolor='#090d16')
    ax1.set_facecolor('#090d16')
    
    # 1. Background Contours: Normalized Velocity Magnitude
    levels = np.linspace(0.0, 1.85, 45)
    cf = ax1.contourf(Xg, Yg, speed_ratio_2d, levels=levels, cmap=cmap_choice, extend='both')
    
    # 2. Horizontal Colorbar
    cbar = fig1.colorbar(cf, ax=ax1, orientation='horizontal', pad=0.14, fraction=0.045, aspect=36)
    cbar.set_label(r'Normalized Flow Velocity Magnitude  $|\vec{V}| / U_\infty$', fontsize=11, color='#e2e8f0', fontweight='bold', labelpad=6)
    cbar.ax.tick_params(colors='#cbd5e1', labelsize=9)
    
    # 3. Streamlines
    ax1.streamplot(
        Xg, Yg, u_2d, v_2d,
        color=(1.0, 1.0, 1.0, 0.42),
        density=streamline_density,
        linewidth=0.75,
        arrowsize=0.85
    )
    
    # 4. Solid Airfoil Body & Outline
    ax1.fill(xb_rot, yb_rot, color='#05070c', zorder=10)
    ax1.plot(xb_rot, yb_rot, color='#38bdf8', lw=2.2, zorder=11, label=f'{naca_designation} Body')
    
    # 5. Stagnation Point
    ax1.plot(x_stag, y_stag, 'o', color='#ef4444', markersize=8, markeredgecolor='white', markeredgewidth=1.8, zorder=15, label='Stagnation Point')
    ax1.annotate(
        f'Stagnation Point\n(Cp = {cp_stag:.2f})',
        xy=(x_stag, y_stag),
        xytext=(x_stag - 0.28, y_stag - 0.24),
        arrowprops=dict(facecolor='#ef4444', edgecolor='white', arrowstyle='->', lw=1.4),
        color='#ffffff', fontsize=8.8, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#1e293b', edgecolor='#ef4444', lw=1.2, alpha=0.92),
        zorder=16
    )
    
    # 6. Boundary Layer Wake Region
    if show_wake_deficit:
        x_wake = np.linspace(x_te, 1.75, 50)
        y_wake = y_te - (x_wake - x_te) * np.tan(downwash_ang)
        ax1.plot(x_wake, y_wake, '--', color='#c084fc', lw=1.3, alpha=0.75, zorder=12, label='Wake Centerline')
        ax1.annotate(
            'Viscous Wake Deficit Region',
            xy=(x_wake[22], y_wake[22]),
            xytext=(x_wake[22] - 0.12, y_wake[22] - 0.22),
            arrowprops=dict(facecolor='#c084fc', edgecolor='none', arrowstyle='->', lw=1.2),
            color='#e9d5ff', fontsize=8.5, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1b4b', edgecolor='#a855f7', alpha=0.88),
            zorder=16
        )
        
    # Axis bounds & styling
    ax1.set_xlim(-0.45, 1.75)
    ax1.set_ylim(-0.75, 0.75)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x / c  (Normalized Spatial Position)', fontsize=10.5, color='#cbd5e1', labelpad=4)
    ax1.set_ylabel('y / c  (Normalized Spatial Position)', fontsize=10.5, color='#cbd5e1', labelpad=4)
    ax1.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#334155')
        
    ax1.legend(loc='upper right', framealpha=0.88, facecolor='#0f172a', edgecolor='#334155', fontsize=8.8, labelcolor='#e2e8f0')
    ax1.set_title(
        f"2D Flow Field Velocity Contours & Streamlines ({naca_designation}, α = {alpha_deg:+.1f}°, U∞ = {U_inf:.0f} m/s)",
        fontsize=12.5, fontweight='bold', color='#f8fafc', pad=12
    )
    
    fig1.subplots_adjust(bottom=0.20, top=0.92, left=0.08, right=0.96)
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)
    
    # Insight notes
    st.info(
        f"💡 **Flow Field Insights:** At $\\alpha = {alpha_deg:+.1f}^\\circ$, the suction peak on the upper surface "
        f"accelerates flow up to **{np.nanmax(speed_ratio_2d):.2f} × $U_\\infty$** ({np.nanmax(speed_ratio_2d)*U_inf:.1f} m/s). "
        f"Flow stagnation occurs at the leading edge at $(x/c = {x_stag:.3f}, y/c = {y_stag:.3f})$ where $C_p = {cp_stag:.2f}$."
    )


# ==============================================================================
# TAB 2: SURFACE PRESSURE DISTRIBUTION (-Cp)
# ==============================================================================
with tab2:
    st.markdown("### Chordwise Surface Pressure Coefficient Distribution ($C_p$)")
    st.caption("Standard aeronautical convention: the vertical axis is inverted so that suction (-Cp) points upward.")
    
    # Solve panel flow on unrotated body for chordwise unprojected x/c distribution
    unrot_res = solve_panel_method(xb, yb, alpha_deg=alpha_deg, U_inf=U_inf)
    xc_unrot = unrot_res['xc']
    Cp_unrot = unrot_res['Cp']
    num_panels = len(xb) - 1
    half = num_panels // 2
    
    # Upper surface: panels 0..half-1 (TE -> LE)
    x_u = xc_unrot[:half]
    cp_u = Cp_unrot[:half]
    sort_u = np.argsort(x_u)
    x_u_s = x_u[sort_u]
    cp_u_s = cp_u[sort_u]
    
    # Lower surface: panels half..end (LE -> TE)
    x_l = xc_unrot[half:]
    cp_l = Cp_unrot[half:]
    sort_l = np.argsort(x_l)
    x_l_s = x_l[sort_l]
    cp_l_s = cp_l[sort_l]
    
    # Interpolate lower Cp onto upper x-grid to calculate net normal force (area)
    cp_l_interp = np.interp(x_u_s, x_l_s, cp_l_s)
    Cn_approx = integrate.trapezoid(cp_l_interp - cp_u_s, x_u_s)
    
    # Center of Pressure (x_cp)
    x_cp_approx = integrate.trapezoid(x_u_s * (cp_l_interp - cp_u_s), x_u_s) / max(Cn_approx, 1e-4)
    # Pitching moment about quarter-chord
    Cm_c4_approx = integrate.trapezoid((0.25 - x_u_s) * (cp_l_interp - cp_u_s), x_u_s)
    
    fig2, ax2 = plt.subplots(figsize=(10.5, 5.2), dpi=120, facecolor='#090d16')
    ax2.set_facecolor('#090d16')
    
    # Upper & Lower Curves
    ax2.plot(x_u_s, cp_u_s, color='#38bdf8', lw=2.4, label='Upper Surface (Suction)')
    ax2.plot(x_l_s, cp_l_s, color='#f43f5e', lw=2.4, label='Lower Surface (Pressure)')
    
    # Shaded Lift Area
    ax2.fill_between(
        x_u_s, cp_u_s, cp_l_interp,
        where=(cp_l_interp >= cp_u_s),
        color='#38bdf8', alpha=0.18,
        label=r'Net Normal Force Area $\oint (C_{p,l} - C_{p,u})\,d(x/c)$'
    )
    
    # Invert Y-axis (Standard Aero Convention: suction upwards)
    ax2.invert_yaxis()
    
    # Key Markers
    min_cp_idx = np.argmin(cp_u_s)
    ax2.plot(x_u_s[min_cp_idx], cp_u_s[min_cp_idx], 'o', color='#38bdf8', markersize=7)
    ax2.annotate(
        f'Suction Peak: Cp = {cp_u_s[min_cp_idx]:.2f}',
        xy=(x_u_s[min_cp_idx], cp_u_s[min_cp_idx]),
        xytext=(x_u_s[min_cp_idx] + 0.08, cp_u_s[min_cp_idx] - 0.25),
        arrowprops=dict(facecolor='#38bdf8', edgecolor='white', arrowstyle='->', lw=1.2),
        color='#ffffff', fontsize=8.8, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.9)
    )
    
    ax2.axhline(0.0, color='#64748b', linestyle=':', lw=1.0, alpha=0.7)
    ax2.axhline(1.0, color='#ef4444', linestyle=':', lw=1.0, alpha=0.7, label='Stagnation Limit (Cp = 1.0)')
    
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_xlabel('Chordwise Position (x / c)', fontsize=10.5, color='#cbd5e1')
    ax2.set_ylabel(r'Pressure Coefficient ($C_p$) [Inverted: Suction $\uparrow$]', fontsize=10.5, color='#cbd5e1')
    ax2.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    ax2.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
    ax2.legend(loc='lower right', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=9, labelcolor='#e2e8f0')
    ax2.set_title(f"Surface Pressure Distribution -Cp for {naca_designation} (α = {alpha_deg:+.1f}°)", fontsize=12, fontweight='bold', color='#f8fafc')
    
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)
    
    # Pressure metrics summary
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Minimum Pressure (Peak Suction)", f"{cp_u_s[min_cp_idx]:.2f}", f"at x/c = {x_u_s[min_cp_idx]:.2f}")
    with m_col2:
        st.metric("Stagnation Pressure", f"{np.max(Cp_unrot):.2f}", "Theoretical Max: 1.00")
    with m_col3:
        st.metric("Center of Pressure (x_cp / c)", f"{x_cp_approx:.3f}", f"Δ to c/4: {x_cp_approx-0.25:+.3f}")
    with m_col4:
        st.metric("Quarter-Chord Moment (Cm,c/4)", f"{Cm_c4_approx:+.4f}", "Pitching moment")


# ==============================================================================
# TAB 3: AERODYNAMIC POLARS & STALL
# ==============================================================================
with tab3:
    st.markdown("### Aerodynamic Performance Polars & Non-Linear Stall Modeling")
    st.caption("Viscous polar analysis spanning angles of attack from -10° to +20°, highlighting the current operating condition.")
    
    alpha_range = np.linspace(-10.0, 20.0, 61)
    aero_polars = calculate_aero_performance(
        alpha_range, m_pct, p_pct, t_pct, U_inf=U_inf, rho=rho_air, nu=nu_air
    )
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=120, facecolor='#090d16')
    for ax in (ax3a, ax3b):
        ax.set_facecolor('#090d16')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
        
    # --- Subplot 1: Cl vs Alpha ---
    # Linear inviscid thin airfoil slope
    linear_slope_line = aero['cla_deg'] * (alpha_range - aero['alpha_0_deg'])
    ax3a.plot(alpha_range, linear_slope_line, '--', color='#64748b', lw=1.3, alpha=0.75, label=r'Inviscid Theory ($2\pi\alpha$)')
    
    # Real viscous lift curve with stall
    ax3a.plot(alpha_range, aero_polars['Cl'], color='#38bdf8', lw=2.6, label='Viscous Lift Curve (Stall Model)')
    
    # Stall boundary indicator
    ax3a.axvline(aero['alpha_stall_pos'], color='#f59e0b', linestyle=':', lw=1.6, label=f'Stall Onset (α = {aero["alpha_stall_pos"]:.1f}°)')
    ax3a.axhline(0.0, color='#475569', lw=0.8, alpha=0.6)
    
    # Operating point marker
    ax3a.plot(alpha_deg, aero['Cl'], 'o', color='#ef4444', markersize=9, markeredgecolor='white', markeredgewidth=1.8,
              zorder=10, label=f'Operating Point (α = {alpha_deg:.1f}°, Cl = {aero["Cl"]:.3f})')
    
    ax3a.set_xlim(-10.5, 20.5)
    ax3a.set_xlabel('Angle of Attack α (°)', fontsize=10.5, color='#cbd5e1')
    ax3a.set_ylabel('Lift Coefficient (Cl)', fontsize=10.5, color='#cbd5e1')
    ax3a.set_title(f'Lift Curve (Cl vs α) for {naca_designation}', fontsize=11.5, fontweight='bold', color='#f8fafc')
    ax3a.legend(loc='upper left', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=8.6, labelcolor='#e2e8f0')
    
    # --- Subplot 2: Drag Polar (Cl vs Cd) ---
    ax3b.plot(aero_polars['Cd'], aero_polars['Cl'], color='#10b981', lw=2.6, label='Drag Polar (Cl vs Cd)')
    
    # Operating point marker on drag polar
    ax3b.plot(aero['Cd'], aero['Cl'], 'o', color='#ef4444', markersize=9, markeredgecolor='white', markeredgewidth=1.8,
              zorder=10, label=f'Operating Point (Cd = {aero["Cd"]:.4f})')
    
    # Max L/D tangent line point
    max_ld_idx = np.argmax(aero_polars['L_D'])
    ax3b.plot(aero_polars['Cd'][max_ld_idx], aero_polars['Cl'][max_ld_idx], 's', color='#f59e0b', markersize=7,
              label=f'Max L/D = {aero_polars["L_D"][max_ld_idx]:.1f} (Cl={aero_polars["Cl"][max_ld_idx]:.2f})')
    
    ax3b.axhline(0.0, color='#475569', lw=0.8, alpha=0.6)
    ax3b.set_xlabel('Drag Coefficient (Cd)', fontsize=10.5, color='#cbd5e1')
    ax3b.set_ylabel('Lift Coefficient (Cl)', fontsize=10.5, color='#cbd5e1')
    ax3b.set_title(f'Drag Polar (Cl vs Cd) for {naca_designation}', fontsize=11.5, fontweight='bold', color='#f8fafc')
    ax3b.legend(loc='lower right', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=8.6, labelcolor='#e2e8f0')
    
    plt.tight_layout()
    st.pyplot(fig3, clear_figure=True)
    plt.close(fig3)
    
    # Polar characteristics table
    st.markdown("#### Airfoil Aerodynamic Characteristics Summary")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.metric("Zero-Lift Angle (α_L=0)", f"{aero['alpha_0_deg']:.2f}°", "Theoretical Camber Intercept")
    with p_col2:
        st.metric("Lift Curve Slope (dCl/dα)", f"{aero['cla_deg']:.4f} /°", f"{aero['cla_deg']*180/np.pi:.2f} /rad")
    with p_col3:
        st.metric("Maximum Lift (Cl,max)", f"{np.max(aero_polars['Cl']):.2f}", f"at α = {alpha_range[np.argmax(aero_polars['Cl'])]:.1f}°")
    with p_col4:
        st.metric("Minimum Drag (Cd,min)", f"{np.min(aero_polars['Cd']):.4f}", f"Cd0 = {aero['Cd0']:.4f}")
    with p_col5:
        st.metric("Peak Efficiency (L/D)max", f"{np.max(aero_polars['L_D']):.1f}", f"at α = {alpha_range[max_ld_idx]:.1f}°")


# ==============================================================================
# TAB 4: PHYSICS & MATHEMATICAL FORMULATION
# ==============================================================================
with tab4:
    st.markdown("### Underlying Aerodynamic Theory & Numerical Methods")
    
    st.markdown("""
    #### 1. NACA 4-Digit Analytical Geometry
    The profile is generated using the classical NACA 4-digit analytical equations with cosine spacing across chord $c$:
    
    $$\\beta \\in [0, \\pi], \\quad x = \\frac{c}{2}(1 - \\cos\\beta)$$
    
    Thickness distribution $y_t(x)$ with closed trailing edge:
    $$y_t(x) = 5 t c \\left[ 0.2969\\sqrt{\\frac{x}{c}} - 0.1260\\left(\\frac{x}{c}\\right) - 0.3516\\left(\\frac{x}{c}\\right)^2 + 0.2843\\left(\\frac{x}{c}\\right)^3 - 0.1036\\left(\\frac{x}{c}\\right)^4 \\right]$$
    
    Mean camber line $y_c(x)$ and slope $\\theta(x) = \\arctan(dy_c/dx)$:
    $$\\text{For } 0 \\le x < pc: \\quad y_c = \\frac{m}{p^2}\\left(2p\\frac{x}{c} - \\left(\\frac{x}{c}\\right)^2\\right) c, \\quad \\frac{dy_c}{dx} = \\frac{2m}{p^2}\\left(p - \\frac{x}{c}\\right)$$
    $$\\text{For } pc \\le x \\le c: \\quad y_c = \\frac{m}{(1-p)^2}\\left((1-2p) + 2p\\frac{x}{c} - \\left(\\frac{x}{c}\\right)^2\\right) c, \\quad \\frac{dy_c}{dx} = \\frac{2m}{(1-p)^2}\\left(p - \\frac{x}{c}\\right)$$
    
    #### 2. Hess-Smith 2D Potential Panel Solver
    The airfoil boundary is discretized into $N$ flat panels. Each panel $j$ carries a constant source strength $q_j$ and a global uniform vortex strength $\\gamma$:
    
    - **Flow Tangency Condition** at panel midpoints:
      $$\\vec{V}_i \\cdot \\vec{n}_i = 0 \\implies \\sum_{j=1}^N A_{ij} q_j + \\gamma \\sum_{j=1}^N B_{ij} = -\\vec{V}_\\infty \\cdot \\vec{n}_i$$
      
    - **Kutta Condition** ensuring smooth tangential separation at the sharp trailing edge:
      $$V_{t, 1} + V_{t, N} = 0$$
      
    - **Surface Pressure Distribution**:
      $$C_p = 1 - \\left(\\frac{V_t}{U_\\infty}\\right)^2$$
      
    #### 3. Viscous Drag & Separation Modeling
    - **Skin Friction (Schlichting Formula)**:
      $$C_f = \\frac{0.455}{(\\log_{10} Re)^{2.58}}, \\quad k_f = 1 + 2\\left(\\frac{t}{c}\\right) + 60\\left(\\frac{t}{c}\\right)^4, \\quad C_{d0} = 2 C_f k_f$$
      
    - **Profile & Lift-Dependent Drag**:
      $$C_{d, \\text{profile}} = C_{d0} + k_p (C_l - C_{l, \\text{ideal}})^2$$
      
    - **Non-Linear Stall (Kirchhoff-Helmholtz Sigmoid Model)**:
      Separation factor $\\sigma(\\alpha) = [1 + \\exp(-(\\alpha - \\alpha_{\\text{stall}})/\\Delta)]^{-1}$:
      $$C_l(\\alpha) = (1 - \\sigma(\\alpha)) C_{l, \\text{linear}} + \\sigma(\\alpha) [1.8 \\sin\\alpha \\cos\\alpha]$$
      $$C_d(\\alpha) = C_{d, \\text{profile}} + \\sigma(\\alpha) [1.8 \\sin^2\\alpha + 0.035]$$
    """)

# ------------------------------------------------------------------------------
# 10. EXPORT / DATA DOWNLOAD EXPANDER
# ------------------------------------------------------------------------------
with st.expander("💾 Export Coordinates & Aerodynamic Data", expanded=False):
    col_dl1, col_dl2 = st.columns(2)
    
    # Airfoil Coordinates CSV
    csv_coords = io.StringIO()
    csv_coords.write("x,y_upper,y_lower,y_camber\n")
    for i in range(len(xu)):
        csv_coords.write(f"{xu[i]:.6f},{yu[i]:.6f},{yl[i]:.6f},{yc_line[i]:.6f}\n")
    
    with col_dl1:
        st.download_button(
            label=f"📥 Download {naca_designation} Coordinates (CSV)",
            data=csv_coords.getvalue(),
            file_name=f"{naca_designation}_coordinates.csv",
            mime="text/csv"
        )
        
    # Pressure Distribution CSV
    csv_cp = io.StringIO()
    csv_cp.write("x_upper,Cp_upper,x_lower,Cp_lower\n")
    for i in range(len(x_u_s)):
        csv_cp.write(f"{x_u_s[i]:.6f},{cp_u_s[i]:.6f},{x_l_s[i]:.6f},{cp_l_s[i]:.6f}\n")
        
    with col_dl2:
        st.download_button(
            label=f"📥 Download Surface Cp Data at α={alpha_deg:.1f}° (CSV)",
            data=csv_cp.getvalue(),
            file_name=f"{naca_designation}_Cp_alpha_{alpha_deg:.1f}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Crafted with aerodynamic rigor & Streamlit • Pure NumPy & SciPy • Streamlit Cloud Ready 🚀")
