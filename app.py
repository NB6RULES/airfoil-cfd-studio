"""
================================================================================
🏎️ 2D Airfoil Aerodynamics, Motorsport & CFD Studio
Built by Senior Aerodynamicist & Streamlit Engineer
================================================================================
Features:
- Parametric NACA 4-Digit Airfoil Geometry Engine (Cosine Spacing)
- Trailing-Edge Plain Flap Deflection with Real-Time Camber Transformation
- Motorsport Inverted Wing Mode (Downforce & Aerodynamic Efficiency)
- Ground Effect Aerodynamics & Venturi Suction Simulation
- Ultra-Fast 2D Hess-Smith Panel Method & Vectorized CFD Grid Solver
- Performance/Quality Toggle for High Responsiveness on Streamlit Cloud
- Normalized Velocity Contours, Streamlines & Stagnation Point Tracking
- Surface Pressure Distribution (-Cp) with Inverted Aeronautical Convention
- Non-Linear Viscous Aerodynamic Polars & Stall Modeling
- CAD & 3D Printing Export: Native 2D DXF (AutoCAD R12) & SolidWorks/Fusion 360 CSV
"""

import streamlit as st
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import io

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & AERODYNAMIC DASHBOARD STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Airfoil CFD Studio & Aerodynamics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark engineering styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.28);
        padding: 12px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.65);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 0.80rem !important;
    }

    /* Tabs Aesthetic */
    button[data-baseweb="tab"] {
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. GEOMETRY GENERATOR (NACA 4-DIGIT + FLAP + INVERTED WING)
# ------------------------------------------------------------------------------
@st.cache_data
def generate_airfoil_geometry(m_pct: float, p_pct: float, t_pct: float,
                              flap_enabled: bool = False, flap_deg: float = 0.0, flap_hinge: float = 0.75,
                              inverted: bool = False, chord: float = 1.0, num_points: int = 70):
    """
    Generate coordinates for NACA 4-digit airfoil with optional trailing-edge plain flap
    and motorsport inverted wing orientation.
    """
    m = m_pct / 100.0
    p = p_pct / 100.0
    t = t_pct / 100.0
    c = chord
    
    # Cosine clustering along chord
    beta = np.linspace(0.0, np.pi, num_points)
    x = c * 0.5 * (1.0 - np.cos(beta))
    
    # Thickness distribution (closed trailing edge)
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
        fwd = x < p * c
        aft = ~fwd
        yc[fwd] = (m / (p**2)) * (2.0 * p * (x[fwd] / c) - (x[fwd] / c)**2) * c
        dyc_dx[fwd] = (2.0 * m / (p**2)) * (p - (x[fwd] / c))
        yc[aft] = (m / ((1.0 - p)**2)) * ((1.0 - 2.0 * p) + 2.0 * p * (x[aft] / c) - (x[aft] / c)**2) * c
        dyc_dx[aft] = (2.0 * m / ((1.0 - p)**2)) * (p - (x[aft] / c))
        
    theta = np.arctan(dyc_dx)
    
    # Base Upper and Lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Apply Trailing-Edge Plain Flap Camber Deflection
    if flap_enabled and abs(flap_deg) > 1e-3:
        d_rad = np.radians(flap_deg)
        cos_d = np.cos(d_rad)
        sin_d = np.sin(d_rad)
        
        # Hinge elevation at camber line
        yh = float(np.interp(flap_hinge * c, x, yc))
        xh = flap_hinge * c
        
        # Upper surface deflection
        u_aft = xu >= xh
        dx_u = xu[u_aft] - xh
        dy_u = yu[u_aft] - yh
        xu[u_aft] = xh + dx_u * cos_d + dy_u * sin_d
        yu[u_aft] = yh - dx_u * sin_d + dy_u * cos_d
        
        # Lower surface deflection
        l_aft = xl >= xh
        dx_l = xl[l_aft] - xh
        dy_l = yl[l_aft] - yh
        xl[l_aft] = xh + dx_l * cos_d + dy_l * sin_d
        yl[l_aft] = yh - dx_l * sin_d + dy_l * cos_d
        
        # Camber line deflection
        c_aft = x >= xh
        dx_c = x[c_aft] - xh
        dy_c = yc[c_aft] - yh
        x[c_aft] = xh + dx_c * cos_d + dy_c * sin_d
        yc[c_aft] = yh - dx_c * sin_d + dy_c * cos_d
        
    # Apply Motorsport Inverted Wing Transformation (Downforce)
    if inverted:
        yu = -yu
        yl = -yl
        yc = -yc
        # Invert surface identities so suction side is appropriately oriented
        xu_orig = xu.copy()
        yu_orig = yu.copy()
        xu = xl.copy()
        yu = yl.copy()
        xl = xu_orig
        yl = yu_orig
        
    # Form closed boundary polygon (Clockwise: TE upper -> LE -> TE lower)
    xb = np.concatenate([xu[::-1], xl[1:]])
    yb = np.concatenate([yu[::-1], yl[1:]])
    
    # Format designation
    m_int = int(round(m_pct))
    p_int = int(round(p_pct / 10.0))
    t_int = int(round(t_pct))
    base_name = f"NACA 00{t_int:02d}" if m_int == 0 else f"NACA {m_int}{p_int}{t_int:02d}"
    if flap_enabled and abs(flap_deg) > 0:
        base_name += f" (Flap {flap_deg:+.0f}°)"
    if inverted:
        base_name += " [Inverted Wing]"
        
    return xb, yb, xu, yu, xl, yl, x, yc, base_name


# ------------------------------------------------------------------------------
# 3. 2D HESS-SMITH PANEL METHOD (POTENTIAL FLOW SOLVER)
# ------------------------------------------------------------------------------
@st.cache_data
def solve_panel_method(xb: np.ndarray, yb: np.ndarray, alpha_deg: float, U_inf: float):
    """
    Exact 2D Hess-Smith Panel Method with Kutta Condition.
    Panels consist of constant-strength source distributions + constant uniform vortex sheet.
    """
    alpha = np.radians(alpha_deg)
    num_panels = len(xb) - 1
    
    dx = xb[1:] - xb[:-1]
    dy = yb[1:] - yb[:-1]
    L = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    
    xc = 0.5 * (xb[:-1] + xb[1:])
    yc = 0.5 * (yb[:-1] + yb[1:])
    
    # Tangent and Outward Normal vectors
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
    Xj = xb[:-1][np.newaxis, :]
    Yj = yb[:-1][np.newaxis, :]
    Lj = L[np.newaxis, :]
    phij = phi[np.newaxis, :]
    
    Xi = xc[:, np.newaxis]
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
    
    # Transform to global frame
    u_s = u_s_loc * cos_p - v_s_loc * sin_p
    v_s = u_s_loc * sin_p + v_s_loc * cos_p
    
    u_v = u_v_loc * cos_p - v_v_loc * sin_p
    v_v = u_v_loc * sin_p + v_v_loc * cos_p
    
    np.fill_diagonal(u_s, 0.0)
    np.fill_diagonal(v_s, 0.0)
    np.fill_diagonal(u_v, 0.0)
    np.fill_diagonal(v_v, 0.0)
    
    n_xi = nx[:, np.newaxis]
    n_yi = ny[:, np.newaxis]
    t_xi = tx[:, np.newaxis]
    t_yi = ty[:, np.newaxis]
    
    A = u_s * n_xi + v_s * n_yi
    C = u_s * t_xi + v_s * t_yi
    B = u_v * n_xi + v_v * n_yi
    D = u_v * t_xi + v_v * t_yi
    
    for i in range(num_panels):
        A[i, i] = 0.5
        B[i, i] = 0.0
        C[i, i] = 0.0
        D[i, i] = 0.5
        
    M = np.zeros((num_panels + 1, num_panels + 1))
    RHS = np.zeros(num_panels + 1)
    
    M[:num_panels, :num_panels] = A
    M[:num_panels, num_panels] = np.sum(B, axis=1)
    RHS[:num_panels] = -V_inf_n
    
    M[num_panels, :num_panels] = C[0, :] + C[-1, :]
    M[num_panels, num_panels] = np.sum(D[0, :]) + np.sum(D[-1, :])
    RHS[num_panels] = -(V_inf_t[0] + V_inf_t[-1])
    
    solution = np.linalg.solve(M, RHS)
    q = solution[:num_panels]
    gamma = solution[num_panels]
    
    Vt = V_inf_t + C @ q + gamma * np.sum(D, axis=1)
    Cp = 1.0 - (Vt / U_inf)**2
    
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
# 4. FAST 2D CARTESIAN FLOW FIELD EVALUATOR (WITH GROUND EFFECT & WAKE)
# ------------------------------------------------------------------------------
@st.cache_data
def evaluate_flow_field(xb_rot: np.ndarray, yb_rot: np.ndarray, q: np.ndarray, gamma: float,
                        L: np.ndarray, phi: np.ndarray, U_inf: float, alpha_deg: float,
                        Cd_val: float, nx: int = 70, ny: int = 50,
                        enable_wake: bool = True, ground_effect: bool = False, ground_y: float = -0.5):
    """
    Evaluates 2D velocity vector field (u, v) and normalized speed on a Cartesian grid.
    Optimized for high-speed execution with custom grid resolution.
    """
    x_lin = np.linspace(-0.45, 1.75, nx)
    y_lin = np.linspace(-0.75, 0.75, ny)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    
    X_flat = Xg.ravel()
    Y_flat = Yg.ravel()
    M = len(X_flat)
    num_panels = len(q)
    
    poly_verts = np.column_stack([xb_rot, yb_rot])
    body_path = Path(poly_verts)
    inside_mask = body_path.contains_points(np.column_stack([X_flat, Y_flat]))
    
    u_ind = np.zeros(M)
    v_ind = np.zeros(M)
    
    # Vectorized panel induction accumulation
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
    
    speed = np.sqrt(u_total**2 + v_total**2)
    speed_wake = speed.copy()
    
    # Downstream Viscous Wake Momentum Deficit
    x_te = xb_rot[0]
    y_te = yb_rot[0]
    downwash_angle = np.radians(alpha_deg * 0.7)
    
    if enable_wake:
        wake_mask = (X_flat > x_te)
        dx_w = X_flat[wake_mask] - x_te
        y_wake_center = y_te - dx_w * np.tan(downwash_angle)
        dy_w = Y_flat[wake_mask] - y_wake_center
        wake_width = 0.035 + 0.12 * np.sqrt(dx_w)
        deficit = np.clip(0.40 * (Cd_val / 0.012), 0.15, 0.85) * np.exp(-0.5 * (dy_w / wake_width)**2)
        speed_wake[wake_mask] *= (1.0 - deficit)
        
    # Ground Effect Venturi Channel Suction Acceleration
    if ground_effect:
        # Under-wing ground channel
        h_local = np.abs(np.min(yb_rot) - ground_y)
        under_wing = (X_flat >= np.min(xb_rot) - 0.1) & (X_flat <= np.max(xb_rot) + 0.2) & (Y_flat > ground_y) & (Y_flat < np.min(yb_rot) + 0.15)
        dist_to_ground = np.maximum(Y_flat[under_wing] - ground_y, 0.01)
        venturi_boost = 1.0 + (0.28 / (h_local + 0.10)) * np.exp(-1.5 * dist_to_ground / max(h_local, 0.05))
        speed_wake[under_wing] *= venturi_boost
        
        # Mask beneath ground surface
        below_ground = Y_flat <= ground_y
        speed_wake[below_ground] = 0.0
        
    speed_ratio = speed_wake / U_inf
    
    speed_ratio_2d = speed_ratio.reshape(ny, nx)
    u_2d = u_total.reshape(ny, nx)
    v_2d = v_total.reshape(ny, nx)
    
    return Xg, Yg, speed_ratio_2d, u_2d, v_2d, x_te, y_te, downwash_angle


# ------------------------------------------------------------------------------
# 5. AERODYNAMIC PERFORMANCE MODEL (Cl, Cd, L/D, STALL, FLAP, GROUND EFFECT)
# ------------------------------------------------------------------------------
@st.cache_data
def calculate_aero_performance(alpha_deg: float, m_pct: float, p_pct: float, t_pct: float,
                               flap_enabled: bool = False, flap_deg: float = 0.0, flap_hinge: float = 0.75,
                               inverted: bool = False, ground_effect: bool = False, ground_h: float = 0.30,
                               U_inf: float = 25.0, rho: float = 1.225, nu: float = 1.5e-5):
    """
    Computes viscous lift, drag, downforce, and aerodynamic efficiency
    incorporating Thin Airfoil Theory, flap effectiveness, ground effect, and stall.
    """
    m = m_pct / 100.0
    p = p_pct / 100.0
    t = t_pct / 100.0
    c = 1.0
    
    # Reynolds Number
    Re = max(U_inf * c / nu, 1e4)
    
    # Skin friction (Schlichting flat-plate turbulent)
    Cf = 0.455 / (np.log10(Re)**2.58)
    kf = 1.0 + 2.0 * t + 60.0 * (t**4)
    Cd0 = 2.0 * Cf * kf
    
    # Thin airfoil zero-lift angle
    alpha_0_deg = -1.15 * m * (180.0 / np.pi)
    
    # Flap lift increment (Thin Airfoil Flap Effectiveness)
    delta_a0_flap = 0.0
    Cd_flap = 0.0
    if flap_enabled and abs(flap_deg) > 1e-3:
        theta_f = np.arccos(np.clip(2.0 * flap_hinge - 1.0, -1.0, 1.0))
        flap_eff = (theta_f - np.sin(theta_f)) / np.pi
        delta_a0_flap = -flap_eff * flap_deg
        Cd_flap = 0.0018 * ((abs(flap_deg) / 10.0)**1.8)
        
    alpha_0_total = alpha_0_deg + delta_a0_flap
    
    # Lift curve slope
    cla_rad = 2.0 * np.pi * (1.0 + 0.77 * t) * 0.95
    cla_deg = cla_rad * (np.pi / 180.0)
    
    # Stall boundaries
    alpha_stall_pos = 12.5 + 25.0 * t + 10.0 * m + (0.4 * flap_deg if flap_enabled else 0.0)
    alpha_stall_neg = -(12.5 + 25.0 * t - 10.0 * m)
    
    alpha = np.asarray(alpha_deg, dtype=float)
    a_rad = np.radians(alpha)
    
    # Linear attached lift
    Cl_lin = cla_deg * (alpha - alpha_0_total)
    
    # Non-linear stall blending (Kirchhoff-Helmholtz Sigmoid Model)
    x_pos = (alpha - alpha_stall_pos) / 1.8
    sig_pos = 1.0 / (1.0 + np.exp(-np.clip(x_pos, -25, 25)))
    x_neg = -(alpha - alpha_stall_neg) / 1.8
    sig_neg = 1.0 / (1.0 + np.exp(-np.clip(x_neg, -25, 25)))
    sigma = np.maximum(sig_pos, sig_neg)
    
    Cl_sep = 1.8 * np.sin(a_rad) * np.cos(a_rad)
    Cl = (1.0 - sigma) * Cl_lin + sigma * Cl_sep
    
    # Ground Effect Suction Amplification
    ground_factor = 1.0
    if ground_effect:
        # Venturi ground effect suction multiplier
        ground_factor = 1.0 + (0.12 / (ground_h + 0.05)**0.85)
        # Blockage reduction if too close (diffuser stall)
        if ground_h < 0.12:
            ground_factor -= 0.35 * ((0.12 - ground_h) / 0.12)
        Cl = Cl * ground_factor
        
    # Inverted Wing: Invert lift sign (Produces Downforce)
    if inverted:
        Cl = -Cl
        
    # Drag computation
    Cl_ideal = cla_deg * (-alpha_0_total)
    kp = 0.006 + 0.012 * t
    Cd_prof = Cd0 + Cd_flap + kp * (Cl - Cl_ideal)**2
    Cd_sep = sigma * (1.8 * (np.sin(a_rad)**2) + 0.035)
    
    # Ground effect reduces induced/form drag slightly
    if ground_effect:
        Cd_prof *= np.clip(1.0 - 0.08 / (ground_h + 0.15), 0.75, 1.0)
        
    Cd = np.maximum(Cd_prof + Cd_sep, 0.0035)
    
    # Efficiency metrics
    L_D = np.where(Cd > 0, Cl / Cd, 0.0)
    Downforce_Coeff = -Cl if inverted else 0.0
    
    # Dimensional aerodynamic forces per meter span (N/m)
    q_dyn = 0.5 * rho * (U_inf**2)
    lift_force = q_dyn * c * Cl
    drag_force = q_dyn * c * Cd
    downforce_val = -lift_force if inverted else 0.0
    
    return {
        'Cl': float(Cl) if np.ndim(Cl) == 0 else Cl,
        'Cd': float(Cd) if np.ndim(Cd) == 0 else Cd,
        'L_D': float(L_D) if np.ndim(L_D) == 0 else L_D,
        'Downforce_Coeff': float(Downforce_Coeff) if np.ndim(Downforce_Coeff) == 0 else Downforce_Coeff,
        'downforce_val': float(downforce_val) if np.ndim(downforce_val) == 0 else downforce_val,
        'lift_force': float(lift_force) if np.ndim(lift_force) == 0 else lift_force,
        'drag_force': float(drag_force) if np.ndim(drag_force) == 0 else drag_force,
        'Re': Re,
        'Cd0': Cd0,
        'alpha_0_total': alpha_0_total,
        'alpha_stall_pos': alpha_stall_pos,
        'cla_deg': cla_deg,
        'ground_factor': ground_factor
    }


# ------------------------------------------------------------------------------
# 6. CAD & 3D PRINTING EXPORT GENERATORS (NATIVE 2D DXF & SOLIDWORKS CSV)
# ------------------------------------------------------------------------------
def generate_dxf_r12(x_coords: np.ndarray, y_coords: np.ndarray, scale: float = 1.0, layer: str = "AIRFOIL_PROFILE") -> str:
    """
    Generates a clean, standard ASCII DXF (Release 12) with a closed POLYLINE.
    Directly compatible with SolidWorks, Fusion 360, AutoCAD, FreeCAD, Rhino, and 3D Slicers.
    """
    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$ACADVER",
        "1", "AC1009",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
        "0", "POLYLINE",
        "8", layer,
        "66", "1",      # Vertices follow
        "70", "1",      # Closed polyline flag
        "10", "0.0",
        "20", "0.0",
        "30", "0.0"
    ]
    
    for x, y in zip(x_coords, y_coords):
        lines.extend([
            "0", "VERTEX",
            "8", layer,
            "10", f"{x * scale:.6f}",
            "20", f"{y * scale:.6f}",
            "30", "0.000000"
        ])
        
    lines.extend([
        "0", "SEQEND",
        "8", layer,
        "0", "ENDSEC",
        "0", "EOF"
    ])
    return "\n".join(lines)


def generate_cad_xyz_csv(x_coords: np.ndarray, y_coords: np.ndarray, scale: float = 1.0) -> str:
    """
    Generates XYZ coordinate CSV formatted for SolidWorks 'Curve Through XYZ Points'
    and Fusion 360 'Import Spline CSV'.
    """
    out = io.StringIO()
    out.write("X,Y,Z\n")
    for x, y in zip(x_coords, y_coords):
        out.write(f"{x * scale:.6f},{y * scale:.6f},0.000000\n")
    # Ensure closed loop
    out.write(f"{x_coords[0] * scale:.6f},{y_coords[0] * scale:.6f},0.000000\n")
    return out.getvalue()


# ------------------------------------------------------------------------------
# 7. SIDEBAR CONTROLS & PERFORMANCE SETTINGS
# ------------------------------------------------------------------------------
st.sidebar.markdown("## ⚡ Performance & Quality")
perf_mode = st.sidebar.radio(
    "Rendering Mode",
    options=["⚡ Fast (Interactive)", "💎 High Def"],
    index=0,
    help="Fast mode downsamples the CFD grid and streamline trace for ultra-responsive slider adjustments on Streamlit Cloud."
)

if "Fast" in perf_mode:
    grid_nx, grid_ny = 70, 50
    streamline_density = 0.6
    fig_dpi = 85
else:
    grid_nx, grid_ny = 140, 100
    streamline_density = 1.1
    fig_dpi = 115

st.sidebar.markdown("---")
st.sidebar.markdown("## ✈️ Airfoil Geometry (NACA 4-Digit)")

m_pct = st.sidebar.slider(
    "Max Camber (m)",
    min_value=0.0,
    max_value=9.0,
    value=2.0,
    step=0.5,
    format="%.1f%% chord",
    help="Maximum height of the camber line."
)

p_pct = st.sidebar.slider(
    "Camber Position (p)",
    min_value=10.0,
    max_value=90.0,
    value=40.0,
    step=10.0,
    format="%.0f%% chord",
    help="Location of maximum camber along chord from leading edge."
)

t_pct = st.sidebar.slider(
    "Max Thickness (t)",
    min_value=5.0,
    max_value=30.0,
    value=12.0,
    step=1.0,
    format="%.0f%% chord",
    help="Maximum thickness of the airfoil profile."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🛩️ High-Lift Plain Flap")

enable_flap = st.sidebar.checkbox("Enable Trailing-Edge Flap", value=False, help="Adds a plain trailing-edge flap with real-time camber deflection.")
flap_deflection = 0.0
flap_hinge = 0.75
if enable_flap:
    flap_deflection = st.sidebar.slider(
        "Flap Deflection Angle (δf)",
        min_value=-15.0,
        max_value=35.0,
        value=15.0,
        step=1.0,
        format="%.0f°",
        help="Downwards deflection angle of the trailing-edge flap."
    )
    flap_hinge = st.sidebar.slider(
        "Hinge Location (xf/c)",
        min_value=0.60,
        max_value=0.85,
        value=0.75,
        step=0.05,
        format="%.2f chord",
        help="Chordwise position of the flap hinge."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏎️ Motorsport Downforce & Ground Effect")

inverted_wing = st.sidebar.checkbox(
    "Inverted Wing (Downforce Mode)",
    value=False,
    help="Inverts the wing profile as used in Formula 1, GT3, and LMP sports car wings to create aerodynamic downforce."
)

enable_ground_effect = st.sidebar.checkbox(
    "Simulate Ground Effect",
    value=False,
    help="Simulates proximity to the track surface, accelerating flow beneath the wing via the Venturi ground effect."
)

ground_clearance = 0.30
if enable_ground_effect:
    ground_clearance = st.sidebar.slider(
        "Ground Clearance (h/c)",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        format="%.2f chord",
        help="Distance between the lowest point of the airfoil and the road/track plane."
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
    help="Angle between oncoming freestream air and chord line."
)

U_inf = st.sidebar.slider(
    "Freestream Velocity ($U_\\infty$)",
    min_value=5.0,
    max_value=60.0,
    value=25.0,
    step=1.0,
    format="%.0f m/s",
    help="Magnitude of freestream velocity."
)

# Constants
rho_air = 1.225
nu_air = 1.5e-5
chord_len = 1.0

with st.sidebar.expander("🎨 Visual Options", expanded=False):
    cmap_choice = st.selectbox(
        "Colormap",
        options=["plasma", "viridis", "turbo", "inferno", "coolwarm"],
        index=0
    )
    show_wake = st.checkbox("Include Viscous Wake Deficit", value=True)


# ------------------------------------------------------------------------------
# 8. COMPUTATIONAL EXECUTION
# ------------------------------------------------------------------------------
# 1. Generate unrotated geometry
xb, yb, xu, yu, xl, yl, xc_line, yc_line, designation = generate_airfoil_geometry(
    m_pct, p_pct, t_pct,
    flap_enabled=enable_flap, flap_deg=flap_deflection, flap_hinge=flap_hinge,
    inverted=inverted_wing, chord=chord_len, num_points=70
)

# 2. Aero performance
aero = calculate_aero_performance(
    alpha_deg, m_pct, p_pct, t_pct,
    flap_enabled=enable_flap, flap_deg=flap_deflection, flap_hinge=flap_hinge,
    inverted=inverted_wing, ground_effect=enable_ground_effect, ground_h=ground_clearance,
    U_inf=U_inf, rho=rho_air, nu=nu_air
)

# 3. Rotate airfoil by AoA around quarter-chord (0.25, 0)
alpha_rad = np.radians(alpha_deg)
cos_a = np.cos(alpha_rad)
sin_a = np.sin(alpha_rad)

# Inverted wing pitches down for positive downforce AoA
if inverted_wing:
    xb_rot = 0.25 + (xb - 0.25) * cos_a - yb * sin_a
    yb_rot = (xb - 0.25) * sin_a + yb * cos_a
else:
    xb_rot = 0.25 + (xb - 0.25) * cos_a + yb * sin_a
    yb_rot = -(xb - 0.25) * sin_a + yb * cos_a

# 4. Panel method solution
panel_res = solve_panel_method(xb_rot, yb_rot, alpha_deg=0.0, U_inf=U_inf)

# 5. Determine ground plane location if active
ground_y_coord = -0.5
if enable_ground_effect:
    lowest_y = np.min(yb_rot)
    ground_y_coord = lowest_y - ground_clearance

# 6. Evaluate fast 2D flow field
Xg, Yg, speed_ratio_2d, u_2d, v_2d, x_te, y_te, downwash_ang = evaluate_flow_field(
    xb_rot, yb_rot, panel_res['q'], panel_res['gamma'], panel_res['L'], panel_res['phi'],
    U_inf=U_inf, alpha_deg=alpha_deg, Cd_val=aero['Cd'],
    nx=grid_nx, ny=grid_ny, enable_wake=show_wake,
    ground_effect=enable_ground_effect, ground_y=ground_y_coord
)

# 7. Stagnation Point location
xc_rot = panel_res['xc']
yc_rot = panel_res['yc']
stag_idx = np.argmax(panel_res['Cp'])
x_stag = xc_rot[stag_idx]
y_stag = yc_rot[stag_idx]
cp_stag = panel_res['Cp'][stag_idx]


# ------------------------------------------------------------------------------
# 9. MAIN DASHBOARD: HEADER & METRIC CARDS
# ------------------------------------------------------------------------------
header_icon = "🏎️" if inverted_wing else "✈️"
st.title(f"{header_icon} {designation} CFD & Aerodynamics Studio")

mode_badge = "Motorsport Inverted Wing (Downforce)" if inverted_wing else "Aeronautical Lifting Profile"
if enable_ground_effect:
    mode_badge += f" • Ground Effect (h/c = {ground_clearance:.2f})"
if enable_flap:
    mode_badge += f" • TE Flap ({flap_deflection:+.0f}°)"

st.markdown(
    f"**Mode:** `{mode_badge}` &nbsp;|&nbsp; "
    f"**AoA (α):** `{alpha_deg:+.1f}°` &nbsp;|&nbsp; "
    f"**Speed ($U_\\infty$):** `{U_inf:.0f} m/s` ({U_inf * 3.6:.0f} km/h) &nbsp;|&nbsp; "
    f"**Grid:** `{grid_nx}×{grid_ny}` ({perf_mode.split()[1]})"
)

# Top Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    if inverted_wing:
        st.metric(
            label="Downforce Coeff (C_df)",
            value=f"{aero['Downforce_Coeff']:+.3f}",
            delta=f"Downforce: {aero['downforce_val']:.1f} N/m",
            help="Dimensionless downforce coefficient. Positive produces downforce pressing the vehicle onto the track."
        )
    else:
        st.metric(
            label="Lift Coefficient (Cl)",
            value=f"{aero['Cl']:+.3f}",
            delta=f"Lift: {aero['lift_force']:.1f} N/m",
            help="Dimensionless lift coefficient per unit span."
        )

with col2:
    st.metric(
        label="Drag Coefficient (Cd)",
        value=f"{aero['Cd']:.4f}",
        delta=f"Drag: {aero['drag_force']:.2f} N/m",
        delta_color="inverse",
        help="Dimensionless drag coefficient accounting for skin friction, form drag, flap, and separation."
    )

with col3:
    eff_val = abs(aero['L_D'])
    eff_label = "High Efficiency" if eff_val > 40 else ("Moderate" if eff_val > 15 else "High Drag / Stalled")
    st.metric(
        label="Aero Efficiency (|L/D|)" if not inverted_wing else "Downforce / Drag (-L/D)",
        value=f"{eff_val:.1f} : 1",
        delta=eff_label,
        help="Ratio of usable vertical aerodynamic force to aerodynamic drag."
    )

with col4:
    re_exp = int(np.floor(np.log10(aero['Re'])))
    re_base = aero['Re'] / (10**re_exp)
    st.metric(
        label="Reynolds Number (Re)",
        value=f"{re_base:.2f} × 10⁶",
        delta="Motorsport Regime" if inverted_wing else "Subsonic Aero",
        help=f"Exact Re = {aero['Re']:,.0f}."
    )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 10. TABS INTERFACE
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 2D CFD Flow Field",
    "📊 Surface Pressure (-Cp)",
    "📈 Aerodynamic Polars & Stall",
    "📚 Aero Physics & Formulas"
])


# ==============================================================================
# TAB 1: 2D CFD FLOW FIELD VISUALIZATION
# ==============================================================================
with tab1:
    plt.style.use('dark_background')
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.6), dpi=fig_dpi, facecolor='#090d16')
    ax1.set_facecolor('#090d16')
    
    # 1. Background Contours
    levels = np.linspace(0.0, 1.85, 38)
    cf = ax1.contourf(Xg, Yg, speed_ratio_2d, levels=levels, cmap=cmap_choice, extend='both')
    
    # 2. Horizontal Colorbar
    cbar = fig1.colorbar(cf, ax=ax1, orientation='horizontal', pad=0.14, fraction=0.046, aspect=36)
    cbar.set_label(r'Normalized Flow Velocity Magnitude  $|\vec{V}| / U_\infty$', fontsize=10, color='#e2e8f0', fontweight='bold', labelpad=5)
    cbar.ax.tick_params(colors='#cbd5e1', labelsize=8.5)
    
    # 3. Streamlines
    ax1.streamplot(
        Xg, Yg, u_2d, v_2d,
        color=(1.0, 1.0, 1.0, 0.40),
        density=streamline_density,
        linewidth=0.75,
        arrowsize=0.85
    )
    
    # 4. Solid Airfoil Body & Outline
    ax1.fill(xb_rot, yb_rot, color='#05070c', zorder=10)
    ax1.plot(xb_rot, yb_rot, color='#38bdf8', lw=2.0, zorder=11, label='Wing Profile')
    
    # 5. Stagnation Point
    ax1.plot(x_stag, y_stag, 'o', color='#ef4444', markersize=7.5, markeredgecolor='white', markeredgewidth=1.6, zorder=15, label='Stagnation Point')
    ax1.annotate(
        f'Stagnation (Cp={cp_stag:.2f})',
        xy=(x_stag, y_stag),
        xytext=(x_stag - 0.28, y_stag + (0.20 if inverted_wing else -0.22)),
        arrowprops=dict(facecolor='#ef4444', edgecolor='white', arrowstyle='->', lw=1.2),
        color='#ffffff', fontsize=8.2, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#ef4444', lw=1.0, alpha=0.92),
        zorder=16
    )
    
    # 6. Boundary Layer Wake Region
    if show_wake:
        x_wake = np.linspace(x_te, 1.75, 40)
        y_wake = y_te - (x_wake - x_te) * np.tan(downwash_ang)
        ax1.plot(x_wake, y_wake, '--', color='#c084fc', lw=1.2, alpha=0.75, zorder=12, label='Wake Centerline')
        
    # 7. Ground Plane (Road Surface)
    if enable_ground_effect:
        ax1.fill_between([-0.45, 1.75], -0.75, ground_y_coord, color='#1e293b', hatch='//', alpha=0.9, zorder=14)
        ax1.plot([-0.45, 1.75], [ground_y_coord, ground_y_coord], color='#f59e0b', lw=2.4, zorder=15, label='Track / Road Surface')
        ax1.annotate(
            f'Ground Surface (h/c = {ground_clearance:.2f})',
            xy=(0.5, ground_y_coord),
            xytext=(0.35, ground_y_coord - 0.12),
            color='#fbbf24', fontsize=8.2, fontweight='bold', zorder=16
        )
        
    # Axis bounds & styling
    ax1.set_xlim(-0.45, 1.75)
    ax1.set_ylim(-0.75, 0.75)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x / c  (Chord Normalized Position)', fontsize=10, color='#cbd5e1', labelpad=3)
    ax1.set_ylabel('y / c  (Chord Normalized Position)', fontsize=10, color='#cbd5e1', labelpad=3)
    ax1.tick_params(colors='#94a3b8', labelsize=8.5)
    for spine in ax1.spines.values():
        spine.set_color('#334155')
        
    ax1.legend(loc='upper right', framealpha=0.88, facecolor='#0f172a', edgecolor='#334155', fontsize=8.2, labelcolor='#e2e8f0')
    ax1.set_title(
        f"2D Flow Field Contours & Streamlines ({designation}, α = {alpha_deg:+.1f}°, U∞ = {U_inf:.0f} m/s)",
        fontsize=11.5, fontweight='bold', color='#f8fafc', pad=10
    )
    
    fig1.subplots_adjust(bottom=0.20, top=0.92, left=0.08, right=0.96)
    st.pyplot(fig1, clear_figure=True)
    plt.close('all')
    
    if enable_ground_effect:
        st.success(
            f"🏎️ **Ground Effect Active:** Flow channel height $h/c = {ground_clearance:.2f}$. "
            f"Venturi ground suction intensifies under-wing acceleration to **{np.nanmax(speed_ratio_2d):.2f} × $U_\\infty$**, "
            f"boosting downforce by **+{(aero['ground_factor'] - 1.0) * 100:.1f}%**!"
        )


# ==============================================================================
# TAB 2: SURFACE PRESSURE DISTRIBUTION (-Cp)
# ==============================================================================
with tab2:
    st.markdown("### Surface Pressure Coefficient Distribution ($C_p$)")
    st.caption("Standard aeronautical convention: Y-axis is inverted so that suction (-Cp) points upward.")
    
    # Solve unrotated panel flow for chordwise x/c distribution
    unrot_res = solve_panel_method(xb, yb, alpha_deg=alpha_deg, U_inf=U_inf)
    xc_unrot = unrot_res['xc']
    Cp_unrot = unrot_res['Cp']
    half = (len(xb) - 1) // 2
    
    x_u = xc_unrot[:half]
    cp_u = Cp_unrot[:half]
    sort_u = np.argsort(x_u)
    x_u_s = x_u[sort_u]
    cp_u_s = cp_u[sort_u]
    
    x_l = xc_unrot[half:]
    cp_l = Cp_unrot[half:]
    sort_l = np.argsort(x_l)
    x_l_s = x_l[sort_l]
    cp_l_s = cp_l[sort_l]
    
    cp_l_interp = np.interp(x_u_s, x_l_s, cp_l_s)
    Cn_approx = integrate.trapezoid(cp_l_interp - cp_u_s, x_u_s)
    
    # Center of Pressure (undefined at near-zero lift)
    if abs(Cn_approx) > 0.05:
        x_cp_approx = integrate.trapezoid(x_u_s * (cp_l_interp - cp_u_s), x_u_s) / Cn_approx
        x_cp_display = f"{x_cp_approx:.3f}"
        x_cp_delta = f"Δ to c/4: {x_cp_approx - 0.25:+.3f}"
    else:
        x_cp_display = "N/A"
        x_cp_delta = "Near Zero Lift"
        
    Cm_c4_approx = integrate.trapezoid((0.25 - x_u_s) * (cp_l_interp - cp_u_s), x_u_s)
    
    fig2, ax2 = plt.subplots(figsize=(10.0, 4.8), dpi=fig_dpi, facecolor='#090d16')
    ax2.set_facecolor('#090d16')
    
    upper_label = "Upper Surface" if not inverted_wing else "Upper Surface (Pressure Side)"
    lower_label = "Lower Surface" if not inverted_wing else "Lower Surface (Suction Side)"
    
    ax2.plot(x_u_s, cp_u_s, color='#38bdf8', lw=2.2, label=upper_label)
    ax2.plot(x_l_s, cp_l_s, color='#f43f5e', lw=2.2, label=lower_label)
    
    ax2.fill_between(
        x_u_s, cp_u_s, cp_l_interp,
        where=(cp_l_interp >= cp_u_s),
        color='#38bdf8', alpha=0.18,
        label=r'Net Normal Force Area $\oint \Delta C_p\,d(x/c)$'
    )
    
    ax2.invert_yaxis()
    
    min_cp_idx = np.argmin(cp_u_s)
    ax2.plot(x_u_s[min_cp_idx], cp_u_s[min_cp_idx], 'o', color='#38bdf8', markersize=6.5)
    ax2.annotate(
        f'Suction Peak: Cp = {cp_u_s[min_cp_idx]:.2f}',
        xy=(x_u_s[min_cp_idx], cp_u_s[min_cp_idx]),
        xytext=(x_u_s[min_cp_idx] + 0.08, cp_u_s[min_cp_idx] - 0.22),
        arrowprops=dict(facecolor='#38bdf8', edgecolor='white', arrowstyle='->', lw=1.1),
        color='#ffffff', fontsize=8.2, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.9)
    )
    
    ax2.axhline(0.0, color='#64748b', linestyle=':', lw=1.0, alpha=0.7)
    ax2.axhline(1.0, color='#ef4444', linestyle=':', lw=1.0, alpha=0.7, label='Stagnation Limit (Cp = 1.0)')
    
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_xlabel('Chordwise Position (x / c)', fontsize=10, color='#cbd5e1')
    ax2.set_ylabel(r'Pressure Coefficient ($C_p$) [Inverted: Suction $\uparrow$]', fontsize=10, color='#cbd5e1')
    ax2.tick_params(colors='#94a3b8', labelsize=8.5)
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    ax2.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
    ax2.legend(loc='lower right', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=8.5, labelcolor='#e2e8f0')
    ax2.set_title(f"Surface Pressure Distribution -Cp for {designation} (α = {alpha_deg:+.1f}°)", fontsize=11, fontweight='bold', color='#f8fafc')
    
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
    plt.close('all')
    
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.metric("Minimum Pressure (Peak Suction)", f"{cp_u_s[min_cp_idx]:.2f}", f"at x/c = {x_u_s[min_cp_idx]:.2f}")
    with p_col2:
        st.metric("Stagnation Pressure", f"{np.max(Cp_unrot):.2f}", "Theoretical Max: 1.00")
    with p_col3:
        st.metric("Center of Pressure (x_cp / c)", x_cp_display, x_cp_delta)
    with p_col4:

        st.metric("Pitching Moment (Cm,c/4)", f"{Cm_c4_approx:+.4f}", "Quarter-chord moment")


# ==============================================================================
# TAB 3: AERODYNAMIC POLARS & STALL (CACHED & LAZY-LOADED)
# ==============================================================================
with tab3:
    st.markdown("### Aerodynamic Performance Polars & Non-Linear Stall Modeling")
    st.caption("Viscous polar curves calculated across angles of attack from -10° to +20°.")
    
    # 31-point fast polar evaluation
    alpha_polars = np.linspace(-10.0, 20.0, 31)
    aero_polars = calculate_aero_performance(
        alpha_polars, m_pct, p_pct, t_pct,
        flap_enabled=enable_flap, flap_deg=flap_deflection, flap_hinge=flap_hinge,
        inverted=inverted_wing, ground_effect=enable_ground_effect, ground_h=ground_clearance,
        U_inf=U_inf, rho=rho_air, nu=nu_air
    )
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=fig_dpi, facecolor='#090d16')
    for ax in (ax3a, ax3b):
        ax.set_facecolor('#090d16')
        ax.tick_params(colors='#94a3b8', labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', color='#1e293b', alpha=0.7)
        
    # --- Subplot 1: Cl / Downforce vs Alpha ---
    y_polar = aero_polars['Cl'] if not inverted_wing else -aero_polars['Cl']
    y_curr = aero['Cl'] if not inverted_wing else aero['Downforce_Coeff']
    y_label = 'Lift Coefficient (Cl)' if not inverted_wing else 'Downforce Coefficient (C_df)'
    
    # Theoretical linear slope
    linear_slope = aero['cla_deg'] * (alpha_polars - aero['alpha_0_total'])
    if inverted_wing:
        linear_slope = -linear_slope
    ax3a.plot(alpha_polars, linear_slope, '--', color='#64748b', lw=1.2, alpha=0.75, label=r'Inviscid Linear Slope')
    
    # Viscous curve with stall
    ax3a.plot(alpha_polars, y_polar, color='#38bdf8', lw=2.4, label='Viscous Curve (Stall Model)')
    ax3a.axvline(aero['alpha_stall_pos'], color='#f59e0b', linestyle=':', lw=1.5, label=f'Stall Onset ({aero["alpha_stall_pos"]:.1f}°)')
    ax3a.axhline(0.0, color='#475569', lw=0.8, alpha=0.6)
    
    # Current operating point
    ax3a.plot(alpha_deg, y_curr, 'o', color='#ef4444', markersize=8, markeredgecolor='white', markeredgewidth=1.6,
              zorder=10, label=f'Operating Point ({y_label.split()[0]}={y_curr:.3f})')
    
    ax3a.set_xlim(-10.5, 20.5)
    ax3a.set_xlabel('Angle of Attack α (°)', fontsize=9.8, color='#cbd5e1')
    ax3a.set_ylabel(y_label, fontsize=9.8, color='#cbd5e1')
    ax3a.set_title(f'{y_label} vs α', fontsize=11, fontweight='bold', color='#f8fafc')
    ax3a.legend(loc='upper left', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=8.2, labelcolor='#e2e8f0')
    
    # --- Subplot 2: Drag Polar (Cl vs Cd) ---
    ax3b.plot(aero_polars['Cd'], y_polar, color='#10b981', lw=2.4, label='Drag Polar')
    ax3b.plot(aero['Cd'], y_curr, 'o', color='#ef4444', markersize=8, markeredgecolor='white', markeredgewidth=1.6,
              zorder=10, label=f'Operating Point (Cd = {aero["Cd"]:.4f})')
    
    max_ld_idx = np.argmax(np.abs(aero_polars['L_D']))
    ax3b.plot(aero_polars['Cd'][max_ld_idx], y_polar[max_ld_idx], 's', color='#f59e0b', markersize=6.5,
              label=f'Max |L/D| = {abs(aero_polars["L_D"][max_ld_idx]):.1f}')
    
    ax3b.axhline(0.0, color='#475569', lw=0.8, alpha=0.6)
    ax3b.set_xlabel('Drag Coefficient (Cd)', fontsize=9.8, color='#cbd5e1')
    ax3b.set_ylabel(y_label, fontsize=9.8, color='#cbd5e1')
    ax3b.set_title(f'Drag Polar ({y_label.split()[0]} vs Cd)', fontsize=11, fontweight='bold', color='#f8fafc')
    ax3b.legend(loc='lower right', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=8.2, labelcolor='#e2e8f0')
    
    plt.tight_layout()
    st.pyplot(fig3, clear_figure=True)
    plt.close('all')
    
    st.markdown("#### Aerodynamic Characteristics Summary")
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
    with s_col1:
        st.metric("Zero-Lift Angle (α_0)", f"{aero['alpha_0_total']:.2f}°", "Theoretical Intercept")
    with s_col2:
        st.metric("Lift Slope (dCl/dα)", f"{aero['cla_deg']:.4f} /°", f"{aero['cla_deg']*180/np.pi:.2f} /rad")
    with s_col3:
        st.metric("Max Vertical Force", f"{np.max(y_polar):.2f}", f"at α = {alpha_polars[np.argmax(y_polar)]:.1f}°")
    with s_col4:
        st.metric("Minimum Drag (Cd,min)", f"{np.min(aero_polars['Cd']):.4f}", f"Cd0 = {aero['Cd0']:.4f}")
    with s_col5:
        st.metric("Peak Efficiency", f"{np.max(np.abs(aero_polars['L_D'])):.1f} : 1", f"at α = {alpha_polars[max_ld_idx]:.1f}°")


# ==============================================================================
# TAB 4: PHYSICS & MATHEMATICAL FORMULATION
# ==============================================================================
with tab4:
    st.markdown("### Underlying Aerodynamic Theory & Numerical Methods")
    st.markdown("""
    #### 1. Parametric Airfoil Geometry & Flap Transformation
    The profile is generated using the NACA 4-digit analytical equations with cosine distribution along chord $c$:
    
    $$\\beta \\in [0, \\pi], \\quad x = \\frac{c}{2}(1 - \\cos\\beta)$$
    
    When a plain trailing-edge flap is engaged, all coordinates aft of the hinge $x_h = x_f \\cdot c$ are rotated by flap angle $\\delta_f$ around the hinge point $(x_h, y_c(x_h))$:
    $$x' = x_h + (x - x_h)\\cos\\delta_f + (y - y_h)\\sin\\delta_f$$
    $$y' = y_h - (x - x_h)\\sin\\delta_f + (y - y_h)\\cos\\delta_f$$
    
    #### 2. Hess-Smith 2D Potential Panel Solver
    The boundary is discretized into $N$ planar panels with source distribution $q_j$ and global circulation $\\gamma$:
    - **Flow Tangency Condition**: $\\vec{V}_i \\cdot \\vec{n}_i = 0$ at panel control points.
    - **Kutta Condition**: Smooth tangential separation at trailing edge $V_{t, 1} + V_{t, N} = 0$.
    - **Surface Pressure Distribution**: $C_p = 1 - (V_t / U_\\infty)^2$.
    
    #### 3. Motorsport Inverted Wing & Ground Effect Aerodynamics
    - **Inverted Wing Downforce**: The geometry is inverted ($y \\to -y$), shifting the suction peak to the lower surface to generate downward aerodynamic load ($C_L < 0, C_{df} = -C_L$).
    - **Venturi Ground Effect**: When flying near a road surface ($h/c < 0.6$), the ground plane constricts the under-wing channel, accelerating the flow and deepening the suction peak:
      $$K_{\\text{ground}} = 1.0 + \\frac{0.12}{(h/c + 0.05)^{0.85}}$$
      If ground clearance is reduced below the boundary layer blockage threshold ($h/c < 0.12$), diffuser stall occurs.
    """)


# ------------------------------------------------------------------------------
# 11. CAD & 3D PRINTING EXPORT EXPANDER (DXF, SOLIDWORKS, FUSION 360)
# ------------------------------------------------------------------------------
with st.expander("💾 CAD & 3D Printing Export (SolidWorks, Fusion 360, DXF, CSV)", expanded=False):
    st.markdown("Export publication-quality geometry and simulation results directly to CAD, CAM, 3D printers, or laser cutters.")
    
    scale_col, unit_col = st.columns([2, 1])
    with scale_col:
        scale_mode = st.selectbox(
            "Export Dimensions / Scale",
            options=["Normalized (Chord = 1.0 m)", "Full Scale (Chord = 1000 mm)", "Wind Tunnel Model (Chord = 200 mm)", "3D Printing Test (Chord = 150 mm)"],
            index=1
        )
    with unit_col:
        scale_mult = 1.0
        unit_label = "m"
        if "1000 mm" in scale_mode:
            scale_mult = 1000.0
            unit_label = "mm"
        elif "200 mm" in scale_mode:
            scale_mult = 200.0
            unit_label = "mm"
        elif "150 mm" in scale_mode:
            scale_mult = 150.0
            unit_label = "mm"
        st.markdown(f"**Multiplier:** `{scale_mult}× ({unit_label})`")
        
    btn_col1, btn_col2 = st.columns(2)
    
    # 1. Native 2D DXF Export
    dxf_content = generate_dxf_r12(xb, yb, scale=scale_mult, layer="AIRFOIL_PROFILE")
    with btn_col1:
        st.download_button(
            label="📥 Download Native 2D DXF (AutoCAD / SolidWorks / Fusion)",
            data=dxf_content,
            file_name=f"{designation.replace(' ', '_')}_{unit_label}.dxf",
            mime="application/dxf",
            help="Clean standard ASCII DXF Release 12 closed polyline. Compatible with SolidWorks, Fusion 360, AutoCAD, and laser cutters."
        )
        
    # 2. SolidWorks / Fusion 360 XYZ CSV
    cad_csv_content = generate_cad_xyz_csv(xb, yb, scale=scale_mult)
    with btn_col2:
        st.download_button(
            label="📥 Download SolidWorks / Fusion 360 XYZ CSV",
            data=cad_csv_content,
            file_name=f"{designation.replace(' ', '_')}_XYZ_{unit_label}.csv",
            mime="text/csv",
            help="Formatted for SolidWorks 'Curve Through XYZ Points' and Fusion 360 'Import Spline CSV'."
        )
        
    btn_col3, btn_col4 = st.columns(2)
    
    # 3. Standard Coordinates CSV
    csv_coords = io.StringIO()
    csv_coords.write(f"x_{unit_label},y_upper_{unit_label},y_lower_{unit_label},y_camber_{unit_label}\n")
    for i in range(len(xu)):
        csv_coords.write(f"{xu[i]*scale_mult:.6f},{yu[i]*scale_mult:.6f},{yl[i]*scale_mult:.6f},{yc_line[i]*scale_mult:.6f}\n")
    with btn_col3:
        st.download_button(
            label="📥 Download Airfoil Coordinates (CSV)",
            data=csv_coords.getvalue(),
            file_name=f"{designation.replace(' ', '_')}_coordinates.csv",
            mime="text/csv"
        )
        
    # 4. Surface Pressure CSV
    csv_cp = io.StringIO()
    csv_cp.write("x_upper,Cp_upper,x_lower,Cp_lower\n")
    for i in range(len(x_u_s)):
        csv_cp.write(f"{x_u_s[i]:.6f},{cp_u_s[i]:.6f},{x_l_s[i]:.6f},{cp_l_s[i]:.6f}\n")
    with btn_col4:
        st.download_button(
            label=f"📥 Download Surface Cp at α={alpha_deg:.1f}° (CSV)",
            data=csv_cp.getvalue(),
            file_name=f"{designation.replace(' ', '_')}_Cp_alpha_{alpha_deg:.1f}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("🏎️ Airfoil CFD Studio • High-Speed Streamlit Cloud Optimized • Pure NumPy & SciPy • SolidWorks / Fusion 360 CAD Ready")
