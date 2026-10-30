"""
Airfoil CFD Studio
==================
Interactive 2D airfoil aerodynamics for aeronautics and motorsport.

- Parametric NACA 4-digit geometry (cosine spacing) with plain trailing-edge flap
- Inverted-wing (downforce) mode and ground-effect simulation
- Hess-Smith panel method with Kutta condition + vectorised flow-field evaluation
- Surface pressure (Cp), viscous polars with stall model
- CAD export: DXF (R12), SolidWorks / Fusion 360 XYZ CSV, coordinate and Cp CSVs
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path
import numpy as np
import plotly.graph_objects as go
import scipy.integrate as integrate
import streamlit as st

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & DESIGN SYSTEM
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Airfoil CFD Studio",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Palette (kept in sync with .streamlit/config.toml)
BG = "#0B0D11"
SURFACE = "#151922"
BORDER = "#232834"
GRID = "#1A1F29"
TEXT = "#E6E8EC"
MUTED = "#8B93A1"
ACCENT = "#FF6B2C"
BLUE = "#4DA3FF"
RED = "#FF5C6C"
GREEN = "#3DD68C"
AMBER = "#F5B83D"

st.markdown(
    f"""
<style>
:root {{
    --bg: {BG}; --surface: {SURFACE}; --border: {BORDER};
    --text: {TEXT}; --muted: {MUTED}; --accent: {ACCENT};
    --green: {GREEN}; --amber: {AMBER}; --red: {RED}; --blue: {BLUE};
}}

/* Chrome */
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1360px; }}

/* Sidebar */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
.brand {{ font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; color: var(--text);
          display: flex; align-items: center; gap: .5rem; margin-bottom: .25rem; }}
.brand .mark {{ width: 22px; height: 22px; border-radius: 6px; background: var(--accent);
                display: inline-block; clip-path: polygon(0 70%, 100% 20%, 100% 45%, 0 100%); }}
.brand-sub {{ font-size: .78rem; color: var(--muted); margin-bottom: .5rem; }}
.sb-label {{ font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
             font-weight: 600; margin: 1.5rem 0 .35rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
.sb-label.first {{ border-top: none; padding-top: 0; margin-top: 1rem; }}

/* Hero */
.eyebrow {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 600; }}
.hero-title {{ font-size: 2.2rem; font-weight: 700; letter-spacing: -0.025em; color: var(--text);
               margin: .15rem 0 0; line-height: 1.15; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem; }}
.chip {{ font-size: .78rem; padding: .28rem .65rem; border: 1px solid var(--border); border-radius: 999px;
         color: var(--muted); background: var(--surface); white-space: nowrap; }}
.chip b {{ color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }}
.chip.accent {{ border-color: rgba(255,107,44,.45); color: var(--accent); background: rgba(255,107,44,.08); }}

/* KPI cards */
.kpis {{ display: grid; gap: .75rem; margin: 1.4rem 0 .6rem; }}
.kpis.c4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.kpis.c5 {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
.kpis.c3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.kpis.c1 {{ grid-template-columns: 1fr; margin-top: .4rem; }}
.legend.stack {{ flex-direction: column; gap: .55rem; margin: .4rem 0 0; }}
@media (max-width: 1000px) {{ .kpis.c4, .kpis.c5, .kpis.c3 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .95rem 1.1rem; }}
.kpi .label {{ font-size: .76rem; color: var(--muted); font-weight: 500; }}
.kpi .value {{ font-size: 1.6rem; font-weight: 650; color: var(--text); margin-top: .2rem;
               letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }}
.kpi .value small {{ font-size: .85rem; color: var(--muted); font-weight: 500; margin-left: .2rem; }}
.kpi .sub {{ font-size: .78rem; color: var(--muted); margin-top: .15rem; }}
.kpis.compact .kpi {{ padding: .75rem .95rem; }}
.kpis.compact .value {{ font-size: 1.2rem; }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: .35rem; vertical-align: middle; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ height: 2.7rem; padding: 0 .1rem; font-weight: 500; background: transparent; }}

/* Small helpers */
.section-title {{ font-size: 1rem; font-weight: 600; color: var(--text); margin: .2rem 0 .1rem; }}
.section-sub {{ font-size: .84rem; color: var(--muted); margin-bottom: .6rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: .8rem; color: var(--muted); margin: .3rem 0 0; }}
.legend .sw {{ display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: .4rem; vertical-align: middle; }}
.callout {{ border: 1px solid rgba(245,184,61,.35); background: rgba(245,184,61,.07); color: var(--text);
            border-radius: 10px; padding: .7rem .95rem; font-size: .86rem; margin-top: .8rem; }}
.foot {{ font-size: .76rem; color: var(--muted); margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
</style>
""",
    unsafe_allow_html=True,
)

# Matplotlib base style for the flow-field render
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
})

# Velocity colormap centred on freestream: slow = blue, freestream = neutral, fast = orange
VELOCITY_CMAP = LinearSegmentedColormap.from_list(
    "velocity",
    [(0.0, "#0B2550"), (0.28, "#2E6BD6"), (0.5, "#262C38"), (0.75, "#F0803A"), (1.0, "#FFE1B0")],
)


def kpi_grid(items, cols=4, compact=False):
    """Render a row of stat cards. items = [(label, value_html, sub_html), ...]"""
    cells = "".join(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in items
    )
    cls = f"kpis c{cols}" + (" compact" if compact else "")
    st.markdown(f'<div class="{cls}">{cells}</div>', unsafe_allow_html=True)


def sb_label(text, first=False):
    st.sidebar.markdown(f'<div class="sb-label{" first" if first else ""}">{text}</div>', unsafe_allow_html=True)


def sci(value):
    """Format a number as a.bc × 10ⁿ with unicode superscripts."""
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    exp = int(np.floor(np.log10(abs(value)))) if value else 0
    return f"{value / 10**exp:.2f} × 10{str(exp).translate(sup)}"


def style_plotly(fig, height=400, xtitle="", ytitle=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11.5, color=TEXT)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, size=12)),
        hovermode="x unified",
    )
    fig.update_layout(legend_traceorder="normal")
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(size=11),
                title_font=dict(size=12, color=MUTED))
    fig.update_xaxes(title_text=xtitle, **axis)
    fig.update_yaxes(title_text=ytitle, **axis)
    return fig


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# ------------------------------------------------------------------------------
# 2-6. PHYSICS: GEOMETRY, PANEL METHOD, FLOW FIELD, AERO MODEL, CAD EXPORT
#      (unchanged from the original solver)
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
# 7. SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
NACA_PRESETS = {
    "NACA 2412 · General aviation": "2412",
    "NACA 0012 · Symmetric (tails, fins, rotors)": "0012",
    "NACA 4412 · High-lift trainer / STOL": "4412",
    "NACA 2415 · Transport wing": "2415",
    "NACA 0009 · Thin aerobatic": "0009",
    "NACA 0018 · Thick wing root": "0018",
    "NACA 6412 · High-camber glider": "6412",
    "NACA 4415 · Endurance UAV": "4415",
    "NACA 0006 · Thin stabiliser": "0006",
    "NACA 2408 · Pylon racer": "2408",
    "NACA 1408 · Low-drag high-speed": "1408",
    "Custom": None,
}
CUSTOM = "Custom"

# Session-state defaults for two-way sync between preset / code / sliders
st.session_state.setdefault("m_slider", 2.0)
st.session_state.setdefault("p_slider", 40.0)
st.session_state.setdefault("t_slider", 12.0)
st.session_state.setdefault("naca_text_in", "2412")
st.session_state.setdefault("naca_preset_choice", "NACA 2412 · General aviation")
if st.session_state.naca_preset_choice not in NACA_PRESETS:
    st.session_state.naca_preset_choice = CUSTOM


def _match_preset(code):
    for name, p_code in NACA_PRESETS.items():
        if p_code == code:
            return name
    return CUSTOM


def on_naca_preset_change():
    code = NACA_PRESETS.get(st.session_state.naca_preset_choice)
    if code:
        m = float(code[0])
        st.session_state.naca_text_in = code
        st.session_state.m_slider = m
        st.session_state.p_slider = float(code[1]) * 10.0 if m > 0 else 40.0
        st.session_state.t_slider = float(code[2:])


def on_naca_text_change():
    raw = st.session_state.naca_text_in.strip().upper().replace("NACA", "").replace(" ", "").replace("-", "")
    if len(raw) == 4 and raw.isdigit():
        m = float(raw[0])
        p = float(raw[1]) * 10.0 if m > 0 else 40.0
        st.session_state.m_slider = m
        st.session_state.p_slider = max(10.0, min(90.0, p))
        st.session_state.t_slider = max(5.0, min(30.0, float(raw[2:])))
        st.session_state.naca_preset_choice = _match_preset(raw)


def on_naca_slider_change():
    m = int(round(st.session_state.m_slider))
    p = int(round(st.session_state.p_slider / 10.0))
    t = int(round(st.session_state.t_slider))
    code = f"00{t:02d}" if m == 0 else f"{m}{p}{t:02d}"
    st.session_state.naca_text_in = code
    st.session_state.naca_preset_choice = _match_preset(code)


with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="mark"></span>Airfoil CFD Studio</div>'
        '<div class="brand-sub">2D panel-method aerodynamics</div>',
        unsafe_allow_html=True,
    )

    # --- Airfoil -----------------------------------------------------------
    sb_label("Airfoil", first=True)
    st.selectbox(
        "Preset",
        options=list(NACA_PRESETS.keys()),
        key="naca_preset_choice",
        on_change=on_naca_preset_change,
    )
    st.text_input(
        "NACA 4-digit code",
        key="naca_text_in",
        on_change=on_naca_text_change,
        max_chars=9,
        help="1st digit = max camber %, 2nd = camber position (tenths of chord), last two = thickness %.",
    )
    clean_code = st.session_state.naca_text_in.strip().upper().replace("NACA", "").replace(" ", "").replace("-", "")
    if len(clean_code) == 4 and clean_code.isdigit():
        m_dec, p_dec, t_dec = int(clean_code[0]), int(clean_code[1]) * 10, int(clean_code[2:])
        camber_desc = f"{m_dec}% camber at {p_dec}% chord" if m_dec > 0 else "Symmetric"
        st.caption(f"{camber_desc} · {t_dec}% thick")
    elif clean_code:
        st.caption(":orange[Enter exactly 4 digits, e.g. 2412]")

    with st.expander("Fine-tune geometry"):
        m_pct = st.slider("Max camber", 0.0, 9.0, step=0.5, key="m_slider",
                          format="%.1f%%", on_change=on_naca_slider_change)
        p_pct = st.slider("Camber position", 10.0, 90.0, step=10.0, key="p_slider",
                          format="%.0f%% chord", on_change=on_naca_slider_change)
        t_pct = st.slider("Thickness", 5.0, 30.0, step=1.0, key="t_slider",
                          format="%.0f%%", on_change=on_naca_slider_change)

    # --- Flow --------------------------------------------------------------
    sb_label("Flow conditions")
    alpha_deg = st.slider("Angle of attack", -10.0, 20.0, 4.0, 0.5, format="%.1f°")
    U_inf = st.slider("Freestream velocity", 5.0, 60.0, 25.0, 1.0, format="%.0f m/s")
    st.caption(f"{U_inf * 3.6:.0f} km/h · sea-level air")

    # --- Configuration -----------------------------------------------------
    sb_label("Configuration")
    enable_flap = st.toggle("Trailing-edge flap", value=False)
    flap_deflection, flap_hinge = 0.0, 0.75
    if enable_flap:
        flap_deflection = st.slider("Flap deflection", -15.0, 35.0, 15.0, 1.0, format="%.0f°")
        flap_hinge = st.slider("Hinge position", 0.60, 0.85, 0.75, 0.05, format="%.2f c")

    inverted_wing = st.toggle("Inverted wing (downforce)", value=False,
                              help="Flip the profile as on a racing car wing so it pushes down instead of up.")
    enable_ground_effect = st.toggle("Ground effect", value=False,
                                     help="Add a track surface below the wing and model the Venturi suction under it.")
    ground_clearance = 0.30
    if enable_ground_effect:
        ground_clearance = st.slider("Ground clearance", 0.10, 1.00, 0.25, 0.05, format="%.2f c")

    # --- Display -----------------------------------------------------------
    sb_label("Display")
    quality = st.segmented_control("Render quality", ["Fast", "High detail"], default="Fast",
                                   selection_mode="single")
    cmap_label = st.selectbox("Colour map", ["Velocity (diverging)", "Turbo", "Viridis", "Plasma", "Inferno"])
    show_wake = st.toggle("Show viscous wake", value=True)

if quality == "High detail":
    grid_nx, grid_ny, streamline_density, fig_dpi = 140, 100, 1.1, 150
else:
    grid_nx, grid_ny, streamline_density, fig_dpi = 70, 50, 0.65, 110

rho_air, nu_air, chord_len = 1.225, 1.5e-5, 1.0


# ------------------------------------------------------------------------------
# 8. SOLVE
# ------------------------------------------------------------------------------
xb, yb, xu, yu, xl, yl, xc_line, yc_line, designation = generate_airfoil_geometry(
    m_pct, p_pct, t_pct,
    flap_enabled=enable_flap, flap_deg=flap_deflection, flap_hinge=flap_hinge,
    inverted=inverted_wing, chord=chord_len, num_points=70,
)

aero = calculate_aero_performance(
    alpha_deg, m_pct, p_pct, t_pct,
    flap_enabled=enable_flap, flap_deg=flap_deflection, flap_hinge=flap_hinge,
    inverted=inverted_wing, ground_effect=enable_ground_effect, ground_h=ground_clearance,
    U_inf=U_inf, rho=rho_air, nu=nu_air,
)

# Rotate by AoA about the quarter chord (inverted wing pitches the other way)
alpha_rad = np.radians(alpha_deg)
cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
if inverted_wing:
    xb_rot = 0.25 + (xb - 0.25) * cos_a - yb * sin_a
    yb_rot = (xb - 0.25) * sin_a + yb * cos_a
else:
    xb_rot = 0.25 + (xb - 0.25) * cos_a + yb * sin_a
    yb_rot = -(xb - 0.25) * sin_a + yb * cos_a

panel_res = solve_panel_method(xb_rot, yb_rot, alpha_deg=0.0, U_inf=U_inf)

ground_y_coord = -0.5
if enable_ground_effect:
    ground_y_coord = np.min(yb_rot) - ground_clearance

Xg, Yg, speed_ratio_2d, u_2d, v_2d, x_te, y_te, downwash_ang = evaluate_flow_field(
    xb_rot, yb_rot, panel_res['q'], panel_res['gamma'], panel_res['L'], panel_res['phi'],
    U_inf=U_inf, alpha_deg=alpha_deg, Cd_val=aero['Cd'],
    nx=grid_nx, ny=grid_ny, enable_wake=show_wake,
    ground_effect=enable_ground_effect, ground_y=ground_y_coord,
)

stag_idx = np.argmax(panel_res['Cp'])
x_stag, y_stag = panel_res['xc'][stag_idx], panel_res['yc'][stag_idx]
cp_stag = panel_res['Cp'][stag_idx]

# Surface pressure in the body frame (used by the Pressure and Export tabs)
unrot_res = solve_panel_method(xb, yb, alpha_deg=alpha_deg, U_inf=U_inf)
xc_unrot, Cp_unrot = unrot_res['xc'], unrot_res['Cp']
half = (len(xb) - 1) // 2
sort_u = np.argsort(xc_unrot[:half])
x_u_s, cp_u_s = xc_unrot[:half][sort_u], Cp_unrot[:half][sort_u]
sort_l = np.argsort(xc_unrot[half:])
x_l_s, cp_l_s = xc_unrot[half:][sort_l], Cp_unrot[half:][sort_l]
cp_l_interp = np.interp(x_u_s, x_l_s, cp_l_s)
Cn_approx = integrate.trapezoid(cp_l_interp - cp_u_s, x_u_s)
Cm_c4_approx = integrate.trapezoid((0.25 - x_u_s) * (cp_l_interp - cp_u_s), x_u_s)
x_cp_approx = (integrate.trapezoid(x_u_s * (cp_l_interp - cp_u_s), x_u_s) / Cn_approx
               if abs(Cn_approx) > 0.05 else None)
if x_cp_approx is not None and not (0.0 <= x_cp_approx <= 1.0):
    x_cp_approx = None  # only meaningful when it lands on the chord

# Polars
alpha_polars = np.linspace(-10.0, 20.0, 31)
aero_polars = calculate_aero_performance(
    alpha_polars, m_pct, p_pct, t_pct,
    flap_enabled=enable_flap, flap_deg=flap_deflection, flap_hinge=flap_hinge,
    inverted=inverted_wing, ground_effect=enable_ground_effect, ground_h=ground_clearance,
    U_inf=U_inf, rho=rho_air, nu=nu_air,
)

naca_name = designation.split(" (")[0].split(" [")[0]
file_stub = designation.replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("°", "deg")


# ------------------------------------------------------------------------------
# 9. HEADER & KEY RESULTS
# ------------------------------------------------------------------------------
chips = [
    '<span class="chip accent">Inverted · downforce</span>' if inverted_wing
    else '<span class="chip">Lifting profile</span>',
    f'<span class="chip">α <b>{alpha_deg:+.1f}°</b></span>',
    f'<span class="chip">U∞ <b>{U_inf:.0f} m/s</b></span>',
    f'<span class="chip">Re <b>{sci(aero["Re"])}</b></span>',
]
if enable_flap:
    chips.append(f'<span class="chip">Flap <b>{flap_deflection:+.0f}°</b> at {flap_hinge:.2f}c</span>')
if enable_ground_effect:
    chips.append(f'<span class="chip">Ground <b>h/c {ground_clearance:.2f}</b></span>')

st.markdown(
    f'<div class="eyebrow">Airfoil analysis</div>'
    f'<div class="hero-title">{naca_name}</div>'
    f'<div class="chips">{"".join(chips)}</div>',
    unsafe_allow_html=True,
)

stall_margin = aero['alpha_stall_pos'] - alpha_deg
if stall_margin <= 0:
    flow_state = f'<span class="dot" style="background:{RED}"></span>Stalled'
elif stall_margin < 3:
    flow_state = f'<span class="dot" style="background:{AMBER}"></span>Close to stall'
else:
    flow_state = f'<span class="dot" style="background:{GREEN}"></span>Attached flow'

if inverted_wing:
    force_kpi = ("Downforce coefficient", f"{aero['Downforce_Coeff']:+.3f}",
                 f"{aero['downforce_val']:,.0f} N per metre span")
    eff_label = "Downforce / drag"
else:
    force_kpi = ("Lift coefficient", f"{aero['Cl']:+.3f}", f"{aero['lift_force']:,.0f} N per metre span")
    eff_label = "Lift / drag"

kpi_grid([
    force_kpi,
    ("Drag coefficient", f"{aero['Cd']:.4f}", f"{aero['drag_force']:,.1f} N per metre span"),
    (eff_label, f"{abs(aero['L_D']):.1f}", flow_state),
    ("Stall margin", f"{stall_margin:+.1f}°", f"Stall onset at {aero['alpha_stall_pos']:.1f}°"),
])


# ------------------------------------------------------------------------------
# 10. TABS
# ------------------------------------------------------------------------------
tab_flow, tab_cp, tab_polars, tab_export, tab_theory = st.tabs(
    ["Flow field", "Pressure", "Polars", "Export", "Theory"]
)

# ---- Flow field --------------------------------------------------------------
with tab_flow:
    col_plot, col_side = st.columns([3.3, 1], gap="medium")
    with col_plot:
        fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=fig_dpi)
        fig1.patch.set_facecolor(BG)
        ax1.set_facecolor(BG)

        levels = np.linspace(0.0, 1.85, 40)
        if cmap_label.startswith("Velocity"):
            cf = ax1.contourf(Xg, Yg, speed_ratio_2d, levels=levels, cmap=VELOCITY_CMAP,
                              norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=1.85), extend="both")
        else:
            cf = ax1.contourf(Xg, Yg, speed_ratio_2d, levels=levels, cmap=cmap_label.lower(), extend="both")

        ax1.streamplot(Xg, Yg, u_2d, v_2d, color=(1, 1, 1, 0.32), density=streamline_density,
                       linewidth=0.6, arrowsize=0.6)

        ax1.fill(xb_rot, yb_rot, color=SURFACE, zorder=10)
        ax1.plot(xb_rot, yb_rot, color=TEXT, lw=1.3, zorder=11)
        ax1.plot(x_stag, y_stag, "o", color=ACCENT, markersize=6, markeredgecolor=BG,
                 markeredgewidth=1.2, zorder=15)

        if show_wake:
            x_wake = np.linspace(x_te, 1.75, 40)
            y_wake = y_te - (x_wake - x_te) * np.tan(downwash_ang)
            ax1.plot(x_wake, y_wake, "--", color="#C9B6FF", lw=1.0, alpha=0.7, zorder=12)

        if enable_ground_effect:
            ax1.fill_between([-0.45, 1.75], -0.75, ground_y_coord, color="#12151B", zorder=14)
            ax1.plot([-0.45, 1.75], [ground_y_coord, ground_y_coord], color=AMBER, lw=1.8, zorder=15)

        ax1.set_xlim(-0.45, 1.75)
        ax1.set_ylim(-0.75, 0.75)
        ax1.set_aspect("equal")
        ax1.set_xlabel("x / c", labelpad=4, fontsize=10.5)
        ax1.set_ylabel("y / c", labelpad=4, fontsize=10.5)
        ax1.tick_params(length=0, labelsize=10)
        for spine in ax1.spines.values():
            spine.set_visible(False)

        cax = ax1.inset_axes([0.0, -0.13, 1.0, 0.022])
        cbar = fig1.colorbar(cf, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1.0, 1.5])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=10, colors=MUTED)
        cbar.set_label("Local speed ÷ freestream speed", fontsize=10.5, color=MUTED, labelpad=4)

        st.pyplot(fig1, clear_figure=True)
        plt.close("all")

    with col_side:
        legend_items = [
            f'<span><span class="dot" style="background:{ACCENT}"></span>Stagnation point (Cp {cp_stag:.2f})</span>',
            '<span><span class="sw" style="background:rgba(255,255,255,.5)"></span>Streamlines</span>',
        ]
        if show_wake:
            legend_items.append('<span><span class="sw" style="background:#C9B6FF"></span>Wake centreline</span>')
        if enable_ground_effect:
            legend_items.append(f'<span><span class="sw" style="background:{AMBER}"></span>Track surface</span>')
        st.markdown(f'<div class="legend stack">{"".join(legend_items)}</div>', unsafe_allow_html=True)

        kpi_grid([
            ("Peak local speed", f"{np.nanmax(speed_ratio_2d):.2f}<small>× U∞</small>",
             f"{np.nanmax(speed_ratio_2d) * U_inf:.1f} m/s"),
            ("Stagnation point", f"{x_stag:.3f}<small>x/c</small>", "Where the flow splits"),
            ("Grid", f"{grid_nx} × {grid_ny}", f"{quality or 'Fast'} render"),
        ], cols=1, compact=True)

        if enable_ground_effect:
            st.markdown(
                f'<div class="callout"><b>Ground effect on.</b> The narrowed channel under the wing speeds the flow up to '
                f'{np.nanmax(speed_ratio_2d):.2f}× freestream, adding '
                f'<b>{(aero["ground_factor"] - 1.0) * 100:+.1f}%</b> vertical force.</div>',
                unsafe_allow_html=True,
            )

# ---- Pressure ----------------------------------------------------------------
with tab_cp:
    st.markdown('<div class="section-title">Surface pressure distribution</div>'
                '<div class="section-sub">Y-axis is flipped (aero convention), so suction points up. '
                'The shaded gap between the curves is what makes the force.</div>', unsafe_allow_html=True)

    upper_label = "Upper surface" + (" (pressure side)" if inverted_wing else "")
    lower_label = "Lower surface" + (" (suction side)" if inverted_wing else "")
    min_cp_idx = int(np.argmin(cp_u_s))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_u_s, y=cp_l_interp, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=x_u_s, y=cp_u_s, mode="lines", name=upper_label,
                              line=dict(color=BLUE, width=2.4), fill="tonexty",
                              fillcolor="rgba(77,163,255,0.10)",
                              hovertemplate="Cp %{y:.3f}<extra>Upper</extra>"))
    fig2.add_trace(go.Scatter(x=x_l_s, y=cp_l_s, mode="lines", name=lower_label,
                              line=dict(color=RED, width=2.4),
                              hovertemplate="Cp %{y:.3f}<extra>Lower</extra>"))
    fig2.add_trace(go.Scatter(x=[x_u_s[min_cp_idx]], y=[cp_u_s[min_cp_idx]], mode="markers",
                              name="Peak suction", marker=dict(color=ACCENT, size=9, line=dict(color=BG, width=1.5)),
                              hovertemplate="Peak suction Cp %{y:.2f}<extra></extra>"))
    fig2.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig2.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dot"),
                   annotation_text="Stagnation (Cp = 1)", annotation_font=dict(size=11, color=MUTED),
                   annotation_position="bottom right")
    style_plotly(fig2, height=430, xtitle="Chord position  x / c", ytitle="Pressure coefficient  Cp")
    fig2.update_yaxes(autorange="reversed")
    fig2.update_xaxes(range=[-0.02, 1.02])
    st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)

    kpi_grid([
        ("Peak suction", f"{cp_u_s[min_cp_idx]:.2f}", f"at x/c = {x_u_s[min_cp_idx]:.2f}"),
        ("Stagnation pressure", f"{np.max(Cp_unrot):.2f}", "Theoretical max 1.00"),
        ("Centre of pressure", f"{x_cp_approx:.3f}<small>x/c</small>" if x_cp_approx is not None else "—",
         f"{x_cp_approx - 0.25:+.3f} from quarter chord" if x_cp_approx is not None else "Not defined at this load"),
        ("Pitching moment", f"{Cm_c4_approx:+.4f}", "About the quarter chord"),
    ], cols=4, compact=True)

# ---- Polars ------------------------------------------------------------------
with tab_polars:
    st.markdown('<div class="section-title">Performance polars</div>'
                '<div class="section-sub">Viscous model with stall, swept from −10° to +20°. '
                'The orange dot is your current setting.</div>', unsafe_allow_html=True)

    y_polar = aero_polars['Cl'] if not inverted_wing else -aero_polars['Cl']
    y_curr = aero['Cl'] if not inverted_wing else aero['Downforce_Coeff']
    y_name = "Lift coefficient  Cl" if not inverted_wing else "Downforce coefficient  Cdf"
    y_short = "Cl" if not inverted_wing else "Cdf"
    linear_slope = aero['cla_deg'] * (alpha_polars - aero['alpha_0_total'])
    if inverted_wing:
        linear_slope = -linear_slope
    max_ld_idx = int(np.argmax(np.abs(aero_polars['L_D'])))

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        fig3a = go.Figure()
        fig3a.add_trace(go.Scatter(x=alpha_polars, y=linear_slope, mode="lines", name="Inviscid (linear)",
                                   line=dict(color=MUTED, width=1.3, dash="dash"),
                                   hovertemplate="%{y:.3f}<extra>Linear</extra>"))
        fig3a.add_trace(go.Scatter(x=alpha_polars, y=y_polar, mode="lines", name="Viscous with stall",
                                   line=dict(color=BLUE, width=2.6, shape="spline"),
                                   hovertemplate="%{y:.3f}<extra>Viscous</extra>"))
        fig3a.add_trace(go.Scatter(x=[alpha_deg], y=[y_curr], mode="markers", name="Current",
                                   marker=dict(color=ACCENT, size=11, line=dict(color=BG, width=2)),
                                   hovertemplate="%{y:.3f}<extra>Current</extra>"))
        fig3a.add_vline(x=aero['alpha_stall_pos'], line=dict(color=AMBER, width=1.2, dash="dot"),
                        annotation_text=f"Stall {aero['alpha_stall_pos']:.1f}°",
                        annotation_font=dict(size=11, color=AMBER), annotation_position="top left")
        style_plotly(fig3a, height=380, xtitle="Angle of attack  α (°)", ytitle=y_name)
        fig3a.update_xaxes(range=[-10.5, 20.5])
        st.plotly_chart(fig3a, width="stretch", config=PLOTLY_CONFIG)

    with col_b:
        fig3b = go.Figure()
        fig3b.add_trace(go.Scatter(x=aero_polars['Cd'], y=y_polar, mode="lines", name="Drag polar",
                                   line=dict(color=BLUE, width=2.6, shape="spline"),
                                   customdata=alpha_polars,
                                   hovertemplate=f"Cd %{{x:.4f}} · {y_short} %{{y:.3f}} · α %{{customdata:.0f}}°<extra></extra>"))
        fig3b.add_trace(go.Scatter(x=[aero_polars['Cd'][max_ld_idx]], y=[y_polar[max_ld_idx]], mode="markers",
                                   name=f"Best L/D ({abs(aero_polars['L_D'][max_ld_idx]):.1f})",
                                   marker=dict(color=AMBER, size=9, symbol="diamond"),
                                   hovertemplate="Best L/D<extra></extra>"))
        fig3b.add_trace(go.Scatter(x=[aero['Cd']], y=[y_curr], mode="markers", name="Current",
                                   marker=dict(color=ACCENT, size=11, line=dict(color=BG, width=2)),
                                   hovertemplate="Current<extra></extra>"))
        style_plotly(fig3b, height=380, xtitle="Drag coefficient  Cd", ytitle=y_name)
        fig3b.update_layout(hovermode="closest")
        st.plotly_chart(fig3b, width="stretch", config=PLOTLY_CONFIG)

    kpi_grid([
        ("Zero-lift angle", f"{aero['alpha_0_total']:.2f}°", "Thin-airfoil theory"),
        ("Lift slope", f"{aero['cla_deg']:.4f}<small>/°</small>", f"{aero['cla_deg'] * 180 / np.pi:.2f} per radian"),
        ("Max vertical force", f"{np.max(y_polar):.2f}", f"at α = {alpha_polars[np.argmax(y_polar)]:.0f}°"),
        ("Minimum drag", f"{np.min(aero_polars['Cd']):.4f}", f"Skin friction Cd0 = {aero['Cd0']:.4f}"),
        ("Best efficiency", f"{np.max(np.abs(aero_polars['L_D'])):.1f}", f"at α = {alpha_polars[max_ld_idx]:.0f}°"),
    ], cols=5, compact=True)

# ---- Export ------------------------------------------------------------------
with tab_export:
    st.markdown('<div class="section-title">Export geometry and results</div>'
                '<div class="section-sub">Ready for SolidWorks, Fusion 360, AutoCAD, slicers and laser cutters. '
                'Exports use the profile shown, including flap and inversion.</div>', unsafe_allow_html=True)

    SCALES = {
        "Normalised · chord 1.0 m": (1.0, "m"),
        "Full scale · chord 1000 mm": (1000.0, "mm"),
        "Wind-tunnel model · chord 200 mm": (200.0, "mm"),
        "3D-print test piece · chord 150 mm": (150.0, "mm"),
    }
    scale_mode = st.selectbox("Scale", list(SCALES.keys()), index=1)
    scale_mult, unit_label = SCALES[scale_mode]

    dxf_content = generate_dxf_r12(xb, yb, scale=scale_mult, layer="AIRFOIL_PROFILE")
    cad_csv_content = generate_cad_xyz_csv(xb, yb, scale=scale_mult)

    csv_coords = io.StringIO()
    csv_coords.write(f"x_{unit_label},y_upper_{unit_label},y_lower_{unit_label},y_camber_{unit_label}\n")
    for i in range(len(xu)):
        csv_coords.write(f"{xu[i]*scale_mult:.6f},{yu[i]*scale_mult:.6f},{yl[i]*scale_mult:.6f},{yc_line[i]*scale_mult:.6f}\n")

    csv_cp = io.StringIO()
    csv_cp.write("x_upper,Cp_upper,x_lower,Cp_lower\n")
    for i in range(len(x_u_s)):
        csv_cp.write(f"{x_u_s[i]:.6f},{cp_u_s[i]:.6f},{x_l_s[i]:.6f},{cp_l_s[i]:.6f}\n")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**CAD profile**")
            st.caption("Closed outline for sketches, extrusions and cutting.")
            st.download_button("DXF · AutoCAD R12", data=dxf_content,
                               file_name=f"{file_stub}_{unit_label}.dxf", mime="application/dxf",
                               icon=":material/download:", type="primary", width="stretch")
            st.download_button("XYZ curve CSV · SolidWorks / Fusion", data=cad_csv_content,
                               file_name=f"{file_stub}_XYZ_{unit_label}.csv", mime="text/csv",
                               icon=":material/download:", width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("**Data**")
            st.caption("Raw numbers for spreadsheets, MATLAB or Python.")
            st.download_button("Airfoil coordinates CSV", data=csv_coords.getvalue(),
                               file_name=f"{file_stub}_coordinates.csv", mime="text/csv",
                               icon=":material/table_chart:", width="stretch")
            st.download_button(f"Surface Cp at α = {alpha_deg:.1f}° CSV", data=csv_cp.getvalue(),
                               file_name=f"{file_stub}_Cp_alpha_{alpha_deg:.1f}.csv", mime="text/csv",
                               icon=":material/table_chart:", width="stretch")

# ---- Theory ------------------------------------------------------------------
with tab_theory:
    st.markdown("""
#### Geometry
NACA 4-digit equations with cosine spacing along the chord $c$:

$$\\beta \\in [0, \\pi], \\quad x = \\tfrac{c}{2}(1 - \\cos\\beta)$$

With a plain flap, everything aft of the hinge $x_h$ is rotated by $\\delta_f$ about $(x_h, y_c(x_h))$:

$$x' = x_h + (x - x_h)\\cos\\delta_f + (y - y_h)\\sin\\delta_f, \\qquad y' = y_h - (x - x_h)\\sin\\delta_f + (y - y_h)\\cos\\delta_f$$

#### Hess–Smith panel method
The surface is split into $N$ flat panels, each with a source strength $q_j$, plus one shared vortex strength $\\gamma$.

- **No flow through the surface:** $\\vec{V}_i \\cdot \\vec{n}_i = 0$ at each panel midpoint
- **Kutta condition:** flow leaves the trailing edge smoothly, $V_{t,1} + V_{t,N} = 0$
- **Surface pressure:** $C_p = 1 - (V_t / U_\\infty)^2$

#### Viscous model
Lift follows thin-airfoil theory, blended into a separated-flow model past stall with a sigmoid. Drag combines turbulent flat-plate skin friction, a form factor, profile drag growth with $C_l$, and separation drag.

#### Motorsport: inverted wing and ground effect
- **Inverted wing:** the profile is mirrored ($y \\to -y$), so suction moves underneath and the wing pushes down ($C_{df} = -C_L$).
- **Ground effect:** near the track the channel under the wing narrows, speeding up the flow and deepening suction:

$$K_{\\text{ground}} = 1 + \\frac{0.12}{(h/c + 0.05)^{0.85}}$$

Below $h/c \\approx 0.12$ the boundary layer chokes the channel and the gain drops off (diffuser stall).

*This is a fast teaching and concept tool, not a replacement for full RANS CFD or wind-tunnel data.*
""")

st.markdown(
    '<div class="foot">Airfoil CFD Studio · Hess–Smith panel method with empirical viscous corrections · '
    'NumPy, SciPy, Matplotlib, Plotly</div>',
    unsafe_allow_html=True,
)
