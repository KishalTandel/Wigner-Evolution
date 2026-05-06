import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, factorial
import matplotlib.animation as animation
from IPython.display import HTML
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap


# Custom DARK colormap (0 = #11111a)
cmap = LinearSegmentedColormap.from_list(
    "dark_diverging",
    [
        "#FF1493",   # negative (pink)
        "#1a1a26",   # ZERO (background)
        "#FF0000"    # positive (red)
    ],
    N=256
)


# Parameters
hbar = 1.0
m = 1.0
omega = 1.0
x0 = np.sqrt(hbar/(m*omega))


# Phase space grid
N = 150
x = np.linspace(-4*x0, 4*x0, N)
p = np.linspace(-4*hbar/x0, 4*hbar/x0, N)
X, P = np.meshgrid(x, p)


# Wigner matrix element
def W_mn(m, n, xi, pi_, r2):
    if n >= m:
        pref = (2 * (-1)**m) / (np.pi * hbar)
        pref *= np.sqrt(factorial(m) / factorial(n))

        z = 2 * (xi + 1j*pi_)
        lag = eval_genlaguerre(m, n-m, 4*r2)

        return pref * np.exp(-2*r2) * (z**(n-m)) * lag
    else:
        return np.conj(W_mn(n, m, xi, pi_, r2))


# Build Wigner function
def Wigner_from_rho(rho, xi, pi_, r2):
    W = np.zeros_like(xi, dtype=complex)
    dim = rho.shape[0]

    for m in range(dim):
        for n in range(dim):
            if abs(rho[m, n]) > 1e-12:
                W += rho[m, n] * W_mn(m, n, xi, pi_, r2)

    return np.real(W)


# States
def pure_fock(n, dim=6):
    rho = np.zeros((dim, dim), dtype=complex)
    rho[n, n] = 1.0
    return rho

def superposition_state(coeffs):
    coeffs = np.array(coeffs, dtype=complex)
    return np.outer(coeffs, np.conj(coeffs))

def mixed_state(states, probs):
    rho = np.zeros_like(states[0])
    for s, p in zip(states, probs):
        rho += p * s
    return rho


# Time evolution (rotation)
def Wigner_time(rho, t):
    X_t = X*np.cos(omega*t) + (P/(m*omega))*np.sin(omega*t)
    P_t = P*np.cos(omega*t) - (m*omega*X)*np.sin(omega*t)

    xi = X_t/x0
    pi_ = P_t*x0/hbar
    r2 = xi**2 + pi_**2

    return Wigner_from_rho(rho, xi, pi_, r2)


# Animation
def animate_wigner(rho, title):
    fig, ax = plt.subplots(figsize=(8,6))
    
    # DARK BACKGROUND
    fig.patch.set_facecolor('#11111a')
    ax.set_facecolor('#11111a')

    # Colorbar inside axis
    cax = inset_axes(
        ax,
        width="5%",
        height="85%",
        loc='center right',
        bbox_to_anchor=(-0.08, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    def update(frame):
        ax.clear()
        cax.clear()

        ax.set_facecolor('#11111a')

        t = frame * 0.1
        W = Wigner_time(rho, t)

        vmax = np.max(np.abs(W))

        cf = ax.contourf(
        X, P, W, 120,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        antialiased=False   # important
        )

        # Colorbar
        cb = fig.colorbar(cf, cax=cax)
        cax.set_facecolor('#11111a')
        cb.outline.set_edgecolor('#e6e6e6')
        cb.ax.tick_params(labelsize=8, colors='#e6e6e6')

        # Title
        ax.text(
            0.5, 0.97,
            f"{title}   t={t:.2f}",
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=11,
            color='#e6e6e6'
        )

        # Axis labels
        ax.text(0.5, 0.06, "x",
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=10, color='#e6e6e6')

        ax.text(0.06, 0.5, "p",
                transform=ax.transAxes,
                ha='left', va='center',
                rotation=90,
                fontsize=10, color='#e6e6e6')

        # Ticks
        ax.tick_params(direction='in', colors='#e6e6e6', labelsize=8)
        ax.tick_params(axis='x', pad=-15)
        ax.tick_params(axis='y', pad=-20)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(p.min(), p.max())

        # Clean ticks
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticks([t for t in xticks if abs(t) < 4])
        ax.set_yticks([t for t in yticks if abs(t) < 4])

        # Spines
        for spine in ax.spines.values():
            spine.set_color('#e6e6e6')

    ani = animation.FuncAnimation(
        fig, update,
        frames=180,
        interval=100
    )

    return ani


# Create states
rho_super = superposition_state([1/np.sqrt(2), 1/np.sqrt(2)])


# Display with DARK container
ani = animate_wigner(rho_super, "")

html = ani.to_jshtml()

styled_html = f"""
<div style="
    background-color:#11111a;
    color:#e6e6e6;
    padding:10px;
    border-radius:10px;
">
{html}
</div>
"""

HTML(styled_html)
