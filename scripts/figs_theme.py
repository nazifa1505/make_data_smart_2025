# scripts/figs_theme.py
import matplotlib.pyplot as plt

def apply_simple_theme():
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "#cccccc",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.autolayout": True,
        # bakgrunn
        "figure.facecolor": "#fcf6ee",
        "axes.facecolor": "#fcf6ee",
        # palett for plott
        "axes.prop_cycle": plt.cycler(color=["#ff8c45", "#ffd865", "#97d2ec"]),
    })
