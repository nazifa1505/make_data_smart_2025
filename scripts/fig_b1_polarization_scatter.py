import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figs_theme import apply_simple_theme

COL_NEG = "#ff8c45"; COL_NEU = "#ffd865"; COL_POS = "#97d2ec"
TOP_K = 8  # hvor mange spørsmål som merkes

def read_parties():
    with open("figs/parties.txt", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def main():
    apply_simple_theme()
    parties = read_parties()
    df = pd.read_parquet("figs/nrk_h.parquet")
    df["mean"] = df[parties].mean(axis=1)   # μ
    df["var"]  = df[parties].var(axis=1)    # σ²
    df["std"]  = df[parties].std(axis=1)    # σ

    def pick_color(m): return COL_NEG if m < -0.2 else (COL_POS if m > 0.2 else COL_NEU)
    colors = df["mean"].apply(pick_color)
    sizes  = 70 + (df["var"] - df["var"].min())/(df["var"].max()-df["var"].min()+1e-6) * 240

    med_var = float(df["var"].median()); med_std = float(df["std"].median())

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(df["mean"], df["var"], s=sizes, c=colors,
               edgecolor="black", linewidth=0.25, alpha=0.90, zorder=2)

    ax.axvline(0, color="#888", ls="--", lw=1)
    ax.axhline(med_var, color="#888", ls="--", lw=1)

    ax.set_title("Spørsmål som splitter: retning (x) vs. uenighet (y) – NRK")
    ax.set_xlabel("Retning (snitt, μ)  ← negativ | positiv →")
    ax.set_ylabel("Uenighet (varians, σ²)")

    # Medianboks
    ax.text(0.015, 0.985, f"Median uenighet: σ²={med_var:.2f}  (σ={med_std:.2f})",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fcf6ee", ec="#cccccc"))

    # Nummerer kun topp-K bobler (mest polariserende)
    top = df.nlargest(TOP_K, "var").copy().reset_index(drop=True)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        ax.text(r["mean"], r["var"], str(i), ha="center", va="center",
                fontsize=11, fontweight="bold", color="black", zorder=3)

    # Legende
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', label='Heller negativt', markerfacecolor=COL_NEG, markeredgecolor='black', markersize=10),
        Line2D([0],[0], marker='o', color='w', label='Nøytral',        markerfacecolor=COL_NEU, markeredgecolor='black', markersize=10),
        Line2D([0],[0], marker='o', color='w', label='Heller positivt',markerfacecolor=COL_POS, markeredgecolor='black', markersize=10),
    ], loc="upper left")

    fig.tight_layout()
    fig.savefig("figs/B1_polarization_scatter.png", dpi=220)

    # lag CSV for liste-sliden
    top_out = top[["SpmID", "Spm", "mean", "var"]].copy()
    top_out.to_csv("figs/B_polarization_top.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
