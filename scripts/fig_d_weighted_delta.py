# scripts/fig_d_weighting_before_after.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from figs_theme import apply_simple_theme

COLORS = ["#ff8c45", "#ffd865", "#97d2ec"]
NQ = 30
BOOST_CATEGORY = "Barn og familie"
WEIGHT = 2.0

def load_tv2():
    if os.path.exists("figs/tv2_full.parquet"):
        return pd.read_parquet("figs/tv2_full.parquet")
    return pd.read_csv("figs/tv2_full.csv")

def read_parties():
    with open("figs/parties.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    apply_simple_theme()
    parties = read_parties()
    tv2 = load_tv2()

    sel = tv2.head(NQ).copy()
    base = sel[parties].sum()
    weights = sel["Kategori"].apply(lambda k: WEIGHT if k == BOOST_CATEGORY else 1.0)
    weighted = (sel[parties].multiply(weights.values, axis=0)).sum()

    df = pd.DataFrame({"Før": base, "Etter": weighted})
    df = df.sort_values("Før", ascending=False)

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, df["Før"], width, label="Før", color=COLORS[0])
    bars2 = ax.bar(x + width/2, df["Etter"], width, label="Etter", color=COLORS[2])

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_ylabel("Sum score")
    ax.set_title(f"Vektingseffekt: «{BOOST_CATEGORY}» vektes {WEIGHT}× (TV 2, N={NQ})")
    ax.legend()

    plt.tight_layout()
    plt.savefig("figs/D_weighting_before_after.png", dpi=220)

if __name__ == "__main__":
    main()
