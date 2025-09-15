# scripts/fig_a_category_share.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from figs_theme import apply_simple_theme

COLORS = ["#ff8c45", "#ffd865", "#97d2ec"]

def load_tv2():
    if os.path.exists("figs/tv2_full.parquet"):
        return pd.read_parquet("figs/tv2_full.parquet")
    return pd.read_csv("figs/tv2_full.csv")

def main():
    apply_simple_theme()
    tv2 = load_tv2()
    s = tv2["Kategori"].dropna().value_counts()
    total = s.sum()
    pct = (s / total * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.barh(pct.index, pct.values,
                   color=[COLORS[i % len(COLORS)] for i in range(len(pct))])
    ax.set_title("Tema-miks i spørsmål (TV 2)")
    ax.set_xlabel("Andel av spørsmål (%)")
    ax.set_ylabel("Kategori")
    for b, v in zip(bars, pct.values):
        ax.annotate(f"{v:.0f}%", xy=(v, b.get_y()+b.get_height()/2),
                    xytext=(6,0), textcoords="offset points",
                    va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("figs/A_category_share_tv2.png", dpi=220)

if __name__ == "__main__":
    main()
