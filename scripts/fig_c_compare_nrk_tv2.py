import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from figs_theme import apply_simple_theme

COLORS = ["#ff8c45", "#ffd865", "#97d2ec"]
NQ = 30

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
    nrk = pd.read_parquet("figs/nrk_h.parquet")
    tv2 = load_tv2()

    nrk_sum = nrk.head(NQ)[parties].sum()
    tv2_sum = tv2.head(NQ)[parties].sum()

    df = pd.DataFrame({"NRK": nrk_sum, "TV 2": tv2_sum})
    # sorter etter gjennomsnitt for stabil rekkefølge
    df["avg"] = df[["NRK","TV 2"]].mean(axis=1)
    df = df.sort_values("avg", ascending=False).drop(columns="avg")

    x = np.arange(len(df))
    w = 0.42

    fig, ax = plt.subplots(figsize=(14, 6.5))
    b1 = ax.bar(x - w/2, df["NRK"], width=w, label="NRK", color=COLORS[0])
    b2 = ax.bar(x + w/2, df["TV 2"], width=w, label="TV 2", color=COLORS[2])

    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_ylabel("Sum score (N spørsmål)")
    ax.set_title(f"Partipoengsum: NRK vs. TV 2 (N={NQ} per kilde)")
    ax.legend()

    # små verdilapper
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            ax.annotate(f"{v:.0f}", xy=(r.get_x()+r.get_width()/2, v),
                        xytext=(0, 3 if v>=0 else -12), textcoords="offset points",
                        ha="center", va="bottom" if v>=0 else "top", fontsize=8)

    plt.tight_layout()
    plt.savefig("figs/C_grouped_nrk_tv2.png", dpi=220)

if __name__ == "__main__":
    main()
