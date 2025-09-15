# scripts/scenario_remove_category.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from figs_theme import apply_simple_theme

COLORS = ["#ff8c45", "#ffd865", "#97d2ec"]

def read_parties():
    with open("figs/parties.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main(category_to_drop="Barn og familie", NQ=10):
    apply_simple_theme()
    parties = read_parties()
    tv2 = pd.read_parquet("figs/tv2_full.parquet")

    base = tv2.head(NQ)[parties].sum()
    after = tv2[tv2["Kategori"] != category_to_drop].head(NQ)[parties].sum()

    df = pd.DataFrame({"Med alle kategorier": base, f"Uten {category_to_drop}": after})
    df = df.sort_values("Med alle kategorier", ascending=False)

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, df["Med alle kategorier"], width,
                   label="Med alle", color=COLORS[0])
    bars2 = ax.bar(x + width/2, df[f"Uten {category_to_drop}"], width,
                   label=f"Uten {category_to_drop}", color=COLORS[2])

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_ylabel("Sum score")
    ax.set_title(f"Scenario: fjerner kategori «{category_to_drop}» (TV 2, N={NQ})")
    ax.legend()

    plt.tight_layout()
    plt.savefig("figs/S1_remove_category.png", dpi=220)

if __name__ == "__main__":
    main()
