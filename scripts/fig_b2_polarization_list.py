# scripts/fig_b2_polarization_list.py
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import math
from figs_theme import apply_simple_theme

TOP_K     = 8      # topp N
WRAP      = 62     # maks tegn per linje (senk for kortere linjer)
LINE_H    = 0.055  # linjeavstand i aksis-ytelse (øk for mer luft per linje)
BLOCK_GAP = 0.030  # ekstra luft mellom punkter (etter μ/σ²-linjen)
N_COLS    = 1      # 1 = én spalte (scenevennlig), 2 = to spalter

def wrap_lines(txt, width=WRAP, max_lines=2):
    """Returner liste av linjer (maks max_lines) med ellipses på siste."""
    txt = (txt or "").strip()
    lines = textwrap.wrap(txt, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " …"
    return lines

def draw_entries(ax, entries, x0, y0):
    """
    Tegn en liste av (idx, text_lines, mu, var) fra start (x0,y0).
    Returner y-posisjonen etter siste blokk.
    """
    y = y0
    for idx, lines, mu, var in entries:
        # 1) Spørsmålslinjene (flere linjer)
        ax.text(x0, y, f"{idx}. {lines[0]}", fontsize=18, ha="left", va="top")
        for extra in lines[1:]:
            y -= LINE_H
            ax.text(x0 + 0.02, y, extra, fontsize=18, ha="left", va="top")

        # 2) Tall-linjen under
        y -= LINE_H
        ax.text(x0 + 0.02, y, f"μ={mu:+.2f}   σ²={var:.2f}", fontsize=14, color="#555",
                ha="left", va="top")

        # 3) Ekstra luft før neste punkt
        y -= BLOCK_GAP
        # og flytt ned tilsvarende *én* ekstra linje hvis teksten var to linjer
        # (dette er dekket av løkken over, så vi trenger ikke mer her)

    return y

def main():
    apply_simple_theme()

    # les topp-listen lagd av fig_b1 (eller lag fra parquet selv)
    top = pd.read_csv("figs/B_polarization_top.csv").nlargest(TOP_K, "var").reset_index(drop=True)

    # Forbered datastruktur: [(idx, [linjer], mu, var), ...]
    items = []
    for i, row in enumerate(top.itertuples(index=False), 1):
        lines = wrap_lines(row.Spm, width=WRAP, max_lines=2)
        items.append((i, lines, row.mean, row.var))

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title(f"Mest polariserende spørsmål (NRK) – topp {TOP_K}",
                 pad=20, fontsize=20)

    if N_COLS == 1:
        # Én spalte – maks scenevennlig
        draw_entries(ax, items, x0=0.06, y0=0.90)

    else:
        # To spalter – del i to like lister
        half = math.ceil(len(items) / 2)
        left, right = items[:half], items[half:]
        y_left_end  = draw_entries(ax, left,  x0=0.06, y0=0.90)
        draw_entries(ax, right, x0=0.56, y0=0.90)

    fig.tight_layout()
    fig.savefig("figs/B2_polarization_list.png", dpi=220)

if __name__ == "__main__":
    main()
