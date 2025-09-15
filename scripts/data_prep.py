# scripts/data_prep.py
from utils import load_raw, harmonize, quality_report
from figs_theme import apply_simple_theme

def main():
    nrk, tv2, tv2d = load_raw("data")
    nrk_h, tv2_full, parties = harmonize(nrk, tv2, tv2d)

    # skriv ut datakvalitetsrapport
    report = quality_report(nrk_h, tv2_full, parties)
    with open("figs/QUALITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    # cache harmoniserte data for andre skript
    nrk_h.to_parquet("figs/nrk_h.parquet", index=False)
    tv2_full.to_parquet("figs/tv2_full.parquet", index=False)
    with open("figs/parties.txt","w",encoding="utf-8") as f:
        f.write("\n".join(parties))

if __name__ == "__main__":
    apply_simple_theme()
    main()
