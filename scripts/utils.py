# scripts/utils.py
from __future__ import annotations
import pandas as pd
import re
from typing import List

META_COLS = {
    "SpmID", "Spm", "Rekkefølge", "Kategori ID", "Kategori",
    "Kategori Slug", "Beskrivelse", "Original ID"
}

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def _normalize_text(s: pd.Series) -> pd.Series:
    # strip, lower, komprimer mellomrom, fjern "smart quotes"/non-breaking space
    s = s.astype(str).str.replace("\u00A0", " ", regex=False)
    s = s.str.strip().str.lower()
    s = s.apply(lambda x: re.sub(r"\s+", " ", x))
    return s

def load_raw(data_dir: str = "data"):
    nrk  = pd.read_csv(f"{data_dir}/nrk.csv", encoding="utf-8")
    tv2  = pd.read_csv(f"{data_dir}/tv2.csv", encoding="utf-8")
    tv2d = pd.read_csv(f"{data_dir}/tv2_description.csv", encoding="utf-8")

    nrk  = _clean_columns(nrk)
    tv2  = _clean_columns(tv2)
    tv2d = _clean_columns(tv2d)

    # harmoniser ID/tekst-kolonner
    for df in (nrk, tv2, tv2d):
        if "Spørsmål ID" in df.columns:
            df.rename(columns={"Spørsmål ID": "SpmID"}, inplace=True)
        if "Spørsmål" in df.columns:
            df.rename(columns={"Spørsmål": "Spm"}, inplace=True)

    # lag normalisert tekst for robust join
    for df in (tv2, tv2d):
        if "Spm" in df.columns:
            df["Spm_norm"] = _normalize_text(df["Spm"])

    return nrk, tv2, tv2d

def party_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS and c not in {"Spm_norm"}]

def harmonize(nrk: pd.DataFrame, tv2: pd.DataFrame, tv2d: pd.DataFrame):
    nrk_parties = party_columns(nrk)
    tv2_parties = party_columns(tv2)
    parties = [p for p in nrk_parties if p in tv2_parties]  # felles partisett

    nrk_h = nrk[["SpmID", "Spm"] + parties].copy()

    # TV2: behold SpmID/Spm/partier + normalisert tekst
    base_cols = ["SpmID", "Spm", "Spm_norm"] + parties
    tv2_h = tv2[base_cols].copy()

    # Beskrivelse/kategori: dedupliser på normalisert tekst (1 rad per spørsmålstekst)
    tv2d_ = tv2d.copy()
    if "Spm" in tv2d_.columns:
        tv2d_["Spm_norm"] = _normalize_text(tv2d_["Spm"])
    keep_cols = ["Spm_norm", "Kategori", "Beskrivelse"]
    tv2d_ = tv2d_[keep_cols].drop_duplicates(subset=["Spm_norm"])

    # JOIN PÅ TEKST (ikke SpmID)
    tv2_full = tv2_h.merge(tv2d_, on="Spm_norm", how="left", validate="m:1")

    # konverter partiscore til numerisk
    for p in parties:
        nrk_h[p] = pd.to_numeric(nrk_h[p], errors="coerce")
        tv2_full[p] = pd.to_numeric(tv2_full[p], errors="coerce")

    # rydd bort hjelpekolonne
    tv2_full.drop(columns=["Spm_norm"], inplace=True)

    return nrk_h, tv2_full, parties

def quality_report(nrk: pd.DataFrame, tv2_full: pd.DataFrame, parties: list) -> str:
    lines = []
    lines.append("# Datakvalitetsrapport\n")
    nrk_na = nrk[parties].isna().sum().sum()
    tv2_na = tv2_full[parties].isna().sum().sum()
    lines += [
        f"- Manglende verdier (NRK): {nrk_na}",
        f"- Manglende verdier (TV2): {tv2_na}",
    ]
    bad_nrk = nrk[parties].stack().pipe(lambda s: s[(s < -2) | (s > 2)]).shape[0]
    bad_tv2 = tv2_full[parties].stack().pipe(lambda s: s[(s < -2) | (s > 2)]).shape[0]
    lines += [
        f"- Utenfor [-2,2] (NRK): {bad_nrk}",
        f"- Utenfor [-2,2] (TV2): {bad_tv2}",
    ]
    lines.append(f"- Antall partier (felles): {len(parties)}")
    # hvor mange fikk kategori/beskrivelse
    if "Kategori" in tv2_full.columns:
        lines.append(f"- TV2 rader med Kategori: {tv2_full['Kategori'].notna().sum()} av {len(tv2_full)}")
    if "Beskrivelse" in tv2_full.columns:
        lines.append(f"- TV2 rader med Beskrivelse: {tv2_full['Beskrivelse'].notna().sum()} av {len(tv2_full)}")
    return "\n".join(lines) + "\n"
