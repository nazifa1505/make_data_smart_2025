# Workshop: Valgomat som datakvalitet + KI-case

## 1) Oppsett
- python -m venv .venv
- source .venv/bin/activate # Win: .venv\Scripts\activate
- pip install -r requirements.txt

## 2) Forbered data og kvalitetsrapport
- python scripts/data_prep.py

- Output: `figs/QUALITY_REPORT.md`, `figs/nrk_h.parquet`, `figs/tv2_full.parquet`, `figs/parties.txt`

## 3) Lag figurene
- python scripts/fig_a_categories_tv2.py
- python scripts/fig_b_heatmap_nrk.py
- python scripts/fig_c_naive_recommender.py

- Output: `figs/A_kategorifordeling_tv2.png`, `figs/B_heatmap_nrk.png`,
  `figs/C1_naiv_recommender_nrk.png`, `figs/C2_naiv_recommender_tv2.png`

## 4) Skjermopptak (uten live-demo)
Kjør ett scenario-skript, og gjør et kort opptak mens du viser PNG før/etter.

- Fjern kategori:
python scripts/scenario_remove_category.py

- Dobbel-vekting:
python scripts/scenario_weighting_tv2.py


## 5) Bruk i presentasjon
- Figur A = *representativitet/tema-utvalg*
- Figur B = *skala/proveniens/konsistens*
- Figur C = *input→output (samme «KI», ulike svar)*
- Scenario PNG (S1/S2) til korte videoer

## 6) Snakkepunkter (datakvalitet → KI-kvalitet)
- Kompletthet/representativitet (valg av tema)
- Validitet (formulering/ledende tekst)
- Vekting og transparens
- Konsistens og aktualitet
- Proveniens: hvem satte ±2, når, og etter hvilke kilder?

