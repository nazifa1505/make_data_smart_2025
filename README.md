
## Metodikk for datakvalitetsvurdering

## Hvordan beregner vi datakvalitet?

Vår app vurderer datakvalitet langs 6 vitenskapelig anerkjente dimensjoner. Her er logikken bak hver beregning:

### 1. 🎯 Nøyaktighet (Accuracy)
**Hva det måler:** Hvor godt dataene reflekterer virkeligheten
**Beregningsmetode:**
```
Nøyaktighet = 100% - (Gjennomsnittlig absolutt forskjell mellom NRK og TV2) / 4 * 100%
```
**Logikk:** Hvis to uavhengige kilder (NRK og TV2) gir lignende resultater, øker tilliten til dataenes nøyaktighet.

**Tolkning:**
- **90-100%:** Høy konsistens mellom kilder = høy nøyaktighet
- **75-89%:** Moderate forskjeller mellom kilder
- **Under 75%:** Store forskjeller = lav tillit til nøyaktighet

### 2. Kompletthet (Completeness)
**Hva det måler:** Hvor mye av dataene som faktisk er tilgjengelig
**Beregningsmetode:**
```
Kompletthet = (Totale celler - Manglende celler) / Totale celler * 100%
```
**Logikk:** Manglende data reduserer AI-systemers evne til å lære korrekte mønstre.

**Tolkning:**
- **95-100%:** Minimalt datamangel
- **90-94%:** Akseptabel mengde manglende data  
- **Under 90%:** For mye manglende data for pålitelig AI-bruk

### 3.Konsistens (Consistency)
**Hva det måler:** Hvor stabile og ikke-motsigelsesfulle dataene er
**Beregningsmetode:**
```
Konsistens = 100% - (Gjennomsnittlig standardavvik * 25)
```
**Logikk:** Høy variabilitet kan indikere inkonsistente målinger eller motstridende data.

**Tolkning:**
- **85-100%:** Stabile, konsistente mønstre
- **70-84%:** Moderate inkonsistenser
- **Under 70%:** Høy variabilitet = lav konsistens

### 4.Aktualitet (Timeliness)
**Hva det måler:** Hvor oppdaterte dataene er
**Beregningsmetode:**
```
Aktualitet = 100% - (Måneder siden innsamling * 2%)
```
**Logikk:** Politiske holdninger endres over tid. Eldre data blir mindre relevant.

**Antakelser:**
- Data antas å være 6 måneder gamle (estimat for valgomatdata)
- 2% verdifall per måned (basert på politisk volatilitet)

### 5.Validitet (Validity)
**Hva det måler:** Om dataene har korrekt format og gyldige verdier
**Beregningsmetode:**
```
Validitet = Antall verdier i gyldig range [-2, +2] / Totale verdier * 100%
```
**Logikk:** Valgomatskalaen går fra -2 (sterkt uenig) til +2 (sterkt enig). Verdier utenfor er ugyldige.

### 6.Unikalitet (Uniqueness)
**Hva det måler:** Grad av duplikater og overrepresentasjon
**Beregningsmetode:**
```
Unikalitet = min(100%, (Antall kategorier * 4) / Totale spørsmål * 100%)
```
**Logikk:** Vi forventer ~4 spørsmål per tematisk kategori som optimal balanse.

## Samlet kvalitetsscore

```
Samlet score = Gjennomsnitt av alle 6 dimensjoner
```

**Vurderingsskala:**
- **90-100%:** Utmerket - Klar for avanserte AI-analyser
- **75-89%:** God - Brukbar for de fleste AI-applikasjoner  
- **60-74%:** Akseptabel - Krever forbedringer før pålitelig AI-bruk
- **Under 60%:** Lav - Omfattende datarengjøring nødvendig

## Begrensninger og antakelser

### Hva vi IKKE kan måle:
- **Faktisk nøyaktighet:** Vi har ingen "fasit" å sammenligne med
- **Skjulte bias:** Systematiske skjevheter kan være usynlige
- **Temporal drift:** Hvordan holdninger endrer seg over tid

### Våre antakelser:
- NRK og TV2 er begge relativt pålitelige kilder
- 6 måneder gammel data (estimat)
- 4 spørsmål per kategori er optimalt
- Politisk volatilitet på 2% per måned

### Transparens-prinsipp:
All vår metodikk er åpen og kan granskes. Vi oppfordrer til kritisk vurdering av våre antakelser og metoder.

## For videre lesing:
- Wang, R. Y., & Strong, D. M. (1996). "Beyond accuracy: What data quality means to data consumers"
- ISO/IEC 25012:2008 - Data Quality Model
- Pipino, L. L., Lee, Y. W., & Wang, R. Y. (2002). "Data quality assessment"


