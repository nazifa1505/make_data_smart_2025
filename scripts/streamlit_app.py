
import os
import math
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import seaborn as sns

# Configuration and color scheme
BG = "#fcf6ee"
COL_NEG, COL_NEU, COL_POS = "#ff8c45", "#ffd865", "#97d2ec"
COLORS = [COL_NEG, COL_NEU, COL_POS]

st.set_page_config(
    page_title="Valgomat-utforsker", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS styling
def load_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback: minimal inline CSS if file not found
        st.markdown(f"""
        <style>
        .stApp {{ background-color: {BG} !important; color: #111111 !important; }}
        .explanation-card {{ 
            background: linear-gradient(135deg, #e8f4f8, #f0f8ff);
            border-left: 4px solid {COL_POS};
            padding: 15px; border-radius: 8px; margin: 10px 0;
        }}
        .warning-box {{ 
            background: #fff3cd; border-left: 4px solid #f39c12;
            padding: 12px; border-radius: 8px; margin: 10px 0;
        }}
        .critical-info {{
            background: #f8d7da; border-left: 4px solid #dc3545;
            padding: 12px; border-radius: 8px; margin: 10px 0;
        }}
        </style>
        """, unsafe_allow_html=True)

load_css()

# Utility functions for creating user-friendly UI elements
def create_explanation_card(title, content, icon="💡", card_type="info"):
    """Create informative explanation cards with different styles"""
    if card_type == "warning":
        css_class = "warning-box"
        bg_color = "#fff3cd"
        border_color = "#f39c12"
    elif card_type == "critical":
        css_class = "critical-info"  
        bg_color = "#f8d7da"
        border_color = "#dc3545"
    else:
        css_class = "explanation-card"
        bg_color = "linear-gradient(135deg, #e8f4f8, #f0f8ff)"
        border_color = COL_POS
    
    st.markdown(f"""
    <div class="{css_class}">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_how_to_read_box(graph_type):
    """Create 'How to read this graph' explanations"""
    explanations = {
        "polarization": """
        <strong>Slik leser du polariseringsgrafen:</strong><br>
        • <strong>Hvert punkt</strong> = Ett spørsmål fra valgomaten<br>
        • <strong>Høyt oppe</strong> = Partiene er svært uenige om dette spørsmålet<br>
        • <strong>Til høyre</strong> = De fleste partier støtter forslaget<br>
        • <strong>Til venstre</strong> = De fleste partier er imot forslaget<br>
        • <strong>Størrelse på punkt</strong> = Hvor kontroversielt spørsmålet er
        """,
        "party_comparison": """
        <strong>Slik leser du sammenligningen:</strong><br>
        • <strong>Stolpehøyde</strong> = Sum av partiets posisjoner<br>
        • <strong>Sammenlign mønster</strong>, ikke eksakte tall<br>
        • <strong>Rangering</strong> er mer pålitelig enn absolutte verdier<br>
        • <strong>Forskjellige spørsmål</strong> = kan ikke sammenligne direkte
        """,
        "party_positions": """
        <strong>Slik leser du partiposisjoner:</strong><br>
        • <strong>+2</strong> = Partiet er sterkt FOR forslaget<br>
        • <strong>+1</strong> = Partiet er moderat FOR<br>
        • <strong>0</strong> = Partiet er nøytral/usikker<br>
        • <strong>-1</strong> = Partiet er moderat MOT<br>
        • <strong>-2</strong> = Partiet er sterkt MOT forslaget
        """
    }
    
    if graph_type in explanations:
        st.markdown(f"""
        <div class="explanation-card">
            <details>
                <summary><strong>❓ Hvordan lese denne grafen?</strong></summary>
                <p style="margin-top: 8px;">{explanations[graph_type]}</p>
            </details>
        </div>
        """, unsafe_allow_html=True)

def create_enhanced_data_warning():
    """Create simple warning using native Streamlit components"""
    
    # Main warning box
    st.warning("**Viktig å forstå før du starter**")
    
    # Main explanation
    st.markdown("""
    Denne appen viser **partienes offisielle svar** på valgomatspørsmål - ikke hva folk flest mener.
    """)
    
    # Key limitation
    st.info("""
    **NRK og TV2 har helt forskjellige spørsmål.** 
    
    Du kan ikke sammenligne tallene direkte - det er som å sammenligne epler med pærer.
    """)
    
    # What you can do
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Hva du KAN bruke appen til:**
        - Se hvilke tema som skaper mest uenighet  
        - Lære om datakvalitet og AI-systemer
        - Utforske hver kilde separat
        """)
    
    with col2:
        st.error("""
        **❌ Hva appen IKKE kan:**
        - Forutsi valgresultater
        - Fortelle deg hva vanlige velgere mener
        - Sammenligne NRK og TV2 direkte
        """)
    
    # Final note
    st.markdown("""
    ---
    💡 **Husk:** Negative tall betyr ikke at partiet er "negativt" - det betyr de er imot de spesifikke forslagene som ble testet.
    """)

def add_source_selection_guidance():
    """Add clear guidance for source selection"""
    with st.sidebar:
        st.info("📋 **Kildeguide**")
        
        st.write("**Velg NRK når:**")
        st.write("• Du vil se bredest mulig spekter (252 spm)")
        st.write("• Du vil fokusere på politisk polarisering") 
        st.write("• Du ikke trenger tematisk kategorisering")
        
        st.write("**Velg TV2 når:**")
        st.write("• Du vil analysere spesifikke tema")
        st.write("• Du vil bruke vektings- eller scenario-analyser")
        st.write("• Du foretrekker færre, men kategoriserte spørsmål")
        
        st.warning(" **Husk:** Kan ikke sammenlignes direkte!")

def add_comparison_blocker():
    """Add visual element that prevents comparison thinking"""
    st.markdown("""
    <div style="background: linear-gradient(45deg, #ff6b6b, #feca57); 
                padding: 10px; border-radius: 10px; text-align: center; 
                font-weight: bold; color: white; margin: 20px 0;">
         NRK og TV2 har forskjellige spørsmål - IKKE sammenlign tallene! 
    </div>
    """, unsafe_allow_html=True)

def create_tv2_only_explanation():
    """Explain why some analyses use only TV2 data"""
    create_explanation_card(
        "Hvorfor bare TV2-data?",
        "Vektings- og scenario-analyser krever tematiske kategorier (økonomi, helse, osv.). " +
        "TV2-data har denne informasjonen, mens NRK-data mangler kategorisering. " +
        "Dette er ikke en svakhet, men reflekterer at ulike kilder strukturerer data forskjellig.",
        "ℹ️"
    )

def create_score_interpretation_box():
    st.markdown("""
    <div class="explanation-card">
        <h4>📊 Slik tolker du scorene</h4>
        <p><strong>Summer vs. Gjennomsnitt:</strong></p>
        <ul>
            <li><strong>Sum -60 over 30 spørsmål</strong> = Gjennomsnitt -2.0 = Konsekvent imot forslagene</li>
            <li><strong>Sum +40 over 20 spørsmål</strong> = Gjennomsnitt +2.0 = Konsekvent for forslagene</li>
            <li><strong>Sum 0 over 25 spørsmål</strong> = Gjennomsnitt 0.0 = Nøytral/balansert</li>
        </ul>
        <p><strong>Negative tall betyr IKKE:</strong> Partiet er "negativt" eller "imot alt"<br>
        <strong>Negative tall betyr:</strong> Partiet er imot de spesifikke forslagene som ble testet</p>
    </div>
    """, unsafe_allow_html=True)

def friendly_translate_stats(mu, std):
    """Translate statistics to friendly language"""
    # Translate direction (mu)
    if abs(mu) < 0.2:
        direction = "Partiene er delte"
    elif mu > 0.5:
        direction = "De fleste partier støtter dette"
    elif mu > 0:
        direction = "Svakt flertall støtter"
    elif mu < -0.5:
        direction = "De fleste partier er imot"
    else:
        direction = "Svakt flertall er imot"
    
    # Translate disagreement (std)
    if std < 0.5:
        agreement = "bred enighet"
    elif std < 1.0:
        agreement = "noe uenighet"
    elif std < 1.5:
        agreement = "mye uenighet"
    else:
        agreement = "dyp splittelse"
        
    return f" {direction} ( {agreement})"

def simple_sample_or_all(df, n_questions, category_col='Kategori'):
    """Ta enten et enkelt sample, eller alle data hvis n_questions er stort nok"""
    if len(df) <= n_questions or n_questions >= 50:
        return df  # Ta alle spørsmålene
    else:
        # Ta et tilfeldig utvalg som bevarer kategori-proporsjonene
        return df.sample(n=n_questions, random_state=42)

def get_consistent_color_scale():
    """Returner konsistent fargepalett for alle charts"""
    return [[0, COL_NEG], [0.5, COL_NEU], [1, COL_POS]]

# Data loading and processing functions
@st.cache_data
def load_files():
    parties = [p.strip() for p in open("figs/parties.txt", encoding="utf-8").read().splitlines() if p.strip()]
    nrk = pd.read_parquet("figs/nrk_h.parquet")
    tv2 = (pd.read_parquet("figs/tv2_full.parquet")
           if os.path.exists("figs/tv2_full.parquet")
           else pd.read_csv("figs/tv2_full.csv"))
    return parties, nrk, tv2

def wrap(text, width=70, lines=2):
    t = (text or "").strip()
    w = textwrap.wrap(t, width=width)
    if len(w) > lines:
        w = w[:lines]
        w[-1] += " …"
    return "<br>".join(w)

def prep_data_with_friendly_labels(df, parties):
    """Prepare data with user-friendly labels and explanations"""
    out = df.copy()
    out["avg_position"] = out[parties].mean(axis=1)
    out["disagreement"] = out[parties].std(axis=1)
    out["disagreement_squared"] = out[parties].var(axis=1)
    
    
    out["direction_label"] = out["avg_position"].apply(lambda x:
        "Sterkt støttende" if x > 1.0 else
        "Moderat støttende" if x > 0.3 else
        "Delt/nøytral" if abs(x) <= 0.3 else
        "Moderat kritisk" if x > -1.0 else
        "Sterkt kritisk"
    )
    
    out["agreement_label"] = out["disagreement"].apply(lambda x:
        "Bred enighet" if x < 0.5 else
        "Noe uenighet" if x < 0.8 else
        "Betydelig uenighet" if x < 1.2 else
        "Dyp polarisering"
    )
    
    out["friendly_summary"] = out.apply(
        lambda row: friendly_translate_stats(row["avg_position"], row["disagreement"]), 
        axis=1
    )
    
    return out

def color_map(mu):
    if mu < -0.2: return COL_NEG
    if mu >  0.2: return COL_POS
    return COL_NEU

def size_scale(x, smin=10, smax=28):
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return smin + x*(smax - smin)

def calculate_quality_scores(parties, nrk, tv2):
    """Calculate actual data quality scores - CORRECTED VERSION"""
    scores = {}
    
    
    nrk_party_consistency = []
    tv2_party_consistency = []
    
    for party in parties:
        if party in nrk.columns:
            party_positions = nrk[party].dropna()
            if len(party_positions) > 1:
                # Måler hvor ofte partiet endrer posisjon drastisk (mer enn 2 poeng)
                position_changes = abs(party_positions.diff()).dropna()
                consistency = 1 - (position_changes > 2).mean()  # Andel små endringer
                nrk_party_consistency.append(consistency)
        
        if party in tv2.columns:
            party_positions = tv2[party].dropna()
            if len(party_positions) > 1:
                position_changes = abs(party_positions.diff()).dropna()
                consistency = 1 - (position_changes > 2).mean()
                tv2_party_consistency.append(consistency)
    
    if nrk_party_consistency and tv2_party_consistency:
        avg_consistency = (np.mean(nrk_party_consistency) + np.mean(tv2_party_consistency)) / 2
    elif nrk_party_consistency:
        avg_consistency = np.mean(nrk_party_consistency)
    elif tv2_party_consistency:
        avg_consistency = np.mean(tv2_party_consistency)
    else:
        avg_consistency = 0.85  # Default reasonable value
    
    scores["Intern_konsistens"] = max(0, min(100, avg_consistency * 100))
    
    # 2. Kompletthet - sjekk for manglende verdier
    total_cells = len(nrk) * len(parties) + len(tv2) * len(parties)
    missing_cells = nrk[parties].isna().sum().sum() + tv2[parties].isna().sum().sum()
    completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 1
    scores["Kompletthet"] = max(0, min(100, completeness * 100))
    
    
    all_values = pd.concat([nrk[parties].stack(), tv2[parties].stack()]).dropna()
    if len(all_values) > 0:
        value_distribution = all_values.value_counts(normalize=True)
        ideal_distribution = 1 / len(value_distribution)
        balance_score = 1 - np.sum(abs(value_distribution - ideal_distribution)) / 2
        scores["Balanse"] = max(0, min(100, balance_score * 100))
    else:
        scores["Balanse"] = 85
    
    # 4. Aktualitet - anta dataene er 6 måneder gamle
    months_old = 6
    timeliness = max(0, 100 - months_old * 2)
    scores["Aktualitet"] = timeliness
    
    # 5. Validitet - sjekk om verdier er i forventet range [-2, 2]
    if len(all_values) > 0:
        valid_values = all_values[(all_values >= -2) & (all_values <= 2)]
        validity = len(valid_values) / len(all_values) * 100
    else:
        validity = 95
    scores["Validitet"] = validity
    
    nrk_contribution = len(nrk) / (len(nrk) + len(tv2))
    balance_deviation = abs(nrk_contribution - 0.5)
    source_balance = max(0, 100 - balance_deviation * 200)
    scores["Kildebalanse"] = source_balance
    
    return scores

def safe_dataset_selector(key_suffix=""):
    """Dataset selector with clear warnings about comparisons"""
    dataset = st.radio("Velg datasett", ["NRK (252 spm)", "TV2 (98 spm)"], key=f"dataset_{key_suffix}")
    
    if "NRK" in dataset:
        st.info("💡 **NRK:** 252 spørsmål uten kategorier")
        return "NRK", nrk
    else:
        st.info("💡 **TV2:** 98 spørsmål organisert i tema")
        return "TV2", tv2

# Load data
parties, nrk, tv2 = load_files()

# Main app
st.title("🗳️ Valgomat-utforsker")
st.subheader("Utforsk hvordan norske partier posisjonerer seg politisk")


create_enhanced_data_warning()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📖 Metodikk",
    "🎯 Hva splitter partiene?",
    "⚖️ Kildeforskjeller (IKKE sammenlign verdier!)",
    "📋 Temaoversikt (TV2)",
    "⚖️ Vektingseffekter", 
    "📈 Scenario-analyse",
    "🤖 AI & Datakvalitet",
    "🧠 AI-Analyser"  
])

with tab1:
    st.header("Metodikk og Transparens")
    
    create_explanation_card(
        "Prosess",
        "All vår metodikk er åpen og kan granskes. Vi oppfordrer til kritisk vurdering av våre " +
        "antakelser og metoder. Dette er ikke 'absolutte sannheter', men analytiske verktøy."
    )
    
    # Expandable methodology sections
    with st.expander("🎯 Hvordan beregner vi datakvalitet?", expanded=True):
        st.markdown("""
        ### De 6 dimensjonene av datakvalitet
        
        Vår app vurderer datakvalitet langs 6 vitenskapelig anerkjente dimensjoner:
        
        #### 1. 🔄 Intern konsistens
        **Hva det måler:** Hvor konsistente partier er i sine posisjoner innenfor hver kilde  
        **Beregning:** `Andel av partier som sjelden endrer posisjon drastisk (>2 poeng)`  
        **Logikk:** Konsistente partier indikerer pålitelige målinger
        
        #### 2. 📋 Kompletthet
        **Hva det måler:** Hvor mye av dataene som faktisk er tilgjengelig  
        **Beregning:** `(Totale celler - Manglende celler) / Totale celler * 100%`  
        **Logikk:** Manglende data reduserer AI-systemers læringsevne
        
        #### 3. ⚖️ Balanse
        **Hva det måler:** Om posisjoner er jevnt fordelt på skalaen  
        **Beregning:** `1 - sum(|faktisk_fordeling - ideal_fordeling|) / 2`  
        **Logikk:** Skjeve fordelinger kan indikere bias i spørsmålsformulering
        
        #### 4. ⏰ Aktualitet
        **Hva det måler:** Hvor oppdaterte dataene er  
        **Beregning:** `100% - (Måneder siden innsamling * 2%)`  
        **Antakelse:** Data antas 6 mnd gamle, 2% verdifall per måned
        
        #### 5. ✅ Validitet
        **Hva det måler:** Om dataene har korrekt format og gyldige verdier  
        **Beregning:** `Antall verdier i range [-2,+2] / Totale verdier * 100%`  
        **Logikk:** Valgomatskalaen har definerte grenser
        
        #### 6. 📊 Kildefordeling
        **Hva det måler:** Balansen mellom datamengde fra hver kilde  
        **Beregning:** `100% - |NRK_andel - 0.5| * 200%`  
        **Logikk:** Ekstrem ubalanse kan skape skjevheter
        """)
    
    with st.expander("📊 Kvalitetsvurderingsskala"):
        st.markdown("""
        ### Hvordan tolke kvalitetsscorer?
        
        **Samlet kvalitetsscore = Gjennomsnitt av alle 6 dimensjoner**
        
        | Score | Vurdering | AI-egnethet | Anbefaling |
        |-------|-----------|-------------|------------|
        | 90-100% | 🟢 Utmerket | Klar for avanserte AI-analyser | Fortsett som normalt |
        | 75-89% | 🟡 God | Brukbar for de fleste AI-applikasjoner | Vurder forbedringer |
        | 60-74% | 🟠 Akseptabel | Krever forbedringer før AI-bruk | Datarengjøring anbefales |
        | Under 60% | 🔴 Lav | Omfattende datarengjøring nødvendig | Ikke egnet for AI |
        """)
    
    with st.expander("⚠️ Begrensninger og antakelser"):
        st.markdown("""
        ### Hva vi IKKE kan måle:
        - **Faktisk nøyaktighet:** Vi har ingen "fasit" å sammenligne med
        - **Skjulte bias:** Systematiske skjevheter kan være usynlige  
        - **Temporal drift:** Hvordan holdninger endrer seg over tid
        - **Kontekstuelle faktorer:** Politisk klima, mediedekning osv.
        
        ### Våre antakelser:
        - NRK og TV2 er begge relativt pålitelige kilder (separat)
        - 6 måneder gammel data (estimat for valgomatdata)  
        - Politiske posisjoner er relativt stabile over korte perioder
        - Partier bør være konsistente i sine standpunkter
        
        ### Viktige forbehold:
        - **Ikke absolutte sannheter:** Våre metoder er analytiske verktøy
        - **Kontekst-avhengig:** Kvalitet avhenger av bruksområde og krav
        - **Forenklede modeller:** Virkeligheten er mer kompleks enn våre algoritmer
        - **Ingen sammenligning mellom kilder:** NRK og TV2 måles separat
        """)
    
    with st.expander("🔬 Vitenskapelig grunnlag"):
        st.markdown("""
        ### Forskningsbasert metodikk
        
        Våre datakvalitetsdimensjoner er basert på etablert forskning:
        
        **Klassiske referanser:**
        - Wang, R. Y., & Strong, D. M. (1996). "Beyond accuracy: What data quality means to data consumers"
        - ISO/IEC 25012:2008 - Data Quality Model  
        - Pipino, L. L., Lee, Y. W., & Wang, R. Y. (2002). "Data quality assessment"
        
        **AI og bias-forskning:**
        - Mehrabi, N., et al. (2021). "A Survey on Bias and Fairness in Machine Learning"
        - Barocas, S., Hardt, M., & Narayanan, A. (2019). "Fairness and Machine Learning"
        
        **Politisk opinion-forskning:**
        - Krosnick, J. A. (1991). "Response strategies for coping with the cognitive demands of attitude measures"
        - Tourangeau, R., et al. (2000). "The Psychology of Survey Response"
        """)
    
    with st.expander("💻 Teknisk implementasjon"):
        st.markdown("""
        ### Hvordan appen fungerer
        
        **Databehandling:**
        ```python
        # Eksempel: Beregning av intern konsistens
        party_positions = nrk[party].dropna()
        position_changes = abs(party_positions.diff()).dropna()
        consistency = 1 - (position_changes > 2).mean()
        ```
        
        **Visualisering:**
        - Plotly for interaktive grafer
        - Streamlit for brukergrensesnitt
        - Pandas for datamanipulasjon
        
        **Ytelse:**
        - Caching av datainnlasting (@st.cache_data)
        - Begrenset til 50 punkter i scatter plots for responsivitet
        - Lazy loading av tunge beregninger
        """)
    
    
    

with tab2:
    st.header("Hvilke spørsmål skaper mest uenighet?")
    
    create_explanation_card(
        "Hva viser denne analysen?",
        "Vi ser på valgomatspørsmål og finner ut hvilke som skaper mest politisk splittelse. " +
        "Noen forslag får bred støtte, andre deler partiene på midten.",
        "🔍"
    )
    
    
    add_source_selection_guidance()
    
    with st.sidebar:
        st.header("Innstillinger")
        source_name, current_data = safe_dataset_selector("polarization")
        
        top_k = st.slider("Vis topp-punkter", 5, 15, 8)
        show_numbers = st.toggle("Vis nummerering på punkter", value=True)

    
    df = prep_data_with_friendly_labels(current_data, parties)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Create main visualization
        create_how_to_read_box("polarization")
        
        df["color"] = df["avg_position"].apply(color_map)
        df["size"] = size_scale(df["disagreement_squared"])
        
     
        fig = px.scatter(
            df.head(50),  # Limit to first 50 for performance
            x="avg_position", 
            y="disagreement",
            size="size",
            color="agreement_label",
            color_discrete_sequence=[COL_POS, COL_NEU, COL_NEG, "#ff4444"],
            hover_data=["Smp"] if "Smp" in df.columns else ["Spm"],
            title=f"Politisk splittelse: {source_name}-data<br><sub>Basert på {len(current_data)} spørsmål - Kan ikke sammenlignes med andre kilder</sub>"
        )
        
        # Customize plot
        fig.update_traces(
            marker=dict(line=dict(width=1, color="white")),
            hovertemplate="<b>%{customdata[0]}</b><br>" + 
                         "Gjennomsnitt: %{x:.2f}<br>" +
                         "Uenighet: %{y:.2f}<br>" +
                         "<extra></extra>"
        )
        
        # Add reference lines
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=df["disagreement"].median(), line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            xaxis_title="Partienes gjennomsnittlige posisjon ← Imot | For →",
            yaxis_title="Hvor mye er partiene uenige?",
            plot_bgcolor=BG,
            paper_bgcolor=BG,
            font=dict(color="#111111"),
            showlegend=True
        )
        
        # Add top K annotations if requested
        if show_numbers:
            top_controversial = df.nlargest(top_k, "disagreement").reset_index(drop=True)
            
            for i, row in enumerate(top_controversial.itertuples(), 1):
                fig.add_annotation(
                    x=row.avg_position,
                    y=row.disagreement,
                    text=str(i),
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="white",
                    borderwidth=1
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics in friendly language
        st.info(f"""
        📈 **Sammendrag for {source_name}:**
        • Totalt {len(df)} spørsmål analysert
        • Gjennomsnittlig uenighet: {df['disagreement'].mean():.2f}
        • Mest kontroversielle spørsmål har uenighet > {df['disagreement'].quantile(0.8):.2f}
        """)
    
    with col_right:
        st.subheader(f"Topp {top_k} mest kontroversielle")
        
        most_controversial = df.nlargest(top_k, "disagreement")
        
        for i, row in enumerate(most_controversial.itertuples(), 1):
            question_text = getattr(row, 'Smp', getattr(row, 'Spm', 'Ukjent spørsmål'))
            
            # Create styled card
            card_class = "supporting" if row.avg_position > 0.2 else "opposing" if row.avg_position < -0.2 else "neutral"
            
            st.markdown(f"""
            <div class="question-card {card_class}">
                <div style="font-weight: bold; margin-bottom: 8px;">
                    ▶ {i}. {textwrap.fill(question_text, width=60)}
                </div>
                <div style="font-size: 12px; color: #666;">
                    {row.friendly_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add explanation of what makes questions controversial
        with st.expander("Hva gjør et spørsmål kontroversielt?"):
            st.write(f"""
            Et spørsmål blir kontroversielt i {source_name}-data når:
            - **Noen partier er sterkt for** (svarer +2)
            - **Andre partier er sterkt imot** (svarer -2)  
            - **Få partier er nøytrale** (svarer 0)
            
            Dette skaper stor "uenighet" som vi måler statistisk.
            """)

with tab3:
    st.header("Sammenlign datakilder - Strukturelt")
    
    add_comparison_blocker()

    
    comparison_type = st.radio("Hva vil du sammenligne?", [
        "📊 Datastruktur og omfang",
        "🎯 Separate polariseringsanalyser", 
        "📈 Separate trendanalyser"
    ])
    
    if comparison_type == "📊 Datastruktur og omfang":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📺 NRK-data")
            st.metric("Antall spørsmål", len(nrk))
            st.metric("Antall partier", len(parties))
            st.metric("Dekningsgrad", f"{(nrk[parties].notna().sum().sum() / (len(nrk) * len(parties)) * 100):.1f}%")
            
            # Show question distribution characteristics
            nrk_positions = nrk[parties].stack().dropna()
            if len(nrk_positions) > 0:
                position_dist = nrk_positions.value_counts().sort_index()
                
                fig_nrk_dist = px.bar(
                    x=position_dist.index, y=position_dist.values,
                    title="NRK: Fordeling av partiposisjoner",
                    labels={'x': 'Posisjon', 'y': 'Antall svar'},
                    color=position_dist.index,
                    color_continuous_scale=get_consistent_color_scale()
                )
                fig_nrk_dist.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, showlegend=False)
                st.plotly_chart(fig_nrk_dist, use_container_width=True)
        
        with col2:
            st.subheader("📺 TV2-data")
            st.metric("Antall spørsmål", len(tv2))
            st.metric("Antall kategorier", len(tv2['Kategori'].unique()) if 'Kategori' in tv2.columns else "N/A")
            st.metric("Dekningsgrad", f"{(tv2[parties].notna().sum().sum() / (len(tv2) * len(parties)) * 100):.1f}%")
            
            # Show question distribution characteristics
            tv2_positions = tv2[parties].stack().dropna()
            if len(tv2_positions) > 0:
                position_dist_tv2 = tv2_positions.value_counts().sort_index()
                
                fig_tv2_dist = px.bar(
                    x=position_dist_tv2.index, y=position_dist_tv2.values,
                    title="TV2: Fordeling av partiposisjoner",
                    labels={'x': 'Posisjon', 'y': 'Antall svar'},
                    color=position_dist_tv2.index,
                    color_continuous_scale=get_consistent_color_scale()
                )
                fig_tv2_dist.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, showlegend=False)
                st.plotly_chart(fig_tv2_dist, use_container_width=True)
        
        # Comparative summary - structural only
        st.subheader("📋 Strukturell sammenligning")
        
        # Safe calculations
        nrk_most_used = nrk_positions.mode().iloc[0] if len(nrk_positions.mode()) > 0 else 'N/A'
        nrk_least_used = position_dist.idxmin() if len(position_dist) > 0 else 'N/A'
        tv2_most_used = tv2_positions.mode().iloc[0] if len(tv2_positions.mode()) > 0 else 'N/A'
        tv2_least_used = position_dist_tv2.idxmin() if len(position_dist_tv2) > 0 else 'N/A'
        
        comparison_table = pd.DataFrame({
            "Aspekt": ["Antall spørsmål", "Kategorisering", "Gj.snitt svarrate", 
                      "Mest brukte posisjon", "Minst brukte posisjon"],
            "NRK": [
                len(nrk),
                "Nei",
                f"{(nrk[parties].notna().mean().mean()*100):.1f}%",
                f"{nrk_most_used}",
                f"{nrk_least_used}"
            ],
            "TV2": [
                len(tv2),
                "Ja" if 'Kategori' in tv2.columns else "Nei",
                f"{(tv2[parties].notna().mean().mean()*100):.1f}%",
                f"{tv2_most_used}",
                f"{tv2_least_used}"
            ]
        })
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
        
    elif comparison_type == "🎯 Separate polariseringsanalyser":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NRK: Mest kontroversielle spørsmål")
            nrk_analysis = prep_data_with_friendly_labels(nrk, parties)
            nrk_top_controversial = nrk_analysis.nlargest(5, "disagreement")
            
            for i, row in enumerate(nrk_top_controversial.itertuples(), 1):
                question = getattr(row, 'Smp', getattr(row, 'Spm', 'Ukjent'))
                st.write(f"**{i}.** {question[:60]}...")
                st.caption(f"Uenighet: {row.disagreement:.2f}")
        
        with col2:
            st.subheader("TV2: Mest kontroversielle spørsmål")
            tv2_analysis = prep_data_with_friendly_labels(tv2, parties)
            tv2_top_controversial = tv2_analysis.nlargest(5, "disagreement")
            
            for i, row in enumerate(tv2_top_controversial.itertuples(), 1):
                question = getattr(row, 'Smp', getattr(row, 'Spm', 'Ukjent'))
                st.write(f"**{i}.** {question[:60]}...")
                st.caption(f"Uenighet: {row.disagreement:.2f}")
        
        st.info("💡 **Viktig:** Disse listene viser kontrovers innenfor hver kilde separat. " +
               "De kan ikke sammenlignes direkte siden spørsmålene er forskjellige.")
        
    else:  
        st.subheader("Separate dataanalyser")
        
        source_choice = st.selectbox("Velg kilde å analysere:", ["NRK (252 spørsmål)", "TV2 (98 spørsmål med kategorier)"])
        
        if "NRK" in source_choice:
            current_data = nrk
            source_name = "NRK"
            st.info("🔍 Analyserer kun NRK-data")
        else:
            current_data = tv2
            source_name = "TV2"
            st.info("🔍 Analyserer kun TV2-data")
        
        
        valid_parties = [p for p in parties if p in current_data.columns and not current_data[p].isna().all()]

        if len(valid_parties) == 0:
            st.error(f"Ingen gyldig partidata funnet i {source_name}")
        else:
            # Beregn scores kun for gyldige partier
            party_scores = current_data[valid_parties].sum().sort_values(ascending=False)
            party_averages = current_data[valid_parties].mean().sort_values(ascending=False)
            
            # Debug info
            st.info(f"📊 **Analyserer {len(party_scores)} partier fra {source_name}** (Basert på {len(current_data)} spørsmål)")
            
            col1, col2 = st.columns([2.5, 1.5])
            
            with col1:
                # Vis både sum og gjennomsnitt
                show_sums = st.toggle("Vis totalsummer i stedet for gjennomsnitt", value=False, key="show_sums_separate")
                
                if show_sums:
                    chart_data = party_scores
                    chart_title = f"Partiposisjoner (totalsummer) - {source_name}"
                    y_title = "Total score"
                    scale_info = f"Skala: -{len(current_data)*2} til +{len(current_data)*2}"
                else:
                    chart_data = party_averages  
                    chart_title = f"Partiposisjoner (gjennomsnitt) - {source_name}"
                    y_title = "Gjennomsnittlig posisjon per spørsmål"
                    scale_info = "Skala: -2.0 til +2.0"
                
                # Lag diskrete farger basert på posisjon-range
                def get_position_color(value):
                    if value > 0.5: return COL_POS
                    elif value < -0.5: return COL_NEG  
                    else: return COL_NEU
                
                colors = [get_position_color(val) for val in chart_data.values]
                
                # Forbedret bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=chart_data.index,
                        y=chart_data.values,
                        marker_color=colors,
                        text=[f"{val:+.1f}" if show_sums else f"{val:+.2f}" for val in chart_data.values],
                        textposition='outside',
                        textfont=dict(size=11, color="#111111"),
                        hovertemplate="<b>%{x}</b><br>" +
                                     ("Total: %{y:+.1f}<br>" if show_sums else "Gjennomsnitt: %{y:+.2f}<br>") +
                                     f"Basert på {len(current_data)} spørsmål" +
                                     "<extra></extra>"
                    )
                ])
                
                # Legg til nullinje for referanse
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                
                fig.update_layout(
                    title=f"{chart_title}<br><sub>{scale_info} - Kan ikke sammenlignes med andre kilder</sub>",
                    xaxis_title="Partier",
                    yaxis_title=y_title,
                    xaxis_tickangle=45,
                    plot_bgcolor=BG, 
                    paper_bgcolor=BG,
                    font=dict(color="#111111"),
                    margin=dict(t=80, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forklaringsboks for negative verdier
                if chart_data.min() < 0:
                    st.markdown(f"""
                    <div class="explanation-card">
                        <h4>💡 Hvorfor negative verdier?</h4>
                        <p><strong>Negative verdier betyr IKKE</strong> at partiet er "negativt"!</p>
                        <p><strong>Negative verdier betyr:</strong> Partiet er oftere imot de spesifikke forslagene i {source_name} enn for dem.</p>
                        <p><strong>Positive verdier betyr:</strong> Partiet støtter oftere forslagene.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("📊 Posisjonsoversikt")
                
                # Vis statistikk
                positive_parties = (chart_data > 0).sum()
                negative_parties = (chart_data < 0).sum()
                neutral_parties = (chart_data == 0).sum()
                
                st.metric("Partier med positiv score", positive_parties)
                st.metric("Partier med negativ score", negative_parties)  
                st.metric("Nøytrale partier", neutral_parties)
                
                # Vis range og spredning
                st.metric("Høyeste score", f"{chart_data.max():+.2f}")
                st.metric("Laveste score", f"{chart_data.min():+.2f}")
                st.metric("Spredning", f"{chart_data.max() - chart_data.min():.2f}")
                
                # Rangering
                st.subheader("🏆 Rangering")
                for i, (party, score) in enumerate(chart_data.head(5).items(), 1):
                    direction = "📈" if score > 0 else "📉" if score < 0 else "➡️"
                    score_str = f"{score:+.2f}" if not show_sums else f"{score:+.0f}"
                    st.write(f"{direction} **{i}. {party}** ({score_str})")
                    
            st.caption(f"📊 Basert på {len(current_data)} spørsmål fra {source_name}")

with tab4:
    st.header("Temaoversikt fra TV2")
    
    if 'Kategori' not in tv2.columns:
        st.error("TV2-data mangler kategori-informasjon")
    else:
        create_explanation_card(
            "Fordeling av politiske tema",
            "TV2 organiserer spørsmålene sine i tema som økonomi, familie, helse osv. " +
            "Her ser vi hvilke tema som får mest oppmerksomhet."
        )
        
        # Category distribution
        categories = tv2['Kategori'].dropna()
        category_counts = categories.value_counts()
        category_pct = (category_counts / category_counts.sum() * 100).round(1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create pie chart
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Fordeling av politiske tema"
            )
            
            fig_pie.update_traces(
                textposition='outside',
                textinfo='percent+label',
                marker=dict(line=dict(color='white', width=2)),
                pull=[0.05] * len(category_counts)  # Slight separation of slices
            )
            
            fig_pie.update_layout(
                plot_bgcolor=BG,
                paper_bgcolor=BG,
                font=dict(color="#111111"),
                margin=dict(t=50, b=50, l=100, r=100),  # More margin for outside labels
                showlegend=False  # Remove legend since labels are on chart
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Temastatistikk")
            
            for i, (tema, antall) in enumerate(category_counts.head(5).items()):
                prosent = category_pct[tema]
                st.markdown(f"""
                <div class="explanation-card" style="margin: 5px 0;">
                    <h4 style="margin: 0; font-size: 14px;">#{i+1}. {tema}</h4>
                    <p style="margin: 0; font-size: 12px;">{antall} spørsmål ({prosent}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.metric("Totalt antall tema", len(category_counts))
            st.metric("Største tema", f"{category_counts.index[0]} ({category_counts.iloc[0]} spm)")
            
            # Balance check
            max_pct = category_pct.max()
            if max_pct > 30:
                st.warning(f"⚠️ Tema '{category_counts.index[0]}' dominerer ({max_pct}%)")
            else:
                st.success("✅ Godt balanserte tema")
        
        
        st.subheader("Hvilke tema skaper mest uenighet?")
        
        tv2_with_analysis = prep_data_with_friendly_labels(tv2, parties)
        
        if len(tv2_with_analysis) > 0:
            topic_controversy = tv2_with_analysis.groupby('Kategori')['disagreement'].agg(['mean', 'max', 'count']).round(3)
            topic_controversy.columns = ['Gj_uenighet', 'Maks_uenighet', 'Antall_sporsmal']
            topic_controversy = topic_controversy.sort_values('Gj_uenighet', ascending=False)
            
            # DEBUG: Sjekk om alle verdier er like
            unique_values = topic_controversy['Gj_uenighet'].nunique()
            if unique_values == 1:
                st.warning(f"⚠️ Alle tema har identisk uenighet ({topic_controversy['Gj_uenighet'].iloc[0]:.3f})")
            
            # Forbedret visualisering med bedre skalering
            fig_controversy = px.bar(
                topic_controversy.reset_index(),
                x='Kategori',
                y='Gj_uenighet',
                color='Gj_uenighet',
                color_continuous_scale=get_consistent_color_scale(),
                title="Gjennomsnittlig uenighet per tema",
                hover_data={'Antall_sporsmal': True, 'Maks_uenighet': ':.3f'},
                text='Gj_uenighet'  # Vis verdier på stolpene
            )
            
            # Eksplisitt y-akse skalering
            y_min = max(0, topic_controversy['Gj_uenighet'].min() - 0.1)
            y_max = topic_controversy['Gj_uenighet'].max() + 0.1
            
            fig_controversy.update_traces(
                texttemplate='%{text:.3f}', 
                textposition='outside',
                textfont_size=10
            )
            
            fig_controversy.update_layout(
                xaxis_tickangle=45,
                plot_bgcolor=BG, 
                paper_bgcolor=BG,
                xaxis_title="Politisk tema",
                yaxis_title="Gjennomsnittlig uenighet mellom partier",
                yaxis=dict(range=[y_min, y_max]),  # Eksplisitt range
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_controversy, use_container_width=True)
            
            
            st.subheader("📊 Detaljert uenighets-tabell")
            display_table = topic_controversy.copy()
            display_table['Uenighet_pct'] = (display_table['Gj_uenighet'] * 100 / display_table['Gj_uenighet'].max()).round(1)
            display_table.columns = ['Gj. uenighet', 'Maks uenighet', 'Antall spm', 'Relativ uenighet (%)']
            st.dataframe(display_table, use_container_width=True)
            
           
            st.write("**Mest kontroversielle tema:**")
            for i, (tema, stats) in enumerate(topic_controversy.head(3).iterrows(), 1):
                uenighet_pct = (stats['Gj_uenighet'] / topic_controversy['Gj_uenighet'].max() * 100)
                st.write(f"{i}. **{tema}** - Uenighet: {stats['Gj_uenighet']:.3f} ({uenighet_pct:.0f}% av maks, {stats['Antall_sporsmal']} spørsmål)")
        else:
            st.error("Ingen data tilgjengelig for analyse")

with tab5:
    st.header("⚖️ Hva skjer hvis vi vekter tema ulikt?")
    
    create_tv2_only_explanation()
    
    create_explanation_card(
        "Utforsk vektingseffekter",
        "I virkeligheten bryr folk seg mer om noen tema enn andre. " +
        "Hva skjer hvis vi gir økonomi-spørsmål dobbelt så mye vekt som andre tema?",
        "⚖️"
    )
    
    create_score_interpretation_box()
    
    if 'Kategori' not in tv2.columns:
        st.error("Denne analysen krever TV2-data med kategorier")
    else:
        # Vis faktisk fordeling av alle kategorier først
        full_category_counts = tv2['Kategori'].value_counts()
        st.info(f"""
        📊 **Alle kategorier i TV2-data:**
        {', '.join([f"{cat}: {count} spm" for cat, count in full_category_counts.items()])}
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_questions = len(tv2)
            n_questions = st.slider(
                "Antall spørsmål å inkludere", 
                10, 
                max_questions, 
                min(50, max_questions), 
                key="weight_q",
                help=f"Maks {max_questions} spørsmål tilgjengelig"
            )
        
        with col2:
            available_categories = tv2['Kategori'].dropna().unique()
            boost_category = st.selectbox("Tema å vektlegge ekstra", available_categories)
        
        with col3:
            weight_factor = st.slider("Hvor mye ekstra vekt?", 1.0, 5.0, 2.0, 0.5)
        
        # Bruk enten sample eller alle data
        if n_questions >= len(tv2):
            subset = tv2.copy()
            st.success(f"✅ Bruker ALLE {len(tv2)} spørsmål fra TV2")
        else:
            subset = tv2.sample(n=n_questions, random_state=42)
            st.info(f"📊 Bruker tilfeldig utvalg på {len(subset)} spørsmål")
        
        # Vis faktisk fordeling i utvalget
        actual_category_counts = subset['Kategori'].value_counts()
        boost_questions_in_sample = actual_category_counts.get(boost_category, 0)
        
        st.write("**Kategorifordeling i utvalget:**")
        for cat, count in actual_category_counts.items():
            if cat == boost_category:
                st.write(f"🎯 **{cat}: {count} spørsmål** (får {weight_factor}x vekt)")
            else:
                st.write(f"• {cat}: {count} spørsmål")
        
        if boost_questions_in_sample == 0:
            st.error(f"❌ Ingen spørsmål fra '{boost_category}' i utvalget! Prøv å øke antall spørsmål.")
        
        # Original scores
        original_scores = subset[parties].sum()
        original_averages = subset[parties].mean()
        
        # Weighted scores
        weights = subset['Kategori'].apply(lambda cat: weight_factor if cat == boost_category else 1.0)
        weighted_scores = (subset[parties].multiply(weights, axis=0)).sum()
        weighted_averages = weighted_scores / len(subset)
        
        # Create comparison
        comparison = pd.DataFrame({
            "Før_vekting_sum": original_scores,
            "Før_vekting_snitt": original_averages,
            "Etter_vekting_sum": weighted_scores,
            "Etter_vekting_snitt": weighted_averages,
            "Endring_sum": weighted_scores - original_scores,
            "Endring_snitt": weighted_averages - original_averages
        }).sort_values("Før_vekting_sum", ascending=False)
        
        col_left, col_right = st.columns([2.5, 1.5])
        
        with col_left:
            show_averages = st.toggle("Vis gjennomsnitt per spørsmål i stedet for totalsummer", value=True)
            
            create_how_to_read_box("party_comparison")
            
            fig = go.Figure()
            
            if show_averages:
                y_col_before = "Før_vekting_snitt" 
                y_col_after = "Etter_vekting_snitt"
                y_title = "Gjennomsnittlig posisjon per spørsmål"
                title_suffix = "(gjennomsnitt per spørsmål)"
            else:
                y_col_before = "Før_vekting_sum"
                y_col_after = "Etter_vekting_sum"  
                y_title = "Total score"
                title_suffix = "(totalsummer)"
            
            # Before weighting
            fig.add_trace(go.Bar(
                name='Før vekting',
                x=comparison.index,
                y=comparison[y_col_before],
                marker_color=COL_NEU,
                text=[f"{val:.1f}" for val in comparison[y_col_before]],
                textposition='auto'
            ))
            
            # After weighting  
            fig.add_trace(go.Bar(
                name=f'Etter {weight_factor}× vekt på "{boost_category}"',
                x=comparison.index,
                y=comparison[y_col_after],
                marker_color=COL_POS,
                text=[f"{val:.1f}" for val in comparison[y_col_after]],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Effekt av å vektlegge '{boost_category}' {weight_factor}× høyere {title_suffix}",
                xaxis_title="Partier",
                yaxis_title=y_title,
                barmode='group',
                xaxis_tickangle=45,
                plot_bgcolor=BG, paper_bgcolor=BG
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("Vektings-påvirkning")
            
            change_col = "Endring_snitt" if show_averages else "Endring_sum"
            max_change = comparison[change_col].abs().max()
            most_affected = comparison.loc[comparison[change_col].abs().idxmax()]
            
            change_value = most_affected[change_col]
            change_format = f"{change_value:+.2f}" if show_averages else f"{change_value:+.1f}"
            
            st.metric("Største endring", f"{change_format} {'per spm' if show_averages else 'poeng'}")
            st.metric("Mest påvirket parti", most_affected.name)
            st.metric("Spørsmål i vektet tema", f"{boost_questions_in_sample}/{len(subset)}")
            
            # Impact assessment
            threshold_major = 0.3 if show_averages else 5
            threshold_minor = 0.1 if show_averages else 2
            
            if boost_questions_in_sample == 0:
                st.error("❌ Ingen effekt - tema ikke i utvalget")
            elif max_change > threshold_major:
                st.warning(f"⚠️ Stor påvirkning!")
            elif max_change > threshold_minor:
                st.info(f"📊 Moderat påvirkning")
            else:
                st.success(f"✅ Minimal påvirkning")
with tab6:
    st.header("Hva om vi fjerner et tema helt?")
    
    create_tv2_only_explanation()
    
    create_explanation_card(
        "Scenario-analyse",
        "Noen ganger er det nyttig å se hva som skjer hvis vi ignorerer visse tema helt. " +
        "For eksempel: Hvordan ser partiene ut hvis vi ser bort fra alle økonomi-spørsmål?"
    )
    
    create_score_interpretation_box()
    
    if 'Kategori' not in tv2.columns:
        st.error("Denne analysen krever TV2-data med kategorier")
    else:
        # Vis faktisk fordeling av alle kategorier først
        full_category_counts = tv2['Kategori'].value_counts()
        st.info(f"""
        📊 **Alle kategorier i TV2-data:**
        {', '.join([f"{cat}: {count} smp" for cat, count in full_category_counts.items()])}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_questions = len(tv2)
            n_questions_scenario = st.slider(
                "Antall spørsmål", 
                20, 
                max_questions, 
                min(60, max_questions), 
                key="scenario_q",
                help=f"Anbefaler minst 40 for meningsfulle resultater. Maks {max_questions}."
            )
        
        with col2:
            available_cats = tv2['Kategori'].dropna().unique()
            remove_category = st.selectbox("Tema å fjerne", available_cats)
        
        # Bruk enten sample eller alle data
        if n_questions_scenario >= len(tv2):
            full_data = tv2.copy()
            st.success(f"✅ Bruker ALLE {len(tv2)} spørsmål fra TV2")
        else:
            full_data = tv2.sample(n=n_questions_scenario, random_state=42)
            st.info(f"📊 Bruker tilfeldig utvalg på {len(full_data)} spørsmål")
        
        # Vis faktisk fordeling FØR fjerning
        full_category_counts_sample = full_data['Kategori'].value_counts()
        questions_to_remove = full_category_counts_sample.get(remove_category, 0)
        
        st.write("**Kategorifordeling FØR fjerning:**")
        for cat, count in full_category_counts_sample.items():
            if cat == remove_category:
                st.write(f"❌ **{cat}: {count} spørsmål** (fjernes)")
            else:
                st.write(f"✅ {cat}: {count} spørsmål (beholdes)")
        
        if questions_to_remove == 0:
            st.error(f"❌ Ingen spørsmål fra '{remove_category}' i utvalget! Prøv å øke antall spørsmål.")
        
        # Fjern kategorien
        without_category = full_data[full_data['Kategori'] != remove_category].copy()
        
        # Beregn scores
        full_scores = full_data[parties].sum()
        full_averages = full_data[parties].mean()
        
        if len(without_category) > 0:
            reduced_scores = without_category[parties].sum()
            reduced_averages = without_category[parties].mean()
        else:
            # Hvis alle spørsmål fjernes
            reduced_scores = pd.Series(0, index=parties)
            reduced_averages = pd.Series(0, index=parties)
        
        # Vis resultat av fjerning
        actual_removed = len(full_data) - len(without_category)
        st.write(f"**Faktisk fjernet: {actual_removed} spørsmål**")
        
        if actual_removed != questions_to_remove:
            st.error(f"⚠️ Feil: Forventet å fjerne {questions_to_remove}, men fjernet {actual_removed}")
        else:
            st.success(f"✅ Korrekt: Fjernet alle {actual_removed} spørsmål fra '{remove_category}'")
        
        scenario_comparison = pd.DataFrame({
            "Med_alle_tema_sum": full_scores,
            "Med_alle_tema_snitt": full_averages,
            f"Uten_{remove_category}_sum": reduced_scores,
            f"Uten_{remove_category}_snitt": reduced_averages,
            "Endring_sum": reduced_scores - full_scores,
            "Endring_snitt": reduced_averages - full_averages
        }).sort_values("Med_alle_tema_sum", ascending=False)
        
        col_left, col_right = st.columns([2.5, 1.5])
        
        with col_left:
            show_averages_scenario = st.toggle("Vis gjennomsnitt per spørsmål i stedet for totalsummer", value=True, key="scenario_avg_toggle")
            
            fig_scenario = go.Figure()
            
            if show_averages_scenario:
                y_col_full = "Med_alle_tema_snitt"
                y_col_reduced = f"Uten_{remove_category}_snitt"
                y_title = "Gjennomsnittlig posisjon per spørsmål"
                title_suffix = "(gjennomsnitt per spørsmål)"
            else:
                y_col_full = "Med_alle_tema_sum"
                y_col_reduced = f"Uten_{remove_category}_sum"
                y_title = "Total score"
                title_suffix = "(totalsummer)"
            
            # With all themes
            fig_scenario.add_trace(go.Bar(
                name='Med alle tema',
                x=scenario_comparison.index,
                y=scenario_comparison[y_col_full],
                marker_color=COL_NEU,
                text=[f"{val:.1f}" for val in scenario_comparison[y_col_full]],
                textposition='auto'
            ))
            
            # Without selected theme
            fig_scenario.add_trace(go.Bar(
                name=f'Uten "{remove_category}"',
                x=scenario_comparison.index, 
                y=scenario_comparison[y_col_reduced],
                marker_color=COL_NEG,
                text=[f"{val:.1f}" for val in scenario_comparison[y_col_reduced]],
                textposition='auto'
            ))
            
            fig_scenario.update_layout(
                title=f"Scenario: Fjernet {actual_removed} spørsmål om '{remove_category}' {title_suffix}",
                xaxis_title="Partier",
                yaxis_title=y_title,
                barmode='group',
                xaxis_tickangle=45,
                plot_bgcolor=BG, paper_bgcolor=BG
            )
            
            st.plotly_chart(fig_scenario, use_container_width=True)
        
        with col_right:
            st.subheader("🎯 Scenario-konsekvenser")
            
            change_col = "Endring_snitt" if show_averages_scenario else "Endring_sum"
            max_impact = scenario_comparison[change_col].abs().max()
            most_impacted = scenario_comparison.loc[scenario_comparison[change_col].abs().idxmax()]
            
            impact_value = most_impacted[change_col]
            impact_format = f"{impact_value:+.2f}" if show_averages_scenario else f"{impact_value:+.1f}"
            
            st.metric("Største påvirkning", f"{impact_format}")
            st.metric("Mest påvirket parti", most_impacted.name)
            st.metric("Spørsmål fjernet", f"{actual_removed}/{len(full_data)}")
            
            # Impact assessment
            threshold_major = 0.3 if show_averages_scenario else 5
            threshold_minor = 0.1 if show_averages_scenario else 2
            
            if actual_removed == 0:
                st.info("ℹ️ Ingen spørsmål fjernet")
            elif actual_removed == len(full_data):
                st.error("🚨 Alle spørsmål fjernet!")
            elif max_impact < threshold_minor:
                st.success(f"✅ Minimal påvirkning")
            elif max_impact < threshold_major:
                st.warning(f"⚠️ Moderat påvirkning") 
            else:
                st.error(f"🚨 Stor påvirkning!")

with tab7:
    st.header("🤖 AI & Datakvalitet")
    st.write("Utforsk hvordan datakvalitet påvirker AI-resultater gjennom valgomatdata.")
    
    create_explanation_card(
        "Hvorfor er datakvalitet viktig?",
        "AI-systemer er kun så gode som dataene de er trent på. Dårlig datakvalitet kan føre til " +
        "feilaktige konklusjoner, skjeve algoritmer og upålitelige anbefalinger. " +
        "Her kan du utforske hvordan ulike datakualitetsutfordringer påvirker AI-ytelse.",
        "🔍"
    )
    
    # Data quality dimensions selector
    quality_dim = st.selectbox(
        "Velg datakvalitetsdimensjon:",
        ["📊 Oversikt", "🔄 Intern konsistens", "📋 Kompletthet", "⚖️ Balanse", 
         "⏰ Aktualitet", "✅ Validitet", "📊 Kildefordeling"]
    )
    
    if quality_dim == "📊 Oversikt":
        st.subheader("De 6 dimensjonene av datakvalitet")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            quality_scores = calculate_quality_scores(parties, nrk, tv2)
            dimensions = list(quality_scores.keys())
            scores = list(quality_scores.values())
            
            # Create interactive radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name='Valgomatdata Score',
                line_color=COL_POS,
                fillcolor=f'rgba(151, 210, 236, 0.3)',
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>"
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont_size=10
                    )),
                showlegend=False,
                title="Datakvalitetsprofil for Valgomatdata",
                plot_bgcolor=BG,
                paper_bgcolor=BG,
                font=dict(color="#111111"),
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Add interactive dimension selector
            st.subheader("🔍 Utforsk dimensjoner")
            selected_dimension = st.selectbox(
                "Klikk for å utforske en datakvalitetsdimensjon:",
                options=dimensions,
                help="Velg en dimensjon fra listen for å se detaljert informasjon"
            )
            
            if selected_dimension:
                clicked_score = quality_scores[selected_dimension]
                st.success(f"🎯 **{selected_dimension}** valgt: {clicked_score:.0f}%")
                
                
                dimension_details = {
                    "Intern_konsistens": {
                        "description": "Måler hvor konsistente partier er i sine posisjoner",
                        "calculation": "Basert på hvor ofte partier endrer posisjon drastisk innenfor hver kilde",
                        "interpretation": "Høy score = partier er konsistente i sine standpunkter"
                    },
                    "Kompletthet": {
                        "description": "Måler hvor mye av dataene som faktisk er tilgjengelig",
                        "calculation": "Prosent av celler som ikke er tomme eller manglende",
                        "interpretation": "Høy score = få manglende verdier"
                    },
                    "Balanse": {
                        "description": "Måler om posisjoner er jevnt fordelt på skalaen",
                        "calculation": "Basert på hvor jevnt fordelt posisjonene -2 til +2 er",
                        "interpretation": "Høy score = balansert fordeling, lav bias i spørsmålsformulering"
                    },
                    "Aktualitet": {
                        "description": "Måler hvor oppdaterte dataene er",
                        "calculation": "Antatt at data er 6 måneder gamle, 2% frafall per måned",
                        "interpretation": "Høy score = ferske data, lav score = utdaterte data"
                    },
                    "Validitet": {
                        "description": "Måler om dataene har korrekt format og gyldige verdier",
                        "calculation": "Prosent av verdier som er i forventet range [-2, +2]",
                        "interpretation": "Høy score = få formatfeil eller ugyldige verdier"
                    },
                    "Kildebalanse": {
                        "description": "Måler balansen mellom datamengde fra hver kilde",
                        "calculation": "Basert på hvor balansert fordelingen mellom NRK og TV2 er",
                        "interpretation": "Høy score = god balanse mellom kilder"
                    }
                }
                
                if selected_dimension in dimension_details:
                    details = dimension_details[selected_dimension]
                    
                    with st.expander(f"📋 Detaljer om {selected_dimension}", expanded=True):
                        st.write(f"**Beskrivelse:** {details['description']}")
                        st.write(f"**Beregning:** {details['calculation']}")  
                        st.write(f"**Tolkning:** {details['interpretation']}")
                        
                        # Visual indicator
                        if clicked_score >= 90:
                            st.success("🟢 Utmerket kvalitet")
                        elif clicked_score >= 75:
                            st.warning("🟡 God kvalitet")
                        elif clicked_score >= 60:
                            st.warning("🟠 Akseptabel kvalitet")
                        else:
                            st.error("🔴 Lav kvalitet")
        
        with col2:
            st.subheader("Kvalitetsvurdering")
            overall_score = np.mean(scores)
            
            st.metric("Samlet kvalitetsscore", f"{overall_score:.0f}%")
            
            if overall_score >= 90:
                st.success("🟢 Utmerket datakvalitet!")
                recommendation = "Dataene er klare for avanserte AI-analyser"
            elif overall_score >= 75:
                st.warning("🟡 God kvalitet - noen forbedringer mulige")
                recommendation = "Brukbar for de fleste AI-applikasjoner, men kan forbedres"
            elif overall_score >= 60:
                recommendation = "Krever forbedringer før pålitelig AI-bruk"
                st.warning("🟠 Akseptabel kvalitet")
            else:
                st.error("🔴 Lav kvalitet - store forbedringer nødvendig")
                recommendation = "Omfattende datarengjøring nødvendig"
            
            st.info(f"**Anbefaling:** {recommendation}")
            
            
            score_df = pd.DataFrame({"Dimensjon": dimensions, "Score": scores}).sort_values("Score", ascending=False)
            
            st.write("**🏆 Best:**")
            st.write(f"{score_df.iloc[0]['Dimensjon']}: {score_df.iloc[0]['Score']:.0f}%")
            
            st.write("**🎯 Trenger forbedring:**")  
            st.write(f"{score_df.iloc[-1]['Dimensjon']}: {score_df.iloc[-1]['Score']:.0f}%")
        
        create_explanation_card(
            "Tips for interaksjon",
            "Velg dimensjoner fra dropdown-menyen ovenfor for å se detaljert informasjon om " +
            "hvordan scoren beregnes og hva den betyr for AI-kvalitet."
        )
    
    elif quality_dim == "🔄 Intern konsistens":
        st.subheader("Intern konsistens: Hvor stabile er partiene?")
        
        create_explanation_card(
            "Problemet med inkonsistente partier",
            "Hvis partier bytter mening ofte eller svarer tilfeldig, blir AI-systemene forvirret. " +
            "Konsistente partier indikerer pålitelige data som AI kan lære fra.",
            "⚠️", "warning"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NRK: Partikonsistens")
            nrk_consistency = []
            for party in parties[:6]:  # Show top 6 for space
                if party in nrk.columns:
                    party_positions = nrk[party].dropna()
                    if len(party_positions) > 1:
                        position_changes = abs(party_positions.diff()).dropna()
                        large_changes = (position_changes > 2).sum()
                        consistency = max(0, 100 - (large_changes / len(position_changes)) * 100)
                        nrk_consistency.append((party, consistency))
            
            if nrk_consistency:
                nrk_df = pd.DataFrame(nrk_consistency, columns=['Parti', 'Konsistens'])
                fig_nrk = px.bar(nrk_df, x='Parti', y='Konsistens', 
                               title="NRK: Partikonsistens (%)", 
                               color='Konsistens',
                               color_continuous_scale=get_consistent_color_scale())
                fig_nrk.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                st.plotly_chart(fig_nrk, use_container_width=True)
        
        with col2:
            st.subheader("TV2: Partikonsistens")
            tv2_consistency = []
            for party in parties[:6]:
                if party in tv2.columns:
                    party_positions = tv2[party].dropna()
                    if len(party_positions) > 1:
                        position_changes = abs(party_positions.diff()).dropna()
                        large_changes = (position_changes > 2).sum()
                        consistency = max(0, 100 - (large_changes / len(position_changes)) * 100)
                        tv2_consistency.append((party, consistency))
            
            if tv2_consistency:
                tv2_df = pd.DataFrame(tv2_consistency, columns=['Parti', 'Konsistens'])
                fig_tv2 = px.bar(tv2_df, x='Parti', y='Konsistens',
                               title="TV2: Partikonsistens (%)",
                               color='Konsistens', 
                               color_continuous_scale=get_consistent_color_scale())
                fig_tv2.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                st.plotly_chart(fig_tv2, use_container_width=True)
        
        st.info("💡 **Konsistens over 80%** = Parti bytter sjelden mening drastisk")
        st.info("⚠️ **Konsistens under 60%** = Parti kan være upålitelig eller data har feil")
    
    elif quality_dim == "📋 Kompletthet":
        st.subheader("Kompletthet: Hvor mye data mangler?")
        
        create_explanation_card(
            "Problemet med manglende data",
            "AI-systemer kan bare lære av dataene de får. Manglende informasjon kan føre til " +
            "feilaktige konklusjoner og unfair behandling av grupper som er underrepresenterte.",
            "⚠️", "warning"  
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NRK: Datakomplettphet")
            nrk_completeness = nrk[parties].notna().mean() * 100
            
            fig_nrk_comp = px.bar(
                x=parties, y=nrk_completeness.values,
                title="NRK: Kompletthet per parti (%)",
                color=nrk_completeness.values,
                color_continuous_scale=get_consistent_color_scale()
            )
            fig_nrk_comp.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, xaxis_tickangle=45)
            st.plotly_chart(fig_nrk_comp, use_container_width=True)
            
            st.metric("NRK samlet kompletthet", f"{nrk_completeness.mean():.1f}%")
        
        with col2:
            st.subheader("TV2: Datakomplettphet") 
            tv2_completeness = tv2[parties].notna().mean() * 100
            
            fig_tv2_comp = px.bar(
                x=parties, y=tv2_completeness.values,
                title="TV2: Kompletthet per parti (%)",
                color=tv2_completeness.values,
                color_continuous_scale=get_consistent_color_scale()
            )
            fig_tv2_comp.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, xaxis_tickangle=45)
            st.plotly_chart(fig_tv2_comp, use_container_width=True)
            
            st.metric("TV2 samlet kompletthet", f"{tv2_completeness.mean():.1f}%")
        
        # AI impact assessment
        overall_completeness = (nrk_completeness.mean() + tv2_completeness.mean()) / 2
        if overall_completeness > 95:
            st.success("🟢 Utmerket kompletthet - AI kan lære optimalt")
        elif overall_completeness > 85:
            st.warning("🟡 God kompletthet - AI håndterer godt")
        else:
            st.error("🔴 Lav kompletthet - AI-resultater kan bli skjeve")
        
        # Interactive missing data simulation
        st.subheader("Simuler effekt av manglende data")
        missing_pct = st.slider("Prosent manglende data:", 0, 50, 20, 5)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show impact of missing data on party rankings
            complete_data = tv2.head(20)[parties].sum().sort_values(ascending=False)
            
            # Simulate missing data by randomly removing questions
            np.random.seed(42)
            n_total = 20
            n_missing = int(missing_pct / 100 * n_total)
            available_indices = np.random.choice(20, n_total - n_missing, replace=False)
            
            incomplete_data = tv2.iloc[available_indices][parties].sum().sort_values(ascending=False)
            
            # Create comparison visualization
            fig_comp = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Komplette data", f"Med {missing_pct}% manglende data"],
                horizontal_spacing=0.15
            )
            
            fig_comp.add_trace(
                go.Bar(x=complete_data.index, y=complete_data.values, 
                      marker_color=COL_POS, name="Komplett"),
                row=1, col=1
            )
            
            fig_comp.add_trace(
                go.Bar(x=incomplete_data.index, y=incomplete_data.values, 
                      marker_color=COL_NEG, name="Ufullstendig"),
                row=1, col=2
            )
            
            fig_comp.update_layout(
                title=f"Hvordan {missing_pct}% manglende data påvirker AI-beslutninger",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111"),
                showlegend=False
            )
            
            fig_comp.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            # Calculate AI impact metrics
            ai_accuracy = max(45, 92 - missing_pct * 0.6)
            ai_confidence = max(20, 95 - missing_pct * 1.2)
            
            st.subheader("AI Påvirkning")
            st.metric("Anbefaling-nøyaktighet", f"{ai_accuracy:.0f}%", 
                     delta=f"{ai_accuracy-92:.0f}% vs komplette data")
            st.metric("AI Tillit", f"{ai_confidence:.0f}%",
                     delta=f"{ai_confidence-95:.0f}% vs optimalt")
            
            # Calculate ranking changes
            if len(incomplete_data) > 0:
                complete_ranking = complete_data.rank(ascending=False, method='min')
                incomplete_ranking = incomplete_data.rank(ascending=False, method='min')
                
                # Find common parties for comparison
                common_parties = set(complete_ranking.index) & set(incomplete_ranking.index)
                if common_parties:
                    ranking_changes = []
                    for party in common_parties:
                        change = abs(complete_ranking[party] - incomplete_ranking[party])
                        ranking_changes.append(change)
                    
                    avg_rank_change = np.mean(ranking_changes)
                    st.metric("Gj.snitt rangeringsendring", f"{avg_rank_change:.1f} plasser")
            
            # Data completeness status
            if missing_pct < 5:
                st.success("🟢 Utmerket kompletthet")
            elif missing_pct < 15:
                st.warning("🟡 Akseptabel kompletthet") 
            else:
                st.error("🔴 Lav kompletthet - AI upålitelig")
        
        # Missing data pattern analysis
        st.subheader("Typer manglende data")
        
        patterns = ["Tilfeldig", "Systematisk (etter kategori)", "Skjevt (mot spesifikke partier)"]
        selected_pattern = st.selectbox("Type manglende data:", patterns)
        
        if selected_pattern == "Tilfeldig":
            st.info("💡 **Tilfeldig manglende**: Minst skadelig - AI kan ofte kompensere")
        elif selected_pattern == "Systematisk (etter kategori)":
            st.warning("⚠️ **Systematisk manglende**: Mer skadelig - skaper kunnskapshull")
        else:
            st.error("🚨 **Skjevt manglende**: Mest skadelig - skaper urettferdige AI-anbefalinger")
    
    elif quality_dim == "⚖️ Balanse":
        st.subheader("Balanse: Hvor jevnt fordelte er posisjonene?")
        
        create_explanation_card(
            "Problemet med ubalanserte data",
            "Hvis de fleste spørsmål leder til samme type svar, kan AI-systemer bli biased. " +
            "En balansert fordeling av posisjoner indikerer mer nøytrale spørsmålsformuleringer.",
            "⚠️", "warning"
        )
        
        st.write("Viser fordelingen av partiposisjoner på skalaen -2 til +2")
        
        all_positions = pd.concat([nrk[parties].stack(), tv2[parties].stack()]).dropna()
        position_counts = all_positions.value_counts().sort_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_balance = px.bar(
                x=position_counts.index, y=position_counts.values,
                title="Fordeling av alle partiposisjoner",
                labels={'x': 'Posisjon', 'y': 'Antall'},
                color=position_counts.index,
                color_continuous_scale=get_consistent_color_scale()
            )
            fig_balance.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig_balance, use_container_width=True)
        
        with col2:
            # Check balance
            expected_per_position = len(all_positions) / len(position_counts)
            max_deviation = max(abs(count - expected_per_position) for count in position_counts.values)
            balance_score = max(0, 100 - (max_deviation / expected_per_position) * 50)
            
            st.metric("Balanse-score", f"{balance_score:.0f}%")
            
            if balance_score > 80:
                st.success("🟢 God balanse - liten bias i spørsmålsformulering")
            elif balance_score > 60:
                st.warning("🟡 Moderat ubalanse - noe bias mulig")
            else:
                st.error("🔴 Høy ubalanse - betydelig bias i spørsmål")
            
            # Show distribution details
            st.write("**Posisjonesfordeling:**")
            for pos, count in position_counts.items():
                pct = count / len(all_positions) * 100
                st.write(f"Posisjon {pos}: {count} ({pct:.1f}%)")
        
        # Interactive bias demonstration
        st.subheader("Simuler skjevhet i spørsmålsformulering")
        
        bias_direction = st.selectbox("Type skjevhet:", ["Nøytral (ingen bias)", "Pro-bias (leder til JA)", "Anti-bias (leder til NEI)"])
        bias_strength = st.slider("Styrke på bias:", 0.0, 1.0, 0.3, 0.1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if bias_direction == "Nøytral (ingen bias)":
                st.write("**Nøytral formulering:**")
                st.write("'Bør Norge øke skatten på høye inntekter?'")
                # Simulate balanced distribution
                np.random.seed(42)
                neutral_dist = np.random.choice([-2, -1, 0, 1, 2], size=1000, p=[0.2, 0.2, 0.2, 0.2, 0.2])
            elif bias_direction == "Pro-bias (leder til JA)":
                st.write("**Pro-bias formulering:**")
                st.write("'Bør Norge sørge for mer rettferdig skattefordeling ved å øke skatten for de rikeste?'")
                # Simulate pro-biased distribution
                np.random.seed(42)
                bias_factor = bias_strength
                probs = [0.1, 0.15, 0.2, 0.25 + bias_factor*0.2, 0.3 + bias_factor*0.3]
                probs = np.array(probs) / sum(probs)  # Normalize
                neutral_dist = np.random.choice([-2, -1, 0, 1, 2], size=1000, p=probs)
            else:  # Anti-bias
                st.write("**Anti-bias formulering:**")  
                st.write("'Bør Norge belaste vanlige arbeidsfolk med enda høyere skatter for de rike?'")
                # Simulate anti-biased distribution
                np.random.seed(42)
                bias_factor = bias_strength
                probs = [0.3 + bias_factor*0.3, 0.25 + bias_factor*0.2, 0.2, 0.15, 0.1]
                probs = np.array(probs) / sum(probs)  # Normalize
                neutral_dist = np.random.choice([-2, -1, 0, 1, 2], size=1000, p=probs)
            
            sim_counts = pd.Series(neutral_dist).value_counts().sort_index()
            fig_sim = px.bar(
                x=sim_counts.index, y=sim_counts.values,
                title="Simulert svarfordeling",
                color=sim_counts.index,
                color_continuous_scale=get_consistent_color_scale()
            )
            fig_sim.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig_sim, use_container_width=True)
        
        with col2:
            # Calculate bias metrics
            mean_position = np.mean(neutral_dist)
            st.metric("Gjennomsnittlig posisjon", f"{mean_position:.2f}")
            
            if abs(mean_position) < 0.2:
                st.success("🟢 Nøytral fordeling")
            elif mean_position > 0.5:
                st.warning("🟡 Pro-bias oppdaget")
            elif mean_position < -0.5:
                st.warning("🟡 Anti-bias oppdaget")
            else:
                st.info("📊 Svak bias")
            
            # AI impact
            ai_bias_score = max(0, 100 - abs(mean_position) * 50)
            st.metric("AI Objektivitet", f"{ai_bias_score:.0f}%")
            
            if ai_bias_score > 80:
                st.success("AI vil lære balanserte mønstre")
            elif ai_bias_score > 60:
                st.warning("AI kan lære skjeve mønstre")
            else:
                st.error("AI vil lære sterkt biased mønstre")
    
    elif quality_dim == "⏰ Aktualitet":
        st.subheader("Aktualitet: AI-ytelse med utdaterte data")
        
        create_explanation_card(
            "Problemet med utdaterte data",
            "AI-modeller som er trent på gamle data kan gi anbefalinger som ikke reflekterer " +
            "dagens situasjon. Spesielt i politikk endrer opinioner seg raskt.",
            "⚠️", "warning"
        )
        
        # Simulate data aging effect
        months_old = st.slider("Alder på data (måneder):", 0, 24, 6, 1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show relevance decay over time
            months = list(range(0, 25, 3))
            stable_relevance = [max(0, 100 - m * 1.5) for m in months]  # Gradual decay
            dynamic_relevance = [max(0, 100 - m * 4) for m in months]   # Faster decay
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=months, y=stable_relevance,
                mode='lines+markers',
                name='Stabile tema (økonomi)',
                line=dict(color=COL_POS, width=3),
                marker=dict(size=8)
            ))
            fig_time.add_trace(go.Scatter(
                x=months, y=dynamic_relevance,
                mode='lines+markers',
                name='Dynamiske tema (teknologi)',
                line=dict(color=COL_NEG, width=3),
                marker=dict(size=8)
            ))
            
            # Add vertical line for current data age
            fig_time.add_vline(x=months_old, line_dash="dash", line_color="red", 
                              annotation_text=f"Dine data: {months_old} mnd")
            
            fig_time.update_layout(
                title="Datarelevans over tid",
                xaxis_title="Måneder siden innsamling",
                yaxis_title="Relevans for AI (%)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Calculate current relevance and AI impact
            stable_relevance = max(0, 100 - months_old * 1.5)
            dynamic_relevance = max(0, 100 - months_old * 4)
            avg_relevance = (stable_relevance + dynamic_relevance) / 2
            
            st.subheader("Aktualitet-påvirkning")
            st.metric("Stabile tema-relevans", f"{stable_relevance:.0f}%")
            st.metric("Dynamiske tema-relevans", f"{dynamic_relevance:.0f}%")
            st.metric("Samlet data-relevans", f"{avg_relevance:.0f}%")
            
            ai_performance = max(20, 95 - (100 - avg_relevance) * 0.8)
            st.metric("AI Ytelse", f"{ai_performance:.0f}%", 
                     delta=f"{ai_performance-95:.0f}% vs ferske data")
            
            # Timeliness recommendations
            if months_old < 3:
                st.success("🟢 Ferske data - AI yter optimalt")
            elif months_old < 12:
                st.warning("🟡 Aldrende data - vurder oppdatering")
            else:
                st.error("🔴 Gamle data - AI-anbefalinger kan være utdaterte")
        
        # Show examples of how politics change over time
        st.subheader("Eksempler på politisk endring over tid")
        
        time_examples = {
            "Klimapolitikk": "Fra perifert tema til hovedsak på 10 år",
            "Digitale rettigheter": "Ny kategori som ikke eksisterte før", 
            "Pandemihåndtering": "Helt ukjent tema før 2020",
            "AI-regulering": "Fremvoksende tema de siste 2-3 årene"
        }
        
        for topic, description in time_examples.items():
            st.write(f"**{topic}:** {description}")
    
    elif quality_dim == "✅ Validitet":
        st.subheader("Validitet: AI-robusthet mot ugyldige data")
        
        create_explanation_card(
            "Problemet med ugyldige data",
            "Data med feil format, impossible verdier eller logiske motsigelser kan få AI-systemer " +
            "til å lære feilaktige mønstre eller krasje helt.",
            "⚠️", "warning"
        )
        
        # Simulate different types of validity issues
        validity_issue = st.selectbox("Type validitetsproblem:", 
                                    ["Formatfeil", "Verdier utenfor rekkevidde", "Logiske motsigelser", "Kodingsproblemer"])
        
        error_rate = st.slider("Feilrate (%):", 0, 25, 5, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Problem: {validity_issue}**")
            
            if validity_issue == "Formatfeil":
                st.code("""
Gyldig: [-2, -1, 0, 1, 2]
Ugyldig: ["sterkt uenig", "", "N/A", 999]
                """, language="text")
                
            elif validity_issue == "Verdier utenfor rekkevidde":
                st.code("""
Gyldig rekkevidde: -2 til +2
Ugyldige verdier: [-5, 7, 15, -10]
                """, language="text")
                
            elif validity_issue == "Logiske motsigelser":
                st.code("""
Spm1: "Øk skatter" -> +2 (sterkt enig)
Spm2: "Senk skatter" -> +2 (sterkt enig)
[Logisk motsigelse!]
                """, language="text")
                
            else:  # Kodingsproblemer
                st.code("""
Forventet: UTF-8 tekst
Faktisk: "Feil kÃ¶ding av norske tegn"
Skulle være: "Feil koding av norske tegn"
                """, language="text")
            
            # Show data distribution with errors
            np.random.seed(42)
            valid_data = np.random.choice([-2, -1, 0, 1, 2], size=100-error_rate)
            
            if validity_issue == "Verdier utenfor rekkevidde":
                invalid_data = np.random.choice([5, 7, -5, 10], size=error_rate)
                
                fig_val = go.Figure()
                
                # Valid data
                if len(valid_data) > 0:
                    valid_counts = np.bincount(valid_data + 2, minlength=5)
                    fig_val.add_trace(go.Bar(
                        x=[-2, -1, 0, 1, 2], y=valid_counts,
                        name="Gyldige data", marker_color=COL_POS
                    ))
                
                # Invalid data  
                if len(invalid_data) > 0:
                    unique_invalid, invalid_counts = np.unique(invalid_data, return_counts=True)
                    fig_val.add_trace(go.Bar(
                        x=unique_invalid, y=invalid_counts,
                        name="Ugyldige data", marker_color=COL_NEG
                    ))
            else:
                fig_val = go.Figure()
                if len(valid_data) > 0:
                    valid_counts = np.bincount(valid_data + 2, minlength=5)
                    fig_val.add_trace(go.Bar(
                        x=[-2, -1, 0, 1, 2], y=valid_counts,
                        name="Gyldige data", marker_color=COL_POS
                    ))
            
            fig_val.update_layout(
                title=f"Datafordeling med {error_rate}% feil",
                xaxis_title="Verdier", yaxis_title="Antall",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_val, use_container_width=True)
        
        with col2:
            # AI impact of validity issues
            validity_score = max(0, 100 - error_rate * 4)
            ai_robustness = max(20, 95 - error_rate * 3)
            
            st.subheader("Validitet-påvirkning")
            st.metric("Data-validitetsscore", f"{validity_score:.0f}%")
            st.metric("AI Robusthet", f"{ai_robustness:.0f}%", 
                     delta=f"{ai_robustness-95:.0f}% vs rene data")
            
            # Error handling strategies
            st.subheader("🛠️ Feilhåndtering")
            if error_rate < 2:
                st.success("🟢 Minimale feil - AI håndterer elegant")
                st.info("Strategi: Automatisk outlier-deteksjon")
            elif error_rate < 10:
                st.warning("🟡 Moderate feil - krever forbehandling")
                st.info("Strategi: Data-rensing + valideringsregler")
            else:
                st.error("🔴 Høy feilrate - AI-resultater upålitelige")
                st.info("Strategi: Manuell data-revisjon nødvendig")
            
            # Show AI error recovery capability
            recovery_methods = ["Hopp over ugyldige poster", "Fyll inn manglende verdier", 
                              "Flagg for manuell gjennomgang", "Bruk ensemble-metoder"]
            selected_recovery = st.selectbox("AI Gjenopprettingsmetode:", recovery_methods)
            
            recovery_effectiveness = {
                "Hopp over ugyldige poster": max(60, 100 - error_rate * 2),
                "Fyll inn manglende verdier": max(70, 100 - error_rate * 1.5),
                "Flagg for manuell gjennomgang": max(80, 100 - error_rate * 1),
                "Bruk ensemble-metoder": max(75, 100 - error_rate * 1.2)
            }
            
            st.metric("Gjenopprettings-effektivitet", f"{recovery_effectiveness[selected_recovery]:.0f}%")
        
        # Real-world examples
        st.subheader("Virkelige eksempler på validitetsproblemer")
        
        real_examples = {
            "Manglende svar": "Partier som ikke svarer på kontroversielle spørsmål",
            "Tekniske feil": "Feil i dataoverføring som endrer verdier",
            "Kodefeil": "Feil mapping mellom svar og tallverdier",
            "Inkonsekvente skalaer": "Noen spørsmål bruker 1-5, andre -2 til +2"
        }
        
        for problem, description in real_examples.items():
            st.write(f"**{problem}:** {description}")
    
    else:  # Kildefordeling
        st.subheader("Kildefordeling: Balansen mellom NRK og TV2")
        
        create_explanation_card(
            "Problemet med ubalanserte kilder",
            "Hvis en kilde dominerer datasettet, kan AI-systemer bli biased mot den kildens " +
            "spørsmålsformuleringer og temavalg. Balanserte kilder gir mer robuste AI-modeller.",
            "⚠️", "warning"
        )
        
        st.write("Viser balansen mellom NRK og TV2 datamengde")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            nrk_count = len(nrk)
            tv2_count = len(tv2)
            total = nrk_count + tv2_count
            
            fig_source = px.pie(
                values=[nrk_count, tv2_count],
                names=['NRK', 'TV2'],
                title="Fordeling av spørsmål mellom kilder",
                color_discrete_sequence=[COL_NEG, COL_POS]
            )
            fig_source.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig_source, use_container_width=True)
            
            # Show detailed breakdown
            st.subheader("Detaljert sammenligning")
            source_comparison = pd.DataFrame({
                "Kilde": ["NRK", "TV2"],
                "Antall_spørsmål": [nrk_count, tv2_count],
                "Prosent": [nrk_count/total*100, tv2_count/total*100],
                "Kategorisert": ["Nei", "Ja"],
                "Partier_dekket": [len([p for p in parties if p in nrk.columns]), 
                                  len([p for p in parties if p in tv2.columns])]
            })
            st.dataframe(source_comparison, use_container_width=True, hide_index=True)
        
        with col2:
            nrk_pct = nrk_count / total * 100
            tv2_pct = tv2_count / total * 100
            
            col1_metrics, col2_metrics = st.columns(2)
            col1_metrics.metric("NRK andel", f"{nrk_pct:.1f}%")
            col2_metrics.metric("TV2 andel", f"{tv2_pct:.1f}%")
            
            balance_deviation = abs(nrk_pct - 50)
            source_balance = max(0, 100 - balance_deviation * 2)
            st.metric("Kildebalanse-score", f"{source_balance:.0f}%")
            
            if balance_deviation < 10:
                st.success("🟢 God kildebalanse")
            elif balance_deviation < 25:
                st.warning("🟡 Moderat ubalanse")
            else:
                st.error("🔴 Sterk ubalanse - en kilde dominerer")
            
            # AI impact assessment
            ai_diversity = max(30, 100 - balance_deviation)
            st.metric("AI Diversitet", f"{ai_diversity:.0f}%")
            
            if ai_diversity > 80:
                st.success("AI lærer fra balanserte kilder")
            elif ai_diversity > 60:
                st.warning("AI kan lære skjevt")
            else:
                st.error("AI får ensidige læringsinput")
        
        # Interactive source weighting simulation
        st.subheader("Simuler effekt av kildevekting")
        
        nrk_weight = st.slider("NRK vekt:", 0.1, 2.0, 1.0, 0.1)
        tv2_weight = st.slider("TV2 vekt:", 0.1, 2.0, 1.0, 0.1)
        
        # Calculate effective distribution
        weighted_nrk = nrk_count * nrk_weight
        weighted_tv2 = tv2_count * tv2_weight
        weighted_total = weighted_nrk + weighted_tv2
        
        effective_nrk_pct = weighted_nrk / weighted_total * 100
        effective_tv2_pct = weighted_tv2 / weighted_total * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Effektiv fordeling med vekting:**")
            fig_weighted = px.pie(
                values=[effective_nrk_pct, effective_tv2_pct],
                names=['NRK (vektet)', 'TV2 (vektet)'],
                title="Vektet kildefordeling"
            )
            st.plotly_chart(fig_weighted, use_container_width=True)
        
        with col2:
            st.metric("NRK effektiv andel", f"{effective_nrk_pct:.1f}%")
            st.metric("TV2 effektiv andel", f"{effective_tv2_pct:.1f}%")
            
            weighted_balance_deviation = abs(effective_nrk_pct - 50)
            if weighted_balance_deviation < 10:
                st.success("🟢 Vektingen forbedrer balansen")
            elif weighted_balance_deviation < balance_deviation:
                st.info("📊 Vektingen hjelper litt")
            else:
                st.warning("⚠️ Vektingen forverrer ubalansen")

    def perform_party_clustering(df, parties, n_clusters=4):
        """Cluster partier basert på deres politiske posisjoner"""
        # Forbered data: partier som rader, spørsmål som kolonner
        party_data = df[parties].T.fillna(0)
        
        # Standardiser data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(party_data)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        party_clusters = kmeans.fit_predict(scaled_data)
        
        # Beregn silhouette score for kvalitet
        sil_score = silhouette_score(scaled_data, party_clusters)
        
        # DBSCAN som alternativ
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_clusters = dbscan.fit_predict(scaled_data)
        
        return {
            'kmeans_clusters': party_clusters,
            'dbscan_clusters': dbscan_clusters,
            'silhouette_score': sil_score,
            'scaled_data': scaled_data,
            'party_names': parties
        }

    def perform_pca_analysis_with_interpretation(df, parties, n_components=3):
        """Finn underliggende politiske dimensjoner med PCA og tolkning"""
        party_data = df[parties].T.fillna(0)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(party_data)
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Analyser komponenter
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=df.index
        )
        
        
        dimension_interpretations = interpret_pca_dimensions(components_df, df, n_components)
        
        return {
            'pca_result': pca_result,
            'explained_variance': pca.explained_variance_ratio_,
            'components': components_df,
            'party_names': parties,
            'dimension_names': dimension_interpretations  
        }

    def interpret_pca_dimensions(components_df, df, n_components, top_n=5):
        """Tolkér PCA-dimensjoner basert på spørsmål med høyeste loadings"""
        
        interpretations = {}
        
        for i in range(n_components):
            pc_name = f'PC{i+1}'
            
            # Finn spørsmål med høyest absoluttverdi loading for denne komponenten
            loadings = components_df[pc_name].abs().sort_values(ascending=False)
            top_questions = loadings.head(top_n)
            
            # Hent tekst for disse spørsmålene
            top_question_texts = []
            for idx in top_questions.index:
                question_text = df.loc[idx].get('Smp', df.loc[idx].get('Spm', f'Spørsmål {idx}'))
                loading_value = components_df.loc[idx, pc_name]
                top_question_texts.append({
                    'text': question_text,
                    'loading': loading_value,
                    'abs_loading': abs(loading_value)
                })
            
            # Forsøk å tolke basert på mønster
            suggested_name = suggest_dimension_name(top_question_texts, pc_name)
            
            interpretations[pc_name] = {
                'suggested_name': suggested_name,
                'top_questions': top_question_texts,
                'original_name': pc_name
            }
        
        return interpretations

    def suggest_dimension_name(top_questions, original_name):
        """Foreslå et meningsfullt navn basert på spørsmålsmønster"""
        
        # Enkel keyword-basert tolkning
        all_text = ' '.join([q['text'].lower() for q in top_questions])
        
        # FIXED: Mer spesifikke og unike kategorier
        # Økonomiske nøkkelord - mer spesifikke
        if any(word in all_text for word in ['skatt', 'avgift', 'inntekt', 'lønn']):
            return "Skatte- og inntektspolitikk"
        elif any(word in all_text for word in ['arbeid', 'jobb', 'ledig', 'pensjon']):
            return "Arbeids- og velferdspolitikk"
        
        # Sosial/liberal nøkkelord  
        elif any(word in all_text for word in ['abort', 'familie', 'barn', 'mor', 'far']):
            return "Familie- og verdipolitikk"
        elif any(word in all_text for word in ['innvandring', 'integrering', 'flyktning', 'asyl']):
            return "Innvandrings- og integreringspolitikk"
        
        # Miljø/klima nøkkelord
        elif any(word in all_text for word in ['miljø', 'klima', 'co2', 'utslipp']):
            return "Miljø- og klimapolitikk"
        elif any(word in all_text for word in ['energi', 'olje', 'strøm', 'kraft']):
            return "Energipolitikk"
        
        # Stat vs marked nøkkelord
        elif any(word in all_text for word in ['privat', 'offentlig', 'marked', 'konkurranse']):
            return "Stat vs. Marked"
        
        # EU/internasjonalt nøkkelord
        elif any(word in all_text for word in ['eu', 'europa', 'nato', 'utenriks']):
            return "Internasjonal orientering"
        
        # Helse og utdanning
        elif any(word in all_text for word in ['helse', 'sykehus', 'lege', 'behandling']):
            return "Helsepolitikk"
        elif any(word in all_text for word in ['skole', 'utdanning', 'lærer', 'elev']):
            return "Utdanningspolitikk"
        
        # Fallback til mer beskrivende navn
        return f"Politisk dimensjon {original_name[-1]}"

    def visualize_pca_with_meaningful_names(pca_results, st):
        """Visualiser PCA-resultater med tolkede dimensjonsnavn"""
        
        # Vis forklart varians med tolkede navn
        variance_data = []
        for i, variance in enumerate(pca_results['explained_variance']):
            pc_key = f'PC{i+1}'
            meaningful_name = pca_results['dimension_names'][pc_key]['suggested_name']
            variance_data.append({
                'Dimensjon': f"{meaningful_name} ({pc_key})",  # Legg til PC-nummer for å unngå forvirring
                'Original': pc_key,
                'Forklart_Varians': variance * 100
            })
        
        variance_df = pd.DataFrame(variance_data)
        
        
        fig_variance = px.bar(
            variance_df, 
            x='Dimensjon', 
            y='Forklart_Varians',
            title=f"Politiske dimensjoner funnet av AI ({len(variance_data)} dimensjoner)",
            labels={'Forklart_Varians': 'Prosent av total variasjon'},
            text='Forklart_Varians'
        )
        fig_variance.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_variance.update_layout(
            xaxis_tickangle=45,
            xaxis_title="Dimensjon",
            height=400  # Sett fast høyde
        )
        st.plotly_chart(fig_variance, use_container_width=True)
        
        # Vis total forklart varians
        total_variance = sum(variance_data[i]['Forklart_Varians'] for i in range(len(variance_data)))
        st.info(f"🎯 **Total forklart varians: {total_variance:.1f}%** ({len(variance_data)} dimensjoner)")
        
        
    def enhanced_pca_analysis_tab(current_data_ai, parties):
        """Erstatning for PCA-delen i tab8"""
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_dimensions = st.slider("Antall dimensjoner", 2, 4, 3)
            run_pca = st.button("🚀 Kjør PCA-Analyse")
            
            # Manuell override for dimensjonsnavn
            st.subheader("🎯 Manuel justering")
            manual_names = {}
            
            
            if 'pca_done' in st.session_state and 'pca_results' in st.session_state:
                existing_results = st.session_state.pca_results
                available_dimensions = list(existing_results['dimension_names'].keys())
                
                # Only create inputs for dimensions that exist in the stored results
                for i, pc_key in enumerate(available_dimensions):
                    dimension_num = i + 1
                    manual_names[pc_key] = st.text_input(
                        f"Navn for dimensjon {dimension_num}:", 
                        value=existing_results['dimension_names'][pc_key]['suggested_name'],
                        key=f'manual_dim_{dimension_num}'
                    )
                
                # Show info if user requests more dimensions than currently available
                if n_dimensions > len(available_dimensions):
                    st.info(f"💡 For å få {n_dimensions} dimensjoner, kjør PCA-analyse på nytt")
        
        with col2:
            if run_pca:
                with st.spinner("AI finner politiske dimensjoner..."):
                    pca_results = perform_pca_analysis_with_interpretation(current_data_ai, parties, n_dimensions)
                
                # Lagre i session state
                st.session_state.pca_results = pca_results
                st.session_state.pca_done = True
                st.session_state.pca_dimensions = n_dimensions  # Store number of dimensions used
                
                # Initialize dimension names in session state
                for i, pc_key in enumerate([f'PC{j+1}' for j in range(n_dimensions)]):
                    st.session_state[f'dim_{i+1}_name'] = pca_results['dimension_names'][pc_key]['suggested_name']
                
                st.success(f"✅ AI fant {n_dimensions} politiske dimensjoner!")
                
                # Vis resultatene
                visualize_pca_with_meaningful_names(pca_results, st)
                
                # 2D visualisering med meningsfulle navn
                if n_dimensions >= 2:
                    st.subheader("Partier i 2D politisk rom")
                    
                    dim1_name = pca_results['dimension_names']['PC1']['suggested_name']
                    dim2_name = pca_results['dimension_names']['PC2']['suggested_name']
                    
                    party_coords = pd.DataFrame({
                        'Parti': pca_results['party_names'],
                        dim1_name: pca_results['pca_result'][:, 0],
                        dim2_name: pca_results['pca_result'][:, 1]
                    })
                    
                    fig_2d = px.scatter(
                        party_coords, x=dim1_name, y=dim2_name,
                        text='Parti', title="Partier i AI-oppdaget politisk rom",
                        labels={
                            dim1_name: f'{dim1_name} ({pca_results["explained_variance"][0]*100:.1f}%)',
                            dim2_name: f'{dim2_name} ({pca_results["explained_variance"][1]*100:.1f}%)'
                        }
                    )
                    fig_2d.update_traces(textposition="top center")
                    fig_2d.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                    st.plotly_chart(fig_2d, use_container_width=True)
            
            # Vis eksisterende resultater hvis tilgjengelig
            elif 'pca_results' in st.session_state:
                pca_results = st.session_state.pca_results
                
                # FIXED: Only update names for dimensions that exist
                if manual_names:
                    for pc_key, manual_name in manual_names.items():
                        if manual_name.strip() and pc_key in pca_results['dimension_names']:
                            pca_results['dimension_names'][pc_key]['suggested_name'] = manual_name
                
                visualize_pca_with_meaningful_names(pca_results, st)
                
                if len(pca_results['pca_result'][0]) >= 2:
                    dim1_name = pca_results['dimension_names']['PC1']['suggested_name']
                    dim2_name = pca_results['dimension_names']['PC2']['suggested_name']
                    
                    party_coords = pd.DataFrame({
                        'Parti': pca_results['party_names'],
                        dim1_name: pca_results['pca_result'][:, 0],
                        dim2_name: pca_results['pca_result'][:, 1]
                    })
                    
                    fig_2d = px.scatter(
                        party_coords, x=dim1_name, y=dim2_name,
                        text='Parti', title="Partier i politisk rom (med tolkede dimensjoner)"
                    )
                    fig_2d.update_traces(textposition="top center")
                    fig_2d.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                    st.plotly_chart(fig_2d, use_container_width=True)
        
        return manual_names

    def create_recommendation_system(df, parties):
        """Lag et enkelt anbefalingssystem"""
        def recommend_parties(user_answers, top_n=3):
            """
            user_answers: dict med spørsmål_index: svar_verdi
            """
            if len(user_answers) == 0:
                return []
            
            # Konverter brukerinput til vektor
            user_vector = np.zeros(len(df))
            for q_idx, answer in user_answers.items():
                if q_idx < len(user_vector):
                    user_vector[q_idx] = answer
            
            # Beregn similaritet med alle partier
            party_similarities = {}
            for party in parties:
                if party in df.columns:
                    party_positions = df[party].fillna(0).values
                    # Kun sammenlign på spørsmål brukeren har svart på
                    answered_indices = list(user_answers.keys())
                    if len(answered_indices) > 0:
                        user_subset = [user_vector[i] for i in answered_indices]
                        party_subset = [party_positions[i] for i in answered_indices]
                        
                        similarity = cosine_similarity([user_subset], [party_subset])[0][0]
                        party_similarities[party] = similarity
            
            # Returner topp-N mest like partier
            sorted_parties = sorted(party_similarities.items(), 
                                key=lambda x: x[1], reverse=True)
            return sorted_parties[:top_n]
        
        return recommend_parties

    def train_controversy_classifier(df, parties):
        """Tren en modell til å klassifisere spørsmål som kontroversielle"""
        # Beregn kontroversialitet (standardavvik)
        df_analysis = prep_data_with_friendly_labels(df, parties)
        
        # Lag features basert på tekstlengde og andre egenskaper
        features = []
        labels = []
        
        for idx, row in df_analysis.iterrows():
            # Enkle features (kan utvides)
            question_text = row.get('Spm', '') or row.get('Smp', '')
            feature_vector = [
                len(question_text),  # Tekstlengde
                len(question_text.split()),  # Antall ord
                question_text.count('?'),  # Antall spørsmålstegn
                row['avg_position'],  # Gjennomsnittlig posisjon
                row['disagreement']  # Uenighet (target for klassifisering)
            ]
            features.append(feature_vector[:-1])  # Ikke bruk disagreement som feature
            
            # Klassifiser som kontroversielt hvis disagreement > median
            is_controversial = row['disagreement'] > df_analysis['disagreement'].median()
            labels.append(int(is_controversial))
        
        # Tren modell
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42
        )
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluer
        accuracy = clf.score(X_test, y_test)
        
        return {
            'model': clf,
            'accuracy': accuracy,
            'feature_names': ['Tekstlengde', 'Antall_ord', 'Spørsmålstegn', 'Avg_posisjon']
        }

    with tab8:
        st.header("🤖AI-Analyser")
        
        create_explanation_card(
            "Fra statistikk til kunstig intelligens",
            "Nå bruker vi ekte machine learning algoritmer på valgomatdataene. " +
            "Vi kan clustere partier, finne skjulte dimensjoner og bygge anbefalingssystemer.",
            "🧠"
        )
        
        # Dataset selector for AI analysis
        with st.sidebar:
            st.subheader("AI-Innstillinger")
            source_name_ai, current_data_ai = safe_dataset_selector("ai_analysis")
        
        ai_analysis_type = st.selectbox(
        "Velg AI-analyse:",
        ["🎯 Parti-Clustering", "📊 Politiske Dimensjoner (PCA)", 
        "💡 Anbefalingssystem", "🔍 Kontrovers-Klassifisering"]
)
        
        if ai_analysis_type == "🎯 Parti-Clustering":
            st.subheader("Automatisk gruppering av partier")
            
            create_explanation_card(
                "Hva er clustering?",
                "AI-algoritmer kan automatisk finne partier som har like politiske mønstre " +
                "uten at vi forteller dem hva de skal lete etter. Dette kan avsløre skjulte " +
                "politiske allianser eller ideologiske grupperinger.",
                "🤔"
            )
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                n_clusters = st.slider("Antall grupper", 2, 6, 4)
                run_clustering = st.button("🚀 Kjør AI-Clustering")
            
            with col2:
                if run_clustering:
                    with st.spinner("AI analyserer partimønstre..."):
                        results = perform_party_clustering(current_data_ai, parties, n_clusters)
                    
                    st.success(f"✅ AI fant {n_clusters} partigrupper!")
                    st.metric("Clustering-kvalitet (Silhouette)", f"{results['silhouette_score']:.3f}")
                    
                    # Visualiser clustering-resultater
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # KMeans resultater
                        cluster_df = pd.DataFrame({
                            'Parti': results['party_names'],
                            'KMeans_Gruppe': results['kmeans_clusters']
                        })
                        
                        fig_kmeans = px.scatter(
                            cluster_df, x=range(len(cluster_df)), y='KMeans_Gruppe',
                            color='KMeans_Gruppe', text='Parti',
                            title="KMeans Parti-Gruppering",
                            labels={'x': 'Parti-indeks', 'y': 'Gruppe'}
                        )
                        fig_kmeans.update_traces(textposition="top center")
                        fig_kmeans.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                        st.plotly_chart(fig_kmeans, use_container_width=True)
                    
                    with col_viz2:
                        # Vis gruppe-medlemskap
                        st.write("**AI-oppdagede partigrupper:**")
                        for group_id in range(n_clusters):
                            group_parties = [
                                party for i, party in enumerate(results['party_names'])
                                if results['kmeans_clusters'][i] == group_id
                            ]
                            if group_parties:
                                st.write(f"**Gruppe {group_id + 1}:** {', '.join(group_parties)}")
                    
                    # Fortolk resultatene
                    st.subheader("🔍 AI-tolkninger")
                    
                    if results['silhouette_score'] > 0.5:
                        st.success("🟢 Tydelige partigrupper funnet - AI ser klare mønstre")
                    elif results['silhouette_score'] > 0.3:
                        st.warning("🟡 Moderate partigrupper - noen overlappende mønstre")
                    else:
                        st.error("🔴 Svake partigrupper - partier er politisk spredte")
        
        elif ai_analysis_type == "📊 Politiske Dimensjoner (PCA)":
            st.subheader("Finn skjulte politiske dimensjoner")
            
            create_explanation_card(
                "Hva er PCA?",
                "Principal Component Analysis kan finne de mest viktige 'dimensjonene' " +
                "i politikken - som økonomi-akse, liberal-konservativ akse, etc. " +
                "AI vil også forsøke å gi dimensjonene meningsfulle navn basert på " +
                "hvilke spørsmål som påvirker dem mest.",
                "📈"
            )
            
            # Bruk den forbedrede PCA-analysen
            enhanced_pca_analysis_tab(current_data_ai, parties)
            
            # Vis totalt forklart varians hvis resultater finnes
            if 'pca_results' in st.session_state:
                pca_results = st.session_state.pca_results
                total_variance = pca_results['explained_variance'].sum() * 100
                st.metric(f"Total politisk kompleksitet fanget", f"{total_variance:.1f}%")
                
                if total_variance > 70:
                    st.success("🟢 AI fanger mye av den politiske kompleksiteten")
                elif total_variance > 50:
                    st.warning("🟡 AI fanger en del, men politikk er kompleks")
                else:
                    st.error("🔴 Politikk er svært kompleks - trenger flere dimensjoner")
        
        elif ai_analysis_type == "💡 Anbefalingssystem":
            st.subheader("Bygg din egen AI-drevne valgomat")
            
            create_explanation_card(
                "Hvordan fungerer AI-anbefalinger?",
                "Du kan svare på noen spørsmål, og AI-en vil bruke cosine similarity " +
                "til å finne hvilke partier som er mest like dine svar. Dette er samme " +
                "type algoritme som Netflix bruker til å anbefale filmer!",
                "🎬"
            )
            
            # Lag anbefalingssystem
            recommend_func = create_recommendation_system(current_data_ai, parties)
            
            st.subheader("Test AI-anbefalingssystemet")
            st.write("Svar på noen spørsmål og få AI-drevne partianbefalinger:")
            
            # Velg noen representative spørsmål
            sample_questions = current_data_ai.head(8)
            user_answers = {}
            
            col1, col2 = st.columns(2)
            
            for i, (idx, row) in enumerate(sample_questions.iterrows()):
                question_text = row.get('Smp', row.get('Spm', f'Spørsmål {idx}'))
                short_question = question_text[:60] + "..." if len(question_text) > 60 else question_text
                
                with col1 if i % 2 == 0 else col2:
                    answer = st.select_slider(
                        short_question,
                        options=[-2, -1, 0, 1, 2],
                        format_func=lambda x: {-2: "Sterkt uenig", -1: "Uenig", 0: "Nøytral", 
                                            1: "Enig", 2: "Sterkt enig"}[x],
                        value=0,
                        key=f"q_{idx}"
                    )
                    user_answers[idx] = answer
            
            if st.button("🤖 Få AI-anbefalinger"):
                with st.spinner("AI analyserer dine politiske preferanser..."):
                    recommendations = recommend_func(user_answers, top_n=5)
                
                if recommendations:
                    st.success("🎯 AI har analysert dine svar!")
                    
                    st.subheader("Dine mest kompatible partier:")
                    for i, (party, similarity) in enumerate(recommendations, 1):
                        compatibility = similarity * 100
                        
                        col_rank, col_party, col_score = st.columns([0.5, 2, 1])
                        with col_rank:
                            st.write(f"**{i}.**")
                        with col_party:
                            st.write(f"**{party}**")
                        with col_score:
                            if compatibility > 80:
                                st.success(f"{compatibility:.1f}%")
                            elif compatibility > 60:
                                st.warning(f"{compatibility:.1f}%")
                            else:
                                st.error(f"{compatibility:.1f}%")
                    
                    # Visualiser anbefalinger
                    rec_df = pd.DataFrame(recommendations, columns=['Parti', 'Similaritet'])
                    rec_df['Kompatibilitet_%'] = rec_df['Similaritet'] * 100
                    
                    fig_rec = px.bar(
                        rec_df, x='Kompatibilitet_%', y='Parti',
                        orientation='h',
                        title="AI-beregnet partikompatibilitet",
                        color='Kompatibilitet_%',
                        color_continuous_scale=get_consistent_color_scale()
                    )
                    fig_rec.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                    st.plotly_chart(fig_rec, use_container_width=True)
                else:
                    st.error("AI klarte ikke å lage anbefalinger. Prøv å svare på flere spørsmål.")
        
        else:  # Kontrovers-klassifisering
            st.subheader("AI-klassifisering av kontroversielle spørsmål")
            
            create_explanation_card(
                "Kan AI forutsi kontrovers?",
                "Vi kan trene en Random Forest-klassifiserer til å gjenkjenne " +
                "kontroversielle spørsmål basert på tekstegenskaper. Dette viser " +
                "hvordan AI kan lære mønstre fra data.",
                "🌲"
            )
            
            if st.button("🚀 Tren AI-klassifiserer"):
                with st.spinner("AI lærer å gjenkjenne kontroversielle spørsmål..."):
                    classifier_results = train_controversy_classifier(current_data_ai, parties)
                
                st.success(f"✅ AI-modell trent med {classifier_results['accuracy']:.1%} nøyaktighet!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AI-nøyaktighet", f"{classifier_results['accuracy']:.1%}")
                    
                    if classifier_results['accuracy'] > 0.8:
                        st.success("🟢 AI er svært god til å forutsi kontrovers")
                    elif classifier_results['accuracy'] > 0.7:
                        st.warning("🟡 AI er ganske god til å forutsi kontrovers")
                    else:
                        st.error("🔴 AI har vansker med å forutsi kontrovers")
                
                with col2:
                    # Vis feature importance
                    importances = classifier_results['model'].feature_importances_
                    feature_df = pd.DataFrame({
                        'Feature': classifier_results['feature_names'],
                        'Viktighet': importances
                    }).sort_values('Viktighet', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_df, x='Viktighet', y='Feature',
                        orientation='h',
                        title="Hva bruker AI til å forutsi kontrovers?",
                        color='Viktighet',
                        color_continuous_scale=get_consistent_color_scale()
                    )
                    fig_importance.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Test AI-en på nye "spørsmål"
                st.subheader("Test AI-klassifisereren")
                
                test_questions = [
                    "Bør Norge øke skatten for de rikeste?",
                    "Skal vi ha mer midler til skole?",
                    "Bør abort være lovlig?",
                    "Skal politiet ha mer makt?"
                ]
                
                st.write("**AI-prediksjoner for testspørsmål:**")
                
                for question in test_questions:
                    # Lag features for testspørsmål
                    test_features = [[
                        len(question),
                        len(question.split()),
                        question.count('?'),
                        0.0  # Avg position (ukjent for testspørsmål)
                    ]]
                    
                    prediction = classifier_results['model'].predict(test_features)[0]
                    probability = classifier_results['model'].predict_proba(test_features)[0]
                    
                    controversy_prob = probability[1] * 100  # Sannsynlighet for kontrovers
                    
                    col_q, col_pred = st.columns([3, 1])
                    with col_q:
                        st.write(f"'{question}'")
                    with col_pred:
                        if prediction == 1:
                            st.error(f"🔥 Kontroversielt ({controversy_prob:.0f}%)")
                        else:
                            st.success(f"✅ Ikke kontroversielt ({100-controversy_prob:.0f}%)")
        
        # AI vs Statistikk sammenligning
        st.divider()
        st.subheader("🆚 AI vs Tradisjonell statistikk")
        
        comparison_data = {
            "Aspekt": ["Kompleksitet", "Tolkbarhet", "Prediksjonsevne", "Datavolum-krav", "Bias-risiko"],
            "Statistisk analyse": ["Lav", "Høy", "Moderat", "Lavt", "Lavere"],
            "AI/ML analyse": ["Høy", "Lav", "Høy", "Høyt", "Høyere"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.info("""
        💡 **Lærdommer:** 
        - AI kan finne skjulte mønstre vi ikke ser med tradisjonell statistikk
        - Men AI er mindre tolkbar - vi forstår ikke alltid hvorfor den gjør det den gjør
        - Begge tilnærminger har sine styrker og svakheter
        """)
    st.divider()
   
    st.caption("💡 Tips: Klikk på grafene for å utforske interaktivt. Bruk faner for å se ulike analyser.")