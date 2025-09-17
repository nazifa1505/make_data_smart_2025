# streamlit_app.py - Improved User-Friendly Valgomat Analysis
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
                <summary><strong>🤔 Hvordan lese denne grafen?</strong></summary>
                <p style="margin-top: 8px;">{explanations[graph_type]}</p>
            </details>
        </div>
        """, unsafe_allow_html=True)

def create_data_context_warning():
    """Create a prominent warning about data limitations"""
    st.markdown("""
    <div class="critical-info">
        <h4> Viktig å forstå om dataene</h4>
        <p><strong>Dette viser partienes standpunkter, IKKE folks svar!</strong></p>
        <p>Tallene kommer fra hva partiene har svart på valgomatspørsmål:</p>
        <ul>
            <li><strong>+2</strong> = Partiet er sterkt enig i forslaget</li>
            <li><strong>0</strong> = Partiet er nøytral eller usikker</li>
            <li><strong>-2</strong> = Partiet er sterkt uenig i forslaget</li>
        </ul>
        <p>Vi har <em>ikke</em> data fra vanlige folk som har tatt valgomaten.</p>
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

def friendly_translate_stats(mu, std):
    """Translate statistics to friendly language"""
    # Translate direction (mu)
    if abs(mu) < 0.2:
        direction = "Partiene er delte"
        direction_emoji = "⚖️"
    elif mu > 0.5:
        direction = "De fleste partier støtter dette"
        direction_emoji = "👍"
    elif mu > 0:
        direction = "Svakt flertall støtter"
        direction_emoji = "📈"
    elif mu < -0.5:
        direction = "De fleste partier er imot"
        direction_emoji = "👎"
    else:
        direction = "Svakt flertall er imot"
        direction_emoji = "📉"
    
    # Translate disagreement (std)
    if std < 0.5:
        agreement = "bred enighet"
        agreement_emoji = "🤝"
    elif std < 1.0:
        agreement = "noe uenighet"
        agreement_emoji = "⚡"
    elif std < 1.5:
        agreement = "mye uenighet"
        agreement_emoji = "🔥"
    else:
        agreement = "dyp splittelse"
        agreement_emoji = "💥"
    
    return f"{direction_emoji} {direction} ({agreement_emoji} {agreement})"

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
    
    # User-friendly labels
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
    """Calculate actual data quality scores"""
    scores = {}
    
    # 1. Nøyaktighet - basert på konsistens mellom kilder
    if len(parties) > 0:
        nrk_sample = nrk.head(15)[parties].mean()
        tv2_sample = tv2.head(15)[parties].mean()
        consistency = 1 - abs(nrk_sample - tv2_sample).mean() / 4  # Normalize to 0-1
        scores["Nøyaktighet"] = max(0, min(100, consistency * 100))
    else:
        scores["Nøyaktighet"] = 85
    
    # 2. Kompletthet - sjekk for manglende verdier
    total_cells = len(nrk) * len(parties) + len(tv2) * len(parties)
    missing_cells = nrk[parties].isna().sum().sum() + tv2[parties].isna().sum().sum()
    completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 1
    scores["Kompletthet"] = max(0, min(100, completeness * 100))
    
    # 3. Konsistens - variabilitet i data
    nrk_std = nrk[parties].std().mean()
    tv2_std = tv2[parties].std().mean()
    avg_std = (nrk_std + tv2_std) / 2
    consistency_score = max(0, 100 - avg_std * 25)  # Lower std = higher consistency
    scores["Konsistens"] = consistency_score
    
    # 4. Aktualitet - anta dataene er 6 måneder gamle
    months_old = 6
    timeliness = max(0, 100 - months_old * 2)
    scores["Aktualitet"] = timeliness
    
    # 5. Validitet - sjekk om verdier er i forventet range [-2, 2]
    all_values = pd.concat([nrk[parties].stack(), tv2[parties].stack()]).dropna()
    valid_values = all_values[(all_values >= -2) & (all_values <= 2)]
    validity = len(valid_values) / len(all_values) * 100 if len(all_values) > 0 else 95
    scores["Validitet"] = validity
    
    # 6. Unikalitet - estimat basert på antall spørsmål vs kategorier
    if 'Kategori' in tv2.columns:
        unique_categories = tv2['Kategori'].nunique()
        total_questions = len(tv2)
        # Forvent ~3-5 spørsmål per kategori som optimalt
        expected_ratio = unique_categories * 4
        uniqueness = min(100, (expected_ratio / total_questions) * 100) if total_questions > 0 else 90
    else:
        uniqueness = 90
    scores["Unikalitet"] = uniqueness
    
    return scores

# Load data
parties, nrk, tv2 = load_files()

# Main app
st.title("🗳️ Valgomat-utforsker")
st.subheader("Utforsk hvordan norske partier posisjonerer seg politisk")

# Data context warning at the top
create_data_context_warning()

# Navigation tabs with clearer names
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Hva splitter partiene?", 
    "📊 Sammenlign datakilder",
    "📋 Temaoversikt (TV2)",
    "⚖️ Vektingseffekter", 
    "📈 Scenario-analyse",
    "🤖 AI & Datakvalitet",
    "📖 Metodikk"
])

with tab1:
    st.header("Hvilke spørsmål skaper mest uenighet?")
    
    create_explanation_card(
        "Hva viser denne analysen?",
        "Vi ser på valgomatspørsmål og finner ut hvilke som skaper mest politisk splittelse. " +
        "Noen forslag får bred støtte, andre deler partiene på midten.",
        "🔍"
    )
    
    with st.sidebar:
        st.header("Innstillinger")
        dataset = st.radio("Velg datasett", ["NRK", "TV2"])
        
        if dataset == "NRK":
            st.info("💡 NRK-data har ikke tematiske kategorier")
        else:
            st.info("💡 TV2-data er organisert i tema som familie, økonomi, osv.")
        
        top_k = st.slider("Vis topp-punkter", 5, 15, 8)
        show_numbers = st.toggle("Vis nummerering på punkter", value=True)

    # Select and prepare data
    current_data = nrk if dataset == "NRK" else tv2
    df = prep_data_with_friendly_labels(current_data, parties)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Create main visualization
        create_how_to_read_box("polarization")
        
        df["color"] = df["avg_position"].apply(color_map)
        df["size"] = size_scale(df["disagreement_squared"])
        
        # Create the plot
        fig = px.scatter(
            df.head(50),  # Limit to first 50 for performance
            x="avg_position", 
            y="disagreement",
            size="size",
            color="agreement_label",
            color_discrete_sequence=[COL_POS, COL_NEU, COL_NEG, "#ff4444"],
            hover_data=["Smp"] if "Smp" in df.columns else ["Spm"],
            title="Politisk splittelse: Retning vs. Uenighet"
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
        📈 **Sammendrag for {dataset}:**
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
                     {i}. {textwrap.fill(question_text, width=60)}
                </div>
                <div style="font-size: 12px; color: #666;">
                    {row.friendly_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add explanation of what makes questions controversial
        with st.expander("Hva gjør et spørsmål kontroversielt?"):
            st.write("""
            Et spørsmål blir kontroversielt når:
            - **Noen partier er sterkt for** (svarer +2)
            - **Andre partier er sterkt imot** (svarer -2)  
            - **Få partier er nøytrale** (svarer 0)
            
            Dette skaper stor "uenighet" som vi måler statistisk.
            """)

with tab2:
    st.header("Sammenlign NRK og TV2")
    
    create_explanation_card(
        " Viktig begrensning", 
        "NRK og TV2 har FORSKJELLIGE spørsmål! Vi kan ikke sammenligne tall direkte. " +
        "I stedet ser vi på mønstre: hvilke partier rangeres høyt/lavt i hver kilde?",
        "", "warning"
    )
    
    # Analysis options
    comparison_type = st.radio("Hva vil du sammenligne?", [
        "Partienes relative posisjoner",
        "Enighet vs. uenighet mellom kilder", 
        "Rangering av partier"
    ])
    
    n_questions = st.slider("Antall spørsmål per kilde", 10, 50, 25)
    
    # Prepare data
    nrk_subset = nrk.head(n_questions)
    tv2_subset = tv2.head(n_questions)
    
    if comparison_type == "Partienes relative posisjoner":
        create_how_to_read_box("party_comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NRK - Gjennomsnittlige posisjoner")
            nrk_avg = nrk_subset[parties].mean().sort_values()
            
            fig_nrk = px.bar(
                x=nrk_avg.values,
                y=nrk_avg.index,
                orientation='h',
                color=nrk_avg.values,
                color_continuous_scale=[[0, COL_NEG], [0.5, COL_NEU], [1, COL_POS]],
                title="Partier sortert etter gjennomsnittlig posisjon"
            )
            fig_nrk.update_layout(
                xaxis_title="Gjennomsnitt ← Kritisk | Støttende →",
                plot_bgcolor=BG, paper_bgcolor=BG,
                showlegend=False
            )
            st.plotly_chart(fig_nrk, use_container_width=True)
        
        with col2:
            st.subheader("TV2 - Gjennomsnittlige posisjoner")
            tv2_avg = tv2_subset[parties].mean().sort_values()
            
            fig_tv2 = px.bar(
                x=tv2_avg.values,
                y=tv2_avg.index,
                orientation='h',
                color=tv2_avg.values,
                color_continuous_scale=[[0, COL_NEG], [0.5, COL_NEU], [1, COL_POS]],
                title="Partier sortert etter gjennomsnittlig posisjon"
            )
            fig_tv2.update_layout(
                xaxis_title="Gjennomsnitt ← Kritisk | Støttende →",
                plot_bgcolor=BG, paper_bgcolor=BG,
                showlegend=False
            )
            st.plotly_chart(fig_tv2, use_container_width=True)
        
        # Ranking comparison
        st.subheader("📊 Rangering-sammenligning")
        nrk_ranks = nrk_avg.rank(method='min')
        tv2_ranks = tv2_avg.rank(method='min')
        
        rank_comparison = pd.DataFrame({
            "Parti": parties,
            "NRK_rangering": [nrk_ranks[p] for p in parties],
            "TV2_rangering": [tv2_ranks[p] for p in parties],
        })
        rank_comparison["Forskjell"] = rank_comparison["TV2_rangering"] - rank_comparison["NRK_rangering"]
        rank_comparison = rank_comparison.sort_values("Forskjell", key=abs, ascending=False)
        
        st.dataframe(rank_comparison, use_container_width=True)
        
        # Interpretation
        max_diff = rank_comparison["Forskjell"].abs().max()
        if max_diff <= 2:
            st.success("🟢 Kildene rangerer partiene ganske likt!")
        elif max_diff <= 4:
            st.warning("🟡 Noen forskjeller i rangering mellom kildene")
        else:
            st.error("🔴 Store forskjeller - kildene rangerer partiene ulikt")
    
    elif comparison_type == "Enighet vs. uenighet mellom kilder":
        st.subheader("🤝 Hvor enige er kildene?")
        
        # Calculate agreement statistics
        nrk_polarization = prep_data_with_friendly_labels(nrk_subset, parties)
        tv2_polarization = prep_data_with_friendly_labels(tv2_subset, parties)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "NRK: Spørsmål med bred enighet",
                f"{(nrk_polarization['disagreement'] < 0.5).sum()}/{len(nrk_polarization)}"
            )
            st.metric(
                "NRK: Kontroversielle spørsmål", 
                f"{(nrk_polarization['disagreement'] > 1.2).sum()}/{len(nrk_polarization)}"
            )
        
        with col2:
            st.metric(
                "TV2: Spørsmål med bred enighet",
                f"{(tv2_polarization['disagreement'] < 0.5).sum()}/{len(tv2_polarization)}"
            )
            st.metric(
                "TV2: Kontroversielle spørsmål",
                f"{(tv2_polarization['disagreement'] > 1.2).sum()}/{len(tv2_polarization)}"
            )
        
        with col3:
            nrk_avg_disagreement = nrk_polarization['disagreement'].mean()
            tv2_avg_disagreement = tv2_polarization['disagreement'].mean()
            
            st.metric("NRK: Gj.snitt uenighet", f"{nrk_avg_disagreement:.2f}")
            st.metric("TV2: Gj.snitt uenighet", f"{tv2_avg_disagreement:.2f}")
            
            if abs(nrk_avg_disagreement - tv2_avg_disagreement) < 0.1:
                st.success("Likt konflikt-nivå")
            else:
                diff_source = "NRK" if nrk_avg_disagreement > tv2_avg_disagreement else "TV2"
                st.info(f"{diff_source} har mer kontrovers")
    
    else:  # Ranking comparison - FIXED VERSION
        st.subheader("🏆 Hvem rangeres høyest/lavest?")
        
        try:
            nrk_sums = nrk_subset[parties].sum().sort_values(ascending=False)
            tv2_sums = tv2_subset[parties].sum().sort_values(ascending=False)
            
            # Create ranking dataframe for easier handling
            ranking_df = pd.DataFrame({
                'Party': parties,
                'NRK_Sum': [nrk_sums.get(p, 0) for p in parties],
                'TV2_Sum': [tv2_sums.get(p, 0) for p in parties]
            })
            
            # Calculate rankings
            ranking_df['NRK_Rank'] = ranking_df['NRK_Sum'].rank(ascending=False, method='min')
            ranking_df['TV2_Rank'] = ranking_df['TV2_Sum'].rank(ascending=False, method='min')
            
            # Sort by NRK ranking for consistent display
            ranking_df = ranking_df.sort_values('NRK_Rank')
            
            # Create ranking visualization
            fig = go.Figure()
            
            # NRK rankings
            fig.add_trace(go.Scatter(
                x=ranking_df['NRK_Rank'],
                y=ranking_df['Party'],
                mode='markers+lines',
                name='NRK rangering',
                marker=dict(size=12, color=COL_NEG),
                line=dict(color=COL_NEG, width=2, dash='dot'),
                hovertemplate="<b>%{y}</b><br>NRK rangering: #%{x}<br>Sum: %{customdata}<extra></extra>",
                customdata=ranking_df['NRK_Sum'].round(1)
            ))
            
            # TV2 rankings
            fig.add_trace(go.Scatter(
                x=ranking_df['TV2_Rank'],
                y=ranking_df['Party'],
                mode='markers+lines',
                name='TV2 rangering',
                marker=dict(size=12, color=COL_POS),
                line=dict(color=COL_POS, width=2, dash='dot'),
                hovertemplate="<b>%{y}</b><br>TV2 rangering: #%{x}<br>Sum: %{customdata}<extra></extra>",
                customdata=ranking_df['TV2_Sum'].round(1)
            ))
            
            fig.update_layout(
                title="Partirangering: NRK vs TV2<br><sub>Lavere tall = bedre rangering</sub>",
                xaxis_title="Rangering (1 = høyest)",
                yaxis_title="",
                plot_bgcolor=BG, 
                paper_bgcolor=BG,
                font=dict(color="#111111"),
                height=600,
                xaxis=dict(
                    tickmode='linear',
                    tick0=1,
                    dtick=1,
                    range=[0.5, len(parties) + 0.5]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show ranking table
            st.subheader("📋 Detaljert rangering")
            display_ranking = ranking_df[['Party', 'NRK_Rank', 'TV2_Rank', 'NRK_Sum', 'TV2_Sum']].copy()
            display_ranking['Rank_Difference'] = (display_ranking['TV2_Rank'] - display_ranking['NRK_Rank']).astype(int)
            display_ranking.columns = ['Parti', 'NRK Rang', 'TV2 Rang', 'NRK Sum', 'TV2 Sum', 'Forskjell']
            
            # Add status column
            display_ranking['Status'] = display_ranking['Forskjell'].apply(
                lambda x: "📈 Bedre i TV2" if x < -1 else 
                         "📉 Bedre i NRK" if x > 1 else 
                         "➡️ Likt"
            )
            
            st.dataframe(
                display_ranking[['Parti', 'NRK Rang', 'TV2 Rang', 'Forskjell', 'Status']], 
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            avg_diff = abs(display_ranking['Forskjell']).mean()
            max_diff = abs(display_ranking['Forskjell']).max()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Gjennomsnittlig forskjell", f"{avg_diff:.1f} plasser")
            col2.metric("Største forskjell", f"{max_diff:.0f} plasser")
            col3.metric("Partier med lik rangering", f"{sum(display_ranking['Forskjell'] == 0)}/{len(parties)}")
            
            # Interpretation
            if avg_diff <= 1:
                st.success("🟢 Kildene rangerer partiene meget likt!")
            elif avg_diff <= 2:
                st.warning("🟡 Moderate forskjeller i rangering")
            else:
                st.error("🔴 Store forskjeller i rangering mellom kildene")
                
        except Exception as e:
            st.error(f"Feil i rangering-analyse: {str(e)}")
            st.info("Prøv å endre antall spørsmål eller kontakt support hvis problemet vedvarer.")
            
            # Fallback: Simple bar chart comparison
            st.subheader("Forenklet sammenligning")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**NRK - Topp partier**")
                nrk_simple = nrk_subset[parties].sum().nlargest(5)
                for i, (party, score) in enumerate(nrk_simple.items(), 1):
                    st.write(f"{i}. {party}: {score:.1f}")
            
            with col2:
                st.write("**TV2 - Topp partier**")
                tv2_simple = tv2_subset[parties].sum().nlargest(5)
                for i, (party, score) in enumerate(tv2_simple.items(), 1):
                    st.write(f"{i}. {party}: {score:.1f}")

with tab3:
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
        
        # Topic-wise controversy analysis
        st.subheader("Hvilke tema skaper mest uenighet?")
        
        tv2_with_analysis = prep_data_with_friendly_labels(tv2, parties)
        
        if len(tv2_with_analysis) > 0:
            topic_controversy = tv2_with_analysis.groupby('Kategori')['disagreement'].agg(['mean', 'max', 'count']).round(3)
            topic_controversy.columns = ['Gj_uenighet', 'Maks_uenighet', 'Antall_sporsmal']
            topic_controversy = topic_controversy.sort_values('Gj_uenighet', ascending=False)
            
            fig_controversy = px.bar(
                topic_controversy.reset_index(),
                x='Kategori',
                y='Gj_uenighet',
                color='Gj_uenighet',
                color_continuous_scale='Reds',
                title="Gjennomsnittlig uenighet per tema"
            )
            
            fig_controversy.update_layout(
                xaxis_tickangle=45,
                plot_bgcolor=BG, paper_bgcolor=BG,
                xaxis_title="Politisk tema",
                yaxis_title="Gjennomsnittlig uenighet mellom partier"
            )
            
            st.plotly_chart(fig_controversy, use_container_width=True)
            
            # Show top controversial topics
            st.write("**Mest kontroversielle tema:**")
            for i, (tema, stats) in enumerate(topic_controversy.head(3).iterrows(), 1):
                st.write(f"{i}. **{tema}** - Uenighet: {stats['Gj_uenighet']:.2f} ({stats['Antall_sporsmal']} spørsmål)")

with tab4:
    st.header("⚖️ Hva skjer hvis vi vekter tema ulikt?")
    
    create_tv2_only_explanation()
    
    create_explanation_card(
        "Utforsk vektingseffekter",
        "I virkeligheten bryr folk seg mer om noen tema enn andre. " +
        "Hva skjer hvis vi gir økonomi-spørsmål dobbelt så mye vekt som andre tema?",
        "⚖️"
    )
    
    if 'Kategori' not in tv2.columns:
        st.error("Denne analysen krever TV2-data med kategorier")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_questions = st.slider("Antall spørsmål å inkludere", 10, 50, 25, key="weight_q")
        
        with col2:
            available_categories = tv2['Kategori'].dropna().unique()
            boost_category = st.selectbox("Tema å vektlegge ekstra", available_categories)
        
        with col3:
            weight_factor = st.slider("Hvor mye ekstra vekt?", 1.0, 5.0, 2.0, 0.5)
        
        # Calculate weighted vs unweighted scores
        subset = tv2.head(n_questions)
        
        # Original scores
        original_scores = subset[parties].sum()
        
        # Weighted scores
        weights = subset['Kategori'].apply(lambda cat: weight_factor if cat == boost_category else 1.0)
        weighted_scores = (subset[parties].multiply(weights, axis=0)).sum()
        
        # Create comparison
        comparison = pd.DataFrame({
            "Før_vekting": original_scores,
            "Etter_vekting": weighted_scores,
            "Endring": weighted_scores - original_scores,
            "Endring_pst": ((weighted_scores - original_scores) / original_scores * 100).round(1)
        }).sort_values("Før_vekting", ascending=False)
        
        col_left, col_right = st.columns([2.5, 1.5])
        
        with col_left:
            create_how_to_read_box("party_comparison")
            
            fig = go.Figure()
            
            # Before weighting
            fig.add_trace(go.Bar(
                name='Før vekting',
                x=comparison.index,
                y=comparison["Før_vekting"],
                marker_color=COL_NEU,
                text=[f"{val:.0f}" for val in comparison["Før_vekting"]],
                textposition='auto'
            ))
            
            # After weighting  
            fig.add_trace(go.Bar(
                name=f'Etter {weight_factor}× vekt på "{boost_category}"',
                x=comparison.index,
                y=comparison["Etter_vekting"],
                marker_color=COL_POS,
                text=[f"{val:.0f}" for val in comparison["Etter_vekting"]],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Effekt av å vektlegge '{boost_category}' {weight_factor}× høyere",
                xaxis_title="Partier",
                yaxis_title="Samlet score",
                barmode='group',
                xaxis_tickangle=45,
                plot_bgcolor=BG, paper_bgcolor=BG
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("Vektings-påvirkning")
            
            max_change = comparison["Endring"].abs().max()
            most_affected = comparison.loc[comparison["Endring"].abs().idxmax()]
            
            st.metric("Største endring", f"{most_affected['Endring']:+.1f} poeng")
            st.metric("Mest påvirket parti", most_affected.name)
            st.metric("Gjennomsnittlig endring", f"{comparison['Endring'].mean():+.1f} poeng")
            
            # Count questions in boosted category
            category_questions = subset[subset['Kategori'] == boost_category].shape[0]
            st.metric("Spørsmål i vektet tema", f"{category_questions}/{len(subset)}")
            
            # Impact assessment
            if max_change > 5:
                st.warning(f"⚠️ Stor påvirkning! Vekting av '{boost_category}' endrer rangeringer betydelig.")
            elif max_change > 2:
                st.info(f"📊 Moderat påvirkning fra vekting av '{boost_category}'.")
            else:
                st.success(f"✅ Minimal påvirkning fra vekting av '{boost_category}'.")
            
        # Detailed results
        with st.expander("📋 Detaljerte endringer"):
            display_comparison = comparison[["Før_vekting", "Etter_vekting", "Endring", "Endring_pst"]].copy()
            display_comparison.columns = ["Før vekting", "Etter vekting", "Endring (poeng)", "Endring (%)"]
            display_comparison["Retning"] = display_comparison["Endring (poeng)"].apply(
                lambda x: "📈 Økt" if x > 0.5 else "📉 Redusert" if x < -0.5 else "➡️ Uendret"
            )
            st.dataframe(display_comparison, use_container_width=True)

with tab5:
    st.header("Hva om vi fjerner et tema helt?")
    
    create_tv2_only_explanation()
    
    create_explanation_card(
        "Scenario-analyse",
        "Noen ganger er det nyttig å se hva som skjer hvis vi ignorerer visse tema helt. " +
        "For eksempel: Hvordan ser partiene ut hvis vi ser bort fra alle økonomi-spørsmål?"
    )
    
    if 'Kategori' not in tv2.columns:
        st.error("Denne analysen krever TV2-data med kategorier")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            n_questions_scenario = st.slider("Antall spørsmål", 10, 40, 20, key="scenario_q")
        
        with col2:
            available_cats = tv2['Kategori'].dropna().unique()
            remove_category = st.selectbox("Tema å fjerne", available_cats)
        
        # Calculate scenarios
        full_data = tv2.head(n_questions_scenario)
        without_category = tv2[tv2['Kategori'] != remove_category].head(n_questions_scenario)
        
        full_scores = full_data[parties].sum()
        reduced_scores = without_category[parties].sum()
        
        scenario_comparison = pd.DataFrame({
            "Med_alle_tema": full_scores,
            f"Uten_{remove_category}": reduced_scores,
            "Endring": reduced_scores - full_scores,
            "Endring_pst": ((reduced_scores - full_scores) / full_scores * 100).round(1)
        }).sort_values("Med_alle_tema", ascending=False)
        
        col_left, col_right = st.columns([2.5, 1.5])
        
        with col_left:
            fig_scenario = go.Figure()
            
            # With all themes
            fig_scenario.add_trace(go.Bar(
                name='Med alle tema',
                x=scenario_comparison.index,
                y=scenario_comparison["Med_alle_tema"],
                marker_color=COL_NEU,
                text=[f"{val:.0f}" for val in scenario_comparison["Med_alle_tema"]],
                textposition='auto'
            ))
            
            # Without selected theme
            fig_scenario.add_trace(go.Bar(
                name=f'Uten "{remove_category}"',
                x=scenario_comparison.index, 
                y=scenario_comparison[f"Uten_{remove_category}"],
                marker_color=COL_NEG,
                text=[f"{val:.0f}" for val in scenario_comparison[f"Uten_{remove_category}"]],
                textposition='auto'
            ))
            
            fig_scenario.update_layout(
                title=f"Scenario: Hva om vi fjerner alle spørsmål om '{remove_category}'?",
                xaxis_title="Partier",
                yaxis_title="Samlet score",
                barmode='group',
                xaxis_tickangle=45,
                plot_bgcolor=BG, paper_bgcolor=BG
            )
            
            st.plotly_chart(fig_scenario, use_container_width=True)
        
        with col_right:
            st.subheader("🎯 Scenario-konsekvenser")
            
            max_impact = scenario_comparison["Endring"].abs().max()
            most_impacted = scenario_comparison.loc[scenario_comparison["Endring"].abs().idxmax()]
            
            st.metric("Største påvirkning", f"{most_impacted['Endring']:+.1f} poeng")
            st.metric("Mest påvirket parti", most_impacted.name)
            
            # Count removed questions
            removed_questions = full_data[full_data['Kategori'] == remove_category].shape[0]
            st.metric("Spørsmål fjernet", f"{removed_questions}/{len(full_data)}")
            
            if removed_questions == 0:
                st.info("ℹ️ Ingen spørsmål i dette temaet")
            elif max_impact < 2:
                st.success(f"✅ Minimal påvirkning av å fjerne '{remove_category}'")
            elif max_impact < 5:
                st.warning(f"⚠️ Moderat påvirkning av å fjerne '{remove_category}'") 
            else:
                st.error(f"🚨 Stor påvirkning! '{remove_category}' er viktig for partirangeringen.")
        
        # Rankings change analysis
        st.subheader("🏆 Endringer i rangering")
        
        full_ranking = full_scores.rank(method='min', ascending=False)
        reduced_ranking = reduced_scores.rank(method='min', ascending=False)
        
        ranking_changes = pd.DataFrame({
            "Parti": parties,
            "Rangering_før": [full_ranking[p] for p in parties],
            "Rangering_etter": [reduced_ranking[p] for p in parties],
            "Endring_i_rangering": [reduced_ranking[p] - full_ranking[p] for p in parties]
        }).sort_values("Endring_i_rangering", key=abs, ascending=False)
        
        # Show parties with biggest ranking changes
        big_changes = ranking_changes[ranking_changes["Endring_i_rangering"].abs() > 1]
        
        if len(big_changes) > 0:
            st.write("**Partier med endret rangering:**")
            for _, row in big_changes.iterrows():
                direction = "📈" if row["Endring_i_rangering"] < 0 else "📉"
                st.write(f"{direction} **{row['Parti']}**: {row['Rangering_før']:.0f}. → {row['Rangering_etter']:.0f}. plass")
        else:
            st.success("✅ Ingen store endringer i partirangering")

with tab6:
    st.header("🤖 AI & Datakvalitet")
    st.write("Utforsk hvordan datakvalitet påvirker AI-resultater gjennom valgomatdata.")
    
    create_explanation_card(
        "Hvorfor er datakvalitet viktig?",
        "AI-systemer er kun så gode som dataene de er trent på. Dårlig datakvalitet kan føre til " +
        "feilaktige konklusjoner, skjeve algoritmer og upålitelige anbefalinger. " +
        "Her kan du utforske hvordan ulike datakvalitetsutfordringer påvirker AI-ytelse.",
        "🔍"
    )
    
    # Data quality dimensions selector
    quality_dim = st.selectbox(
        "Velg datakvalitetsdimensjon:",
        ["📊 Oversikt", "🎯 Nøyaktighet", "📋 Kompletthet", "🔄 Konsistens", 
         "⏰ Aktualitet", "✅ Validitet", "🎭 Unikalitet"]
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
                
                # Show detailed information about selected dimension
                dimension_details = {
                    "Nøyaktighet": {
                        "description": "Måler hvor godt dataene reflekterer virkeligheten",
                        "calculation": "Basert på konsistens mellom NRK og TV2 kilder",
                        "interpretation": "Høy score = god samsvar mellom kilder, lav bias"
                    },
                    "Kompletthet": {
                        "description": "Måler hvor mye av dataene som faktisk er tilgjengelig",
                        "calculation": "Prosent av celler som ikke er tomme eller manglende",
                        "interpretation": "Høy score = få manglende verdier"
                    },
                    "Konsistens": {
                        "description": "Måler hvor stabilt og ikke-motsigelsesfullt dataene er",
                        "calculation": "Basert på variabilitet i partisvar (lavere variabilitet = mer konsistent)",
                        "interpretation": "Høy score = mindre variabilitet, mer stabile mønstre"
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
                    "Unikalitet": {
                        "description": "Måler grad av duplikater og overrepresentasjon",
                        "calculation": "Basert på balanse mellom antall spørsmål og kategorier",
                        "interpretation": "Høy score = god balanse, få duplikater"
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
            
            # Show top/bottom dimensions
            score_df = pd.DataFrame({"Dimensjon": dimensions, "Score": scores}).sort_values("Score", ascending=False)
            
            st.write("**🏆 Best:**")
            st.write(f"{score_df.iloc[0]['Dimensjon']}: {score_df.iloc[0]['Score']:.0f}%")
            
            st.write("**🎯 Trenger forbedring:**")  
            st.write(f"{score_df.iloc[-1]['Dimensjon']}: {score_df.iloc[-1]['Score']:.0f}%")
        
        create_explanation_card(
            "Tips for interaksjon",
            "Klikk på en dimensjon i radar-diagrammet ovenfor for å se detaljert informasjon om " +
            "hvordan scoren beregnes og hva den betyr for AI-kvalitet. " +
            "Velg andre dimensjoner fra dropdown-menyen for å utforske spesifikke problemer.",

        )
    
    elif quality_dim == "🎯 Nøyaktighet":
        st.subheader("Nøyaktighet: Hvordan skjeve spørsmål påvirker AI-anbefalinger")
        
        create_explanation_card(
            "Problemet med skjevhet",
            "Måten spørsmål formuleres på kan påvirke svarene dramatisk. AI-systemer som lærer " +
            "fra skjeve data vil gi skjeve anbefalinger - dette kalles 'bias in, bias out'.",
            "⚠️", "warning"
        )
        
        # Interactive bias demonstration
        bias_level = st.slider("Grad av skjevhet i spørsmålsformulering:", 0.0, 1.0, 0.3, 0.1, 
                              key="bias_slider", help="Høyere verdi = mer skjev formulering")
        
        # Add explicit regeneration trigger
        if st.button("🔄 Oppdater grafer", help="Klikk hvis grafene ikke oppdateres automatisk"):
            st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Nøytral formulering:**")
            st.write("'Bør Norge øke bistanden til utviklingsland?'")
            
            # Simulate neutral AI recommendations - use bias_level as seed modifier
            np.random.seed(42 + int(bias_level * 100))  # Change seed based on bias level
            neutral_parties = parties[:6]  # Use first 6 parties for simplicity
            neutral_scores = 3 + np.random.normal(0, 0.5, len(neutral_parties))
            neutral_scores = np.clip(neutral_scores, 1, 5)
            
            fig_neutral = go.Figure(data=[
                go.Bar(x=neutral_parties, y=neutral_scores, 
                      marker_color=COL_POS,
                      text=[f"{s:.1f}" for s in neutral_scores],
                      textposition='auto')
            ])
            fig_neutral.update_layout(
                title="AI-anbefaling fra nøytrale data",
                yaxis_title="Anbefalt styrke (1-5)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111"),
                yaxis=dict(range=[0, 5])
            )
            st.plotly_chart(fig_neutral, use_container_width=True, key=f"neutral_chart_{bias_level}")
            
            st.metric("AI Tillit", "95%", delta="Høy pålitelighet")
        
        with col2:
            st.markdown("**🔴 Skjev formulering:**")
            
            # Dynamic question based on bias level
            if bias_level < 0.3:
                question_text = "'Bør Norge investere smartere i bistand?'"
                bias_description = "Svakt ladet språk"
            elif bias_level < 0.7:
                question_text = "'Bør Norge kaste bort penger på bistand til korrupte land?'"
                bias_description = "Moderat negativ framing"
            else:
                question_text = "'Bør Norge stoppe meningsløs bistand til håpløse land?'"
                bias_description = "Sterkt negativ framing"
            
            st.write(question_text)
            st.caption(f"*{bias_description}*")
            
            # Simulate biased recommendations
            bias_effect = bias_level * 2
            biased_scores = neutral_scores - bias_effect + np.random.normal(0, 0.3, len(neutral_parties))
            biased_scores = np.clip(biased_scores, 0.5, 5)
            
            fig_biased = go.Figure(data=[
                go.Bar(x=neutral_parties, y=biased_scores, 
                      marker_color=COL_NEG,
                      text=[f"{s:.1f}" for s in biased_scores],
                      textposition='auto')
            ])
            fig_biased.update_layout(
                title="AI-anbefaling fra skjeve data",
                yaxis_title="Anbefalt styrke (1-5)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111"),
                yaxis=dict(range=[0, 5])
            )
            st.plotly_chart(fig_biased, use_container_width=True, key=f"biased_chart_{bias_level}")
            
            ai_trust = max(20, 95 - bias_level * 60)
            st.metric("AI Tillit", f"{ai_trust:.0f}%", delta=f"{ai_trust-95:.0f}%")
        
        # Show real-time impact analysis
        st.subheader("Skjevhets-påvirkning i sanntid")
        diff_scores = np.abs(neutral_scores - biased_scores)
        avg_difference = np.mean(diff_scores)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Gjennomsnittlig avvik", f"{avg_difference:.2f} poeng")
        col2.metric("Maksimalt avvik", f"{np.max(diff_scores):.2f} poeng") 
        col3.metric("Partier sterkt påvirket", f"{np.sum(diff_scores > 1)}")
        
        # Dynamic status based on current bias level
        if avg_difference > 0.8:
            st.error("🚨 **Kritisk**: Skjeve spørsmål endrer AI-anbefalinger dramatisk!")
        elif avg_difference > 0.4:
            st.warning("⚠️ **Advarsel**: Moderate endringer i AI-anbefalinger")
        else:
            st.success("✅ **Akseptabelt**: Minimal påvirkning på AI-anbefalinger")
    
    elif quality_dim == "📋 Kompletthet":
        st.subheader("Kompletthet: Når AI mangler viktige data")
        
        create_explanation_card(
            "Problemet med manglende data",
            "AI-systemer kan bare lære av dataene de får. Manglende informasjon kan føre til " +
            "feilaktige konklusjoner og unfair behandling av grupper som er underrepresenterte i dataene.",
            "⚠️", "warning"  
        )
        
        # Interactive missing data simulation
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
    
    elif quality_dim == "🔄 Konsistens":
        st.subheader("Konsistens: Når AI får motstridende informasjon")
        
        create_explanation_card(
            "Problemet med inkonsistente data",
            "Når ulike kilder gir forskjellige svar på lignende spørsmål, blir AI-systemer forvirret. " +
            "Dette kan føre til ustabile prediksjoner og lav tillit til systemet.",
            "⚠️", "warning"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # NRK vs TV2 consistency analysis
            nrk_sample = nrk.head(15)[parties].mean()
            tv2_sample = tv2.head(15)[parties].mean()
            
            consistency_df = pd.DataFrame({
                "NRK": nrk_sample,
                "TV2": tv2_sample
            })
            consistency_df["Forskjell"] = abs(consistency_df["NRK"] - consistency_df["TV2"])
            
            fig_cons = go.Figure()
            
            # Scatter plot with party labels
            fig_cons.add_trace(go.Scatter(
                x=consistency_df["NRK"], 
                y=consistency_df["TV2"],
                mode='markers+text',
                text=consistency_df.index,
                textposition="top center",
                marker=dict(
                    size=consistency_df["Forskjell"] * 15 + 8,
                    color=consistency_df["Forskjell"],
                    colorscale=[[0, COL_POS], [0.5, COL_NEU], [1, COL_NEG]],
                    colorbar=dict(title="Inkonsistens")
                ),
                name="Partier"
            ))
            
            # Add diagonal line for perfect consistency
            min_val = min(consistency_df["NRK"].min(), consistency_df["TV2"].min())
            max_val = max(consistency_df["NRK"].max(), consistency_df["TV2"].max())
            fig_cons.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="Perfekt konsistens",
                hoverinfo='skip'
            ))
            
            fig_cons.update_layout(
                title="Konsistens mellom NRK og TV2<br>(større punkt = mer inkonsistent)",
                xaxis_title="NRK gjennomsnitt",
                yaxis_title="TV2 gjennomsnitt",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_cons, use_container_width=True)
        
        with col2:
            # Consistency metrics and AI impact
            avg_diff = consistency_df["Forskjell"].mean()
            max_diff = consistency_df["Forskjell"].max()
            
            st.subheader("Konsistens-mål")
            st.metric("Gj.snitt kilde-uenighet", f"{avg_diff:.2f} poeng")
            st.metric("Maks kilde-uenighet", f"{max_diff:.2f} poeng")
            
            # AI impact from inconsistency
            consistency_score = max(0, 100 - avg_diff * 30)
            ai_trust = max(20, 95 - avg_diff * 25)
            
            st.metric("Konsistens-score", f"{consistency_score:.0f}%")
            st.metric("AI Tillit", f"{ai_trust:.0f}%")
            
            # Recommendations based on consistency
            if avg_diff < 0.3:
                st.success("🟢 Høy konsistens - AI kan stole på begge kilder")
            elif avg_diff < 0.7:
                st.warning("🟡 Moderat inkonsistens - AI trenger validering")
            else:
                st.error("🔴 Høy inkonsistens - AI-resultater upålitelige")
        
        # Solution strategies
        st.subheader("🔧 Løsningsstrategier for inkonsistente data")
        strategy = st.selectbox("AI-strategi for å håndtere inkonsistens:", 
                               ["Kilde-vekting", "Usikkerhetsintervaller", "Ensemble-metoder", "Manuell gjennomgang"])
        
        if strategy == "Kilde-vekting":
            st.info("💡 **Kilde-vekting**: Gi mer pålitelige kilder høyere vekt i AI-beslutninger")
        elif strategy == "Usikkerhetsintervaller":
            st.info("💡 **Usikkerhetsintervaller**: Gi usikkerhetsspenn i stedet for eksakte prediksjoner")
        elif strategy == "Ensemble-metoder":
            st.info("💡 **Ensemble-metoder**: Bruk flere AI-modeller og kombiner deres prediksjoner")
        else:
            st.info("💡 **Manuell gjennomgang**: Flagg inkonsistente tilfeller for ekspert-vurdering")
    
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
Faktisk: "Feil kÃ¸ding av norske tegn"
Skulle være: "Feil koding av norske tegn"
                """, language="text")
            
            # Show data distribution with errors
            valid_data = np.random.choice([-2, -1, 0, 1, 2], size=100-error_rate)
            if validity_issue == "Verdier utenfor rekkevidde":
                invalid_data = np.random.choice([5, 7, -5, 10], size=error_rate)
                all_data = np.concatenate([valid_data, invalid_data])
                
                fig_val = go.Figure()
                
                # Valid data
                valid_counts = np.bincount(valid_data + 2, minlength=5)
                fig_val.add_trace(go.Bar(
                    x=[-2, -1, 0, 1, 2], y=valid_counts,
                    name="Gyldige data", marker_color=COL_POS
                ))
                
                # Invalid data  
                unique_invalid, invalid_counts = np.unique(invalid_data, return_counts=True)
                fig_val.add_trace(go.Bar(
                    x=unique_invalid, y=invalid_counts,
                    name="Ugyldige data", marker_color=COL_NEG
                ))
            else:
                fig_val = go.Figure()
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
    
    else:  # Unikalitet
        st.subheader("Unikalitet: Når AI lærer feil fra duplikater")
        
        create_explanation_card(
            "Problemet med duplikater",
            "Duplikate data får AI til å tro at noen ting er viktigere enn de egentlig er. " +
            "Dette kan skape kunstig skjevhet og overvekting av spesifikke temaer.",
            "⚠️", "warning"
        )
        
        # Interactive duplicate demonstration
        duplicate_rate = st.slider("Duplikatrate (%):", 0, 40, 15, 5)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show effect of duplicates on category influence
            categories = ["Økonomi", "Helse", "Utdanning", "Miljø", "Trygghet"]
            
            # Original balanced influence
            original_influence = [20, 20, 20, 20, 20]
            
            # Simulate duplicates in one category
            duplicate_category = "Økonomi"
            duplicate_factor = 1 + (duplicate_rate / 100) * 3  # Up to 4x overrepresentation
            
            skewed_influence = original_influence.copy()
            skewed_influence[0] = skewed_influence[0] * duplicate_factor
            total_new = sum(skewed_influence)
            skewed_influence = [x / total_new * 100 for x in skewed_influence]
            
            fig_uniq = go.Figure()
            fig_uniq.add_trace(go.Bar(
                name='Original (balansert)',
                x=categories,
                y=original_influence,
                marker_color=COL_POS
            ))
            fig_uniq.add_trace(go.Bar(
                name=f'Med {duplicate_rate}% duplikater',
                x=categories,
                y=skewed_influence,
                marker_color=COL_NEG
            ))
            
            fig_uniq.update_layout(
                title="Påvirkning av kategorier på AI-beslutninger",
                xaxis_title="Kategori",
                yaxis_title="Påvirkning (%)",
                barmode='group',
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_uniq, use_container_width=True)
        
        with col2:
            # Calculate uniqueness impact
            uniqueness_score = max(0, 100 - duplicate_rate * 2.5)
            ai_bias = 100 - uniqueness_score
            
            st.subheader("Unikalitet-påvirkning")
            st.metric("Data-unikalitet", f"{uniqueness_score:.0f}%")
            st.metric("AI Skjevhet", f"{ai_bias:.0f}%", 
                     delta=f"+{ai_bias:.0f}% vs balanserte data")
            
            # Show overrepresentation
            overrep = skewed_influence[0] / original_influence[0]
            st.metric(f"{duplicate_category} Overrepresentasjon", f"{overrep:.1f}x")
            
            # Uniqueness status
            if duplicate_rate < 5:
                st.success("🟢 Høy unikalitet - Minimal AI-skjevhet")
            elif duplicate_rate < 15:
                st.warning("🟡 Moderate duplikater - Noe AI-skjevhet")
            else:
                st.error("🔴 Høy duplisering - Betydelig AI-skjevhet")
        
        # Duplicate detection strategies
        st.subheader("🔍 Duplikat-deteksjonsmetoder")
        
        detection_methods = {
            "Eksakt matching": "Finn identiske spørsmål ord-for-ord",
            "Semantisk likhet": "Finn spørsmål med lignende betydning",  
            "Statistisk korrelasjon": "Finn spørsmål med høyt korrelerte svar",
            "Manuell gjennomgang": "Ekspert identifiserer konseptuelle duplikater"
        }
        
        for method, description in detection_methods.items():
            with st.expander(f"📋 {method}"):
                st.write(description)
                
                if method == "Eksakt matching":
                    st.code("""
Eksempel duplikater:
1. "Bør Norge øke skattene?"
2. "Bør Norge øke skattene?" 
[Identiske - Lett å oppdage]
                    """, language="text")
                elif method == "Semantisk likhet":
                    st.code("""
Eksempel duplikater:
1. "Bør Norge øke skatten for de rike?"
2. "Synes du Norge bør ha høyere skatt på høye inntekter?"
[Lignende mening - Vanskeligere å oppdage]
                    """, language="text")
                elif method == "Statistisk korrelasjon":
                    st.code("""
Hvis svar på Spm1 og Smp2 korrelerer > 0.95:
Sannsynligvis måler det samme
                    """, language="text")
                else:
                    st.code("""
Ekspert-gjennomgang nødvendig for:
- Konseptuelle overlapp
- Ulike framing av samme tema  
- Subtile semantiske forskjeller
                    """, language="text")

with tab7:
    st.header("Metodikk og Transparens")
    
    create_explanation_card(
        "Vårt transparens-prinsipp",
        "All vår metodikk er åpen og kan granskes. Vi oppfordrer til kritisk vurdering av våre " +
        "antakelser og metoder. Dette er ikke 'absolutte sannheter', men analytiske verktøy."
    )
    
    # Expandable methodology sections
    with st.expander("🎯 Hvordan beregner vi datakvalitet?", expanded=True):
        st.markdown("""
        ### De 6 dimensjonene av datakvalitet
        
        Vår app vurderer datakvalitet langs 6 vitenskapelig anerkjente dimensjoner:
        
        #### 1. 🎯 Nøyaktighet (Accuracy)
        **Hva det måler:** Hvor godt dataene reflekterer virkeligheten  
        **Beregning:** `100% - (Gj.snitt absolutt forskjell mellom NRK og TV2) / 4 * 100%`  
        **Logikk:** Hvis to uavhengige kilder gir lignende resultater, øker tilliten til nøyaktighet
        
        #### 2. 📋 Kompletthet (Completeness)  
        **Hva det måler:** Hvor mye av dataene som faktisk er tilgjengelig  
        **Beregning:** `(Totale celler - Manglende celler) / Totale celler * 100%`  
        **Logikk:** Manglende data reduserer AI-systemers læringsevne
        
        #### 3. 🔄 Konsistens (Consistency)
        **Hva det måler:** Hvor stabile og ikke-motsigelsesfulle dataene er  
        **Beregning:** `100% - (Gj.snitt standardavvik * 25)`  
        **Logikk:** Høy variabilitet kan indikere inkonsistente målinger
        
        #### 4. ⏰ Aktualitet (Timeliness)
        **Hva det måler:** Hvor oppdaterte dataene er  
        **Beregning:** `100% - (Måneder siden innsamling * 2%)`  
        **Antakelse:** Data antas 6 mnd gamle, 2% verdifall per måned
        
        #### 5. ✅ Validitet (Validity)
        **Hva det måler:** Om dataene har korrekt format og gyldige verdier  
        **Beregning:** `Antall verdier i range [-2,+2] / Totale verdier * 100%`  
        **Logikk:** Valgomatskalaen har definerte grenser
        
        #### 6. 🎭 Unikalitet (Uniqueness)
        **Hva det måler:** Grad av duplikater og overrepresentasjon  
        **Beregning:** `min(100%, (Antall kategorier * 4) / Totale spørsmål * 100%)`  
        **Antakelse:** ~4 spørsmål per kategori som optimal balanse
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
        - NRK og TV2 er begge relativt pålitelige kilder
        - 6 måneder gammel data (estimat for valgomatdata)  
        - 4 spørsmål per kategori er optimalt
        - Politisk volatilitet på 2% per måned
        - Standardavvik reflekterer inkonsistens (kan også være legitim variasjon)
        
        ### Viktige forbehold:
        - **Ikke absolutte sannheter:** Våre metoder er analytiske verktøy, ikke objektive målinger
        - **Kontekst-avhengig:** Kvalitet avhenger av bruksområde og krav
        - **Forenklede modeller:** Virkeligheten er mer kompleks enn våre algoritmer
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
        # Eksempel: Beregning av konsistens
        nrk_std = nrk[parties].std().mean()
        tv2_std = tv2[parties].std().mean() 
        avg_std = (nrk_std + tv2_std) / 2
        consistency = max(0, 100 - avg_std * 25)
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
    
    with st.expander("🎯 Bruksanvisning for forskere"):
        st.markdown("""
        ### Hvordan bruke appen i forskning
        
        **Egnet for:**
        - Eksplorativ dataanalyse av politiske holdninger
        - Identifisering av kontroversielle politiske tema  
        - Sammenligning av mediekilders politiske profiler
        - Undervisning i datakvalitet og AI-bias
        
        **IKKE egnet for:**
        - Predikering av valgresultater
        - Kausal slutning om politiske årsaksforhold
        - Generalisering til befolkningen som helhet
        - Presise målinger av partiforskjeller
        
        **Best practices:**
        1. Kombiner med andre datakilder
        2. Vurder kontekstuelle faktorer
        3. Rapporter metodiske begrensninger
        4. Bruk som utgangspunkt for videre forskning
        """)
    
    with st.expander(" Filosofiske refleksjoner"):
        st.markdown("""
        ### Hva kan vi egentlig vite?
        
        **Epistemologiske spørsmål:**
        - Kan vi objektivt måle "datakvalitet"?
        - Reflekterer partiposisjoner "sanne" politiske standpunkter?
        - Hvor mye påvirker spørsmålsformulering svarene?
        
        **Etiske betraktninger:**
        - Risiko for å forsterke eksisterende bias
        - Ansvar ved automatisering av politiske vurderinger
        - Transparens vs. kompleksitet i AI-systemer
        
        **Pragmatiske kompromisser:**
        - Perfekt objektivitet er umulig, men vi kan strebe etter transparens
        - Forenklede modeller kan være nyttige selv om de ikke er komplette
        - Kritisk tenkning er viktigere enn algoritmisk presisjon
        """)

# Footer with methodology
st.divider()
create_explanation_card(
    "Om metodikken", 
    """
    **Datakilder:** Partisvar på valgomatspørsmål fra NRK og TV2
    
    **Viktige begrensninger:** 
    • Vi har kun partienes offisielle standpunkter, ikke velgernes svar
    • NRK og TV2 har forskjellige spørsmål - kan ikke sammenlignes direkte
    • Analyser viser politiske mønstre, ikke absolutte "sannheter"
    
    **Statistiske mål:**
    • Gjennomsnitt viser om partiene lener mot støtte eller motstand
    • Uenighet måles som standardavvik mellom partiers posisjoner
    • Høyere uenighet = mer kontroversielt/splittende spørsmål
    
    **Åpen kildekode:** All kode og metodikk er tilgjengelig for granskning og forbedring.
    """,
    "📖"
)

st.caption("💡 Tips: Klikk på grafene for å utforske interaktivt. Bruk faner for å se ulike analyser.")