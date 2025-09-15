# streamlit_app.py
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

BG = "#fcf6ee"
COL_NEG, COL_NEU, COL_POS = "#ff8c45", "#ffd865", "#97d2ec"
COLORS = [COL_NEG, COL_NEU, COL_POS]

st.set_page_config(page_title="Valgomat – utforsker", layout="wide")

# Apply matplotlib theme
def apply_simple_theme():
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "#cccccc",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.autolayout": True,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.prop_cycle": plt.cycler(color=COLORS),
    })

apply_simple_theme()

# --- Global styling with dropdown fix
st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {BG} !important;
        color: #111111 !important;
      }}
      .main .block-container {{
        background-color: {BG} !important;
        padding-top: 1rem;
      }}
      .stApp > header {{
        background-color: transparent !important;
      }}
      .stApp > .main {{
        background-color: {BG} !important;
      }}
      div[data-testid="stAppViewContainer"] {{
        background-color: {BG} !important;
      }}
      div[data-testid="stHeader"] {{
        background-color: {BG} !important;
      }}
      div[data-testid="stToolbar"] {{
        background-color: {BG} !important;
      }}
      div[data-testid="stDecoration"] {{
        background-color: {BG} !important;
      }}
      h1, h2, h3, h4, h5, h6, p, li, span, div {{
        color: #111111 !important;
      }}
      .stMarkdown span {{
        color: #111111 !important;
      }}
      .css-1d391kg, .css-1rs6os {{
        background-color: #f0e6d6 !important;
      }}
      
      /* Fix dropdown menu backgrounds */
      .stSelectbox > div > div {{
        background-color: {BG} !important;
      }}
      .stSelectbox [data-testid="stSelectbox"] {{
        background-color: {BG} !important;
      }}
      .stSelectbox [data-baseweb="select"] {{
        background-color: {BG} !important;
      }}
      .stSelectbox [data-baseweb="select"] > div {{
        background-color: {BG} !important;
      }}
      /* Dropdown options */
      .stSelectbox [data-baseweb="menu"] {{
        background-color: {BG} !important;
      }}
      .stSelectbox [data-baseweb="menu"] ul {{
        background-color: {BG} !important;
      }}
      .stSelectbox [data-baseweb="menu"] li {{
        background-color: {BG} !important;
        color: #111111 !important;
      }}
      .stSelectbox [data-baseweb="menu"] li:hover {{
        background-color: #f0e6d6 !important;
        color: #111111 !important;
      }}
      /* Additional selectbox styling */
      div[data-baseweb="select"] {{
        background-color: {BG} !important;
      }}
      div[data-baseweb="select"] > div {{
        background-color: {BG} !important;
        color: #111111 !important;
      }}
      div[data-baseweb="popover"] {{
        background-color: {BG} !important;
      }}
      div[data-baseweb="popover"] ul {{
        background-color: {BG} !important;
      }}
      div[data-baseweb="popover"] li {{
        background-color: {BG} !important;
        color: #111111 !important;
      }}
      div[data-baseweb="popover"] li:hover {{
        background-color: #f0e6d6 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

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

def prep(df, parties):
    out = df.copy()
    out["mu"]  = out[parties].mean(axis=1)
    out["var"] = out[parties].var(axis=1)
    out["std"] = out[parties].std(axis=1)
    return out

def color_map(mu):
    if mu < -0.2: return COL_NEG
    if mu >  0.2: return COL_POS
    return COL_NEU

def size_scale(x, smin=10, smax=28):
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return smin + x*(smax - smin)

# Simulation functions for data quality demos
def simulate_ai_confidence_degradation(quality_score):
    """Simulate how AI confidence degrades with poor data quality"""
    base_confidence = 95
    degradation = (100 - quality_score) * 0.8
    return max(20, base_confidence - degradation)

def simulate_recommendation_accuracy(missing_data_pct):
    """Simulate how recommendation accuracy drops with missing data"""
    base_accuracy = 92
    accuracy_loss = missing_data_pct * 0.6
    return max(45, base_accuracy - accuracy_loss)

def generate_biased_vs_unbiased_data(parties, bias_factor=0.3):
    """Generate simulated data showing bias effects"""
    np.random.seed(42)
    unbiased = np.random.normal(0, 1, len(parties))
    biased = unbiased + np.random.normal(bias_factor, 0.5, len(parties))
    return unbiased, biased

# Load data
parties, nrk, tv2 = load_files()

# --- MAIN NAVIGATION ---
st.title("Valgomat – utforsker")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Polarisering (Hovedvisning)", 
    "📈 Kategori-oversikt", 
    "⚖️ NRK vs TV2", 
    "🔧 Vektingseffekt", 
    "🎯 Scenario-analyse",
    "🤖 AI & Datakvalitet"
])

with tab1:
    st.header("Spørsmål som splitter: polariseringsanalyse")
    
    with st.sidebar:
        st.header("Kontroller")
        dataset = st.radio("Datasett", ["NRK (uten kategorier)","TV 2 (med kategorier)"])
        top_k   = st.slider("Antall topp-punkter å merke", 3, 12, 8, 1)
        wrap_w  = st.slider("Linjelengde (tooltip/tekst)", 40, 100, 70, 2)
        show_labels = st.toggle("Vis nummer på topp-punkter", value=True)
        st.caption("Bakgrunn = #fcf6ee  •  Palett = oransje/gul/blå")

    left, right = st.columns([3.5, 2])

    # --- Velg df
    if dataset.startswith("NRK"):
        df = prep(nrk, parties)
        extra_cols = []
    else:
        df = prep(tv2, parties)
        extra_cols = ["Kategori"]

    # --- Polarization scatter (Plotly)
    df["color"] = df["mu"].apply(color_map)
    df["size"]  = size_scale(df["var"])

    median_var = float(df["var"].median())
    median_std = float(df["std"].median())

    hover_cols = ["Spm", "mu", "var", "std"] + extra_cols
    df["Spm_wrapped"] = df["Spm"].apply(lambda t: wrap(t, width=wrap_w, lines=5))

    fig = px.scatter(
        df,
        x="mu", y="var",
        size="size",
        color="color",
        color_discrete_sequence=[COL_NEG, COL_NEU, COL_POS],
        hover_data=hover_cols,
    )
    fig.update_traces(marker=dict(color=df["color"], line=dict(width=0.6, color="black")), selector=dict(mode="markers"))
    fig.update_traces(hovertemplate=
        "<b>%{customdata[0]}</b><br>"  # Spm
        "μ=%{customdata[1]:+.2f}&nbsp;&nbsp;σ²=%{customdata[2]:.2f}&nbsp;&nbsp;σ=%{customdata[3]:.2f}"
        + ("<br>Kategori=%{customdata[4]}" if extra_cols else "") +
        "<extra></extra>"
    )

    # median-linje + x=0
    fig.add_hline(y=median_var, line_dash="dash", line_color="gray", opacity=0.8)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.8)

    fig.update_layout(
        title="Spørsmål som splitter: retning (μ) vs. uenighet (σ²)",
        xaxis_title="Retning (snitt, μ)  ← negativ | positiv →",
        yaxis_title="Uenighet (varians, σ²)",
        xaxis=dict(title_font=dict(color="#111111"), tickfont=dict(color="#111111")),
        yaxis=dict(title_font=dict(color="#111111"), tickfont=dict(color="#111111")),
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False
    )

    # merk topp-K
    top = df.nlargest(top_k, "var").copy().reset_index(drop=True)
    if show_labels:
        fig.add_trace(go.Scatter(
            x=top["mu"], y=top["var"],
            mode="markers",
            marker=dict(size=top["size"]*1.15, color="rgba(255,255,255,0.1)", line=dict(width=2.5, color="#333333")),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=top["mu"], y=top["var"],
            mode="text",
            text=[str(i) for i in range(1, len(top)+1)],
            textfont=dict(size=14, color="#222222", family="Arial"),
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip"
        ))

    with left:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Median uenighet: σ²={median_var:.2f} (σ={median_std:.2f})")

    # --- Topp-liste
    def make_list_df(top, wrap_w=80):
        rows = []
        for i, r in enumerate(top.itertuples(index=False), 1):
            rows.append({
                "#": i,
                "Spørsmål": textwrap.fill(r.Spm, width=wrap_w),
                "μ (snitt)": f"{r.mu:+.2f}",
                "σ² (varians)": f"{r.var:.2f}"
            })
        return pd.DataFrame(rows)

    with right:
        st.subheader(f"Topp {top_k} mest polariserende")
        list_df = make_list_df(top, wrap_w=wrap_w-10)
        for i, row in list_df.iterrows():
            st.markdown(
                f"**{row['#']}. {row['Spørsmål']}**\n\n"
                f"<span style='color:#555'>μ={row['μ (snitt)']}&nbsp;&nbsp;σ²={row['σ² (varians)']}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

with tab2:
    # Center the header
    st.markdown("<h1 style='text-align: center; color: #111111;'>Kategori-oversikt (TV 2)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666666;'>Fordeling av spørsmål på temaer i TV 2-datasettet.</p>", unsafe_allow_html=True)
    
    s = tv2["Kategori"].dropna().value_counts()
    total = s.sum()
    pct = (s / total * 100).sort_values(ascending=False)  # Largest first for better visual hierarchy
    
    # Create color palette - more colors for better distinction
    extended_colors = [COL_NEG, COL_POS, COL_NEU, "#8B9DC3", "#DEB887", "#F4A460", "#98FB98", "#FFB6C1", "#87CEEB", "#DDA0DD"]
    category_colors = [extended_colors[i % len(extended_colors)] for i in range(len(pct))]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create modern donut chart with Plotly
        fig_donut = go.Figure(data=[go.Pie(
            labels=pct.index, 
            values=pct.values,
            hole=0.4,  # Creates donut effect
            marker=dict(
                colors=category_colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=11, color='#111111'),
            texttemplate='%{label}<br>%{percent}',
            hovertemplate='<b>%{label}</b><br>' +
                         'Antall: %{value:.0f}<br>' +
                         'Prosent: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        # Add center text
        fig_donut.add_annotation(
            text=f"<b>{total}</b><br>Totalt<br>spørsmål",
            x=0.5, y=0.5,
            font=dict(size=16, color='#111111'),
            showarrow=False
        )
        
        fig_donut.update_layout(
            title=dict(
                text="<b>Kategori-fordeling</b>",
                font=dict(size=18, color='#111111'),
                x=0.5
            ),
            plot_bgcolor=BG,
            paper_bgcolor=BG,
            font=dict(color='#111111'),
            showlegend=False,  # Remove legend since labels are on the chart
            margin=dict(t=80, b=80, l=80, r=80),  # Increased margins to prevent label cutoff
            height=600  # Increased height to accommodate labels
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col2:
        st.subheader("📊 Kategori-detaljer")
        
        # Create elegant category cards
        for i, (kategori, prosent) in enumerate(pct.items()):
            antall = s[kategori]
            color = category_colors[i]
            
            # Create a styled card for each category
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border-left: 4px solid {color};
                padding: 12px 16px;
                margin: 8px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-weight: bold; color: #111111; font-size: 14px;">
                            {kategori}
                        </div>
                        <div style="color: #666; font-size: 12px;">
                            {antall} spørsmål
                        </div>
                    </div>
                    <div style="
                        background: {color};
                        color: white;
                        padding: 4px 12px;
                        border-radius: 20px;
                        font-weight: bold;
                        font-size: 13px;
                    ">
                        {prosent:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add summary statistics
        st.markdown("---")
        st.subheader("📈 Sammendrag")
        
        largest_category = pct.index[0]
        largest_pct = pct.iloc[0]
        smallest_category = pct.index[-1]
        smallest_pct = pct.iloc[-1]
        
        st.metric("Største kategori", largest_category, f"{largest_pct:.1f}%")
        st.metric("Minste kategori", smallest_category, f"{smallest_pct:.1f}%")
        st.metric("Antall kategorier", len(pct))
        
        # Diversity index (how evenly distributed the categories are)
        # Higher values = more evenly distributed
        expected_pct = 100 / len(pct)
        diversity_score = 100 - np.std(pct.values)
        st.metric("Diversitetsindeks", f"{diversity_score:.0f}/100")
        
        if diversity_score > 80:
            st.success("🟢 Godt balansert fordeling")
        elif diversity_score > 60:
            st.warning("🟡 Moderat balanse")
        else:
            st.error("🔴 Ubalansert fordeling")
    
    # Optional: Show raw data in an expandable section
    with st.expander("🔍 Se rådata"):
        cat_df = pd.DataFrame({
            "Kategori": pct.index, 
            "Antall": s[pct.index].values, 
            "Prosent": pct.values.round(1),
            "Kumulativ %": pct.values.cumsum().round(1)
        })
        st.dataframe(cat_df, use_container_width=True)

with tab3:
    st.header("Sammenligning: NRK vs TV 2")
    st.write("Partisummer basert på de første N spørsmålene fra hver kilde.")
    
    NQ = st.slider("Antall spørsmål å inkludere", 10, 50, 30, 5, key="nq_slider")
    
    nrk_sum = nrk.head(NQ)[parties].sum()
    tv2_sum = tv2.head(NQ)[parties].sum()
    
    comp_df = pd.DataFrame({"NRK": nrk_sum, "TV 2": tv2_sum})
    comp_df["avg"] = comp_df[["NRK","TV 2"]].mean(axis=1)
    comp_df = comp_df.sort_values("avg", ascending=False).drop(columns="avg")
    
    # Calculate differences for analysis
    comp_df["Difference"] = comp_df["TV 2"] - comp_df["NRK"]
    comp_df["Abs_Difference"] = abs(comp_df["Difference"])
    
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        # Add subtitle above the chart, separate from the chart itself
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #fafafa; margin-bottom: 5px;">Partipoengsum: NRK vs TV 2</h3>
            <p style="color: #cccccc; font-size: 14px; margin: 0;">
                Basert på {NQ} spørsmål per kilde (av {len(nrk)} NRK, {len(tv2)} TV2 tilgjengelig)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create modern grouped bar chart with Plotly
        fig_comp = go.Figure()
        
        # Add NRK bars
        fig_comp.add_trace(go.Bar(
            name='NRK',
            x=comp_df.index,
            y=comp_df["NRK"],
            marker=dict(
                color=COL_NEG,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.0f}" for val in comp_df["NRK"]],
            textposition='auto',
            textfont=dict(color='white', size=11),
            hovertemplate='<b>%{x}</b><br>NRK: %{y:.1f}<extra></extra>'
        ))
        
        # Add TV2 bars
        fig_comp.add_trace(go.Bar(
            name='TV 2',
            x=comp_df.index,
            y=comp_df["TV 2"],
            marker=dict(
                color=COL_POS,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.0f}" for val in comp_df["TV 2"]],
            textposition='auto',
            textfont=dict(color='white', size=11),
            hovertemplate='<b>%{x}</b><br>TV 2: %{y:.1f}<extra></extra>'
        ))
        
        # Add difference indicators (subtle background bars)
        max_val = max(comp_df[["NRK", "TV 2"]].max())
        fig_comp.add_trace(go.Bar(
            name='Forskjell (bakgrunn)',
            x=comp_df.index,
            y=[max_val * 1.1] * len(comp_df),  # Background height
            marker=dict(
                color=['rgba(255,0,0,0.1)' if diff < -2 else 
                       'rgba(0,255,0,0.1)' if diff > 2 else 
                       'rgba(128,128,128,0.05)' for diff in comp_df["Difference"]],
                line=dict(width=0)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_comp.update_layout(
            xaxis=dict(
                title="Partier",
                tickangle=45,
                tickfont=dict(color='#111111'),
                title_font=dict(color='#111111')
            ),
            yaxis=dict(
                title="Sum score",
                tickfont=dict(color='#111111'),
                title_font=dict(color='#111111'),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            plot_bgcolor=BG,
            paper_bgcolor=BG,
            font=dict(color='#111111'),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111111', size=12)
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=450,  # Reduced height since title is now outside
            margin=dict(t=20, b=100, l=60, r=40),  # Reduced top margin
            showlegend=True
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sammenligning-analyse")
        
        # Key metrics
        total_nrk = comp_df["NRK"].sum()
        total_tv2 = comp_df["TV 2"].sum()
        avg_diff = comp_df["Abs_Difference"].mean()
        max_diff_party = comp_df.loc[comp_df["Abs_Difference"].idxmax()]
        
        st.metric("NRK Total", f"{total_nrk:.0f}", delta=f"{total_nrk-total_tv2:+.0f} vs TV2")
        st.metric("TV2 Total", f"{total_tv2:.0f}")
        st.metric("Gj.snitt forskjell", f"{avg_diff:.1f} poeng")
        
        # Biggest difference
        st.markdown("---")
        st.subheader("🎯 Største forskjell")
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {COL_NEU}22, {COL_NEU}11);
            border-left: 4px solid {COL_NEU};
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
        ">
            <div style="font-weight: bold; color: #111111;">
                {max_diff_party.name}
            </div>
            <div style="color: #666; font-size: 12px;">
                NRK: {max_diff_party['NRK']:.1f} | TV2: {max_diff_party['TV 2']:.1f}
            </div>
            <div style="color: #666; font-size: 12px;">
                Forskjell: {max_diff_party['Difference']:+.1f} poeng
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Source bias analysis
        st.markdown("---")
        st.subheader("📈 Kilde-bias")
        
        nrk_favored = (comp_df["Difference"] < -1).sum()
        tv2_favored = (comp_df["Difference"] > 1).sum()
        neutral = len(comp_df) - nrk_favored - tv2_favored
        
        bias_data = ["NRK-favorisert", "Nøytral", "TV2-favorisert"]
        bias_values = [nrk_favored, neutral, tv2_favored]
        bias_colors = [COL_NEG, COL_NEU, COL_POS]
        
        # Mini pie chart for bias
        fig_bias = go.Figure(data=[go.Pie(
            labels=bias_data,
            values=bias_values,
            hole=0.3,
            marker=dict(colors=bias_colors, line=dict(color='white', width=2)),
            textinfo='value',
            textposition='inside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{label}</b><br>Partier: %{value}<br>Prosent: %{percent}<extra></extra>'
        )])
        
        fig_bias.update_layout(
            plot_bgcolor=BG,
            paper_bgcolor=BG,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=10, color='#111111')
            ),
            margin=dict(t=20, b=20, l=20, r=80),
            height=200,
            font=dict(color='#111111')
        )
        
        st.plotly_chart(fig_bias, use_container_width=True)
        
        # Correlation analysis
        correlation = comp_df["NRK"].corr(comp_df["TV 2"])
        st.metric("Korrelasjonskoeffisient", f"{correlation:.3f}")
        
        if correlation > 0.8:
            st.success("🟢 Høy enighet mellom kildene")
        elif correlation > 0.6:
            st.warning("🟡 Moderat enighet")
        else:
            st.error("🔴 Lav enighet mellom kildene")
    
    # Detailed comparison table
    st.subheader("📋 Detaljert sammenligning")
    
    # Create a more visual table
    display_df = comp_df[["NRK", "TV 2", "Difference"]].copy()
    display_df["Difference"] = display_df["Difference"].round(1)
    display_df["NRK"] = display_df["NRK"].round(1)
    display_df["TV 2"] = display_df["TV 2"].round(1)
    
    # Add visual indicators
    display_df["Trend"] = display_df["Difference"].apply(
        lambda x: "📈 TV2 høyere" if x > 1 else 
                  "📉 NRK høyere" if x < -1 else 
                  "➡️ Omtrent likt"
    )
    
    display_df["Abs_Diff"] = abs(display_df["Difference"])
    display_df = display_df.sort_values("Abs_Diff", ascending=False)
    
    # Style the dataframe
    st.dataframe(
        display_df[["NRK", "TV 2", "Difference", "Trend"]].rename(columns={
            "Difference": "Forskjell",
            "Trend": "Retning"
        }),
        use_container_width=True
    )
    
    # Agreement analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_agreement = (comp_df["Abs_Difference"] < 1).sum()
        st.metric("Høy enighet", f"{high_agreement}/{len(comp_df)} partier")
    
    with col2:
        medium_agreement = ((comp_df["Abs_Difference"] >= 1) & (comp_df["Abs_Difference"] < 3)).sum()
        st.metric("Moderat enighet", f"{medium_agreement}/{len(comp_df)} partier")
    
    with col3:
        low_agreement = (comp_df["Abs_Difference"] >= 3).sum()
        st.metric("Lav enighet", f"{low_agreement}/{len(comp_df)} partier")

with tab4:
    st.header("Vektingseffekt")
    st.write("Sammenligner partisummer før og etter vekting av en spesifikk kategori.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        NQ_weight = st.slider("Antall spørsmål", 10, 50, 30, 5, key="nq_weight")
    with col2:
        categories = tv2["Kategori"].dropna().unique()
        boost_category = st.selectbox("Kategori å vekte", categories, 
                                     index=list(categories).index("Barn og familie") if "Barn og familie" in categories else 0)
    with col3:
        weight = st.slider("Vektfaktor", 1.0, 5.0, 2.0, 0.5)
    
    sel = tv2.head(NQ_weight).copy()
    base = sel[parties].sum()
    weights = sel["Kategori"].apply(lambda k: weight if k == boost_category else 1.0)
    weighted = (sel[parties].multiply(weights.values, axis=0)).sum()
    
    weight_df = pd.DataFrame({"Før": base, "Etter": weighted})
    weight_df = weight_df.sort_values("Før", ascending=False)
    
    # Calculate percentage changes for visual emphasis
    weight_df["Endring"] = weight_df["Etter"] - weight_df["Før"]
    weight_df["Endring (%)"] = (weight_df["Endring"] / weight_df["Før"] * 100).round(1)
    
    col_left, col_right = st.columns([2.5, 1.5])
    
    with col_left:
        # Create modern grouped bar chart with Plotly
        fig_weight = go.Figure()
        
        # Add "Før" bars
        fig_weight.add_trace(go.Bar(
            name='Før vekting',
            x=weight_df.index,
            y=weight_df["Før"],
            marker=dict(
                color=COL_NEG,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.0f}" for val in weight_df["Før"]],
            textposition='auto',
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{x}</b><br>Før: %{y:.1f}<extra></extra>'
        ))
        
        # Add "Etter" bars
        fig_weight.add_trace(go.Bar(
            name='Etter vekting',
            x=weight_df.index,
            y=weight_df["Etter"],
            marker=dict(
                color=COL_POS,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.0f}" for val in weight_df["Etter"]],
            textposition='auto',
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{x}</b><br>Etter: %{y:.1f}<extra></extra>'
        ))
        
        # Add change indicators (subtle arrows/markers)
        for i, (party, row) in enumerate(weight_df.iterrows()):
            if abs(row["Endring"]) > 1:  # Only show arrows for significant changes
                arrow_color = COL_POS if row["Endring"] > 0 else COL_NEG
                fig_weight.add_annotation(
                    x=party,
                    y=max(row["Før"], row["Etter"]) + 5,
                    text="↗" if row["Endring"] > 0 else "↘",
                    font=dict(size=16, color=arrow_color),
                    showarrow=False
                )
        
        fig_weight.update_layout(
            title=dict(
                text=f"<b>Vektingseffekt: «{boost_category}» vektes {weight}×</b>",
                font=dict(size=16, color='#111111'),
                x=0.2
            ),
            xaxis=dict(
                title="Partier",
                tickangle=45,
                tickfont=dict(color='#111111'),
                title_font=dict(color='#111111')
            ),
            yaxis=dict(
                title="Sum score",
                tickfont=dict(color='#111111'),
                title_font=dict(color='#111111'),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            plot_bgcolor=BG,
            paper_bgcolor=BG,
            font=dict(color='#111111'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111111', size=12)
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=450,
            margin=dict(t=80, b=100, l=60, r=40)
        )
        
        st.plotly_chart(fig_weight, use_container_width=True)
    
    with col_right:
        st.subheader("📊 Vektingsanalye")
        
        # Key metrics
        max_increase = weight_df["Endring"].max()
        max_decrease = weight_df["Endring"].min()
        avg_change = weight_df["Endring"].mean()
        
        # Party with biggest increase/decrease
        biggest_winner = weight_df.loc[weight_df["Endring"].idxmax()]
        biggest_loser = weight_df.loc[weight_df["Endring"].idxmin()] if weight_df["Endring"].min() < 0 else None
        
        st.metric("Største økning", f"+{max_increase:.1f} poeng", 
                 delta=f"{biggest_winner.name}")
        
        if biggest_loser is not None:
            st.metric("Største nedgang", f"{max_decrease:.1f} poeng", 
                     delta=f"{biggest_loser.name}")
        
        st.metric("Gj.snitt endring", f"{avg_change:+.1f} poeng")
        
        # Impact assessment
        st.markdown("---")
        st.subheader("🎯 Påvirkning")
        
        # Count parties with significant changes
        significant_changes = (abs(weight_df["Endring"]) > 2).sum()
        moderate_changes = ((abs(weight_df["Endring"]) >= 1) & (abs(weight_df["Endring"]) <= 2)).sum()
        
        st.metric("Betydelig påvirket", f"{significant_changes} partier")
        st.metric("Moderat påvirket", f"{moderate_changes} partier")
        
        # Visual impact indicator
        if max_increase > 10:
            st.success(f"🟢 Høy påvirkning fra {weight}× vekting")
        elif max_increase > 5:
            st.warning(f"🟡 Moderat påvirkning fra {weight}× vekting")
        else:
            st.info(f"🔵 Lav påvirkning fra {weight}× vekting")
        
        # Category weight info
        st.markdown("---")
        st.subheader("⚖️ Vektinfo")
        
        # Count questions in selected category
        category_questions = sel[sel["Kategori"] == boost_category].shape[0]
        total_questions = sel.shape[0]
        category_pct = (category_questions / total_questions) * 100
        
        st.metric("Spørsmål i kategori", f"{category_questions}/{total_questions}")
        st.metric("Kategori-andel", f"{category_pct:.1f}%")
        st.metric("Effektiv vekt", f"{category_pct * weight:.1f}%")
    
    # Detailed changes table
    with st.expander("📋 Se detaljerte endringer"):
        display_change_df = weight_df[["Før", "Etter", "Endring", "Endring (%)"]].copy()
        display_change_df["Trend"] = display_change_df["Endring"].apply(
            lambda x: "📈 Økning" if x > 0 else 
                      "📉 Nedgang" if x < 0 else 
                      "➡️ Uendret"
        )
        
        # Style the dataframe
        st.dataframe(
            display_change_df.rename(columns={
                "Før": "Før vekting",
                "Etter": "Etter vekting",
                "Endring": "Endring (poeng)",
                "Endring (%)": "Endring (%)",
                "Trend": "Retning"
            }),
            use_container_width=True
        )

with tab5:
    st.header("Scenario-analyse")
    st.write("Sammenligner partisummer med og uten en spesifikk kategori.")
    
    col1, col2 = st.columns(2)
    with col1:
        NQ_scenario = st.slider("Antall spørsmål", 5, 30, 10, 1, key="nq_scenario")
    with col2:
        categories_scenario = tv2["Kategori"].dropna().unique()
        category_to_drop = st.selectbox("Kategori å fjerne", categories_scenario,
                                       index=list(categories_scenario).index("Barn og familie") if "Barn og familie" in categories_scenario else 0)
    
    base_scenario = tv2.head(NQ_scenario)[parties].sum()
    after_scenario = tv2[tv2["Kategori"] != category_to_drop].head(NQ_scenario)[parties].sum()
    
    scenario_df = pd.DataFrame({"Med alle kategorier": base_scenario, f"Uten {category_to_drop}": after_scenario})
    scenario_df = scenario_df.sort_values("Med alle kategorier", ascending=False)
    
    # Calculate impact metrics
    scenario_df["Endring"] = scenario_df[f"Uten {category_to_drop}"] - scenario_df["Med alle kategorier"]
    scenario_df["Endring (%)"] = (scenario_df["Endring"] / scenario_df["Med alle kategorier"] * 100).round(1)
    
    # Count questions in the category to be removed
    category_questions = tv2.head(NQ_scenario)[tv2["Kategori"] == category_to_drop].shape[0]
    total_questions = tv2.head(NQ_scenario).shape[0]
    
    col_left, col_right = st.columns([2.5, 1.5])
    
    with col_left:
        # Create modern grouped bar chart with Plotly
        fig_scenario = go.Figure()
        
        # Add "Med alle" bars
        fig_scenario.add_trace(go.Bar(
            name='Med alle kategorier',
            x=scenario_df.index,
            y=scenario_df["Med alle kategorier"],
            marker=dict(
                color=COL_NEU,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.0f}" for val in scenario_df["Med alle kategorier"]],
            textposition='auto',
            textfont=dict(color='#111111', size=10),
            hovertemplate='<b>%{x}</b><br>Med alle: %{y:.1f}<extra></extra>'
        ))
        
        # Add "Uten kategori" bars
        fig_scenario.add_trace(go.Bar(
            name=f'Uten «{category_to_drop}»',
            x=scenario_df.index,
            y=scenario_df[f"Uten {category_to_drop}"],
            marker=dict(
                color=COL_NEG,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.0f}" for val in scenario_df[f"Uten {category_to_drop}"]],
            textposition='auto',
            textfont=dict(color='white', size=10),
            hovertemplate=f'<b>%{{x}}</b><br>Uten {category_to_drop}: %{{y:.1f}}<extra></extra>'
        ))
        
        # Add impact indicators (arrows for significant changes)
        for i, (party, row) in enumerate(scenario_df.iterrows()):
            if abs(row["Endring"]) > 1:  # Only show arrows for significant changes
                arrow_color = COL_POS if row["Endring"] > 0 else COL_NEG
                fig_scenario.add_annotation(
                    x=party,
                    y=max(row["Med alle kategorier"], row[f"Uten {category_to_drop}"]) + 3,
                    text="↗" if row["Endring"] > 0 else "↘",
                    font=dict(size=16, color=arrow_color),
                    showarrow=False
                )
        
        fig_scenario.update_layout(
            title=dict(
                text=f"<b>Scenario: Fjerner kategori «{category_to_drop}»</b>",
                font=dict(size=16, color='#111111'),
                x=0.2
                
            ),
            xaxis=dict(
                title="Partier",
                tickangle=45,
                tickfont=dict(color='#111111'),
                title_font=dict(color='#111111')
            ),
            yaxis=dict(
                title="Sum score",
                tickfont=dict(color='#111111'),
                title_font=dict(color='#111111'),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            plot_bgcolor=BG,
            paper_bgcolor=BG,
            font=dict(color='#111111'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111111', size=12)
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=450,
            margin=dict(t=80, b=100, l=60, r=40)
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
    
    with col_right:
        st.subheader("📊 Scenario-påvirkning")
        
        # Key impact metrics
        max_impact = abs(scenario_df["Endring"]).max()
        avg_impact = scenario_df["Endring"].mean()
        
        # Party most affected
        most_affected = scenario_df.loc[scenario_df["Endring"].abs().idxmax()]
        
        st.metric("Største endring", f"{most_affected['Endring']:+.1f} poeng", 
                 delta=f"{most_affected.name}")
        st.metric("Gj.snitt påvirkning", f"{avg_impact:+.1f} poeng")
        
        # Category removal info
        st.markdown("---")
        st.subheader("🗑️ Fjernet kategori")
        
        st.metric("Spørsmål fjernet", f"{category_questions}/{total_questions}")
        
        if total_questions > 0:
            removal_pct = (category_questions / total_questions) * 100
            st.metric("Andel fjernet", f"{removal_pct:.1f}%")
        
        # Impact assessment
        st.markdown("---")
        st.subheader("🎯 Konsekvenser")
        
        # Count parties with different impact levels
        high_impact = (abs(scenario_df["Endring"]) > 3).sum()
        medium_impact = ((abs(scenario_df["Endring"]) >= 1) & (abs(scenario_df["Endring"]) <= 3)).sum()
        low_impact = (abs(scenario_df["Endring"]) < 1).sum()
        
        st.metric("Høy påvirkning", f"{high_impact} partier")
        st.metric("Moderat påvirkning", f"{medium_impact} partier") 
        st.metric("Lav påvirkning", f"{low_impact} partier")
        
        # Overall scenario assessment
        if max_impact > 5:
            st.error(f"🔴 Kritisk: Fjerning av «{category_to_drop}» har stor påvirkning")
        elif max_impact > 2:
            st.warning(f"🟡 Advarsel: Moderat påvirkning fra fjerning")
        else:
            st.success(f"🟢 Minimal påvirkning fra fjerning")
        
        # Scenario recommendation
        st.markdown("---")
        st.subheader("💡 Anbefaling")
        
        if category_questions == 0:
            st.info("ℹ️ Ingen spørsmål i denne kategorien")
        elif max_impact < 1:
            st.success("✅ Trygt å fjerne - minimal påvirkning")
        elif max_impact < 3:
            st.warning("⚠️ Vurder nøye - moderat påvirkning")
        else:
            st.error("🚨 Ikke anbefalt - høy påvirkning")
    
    # Detailed impact table
    with st.expander("📋 Se detaljert påvirkning av fjerning"):
        display_impact_df = scenario_df[["Med alle kategorier", f"Uten {category_to_drop}", "Endring", "Endring (%)"]].copy()
        display_impact_df["Påvirkning"] = display_impact_df["Endring"].apply(
            lambda x: "🔴 Høy nedgang" if x < -3 else 
                      "🟡 Moderat nedgang" if x < -1 else
                      "🟢 Minimal endring" if abs(x) < 1 else
                      "🟡 Moderat økning" if x < 3 else
                      "🔴 Høy økning"
        )
        
        # Style the dataframe
        st.dataframe(
            display_impact_df.rename(columns={
                "Med alle kategorier": "Med alle kategorier",
                f"Uten {category_to_drop}": f"Uten «{category_to_drop}»",
                "Endring": "Endring (poeng)",
                "Endring (%)": "Endring (%)",
                "Påvirkning": "Påvirkningsnivå"
            }),
            use_container_width=True
        )

with tab6:
    st.header("🤖 AI & Datakvalitet")
    st.write("Utforsk hvordan datakvalitet påvirker AI-resultater gjennom valgomatdata.")
    
    # Data quality dimensions selector
    quality_dim = st.selectbox(
        "Velg datakvalitetsdimensjon:",
        ["📊 Oversikt", "🎯 Accuracy", "📋 Completeness", "🔄 Consistency", 
         "⏰ Timeliness", "✅ Validity", "🎭 Uniqueness", "🔬 Advanced Analysis"]
    )
    
    if quality_dim == "📊 Oversikt":
        st.subheader("Datakvalitetens 6 dimensjoner + AI Impact")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create overview visualization
            dimensions = ["Accuracy", "Completeness", "Consistency", "Timeliness", "Validity", "Uniqueness"]
            scores = [85, 92, 78, 88, 95, 90]  # Example scores
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name='Datakvalitet Score',
                line_color=COLORS[0],
                fillcolor=f'rgba(255, 140, 69, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Datakvalitetsprofil for Valgomatdata",
                plot_bgcolor=BG,
                paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("AI Impact Score")
            overall_score = np.mean(scores)
            ai_confidence = simulate_ai_confidence_degradation(overall_score)
            
            st.metric("Overall Data Quality", f"{overall_score:.0f}%", 
                     delta=f"{overall_score-75:.0f}% vs baseline")
            st.metric("AI Confidence", f"{ai_confidence:.0f}%",
                     delta=f"{ai_confidence-95:.0f}% vs optimal")
            
            # Color-coded recommendations
            if overall_score >= 90:
                st.success("🟢 Excellent data quality - AI ready!")
            elif overall_score >= 75:
                st.warning("🟡 Good quality - Minor improvements needed")
            else:
                st.error("🔴 Poor quality - Major issues detected")
        
        st.info("💡 **Workshop-oppgave**: Velg en dimensjon ovenfor for å se hvordan den påvirker AI-resultater!")
    
    elif quality_dim == "🎯 Accuracy":
        st.subheader("Accuracy: Hvordan spørsmålsformulering påvirker AI-anbefalinger")
        
        # Interactive bias demonstration
        bias_level = st.slider("Grad av skjevhet i spørsmål:", 0.0, 1.0, 0.3, 0.1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Nøytral formulering:**")
            st.write("'Bør Norge øke bistanden til utviklingsland?'")
            
            # Generate unbiased data
            unbiased_data, _ = generate_biased_vs_unbiased_data(parties[:6], 0)
            neutral_scores = 3 + unbiased_data * 0.5  # Scale to reasonable range
            
            fig_acc1 = go.Figure(data=[
                go.Bar(x=parties[:6], y=neutral_scores, 
                      marker_color=COLORS[1],
                      text=[f"{s:.1f}" for s in neutral_scores],
                      textposition='auto')
            ])
            fig_acc1.update_layout(
                title="Nøytral formulering - AI Anbefaling",
                yaxis_title="Anbefalt styrke (1-5)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111"),
                yaxis=dict(range=[0, 5])
            )
            st.plotly_chart(fig_acc1, use_container_width=True)
            
            ai_conf_neutral = simulate_ai_confidence_degradation(95)
            st.metric("AI Confidence", f"{ai_conf_neutral:.0f}%")
        
        with col2:
            st.markdown("**Skjeve formuleringer:**")
            if bias_level < 0.3:
                st.write("'Bør Norge investere smartere i bistand til utviklingsland?'")
            elif bias_level < 0.7:
                st.write("'Bør Norge kaste bort penger på bistand til korrupte land?'")
            else:
                st.write("'Bør Norge stoppe all meningsløs bistand til håpløse land?'")
            
            # Generate biased data
            _, biased_data = generate_biased_vs_unbiased_data(parties[:6], bias_level * 2)
            biased_scores = np.maximum(0.5, 3 + biased_data * 0.5 - bias_level * 2)
            
            fig_acc2 = go.Figure(data=[
                go.Bar(x=parties[:6], y=biased_scores, 
                      marker_color=COLORS[0],
                      text=[f"{s:.1f}" for s in biased_scores],
                      textposition='auto')
            ])
            fig_acc2.update_layout(
                title="Skjev formulering - AI Anbefaling",
                yaxis_title="Anbefalt styrke (1-5)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111"),
                yaxis=dict(range=[0, 5])
            )
            st.plotly_chart(fig_acc2, use_container_width=True)
            
            ai_conf_biased = simulate_ai_confidence_degradation(95 - bias_level * 40)
            st.metric("AI Confidence", f"{ai_conf_biased:.0f}%", 
                     delta=f"{ai_conf_biased - ai_conf_neutral:.0f}%")
        
        # Show impact on recommendations
        st.subheader("📊 Konsekvenser for AI-anbefalinger")
        diff_scores = np.abs(neutral_scores - biased_scores)
        avg_difference = np.mean(diff_scores)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Gjennomsnittlig avvik", f"{avg_difference:.2f} poeng")
        col2.metric("Maks avvik", f"{np.max(diff_scores):.2f} poeng")
        col3.metric("Partier med >1 poengs endring", f"{np.sum(diff_scores > 1)}")
        
        if avg_difference > 0.5:
            st.error("🚨 **Kritisk**: Skjeve spørsmål endrer AI-anbefalinger drastisk!")
        elif avg_difference > 0.2:
            st.warning("⚠️ **Advarsel**: Moderate endringer i AI-anbefalinger")
        else:
            st.success("✅ **Bra**: Minimal påvirkning på AI-anbefalinger")
    
    elif quality_dim == "📋 Completeness":
        st.subheader("Completeness: AI-ytelse vs. manglende data")
        
        # Interactive missing data simulation
        missing_pct = st.slider("Prosent manglende data:", 0, 50, 20, 5)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show impact of missing categories on party rankings
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
                horizontal_spacing=0.1
            )
            
            fig_comp.add_trace(
                go.Bar(x=complete_data.index, y=complete_data.values, 
                      marker_color=COLORS[1], name="Komplett"),
                row=1, col=1
            )
            
            fig_comp.add_trace(
                go.Bar(x=incomplete_data.index, y=incomplete_data.values, 
                      marker_color=COLORS[0], name="Ufullstendig"),
                row=1, col=2
            )
            
            fig_comp.update_layout(
                title=f"Hvordan {missing_pct}% manglende data påvirker partirangeringer",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111"),
                showlegend=False
            )
            
            fig_comp.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            # Calculate AI impact metrics
            ai_accuracy = simulate_recommendation_accuracy(missing_pct)
            ai_confidence = simulate_ai_confidence_degradation(100 - missing_pct)
            
            st.subheader("AI Impact Metrics")
            st.metric("Recommendation Accuracy", f"{ai_accuracy:.0f}%", 
                     delta=f"{ai_accuracy-92:.0f}% vs complete data")
            st.metric("AI Confidence", f"{ai_confidence:.0f}%",
                     delta=f"{ai_confidence-95:.0f}% vs optimal")
            
            # Calculate ranking changes
            complete_ranking = complete_data.rank(ascending=False, method='min')
            incomplete_ranking = incomplete_data.rank(ascending=False, method='min')
            
            # Align rankings for comparison
            common_parties = set(complete_ranking.index) & set(incomplete_ranking.index)
            if common_parties:
                ranking_changes = []
                for party in common_parties:
                    change = abs(complete_ranking[party] - incomplete_ranking[party])
                    ranking_changes.append(change)
                
                avg_rank_change = np.mean(ranking_changes)
                st.metric("Avg. Rank Change", f"{avg_rank_change:.1f} positions")
            
            # Completeness status
            if missing_pct < 5:
                st.success("🟢 Excellent completeness")
            elif missing_pct < 15:
                st.warning("🟡 Acceptable completeness")
            else:
                st.error("🔴 Poor completeness - AI unreliable")
        
        # Missing data patterns
        st.subheader("📊 Missing Data Patterns")
        
        # Simulate different missing data patterns
        patterns = ["Random", "Systematic (by category)", "Biased (against certain parties)"]
        selected_pattern = st.selectbox("Type of missing data:", patterns)
        
        if selected_pattern == "Random":
            st.info("💡 **Random missing**: Least harmful - AI can often compensate")
        elif selected_pattern == "Systematic (by category)":
            st.warning("⚠️ **Systematic missing**: More harmful - creates knowledge gaps")
        else:
            st.error("🚨 **Biased missing**: Most harmful - creates unfair AI recommendations")
    
    elif quality_dim == "🔄 Consistency":
        st.subheader("Consistency: Når AI får motstridende signaler")
        
        # Show inconsistency between data sources
        st.write("Sammenlign hvordan inkonsistente data påvirker AI-beslutninger:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # NRK vs TV2 consistency analysis
            nrk_sample = nrk.head(15)[parties].mean()
            tv2_sample = tv2.head(15)[parties].mean()
            
            consistency_df = pd.DataFrame({
                "NRK": nrk_sample,
                "TV2": tv2_sample
            })
            consistency_df["Difference"] = abs(consistency_df["NRK"] - consistency_df["TV2"])
            
            fig_cons = go.Figure()
            
            # Scatter plot with party labels
            fig_cons.add_trace(go.Scatter(
                x=consistency_df["NRK"], 
                y=consistency_df["TV2"],
                mode='markers+text',
                text=consistency_df.index,
                textposition="top center",
                marker=dict(
                    size=consistency_df["Difference"] * 20 + 8,  # Size based on difference
                    color=consistency_df["Difference"],
                    colorscale=[[0, COLORS[1]], [0.5, COLORS[0]], [1, "#ff4444"]],
                    colorbar=dict(title="Avvik mellom kilder")
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
                name="Perfekt konsistens"
            ))
            
            fig_cons.update_layout(
                title="Konsistens mellom NRK og TV2 (større punkt = mer inkonsistent)",
                xaxis_title="NRK gjennomsnitt",
                yaxis_title="TV2 gjennomsnitt",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_cons, use_container_width=True)
        
        with col2:
            # Consistency metrics and AI impact
            avg_diff = consistency_df["Difference"].mean()
            max_diff = consistency_df["Difference"].max()
            
            st.subheader("Consistency Metrics")
            st.metric("Avg. Source Disagreement", f"{avg_diff:.2f} points")
            st.metric("Max Source Disagreement", f"{max_diff:.2f} points")
            
            # AI impact from inconsistency
            consistency_score = max(0, 100 - avg_diff * 30)
            ai_conf_consistency = simulate_ai_confidence_degradation(consistency_score)
            
            st.metric("Consistency Score", f"{consistency_score:.0f}%")
            st.metric("AI Confidence Impact", f"{ai_conf_consistency:.0f}%")
            
            # Recommendations based on consistency
            if avg_diff < 0.3:
                st.success("🟢 High consistency - AI can trust both sources")
            elif avg_diff < 0.7:
                st.warning("🟡 Moderate inconsistency - AI needs validation")
            else:
                st.error("🔴 High inconsistency - AI results unreliable")
        
        # Inconsistency resolution strategies
        st.subheader("🔧 AI Strategies for Handling Inconsistency")
        strategy = st.selectbox("AI resolution strategy:", 
                               ["Source weighting", "Confidence intervals", "Ensemble methods", "Manual review"])
        
        if strategy == "Source weighting":
            st.info("💡 Weight sources by reliability - more trustworthy sources get higher weight")
        elif strategy == "Confidence intervals":
            st.info("💡 Provide uncertainty ranges instead of point estimates")
        elif strategy == "Ensemble methods":
            st.info("💡 Use multiple AI models and combine their predictions")
        else:
            st.info("💡 Flag inconsistent cases for human expert review")
    
    elif quality_dim == "⏰ Timeliness":
        st.subheader("Timeliness: AI-prestasjon med utdaterte data")
        
        # Simulate data aging effect
        months_old = st.slider("Alder på data (måneder):", 0, 24, 6, 1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show relevance decay over time
            months = list(range(0, 25, 3))
            fresh_relevance = [100 - (m * 2) for m in months]  # Gradual decay
            stale_relevance = [100 - (m * 5) for m in months]  # Faster decay for sensitive topics
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=months, y=fresh_relevance,
                mode='lines+markers',
                name='Stable topics (økonomi)',
                line=dict(color=COLORS[2], width=3),
                marker=dict(size=8)
            ))
            fig_time.add_trace(go.Scatter(
                x=months, y=stale_relevance,
                mode='lines+markers',
                name='Dynamic topics (teknologi)',
                line=dict(color=COLORS[0], width=3),
                marker=dict(size=8)
            ))
            
            # Add vertical line for current data age
            fig_time.add_vline(x=months_old, line_dash="dash", line_color="red", 
                              annotation_text=f"Din data: {months_old} mnd")
            
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
            stable_relevance = max(0, 100 - months_old * 2)
            dynamic_relevance = max(0, 100 - months_old * 5)
            avg_relevance = (stable_relevance + dynamic_relevance) / 2
            
            st.subheader("Timeliness Impact")
            st.metric("Stable Topics Relevance", f"{stable_relevance:.0f}%")
            st.metric("Dynamic Topics Relevance", f"{dynamic_relevance:.0f}%")
            st.metric("Overall Data Relevance", f"{avg_relevance:.0f}%")
            
            ai_performance = simulate_ai_confidence_degradation(avg_relevance)
            st.metric("AI Performance", f"{ai_performance:.0f}%", 
                     delta=f"{ai_performance-95:.0f}% vs fresh data")
            
            # Timeliness recommendations
            if months_old < 3:
                st.success("🟢 Fresh data - AI performs optimally")
            elif months_old < 12:
                st.warning("🟡 Aging data - Consider updates for dynamic topics")
            else:
                st.error("🔴 Stale data - AI recommendations may be outdated")
        
        # Data refresh strategy
        st.subheader("📅 Data Refresh Strategy")
        topic_types = ["Political opinions", "Economic preferences", "Social values", "Technology adoption"]
        refresh_intervals = [1, 6, 12, 3]  # months
        
        refresh_df = pd.DataFrame({
            "Topic": topic_types,
            "Recommended Refresh (months)": refresh_intervals,
            "Current Age (months)": [months_old] * 4,
            "Status": ["🔴 Overdue" if months_old > interval else "🟢 Current" 
                      for interval in refresh_intervals]
        })
        
        st.dataframe(refresh_df, use_container_width=True)
    
    elif quality_dim == "✅ Validity":
        st.subheader("Validity: AI robusthet mot ugyldige data")
        
        # Simulate different types of validity issues
        validity_issue = st.selectbox("Type validitetsproblem:", 
                                    ["Format errors", "Out-of-range values", "Logical inconsistencies", "Encoding issues"])
        
        error_rate = st.slider("Feilrate (%):", 0, 25, 5, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Problem: {validity_issue}**")
            
            if validity_issue == "Format errors":
                st.code("""
Valid: [-2, -1, 0, 1, 2]
Invalid: ["strongly disagree", "", "N/A", 999]
                """)
                
            elif validity_issue == "Out-of-range values":
                st.code("""
Valid range: -2 to +2
Invalid values: [-5, 7, 15, -10]
                """)
                
            elif validity_issue == "Logical inconsistencies":
                st.code("""
Q1: "Increase taxes" -> +2 (strongly agree)
Q2: "Lower taxes" -> +2 (strongly agree)
[Logical contradiction!]
                """)
                
            else:  # Encoding issues
                st.code("""
Expected: UTF-8 text
Actual: "SÃ¸ppel data med feil encoding"
Should be: "Søppel data med feil encoding"
                """)
            
            # Show data distribution with errors
            valid_data = np.random.choice([-2, -1, 0, 1, 2], size=100-error_rate)
            if validity_issue == "Out-of-range values":
                invalid_data = np.random.choice([5, 7, -5, 10], size=error_rate)
            else:
                invalid_data = np.full(error_rate, np.nan)
            
            all_data = np.concatenate([valid_data, invalid_data]) if error_rate > 0 else valid_data
            
            fig_val = go.Figure()
            
            # Valid data
            valid_counts = np.bincount(valid_data + 2, minlength=5)
            fig_val.add_trace(go.Bar(
                x=[-2, -1, 0, 1, 2], y=valid_counts,
                name="Valid data", marker_color=COLORS[1]
            ))
            
            # Invalid data
            if error_rate > 0 and validity_issue == "Out-of-range values":
                invalid_counts = np.bincount(invalid_data)
                invalid_values = np.arange(len(invalid_counts))
                fig_val.add_trace(go.Bar(
                    x=invalid_values, y=invalid_counts,
                    name="Invalid data", marker_color=COLORS[0]
                ))
            
            fig_val.update_layout(
                title=f"Data distribution med {error_rate}% feil",
                xaxis_title="Values", yaxis_title="Count",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_val, use_container_width=True)
        
        with col2:
            # AI impact of validity issues
            validity_score = max(0, 100 - error_rate * 4)
            ai_robustness = simulate_ai_confidence_degradation(validity_score)
            
            st.subheader("Validity Impact")
            st.metric("Data Validity Score", f"{validity_score:.0f}%")
            st.metric("AI Robustness", f"{ai_robustness:.0f}%", 
                     delta=f"{ai_robustness-95:.0f}% vs clean data")
            
            # Error handling strategies
            st.subheader("🛠️ Error Handling")
            if error_rate < 2:
                st.success("🟢 Minimal errors - AI handles gracefully")
                st.info("Strategy: Automatic outlier detection")
            elif error_rate < 10:
                st.warning("🟡 Moderate errors - Requires preprocessing")
                st.info("Strategy: Data cleaning + validation rules")
            else:
                st.error("🔴 High error rate - AI results unreliable")
                st.info("Strategy: Manual data audit required")
            
            # Show AI error recovery capability
            recovery_methods = ["Skip invalid records", "Impute missing values", 
                              "Flag for manual review", "Use ensemble methods"]
            selected_recovery = st.selectbox("AI Recovery Method:", recovery_methods)
            
            if selected_recovery == "Skip invalid records":
                recovery_effectiveness = max(60, 100 - error_rate * 2)
            elif selected_recovery == "Impute missing values":
                recovery_effectiveness = max(70, 100 - error_rate * 1.5)
            elif selected_recovery == "Flag for manual review":
                recovery_effectiveness = max(80, 100 - error_rate * 1)
            else:
                recovery_effectiveness = max(75, 100 - error_rate * 1.2)
            
            st.metric("Recovery Effectiveness", f"{recovery_effectiveness:.0f}%")
    
    elif quality_dim == "🎭 Uniqueness":
        st.subheader("Uniqueness: Når AI lærer feil fra duplikater")
        
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
                marker_color=COLORS[1]
            ))
            fig_uniq.add_trace(go.Bar(
                name=f'Med {duplicate_rate}% duplikater',
                x=categories,
                y=skewed_influence,
                marker_color=COLORS[0]
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
            
            st.subheader("Uniqueness Impact")
            st.metric("Data Uniqueness", f"{uniqueness_score:.0f}%")
            st.metric("AI Bias Level", f"{ai_bias:.0f}%", 
                     delta=f"+{ai_bias:.0f}% vs balanced data")
            
            # Show overrepresentation
            overrep = skewed_influence[0] / original_influence[0]
            st.metric(f"{duplicate_category} Overrepresentation", f"{overrep:.1f}x")
            
            # Uniqueness status
            if duplicate_rate < 5:
                st.success("🟢 High uniqueness - Minimal AI bias")
            elif duplicate_rate < 15:
                st.warning("🟡 Moderate duplicates - Some AI bias")
            else:
                st.error("🔴 High duplication - Significant AI bias")
        
        # Duplicate detection strategies
        st.subheader("🔍 Duplicate Detection Methods")
        
        detection_methods = {
            "Exact matching": "Find identical questions word-for-word",
            "Semantic similarity": "Find questions with similar meaning",
            "Statistical correlation": "Find questions with highly correlated responses",
            "Manual review": "Human expert identifies conceptual duplicates"
        }
        
        for method, description in detection_methods.items():
            with st.expander(f"📋 {method}"):
                st.write(description)
                
                if method == "Exact matching":
                    st.code("""
Example duplicates:
1. "Bør Norge øke skatten?"
2. "Bør Norge øke skatten?"
[Identical - Easy to detect]
                    """)
                elif method == "Semantic similarity":
                    st.code("""
Example duplicates:
1. "Bør Norge øke skatten for de rike?"
2. "Synes du Norge bør ha høyere skatt på høye inntekter?"
[Similar meaning - Harder to detect]
                    """)
                elif method == "Statistical correlation":
                    st.code("""
If responses to Q1 and Q2 correlate > 0.95:
Likely measuring the same thing
                    """)
                else:
                    st.code("""
Expert review needed for:
- Conceptual overlaps
- Different framings of same issue
- Subtle semantic differences
                    """)
    
    elif quality_dim == "🔬 Advanced Analysis":
        st.subheader("Advanced Data Quality Analysis")
        
        analysis_type = st.selectbox("Velg analyse:", 
                                   ["Multi-dimensional Quality Assessment", 
                                    "AI Performance Simulation",
                                    "Data Quality ROI Calculator",
                                    "Quality-Accuracy Trade-offs"])
        
        if analysis_type == "Multi-dimensional Quality Assessment":
            st.subheader("📊 Komplett datakvalitetsprofil")
            
            # Allow user to adjust all dimensions
            st.write("Juster hver datakvalitetsdimensjon:")
            
            col1, col2 = st.columns(2)
            with col1:
                accuracy = st.slider("Accuracy", 0, 100, 85, 5)
                completeness = st.slider("Completeness", 0, 100, 92, 5)
                consistency = st.slider("Consistency", 0, 100, 78, 5)
            
            with col2:
                timeliness = st.slider("Timeliness", 0, 100, 88, 5)
                validity = st.slider("Validity", 0, 100, 95, 5)
                uniqueness = st.slider("Uniqueness", 0, 100, 90, 5)
            
            # Calculate overall scores
            dimensions = ["Accuracy", "Completeness", "Consistency", "Timeliness", "Validity", "Uniqueness"]
            scores = [accuracy, completeness, consistency, timeliness, validity, uniqueness]
            
            # Create comprehensive radar chart
            fig_multi = go.Figure()
            
            fig_multi.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name='Current Quality',
                line_color=COLORS[0],
                fillcolor=f'rgba(255, 140, 69, 0.3)'
            ))
            
            # Add benchmark line
            benchmark_scores = [90] * 6  # Target quality level
            fig_multi.add_trace(go.Scatterpolar(
                r=benchmark_scores,
                theta=dimensions,
                fill='toself',
                name='Target Quality',
                line_color='green',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            
            fig_multi.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Multi-dimensional Data Quality Assessment",
                plot_bgcolor=BG,
                paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # Overall assessment
            overall_score = np.mean(scores)
            ai_readiness = simulate_ai_confidence_degradation(overall_score)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Quality Score", f"{overall_score:.0f}%")
            col2.metric("AI Readiness", f"{ai_readiness:.0f}%")
            
            # Identify weakest dimensions
            weakest_dim = dimensions[np.argmin(scores)]
            col3.metric("Weakest Dimension", weakest_dim, delta=f"{min(scores):.0f}%")
            
            # Recommendations
            st.subheader("🎯 Improvement Recommendations")
            for i, (dim, score) in enumerate(zip(dimensions, scores)):
                if score < 85:
                    if dim == "Accuracy":
                        st.warning(f"⚠️ **{dim}**: Review question formulations for bias")
                    elif dim == "Completeness":
                        st.warning(f"⚠️ **{dim}**: Fill data gaps, especially in key categories")
                    elif dim == "Consistency":
                        st.warning(f"⚠️ **{dim}**: Reconcile differences between data sources")
                    elif dim == "Timeliness":
                        st.warning(f"⚠️ **{dim}**: Update outdated questions and responses")
                    elif dim == "Validity":
                        st.warning(f"⚠️ **{dim}**: Implement data validation rules")
                    else:  # Uniqueness
                        st.warning(f"⚠️ **{dim}**: Remove or merge duplicate questions")
        
        elif analysis_type == "AI Performance Simulation":
            st.subheader("🎮 AI Performance Under Different Quality Scenarios")
            
            # Preset quality scenarios
            scenarios = {
                "Perfect Data": [100, 100, 100, 100, 100, 100],
                "Good Data": [90, 95, 85, 90, 95, 90],
                "Average Data": [75, 80, 70, 75, 85, 80],
                "Poor Data": [60, 65, 55, 60, 70, 65],
                "Critical Issues": [40, 50, 40, 45, 50, 45]
            }
            
            scenario_results = []
            for scenario_name, quality_scores in scenarios.items():
                overall_quality = np.mean(quality_scores)
                ai_confidence = simulate_ai_confidence_degradation(overall_quality)
                ai_accuracy = simulate_recommendation_accuracy(100 - overall_quality)
                
                scenario_results.append({
                    "Scenario": scenario_name,
                    "Data Quality": f"{overall_quality:.0f}%",
                    "AI Confidence": f"{ai_confidence:.0f}%",
                    "Recommendation Accuracy": f"{ai_accuracy:.0f}%",
                    "Business Impact": "🟢 Excellent" if overall_quality >= 90 else
                                    "🟡 Good" if overall_quality >= 75 else
                                    "🟠 Fair" if overall_quality >= 60 else "🔴 Poor"
                })
            
            scenario_df = pd.DataFrame(scenario_results)
            st.dataframe(scenario_df, use_container_width=True)
            
            # Visualize performance curves
            quality_range = np.arange(40, 101, 5)
            confidence_curve = [simulate_ai_confidence_degradation(q) for q in quality_range]
            accuracy_curve = [simulate_recommendation_accuracy(100-q) for q in quality_range]
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=quality_range, y=confidence_curve,
                mode='lines+markers',
                name='AI Confidence',
                line=dict(color=COLORS[2], width=3)
            ))
            fig_perf.add_trace(go.Scatter(
                x=quality_range, y=accuracy_curve,
                mode='lines+markers',
                name='Recommendation Accuracy',
                line=dict(color=COLORS[0], width=3)
            ))
            
            fig_perf.update_layout(
                title="AI Performance vs Data Quality",
                xaxis_title="Data Quality Score (%)",
                yaxis_title="AI Performance (%)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
        
        elif analysis_type == "Data Quality ROI Calculator":
            st.subheader("💰 Return on Investment for Data Quality Improvements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Situation:**")
                current_quality = st.slider("Current Data Quality", 40, 95, 70, 5, key="current_qual")
                annual_decisions = st.number_input("AI Decisions per Year", 1000, 100000, 10000, 1000)
                decision_value = st.number_input("Average Decision Value (NOK)", 100, 100000, 5000, 500)
                
                st.write("**Improvement Target:**")
                target_quality = st.slider("Target Data Quality", current_quality, 100, 90, 5, key="target_qual")
                improvement_cost = st.number_input("Improvement Cost (NOK)", 10000, 1000000, 200000, 10000)
            
            with col2:
                # Calculate ROI
                current_accuracy = simulate_recommendation_accuracy(100 - current_quality)
                target_accuracy = simulate_recommendation_accuracy(100 - target_quality)
                
                accuracy_improvement = target_accuracy - current_accuracy
                
                # Calculate financial impact
                total_annual_value = annual_decisions * decision_value
                current_correct_decisions = total_annual_value * (current_accuracy / 100)
                target_correct_decisions = total_annual_value * (target_accuracy / 100)
                
                annual_benefit = target_correct_decisions - current_correct_decisions
                roi_percentage = ((annual_benefit - improvement_cost) / improvement_cost) * 100
                payback_months = (improvement_cost / annual_benefit) * 12 if annual_benefit > 0 else float('inf')
                
                st.write("**ROI Analysis:**")
                st.metric("Accuracy Improvement", f"+{accuracy_improvement:.1f}%")
                st.metric("Annual Benefit", f"{annual_benefit:,.0f} NOK")
                st.metric("ROI", f"{roi_percentage:.0f}%")
                st.metric("Payback Period", f"{payback_months:.1f} months" if payback_months != float('inf') else "Never")
                
                # ROI recommendation
                if roi_percentage > 200:
                    st.success("🟢 Excellent ROI - Invest immediately!")
                elif roi_percentage > 50:
                    st.warning("🟡 Good ROI - Worth considering")
                elif roi_percentage > 0:
                    st.info("🔵 Positive ROI - Marginal benefit")
                else:
                    st.error("🔴 Negative ROI - Reconsider approach")
            
            # ROI breakdown chart
            quality_levels = np.arange(current_quality, 101, 5)
            roi_values = []
            
            for q in quality_levels:
                acc = simulate_recommendation_accuracy(100 - q)
                benefit = annual_decisions * decision_value * (acc / 100) - current_correct_decisions
                roi = ((benefit - improvement_cost) / improvement_cost) * 100 if improvement_cost > 0 else 0
                roi_values.append(roi)
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(
                x=quality_levels, y=roi_values,
                mode='lines+markers',
                name='ROI',
                line=dict(color=COLORS[1], width=3),
                fill='tonexty'
            ))
            
            fig_roi.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
            fig_roi.add_vline(x=target_quality, line_dash="dash", line_color=COLORS[0], 
                             annotation_text=f"Target: {target_quality}%")
            
            fig_roi.update_layout(
                title="ROI vs Data Quality Level",
                xaxis_title="Data Quality (%)",
                yaxis_title="ROI (%)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_roi, use_container_width=True)
        
        else:  # Quality-Accuracy Trade-offs
            st.subheader("⚖️ Quality vs Speed vs Cost Trade-offs")
            
            st.write("Explore how different quality approaches affect project outcomes:")
            
            approach = st.selectbox("Data Quality Approach:", 
                                  ["Quick & Dirty", "Balanced", "High Quality", "Gold Standard"])
            
            approaches = {
                "Quick & Dirty": {
                    "quality": 60, "time_weeks": 2, "cost": 50000, 
                    "description": "Minimal cleaning, basic validation"
                },
                "Balanced": {
                    "quality": 80, "time_weeks": 6, "cost": 150000,
                    "description": "Standard cleaning, automated validation"
                },
                "High Quality": {
                    "quality": 95, "time_weeks": 12, "cost": 300000,
                    "description": "Comprehensive cleaning, manual review"
                },
                "Gold Standard": {
                    "quality": 99, "time_weeks": 20, "cost": 500000,
                    "description": "Perfect data, multiple validation stages"
                }
            }
            
            selected = approaches[approach]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Quality", f"{selected['quality']}%")
                st.metric("Time to Deliver", f"{selected['time_weeks']} weeks")
                
            with col2:
                st.metric("Total Cost", f"{selected['cost']:,} NOK")
                ai_perf = simulate_ai_confidence_degradation(selected['quality'])
                st.metric("AI Performance", f"{ai_perf:.0f}%")
                
            with col3:
                # Calculate business impact
                annual_value = 10000000  # 10M NOK annual AI decisions
                quality_impact = selected['quality'] / 100
                effective_value = annual_value * quality_impact
                
                st.metric("Effective Annual Value", f"{effective_value:,.0f} NOK")
                value_per_week = effective_value / 52
                opportunity_cost = value_per_week * selected['time_weeks']
                st.metric("Opportunity Cost", f"{opportunity_cost:,.0f} NOK")
            
            st.info(f"**Approach**: {selected['description']}")
            
            # Compare all approaches
            st.subheader("📊 Approach Comparison")
            
            comparison_data = []
            for name, data in approaches.items():
                ai_perf = simulate_ai_confidence_degradation(data['quality'])
                annual_value = 10000000
                effective_value = annual_value * (data['quality'] / 100)
                
                comparison_data.append({
                    "Approach": name,
                    "Quality": f"{data['quality']}%",
                    "Time": f"{data['time_weeks']}w",
                    "Cost": f"{data['cost']:,}",
                    "AI Performance": f"{ai_perf:.0f}%",
                    "Annual Value": f"{effective_value:,.0f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Trade-off visualization
            fig_tradeoff = go.Figure()
            
            qualities = [approaches[name]['quality'] for name in approaches.keys()]
            times = [approaches[name]['time_weeks'] for name in approaches.keys()]
            costs = [approaches[name]['cost'] for name in approaches.keys()]
            names = list(approaches.keys())
            
            fig_tradeoff.add_trace(go.Scatter(
                x=times, y=qualities,
                mode='markers+text',
                text=names,
                textposition="top center",
                marker=dict(
                    size=[c/10000 for c in costs],  # Size proportional to cost
                    color=costs,
                    colorscale='Viridis',
                    colorbar=dict(title="Cost (NOK)")
                ),
                name="Approaches"
            ))
            
            fig_tradeoff.update_layout(
                title="Quality vs Time Trade-offs (bubble size = cost)",
                xaxis_title="Time to Deliver (weeks)",
                yaxis_title="Data Quality (%)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color="#111111")
            )
            
            st.plotly_chart(fig_tradeoff, use_container_width=True)
            
            # Decision framework
            st.subheader("🎯 Decision Framework")
            
            priority = st.selectbox("Project Priority:", 
                                  ["Speed (Time to market)", "Quality (Long-term accuracy)", 
                                   "Cost (Budget constraints)", "Risk (Minimize failures)"])
            
            if priority == "Speed (Time to market)":
                st.success("✅ **Recommendation**: Quick & Dirty approach")
                st.info("Accept lower quality for faster delivery. Plan for iterative improvements.")
                
            elif priority == "Quality (Long-term accuracy)":
                st.success("✅ **Recommendation**: High Quality or Gold Standard")
                st.info("Invest in quality upfront. Higher initial cost but better long-term ROI.")
                
            elif priority == "Cost (Budget constraints)":
                st.success("✅ **Recommendation**: Balanced approach")
                st.info("Best quality-to-cost ratio. Good compromise for most situations.")
                
            else:  # Risk
                st.success("✅ **Recommendation**: Gold Standard approach")
                st.info("Minimize risk of AI failures. Critical for high-stakes applications.")

st.divider()
st.caption("💡 Tips: Bruk faner for å utforske ulike analyser. Plotly-figurer kan lastes ned via kamera-ikonet.")