import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st

# -------------------------------
# Page config (demo-like)
# -------------------------------
st.set_page_config(
    page_title="Café México — CGV, precios y calidad",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# CSS: animated hero + glass UI
# -------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2.5rem; max-width: 1200px;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(0,0,0,0.06);}

    /* Animated hero */
    .hero {
      position: relative;
      border-radius: 22px;
      padding: 26px 26px 18px 26px;
      border: 1px solid rgba(0,0,0,0.08);
      overflow: hidden;
      background: linear-gradient(120deg, rgba(20,20,20,0.92), rgba(25,75,70,0.92), rgba(60,35,25,0.92));
      background-size: 300% 300%;
      animation: gradientMove 10s ease infinite;
      box-shadow: 0 18px 45px rgba(0,0,0,0.10);
      color: white;
    }
    @keyframes gradientMove {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
    }
    .hero h1 {font-size: 2.0rem; margin: 0; line-height: 1.2;}
    .hero p {margin: 6px 0 0 0; color: rgba(255,255,255,0.78); font-size: 1.03rem;}
    .hero .badges {margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;}
    .badge {
      display: inline-block;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.14);
      color: rgba(255,255,255,0.9);
      font-size: 0.85rem;
    }

    /* Cards */
    .card {
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.70);
        box-shadow: 0 14px 30px rgba(0,0,0,0.05);
    }
    .muted {color: rgba(0,0,0,0.55); font-size: 0.95rem;}

    /* Metrics look */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 14px 30px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Constants / helpers
# -------------------------------
STATE_COLS = ["Veracruz", "Puebla", "Chiapas", "Oaxaca", "Guerrero"]

PRICE_MINMAX_PAIRS = [
    ("cereza_conv", "Precio mínimo por Kilo de fruto o cereza convencional", "Precio máximo por Kilo de fruto o cereza convencional"),
    ("perg_lav_conv", "Precio mínimo por Kilo de pergamino lavado convencional", "Precio máximo por Kilo de pergamino lavado convencional"),
    ("natural_conv", "Precio mínimo por Kilo de natural convencional", "Precio máximo por Kilo de natural convencional"),
    ("verde_conv", "Precio mínimo por Kilo de verde, oro, morteado convencional", "Precio máximo por Kilo de verde, oro, morteado convencional"),
    ("perg_lav_esp", "Precio mínimo por Kilo de pergamino lavado especial", "Precio máximo por Kilo de Pergamino lavado especial"),
    ("perg_honey_esp", "Precio mínimo por Kilo de pergamino honey especial", "Precio máximo por Kilo de pergamino honey especial"),
    ("perg_semilav_esp", "Precio mínimo por Kilo de pergamino semilavado especial", "Precio máximo por Kilo de pergamino semilavado especial"),
    ("natural_esp", "Precio mínimo por Kilo de natural especial", "Precio máximo por Kilo de natural especial"),
    ("verde_esp", "Precio mínimo por Kilo de café verde, oro, morteado especial", "Precio máximo por Kilo de café verde, oro o morteado especial"),
]

SPECIAL_COLS = ["verde_esp", "natural_esp", "perg_lav_esp", "perg_honey_esp", "perg_semilav_esp"]
CONV_COLS = ["verde_conv", "natural_conv", "perg_lav_conv", "cereza_conv"]
ALL_COLS = SPECIAL_COLS + CONV_COLS

CENTROIDS = {
    "Chiapas": (16.75, -92.63),
    "Veracruz": (19.18, -96.14),
    "Puebla": (19.04, -98.20),
    "Oaxaca": (17.07, -96.72),
    "Guerrero": (17.55, -99.50),
}

def derive_estado(row: pd.Series) -> str:
    for s in STATE_COLS:
        v = row.get(s, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    other = row.get("Otro (especifique)", "")
    if isinstance(other, str) and other.strip():
        return other.strip()
    return "No especificado"

def midpoint_or_single(a, b):
    if pd.notna(a) and pd.notna(b):
        return (a + b) / 2.0
    if pd.notna(a):
        return a
    if pd.notna(b):
        return b
    return np.nan

def first_nonnull(row: pd.Series, cols: list[str]):
    for c in cols:
        v = row.get(c)
        if pd.notna(v):
            return v
    return np.nan

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Build stage prices from min/max
    for key, cmin, cmax in PRICE_MINMAX_PAIRS:
        a = df.get(cmin)
        b = df.get(cmax)
        if a is None and b is None:
            df[key] = np.nan
        else:
            df[key] = [midpoint_or_single(x, y) for x, y in zip(a, b)]

    # Estado & Segmento
    df["Estado"] = df.apply(derive_estado, axis=1)
    df["I_spec"] = df[SPECIAL_COLS].notna().any(axis=1).astype(int)
    df["Segmento"] = np.where(df["I_spec"] == 1, "Especialidad", "Convencional")

    # p_i priority (especialidad -> convencional)
    df["p_i"] = df.apply(
        lambda r: first_nonnull(r, SPECIAL_COLS) if pd.notna(first_nonnull(r, SPECIAL_COLS)) else first_nonnull(r, CONV_COLS),
        axis=1
    )

    # winsor global 1–99
    p = df["p_i"].astype(float)
    lo, hi = np.nanquantile(p, [0.01, 0.99])
    df["p_iW"] = p.clip(lo, hi)
    df.attrs["winsor_lo"] = float(lo)
    df.attrs["winsor_hi"] = float(hi)
    return df

def pca_2d(X: pd.DataFrame, min_nonmissing=2):
    mask = X.notna().sum(axis=1) >= min_nonmissing
    Xs = X.loc[mask].copy()
    imp = SimpleImputer(strategy="mean")
    Ximp = pd.DataFrame(imp.fit_transform(Xs), columns=Xs.columns, index=Xs.index)
    Z = StandardScaler().fit_transform(Ximp)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(Z)
    loadings = pd.DataFrame(pca.components_.T, index=Xs.columns, columns=["PC1", "PC2"])
    evr = pca.explained_variance_ratio_
    return (
        pd.DataFrame(scores, index=Xs.index, columns=["PC1", "PC2"]),
        loadings,
        evr,
    )

# -------------------------------
# Load
# -------------------------------
df = load_data("Base_Cafe.csv")

# -------------------------------
# Hero cover (animated)
# -------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>☕ Café México — precios, calidad y cadenas globales de valor</h1>
      <p>Explora heterogeneidad por estado, segmentación (especialidad vs convencional), robustez (winsor),
      correlaciones y estructura latente (PCA) con visualizaciones tipo “demo-movies”.</p>
      <div class="badges">
        <span class="badge">CGV</span>
        <span class="badge">Prima de calidad</span>
        <span class="badge">Robustez</span>
        <span class="badge">Correlaciones</span>
        <span class="badge">PCA</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# -------------------------------
# Sidebar filters (movie-demo vibe)
# -------------------------------
st.sidebar.markdown("## ☕ Filtros")
states = sorted(df["Estado"].dropna().unique().tolist())
default_states = [s for s in ["Chiapas", "Oaxaca", "Veracruz", "Puebla", "Guerrero"] if s in states]
sel_states = st.sidebar.multiselect("Estado", states, default=default_states or states[:5])

seg = st.sidebar.radio("Segmento", ["Todos", "Especialidad", "Convencional"], index=0)
use_winsor = st.sidebar.toggle("Usar precio winsorizado $p_i^W$", value=True)
price_col = "p_iW" if use_winsor else "p_i"

minp, maxp = float(np.nanmin(df[price_col])), float(np.nanmax(df[price_col]))
pr_range = st.sidebar.slider("Rango de precio (MXN/kg)", min_value=float(minp), max_value=float(maxp), value=(float(minp), float(maxp)))

st.sidebar.markdown("---")
st.sidebar.download_button(
    "⬇️ Descargar CSV (con variables derivadas)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="base_cafe_con_variables.csv",
    mime="text/csv",
)

# Apply filters
dff = df.copy()
dff = dff[dff["Estado"].isin(sel_states)]
if seg != "Todos":
    dff = dff[dff["Segmento"] == seg]
dff = dff[(dff[price_col].isna()) | ((dff[price_col] >= pr_range[0]) & (dff[price_col] <= pr_range[1]))]

# -------------------------------
# KPIs row
# -------------------------------
lo, hi = df.attrs.get("winsor_lo", np.nan), df.attrs.get("winsor_hi", np.nan)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Observaciones (filtradas)", f"{len(dff):,}")
with k2:
    st.metric("Mediana", f"{np.nanmedian(dff[price_col]):,.2f} MXN/kg" if dff[price_col].notna().any() else "—")
with k3:
    st.metric("P10–P90", f"{np.nanquantile(dff[price_col],0.10):,.1f}–{np.nanquantile(dff[price_col],0.90):,.1f}"
              if dff[price_col].notna().sum() >= 5 else "—")
with k4:
    st.metric("Winsor (global)", f"{lo:,.2f} / {hi:,.2f}")

st.markdown("---")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Explorar", "Correlaciones", "PCA", "Datos"])

with tab1:
    c1, c2 = st.columns([1.2, 1.0], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Distribución de precios")
        fig = px.histogram(
            dff.dropna(subset=[price_col]),
            x=price_col,
            color="Segmento" if seg == "Todos" else None,
            nbins=35,
            opacity=0.70,
            marginal="box",
        )
        fig.update_layout(height=430, bargap=0.02, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="muted">Lectura: la separación entre segmentos sugiere prima; la winsorización estabiliza colas sin eliminar registros.</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Boxplot por estado")
        tmp = dff.dropna(subset=[price_col]).copy()
        order = tmp.groupby("Estado")[price_col].median().sort_values().index.tolist()
        fig2 = px.box(tmp, x="Estado", y=price_col, category_orders={"Estado": order}, points="outliers")
        fig2.update_layout(height=430)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            '<div class="muted">Lectura: diferencias persistentes por estado; motiva efectos fijos y lectura territorial.</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Mapa exploratorio (centroides) — mediana estatal")
    agg = dff.dropna(subset=[price_col]).groupby("Estado")[price_col].agg(mediana="median", N="count").reset_index()
    agg = agg[agg["Estado"].isin(CENTROIDS.keys())].copy()

    if not agg.empty:
        agg["lat"] = agg["Estado"].map(lambda s: CENTROIDS[s][0])
        agg["lon"] = agg["Estado"].map(lambda s: CENTROIDS[s][1])
        figm = px.scatter_geo(
            agg,
            lat="lat", lon="lon",
            color="mediana",
            size="N",
            hover_name="Estado",
            hover_data={"mediana":":.2f","N":True,"lat":False,"lon":False},
            projection="natural earth",
        )
        figm.update_geos(
            showcountries=False, showland=True, landcolor="rgb(245,245,245)",
            lataxis_range=[14, 22], lonaxis_range=[-103, -90]
        )
        figm.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=0), coloraxis_colorbar_title="Mediana (MXN/kg)")
        st.plotly_chart(figm, use_container_width=True)
        st.caption("Para mapas Q1 con polígonos por estado: integrar GeoJSON/INEGI. Este mapa es exploratorio.")
    else:
        st.info("No hay suficientes observaciones para el mapa con los filtros actuales.")

with tab2:
    st.markdown("### Correlaciones")
    st.markdown('<div class="muted">Comparación en niveles vs estandarizadas (elimina escala y resalta co-movimiento).</div>', unsafe_allow_html=True)

    X = dff[ALL_COLS].copy()
    corr_raw = X.corr(min_periods=15)
    Z = (X - X.mean()) / X.std(ddof=1)
    corr_std = Z.corr(min_periods=15)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Niveles**")
        figc1 = px.imshow(corr_raw, zmin=-1, zmax=1, aspect="auto", color_continuous_scale="RdBu")
        figc1.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(figc1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Estandarizadas**")
        figc2 = px.imshow(corr_std, zmin=-1, zmax=1, aspect="auto", color_continuous_scale="RdBu")
        figc2.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(figc2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("### PCA (Plotly) — etiquetas legibles")
    st.markdown('<div class="muted">Biplot interactivo: scores + vectores de cargas, con hover.</div>', unsafe_allow_html=True)

    pca_choice = st.radio("Bloque PCA", ["Especialidad", "Convencional", "Todo (precios)"], horizontal=True)

    if pca_choice == "Especialidad":
        X = dff[SPECIAL_COLS]
        title = "PCA — Especialidad"
        min_nonmissing = 2
    elif pca_choice == "Convencional":
        X = dff[CONV_COLS]
        title = "PCA — Convencional"
        min_nonmissing = 2
    else:
        X = dff[ALL_COLS]
        title = "PCA — Todo (precios)"
        min_nonmissing = 3

    scores, loadings, evr = pca_2d(X, min_nonmissing=min_nonmissing)

    ev = pd.DataFrame({"Componente":["PC1","PC2"], "Varianza (%)":[evr[0]*100, evr[1]*100]})
    fig_ev = px.bar(ev, x="Componente", y="Varianza (%)")
    fig_ev.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_ev, use_container_width=True)

    s = scores.copy()
    s["Segmento"] = dff.loc[s.index, "Segmento"].values
    figb = px.scatter(s, x="PC1", y="PC2", color="Segmento", opacity=0.58, title=title)
    figb.update_layout(height=650, legend_title_text="")

    scale = 3.2
    for var in loadings.index:
        x, y = loadings.loc[var, "PC1"]*scale, loadings.loc[var, "PC2"]*scale
        figb.add_trace(go.Scatter(
            x=[0, x], y=[0, y],
            mode="lines+markers+text",
            text=["", var],
            textposition="top center",
            line=dict(width=3),
            marker=dict(size=6),
            showlegend=False
        ))
    figb.add_hline(y=0, line_width=1, opacity=0.25)
    figb.add_vline(x=0, line_width=1, opacity=0.25)
    st.plotly_chart(figb, use_container_width=True)

    st.markdown("#### Interpretación")
    st.markdown(
        "- **PC1** suele capturar un *factor común de valorización* (nivel general asociado a calidad/segmentación).\n"
        "- **PC2** captura *contrastes de proceso/etapa* cuando coexisten tecnologías (lavado/honey/natural) o etapas (verde/pergamino).\n"
        "- Flechas en la misma dirección ⇒ co-movimiento; direcciones opuestas ⇒ trade-off; flechas largas ⇒ variable influyente."
    )

with tab4:
    st.markdown("### Datos")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(dff.head(300), use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Cómo se construyen p_i y p_iW"):
        st.markdown(
            r"""
- La base reporta **mínimos y máximos** por etapa/segmento; usamos el **punto medio** como aproximación.
- Se construye \(p_i\) por prioridad **especialidad → convencional** y, dentro de cada bloque, por etapa.
- \(p_i^W\) aplica **winsorización global 1%–99%** para estabilizar colas sin borrar datos.
            """
        )


