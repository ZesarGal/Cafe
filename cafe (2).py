# ======================================================
# Café México — Dashboard (Altair, Q1)
# Archivo único: cafe.py
# - Sin Plotly (evita errores de dependencias en Streamlit Cloud)
# - CSV upload + filtros + tablas pandas
# - Estadísticas descriptivas (con y sin winsor)
# - Histogramas + boxplots (espaciados, interactivos)
# - Correlaciones (niveles vs estandarizadas) + tabla opcional
# - PCA (biplot) con etiquetas legibles y lectura matemática
# ======================================================

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Café México — Dashboard (Altair, Q1)",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Altair theme (Q1 style)
# -------------------------------
def _q1_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFont": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
                "titleFont": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
                "labelFontSize": 12,
                "titleFontSize": 12,
                "gridColor": "rgba(0,0,0,0.08)",
                "tickColor": "rgba(0,0,0,0.20)",
                "domainColor": "rgba(0,0,0,0.25)",
            },
            "legend": {
                "labelFont": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
                "titleFont": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
                "labelFontSize": 12,
                "titleFontSize": 12,
                "orient": "right",
                "symbolSize": 120,
            },
            "title": {
                "font": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
                "fontSize": 15,
                "anchor": "start",
            },
        }
    }

alt.themes.register("q1", _q1_theme)
alt.themes.enable("q1")
alt.data_transformers.disable_max_rows()

# -------------------------------
# CSS (spacing + cards)
# -------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2.8rem; max-width: 1250px;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(0,0,0,0.06);}

    .hero {
      border-radius: 22px;
      padding: 22px 22px 14px 22px;
      border: 1px solid rgba(0,0,0,0.08);
      overflow: hidden;
      background: linear-gradient(120deg, rgba(20,20,20,0.92), rgba(25,75,70,0.92), rgba(60,35,25,0.92));
      background-size: 300% 300%;
      animation: gradientMove 10s ease infinite;
      box-shadow: 0 18px 45px rgba(0,0,0,0.10);
      color: white;
      margin-bottom: 0.75rem;
    }
    @keyframes gradientMove {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
    }
    .hero h1 {font-size: 2.0rem; margin: 0; line-height: 1.15;}
    .hero p {margin: 6px 0 0 0; color: rgba(255,255,255,0.78); font-size: 1.03rem;}
    .badge {
      display: inline-block;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.14);
      color: rgba(255,255,255,0.9);
      font-size: 0.85rem;
      margin-right: 6px;
      margin-top: 10px;
    }
    .card {
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.72);
        box-shadow: 0 14px 30px rgba(0,0,0,0.05);
        margin-bottom: 1.0rem;
    }
    .muted {color: rgba(0,0,0,0.55); font-size: 0.95rem;}
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 14px 30px rgba(0,0,0,0.05);
    }
    hr {border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 1.0rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# Schema helpers (adapted to your base)
# ======================================================

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
ALL_STAGE_COLS = SPECIAL_COLS + CONV_COLS


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def midpoint_or_single(a, b):
    if pd.notna(a) and pd.notna(b):
        return (a + b) / 2.0
    if pd.notna(a):
        return a
    if pd.notna(b):
        return b
    return np.nan


def derive_estado(row: pd.Series) -> str:
    for s in STATE_COLS:
        v = row.get(s, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    other = row.get("Otro (especifique)", "")
    if isinstance(other, str) and other.strip():
        return other.strip()
    return "No especificado"


def first_nonnull(row: pd.Series, cols: list[str]):
    for c in cols:
        v = row.get(c)
        if pd.notna(v):
            return v
    return np.nan


def winsorize_series(x: pd.Series, qlo=0.01, qhi=0.99):
    x = safe_numeric(x)
    if x.notna().sum() < 10:
        return x, (np.nan, np.nan)
    lo, hi = np.nanquantile(x, [qlo, qhi])
    return x.clip(lo, hi), (float(lo), float(hi))


@st.cache_data(show_spinner=False)
def build_variables(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # (A) Stage prices from midpoint(min,max)
    for key, cmin, cmax in PRICE_MINMAX_PAIRS:
        if (cmin in df.columns) or (cmax in df.columns):
            a = safe_numeric(df[cmin]) if cmin in df.columns else pd.Series([np.nan] * len(df))
            b = safe_numeric(df[cmax]) if cmax in df.columns else pd.Series([np.nan] * len(df))
            df[key] = [midpoint_or_single(x, y) for x, y in zip(a, b)]
        else:
            df[key] = np.nan

    # (B) Estado
    if any(c in df.columns for c in STATE_COLS):
        df["Estado"] = df.apply(derive_estado, axis=1)
    else:
        df["Estado"] = "No especificado"

    # (C) Segmento (Especialidad vs Convencional)
    if all(c in df.columns for c in SPECIAL_COLS):
        df["I_spec"] = df[SPECIAL_COLS].notna().any(axis=1).astype(int)
    else:
        df["I_spec"] = 0
    df["Segmento"] = np.where(df["I_spec"] == 1, "Especialidad", "Convencional")

    # (D) Precio base p_i with priority: specialty -> conventional
    if all(c in df.columns for c in SPECIAL_COLS) and all(c in df.columns for c in CONV_COLS):
        df["p_i"] = df.apply(
            lambda r: first_nonnull(r, SPECIAL_COLS) if pd.notna(first_nonnull(r, SPECIAL_COLS)) else first_nonnull(r, CONV_COLS),
            axis=1,
        )
    else:
        df["p_i"] = np.nan

    # (E) Winsorized p_iW
    df["p_iW"], cuts = winsorize_series(df["p_i"], 0.01, 0.99)
    df.attrs["winsor_lo"], df.attrs["winsor_hi"] = cuts

    return df


def describe_prices(d: pd.DataFrame, col: str) -> pd.DataFrame:
    x = safe_numeric(d[col])
    if x.notna().sum() == 0:
        return pd.DataFrame()
    q = x.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    out = pd.DataFrame(
        {
            "n": [int(x.notna().sum())],
            "mean": [float(x.mean())],
            "std": [float(x.std(ddof=1))],
            "min": [float(x.min())],
            "p01": [float(q.loc[0.01])],
            "p05": [float(q.loc[0.05])],
            "p10": [float(q.loc[0.10])],
            "p25": [float(q.loc[0.25])],
            "p50": [float(q.loc[0.50])],
            "p75": [float(q.loc[0.75])],
            "p90": [float(q.loc[0.90])],
            "p95": [float(q.loc[0.95])],
            "p99": [float(q.loc[0.99])],
            "max": [float(x.max())],
        }
    )
    return out


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


def premium_descriptive(d: pd.DataFrame, price_col: str):
    """Simple descriptive premium: difference in means and log-difference."""
    tmp = d.copy()
    tmp["_p"] = safe_numeric(tmp[price_col])
    tmp = tmp.dropna(subset=["_p", "Segmento"])
    if tmp["Segmento"].nunique() < 2:
        return None

    mu1 = tmp.loc[tmp["Segmento"] == "Especialidad", "_p"].mean()
    mu0 = tmp.loc[tmp["Segmento"] == "Convencional", "_p"].mean()
    med1 = tmp.loc[tmp["Segmento"] == "Especialidad", "_p"].median()
    med0 = tmp.loc[tmp["Segmento"] == "Convencional", "_p"].median()

    # semi-log approximate premium (on positive)
    tmp_pos = tmp[tmp["_p"] > 0].copy()
    if tmp_pos["Segmento"].nunique() < 2:
        dlog = np.nan
        pct = np.nan
    else:
        mlog1 = np.log(tmp_pos.loc[tmp_pos["Segmento"] == "Especialidad", "_p"]).mean()
        mlog0 = np.log(tmp_pos.loc[tmp_pos["Segmento"] == "Convencional", "_p"]).mean()
        dlog = float(mlog1 - mlog0)
        pct = float(100.0 * (np.exp(dlog) - 1.0))

    return {
        "mu1": float(mu1),
        "mu0": float(mu0),
        "delta": float(mu1 - mu0),
        "med1": float(med1),
        "med0": float(med1 - med0),
        "dlog": dlog,
        "pct": pct,
    }


# ======================================================
# Header
# ======================================================
st.markdown(
    """
    <div class="hero">
      <h1>☕ Café México — Dashboard (Altair, Q1)</h1>
      <p>Sube tu CSV y explora precios (con y sin winsor), heterogeneidad territorial, correlaciones y PCA.</p>
      <span class="badge">Altair-only</span>
      <span class="badge">Filtros</span>
      <span class="badge">Tablas</span>
      <span class="badge">Hist & Box</span>
      <span class="badge">Correlaciones</span>
      <span class="badge">PCA</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# Mathematical intro (premium definition)
# ======================================================
st.markdown(r"""
## Prima de calidad: definición matemática e interpretación

En este proyecto, una **prima (premium)** es una diferencia sistemática de precios asociada al segmento de mercado
(**especialidad** vs **convencional**), y no meramente fluctuación idiosincrática.

Sea \(p_i>0\) el precio (MXN/kg) observado para la unidad \(i\). Sea \(D_i\in\{0,1\}\) un indicador de especialidad
(\(D_i=1\) especialidad, \(D_i=0\) convencional).

### Prima en niveles
\[
\Delta \;=\; \mathbb{E}[p_i \mid D_i=1] \;-\; \mathbb{E}[p_i \mid D_i=0].
\]
Si \(\Delta>0\), el mercado paga más (en promedio) por especialidad.

### Prima porcentual (semi-log)
Con colas pesadas y heterogeneidad, es natural usar logaritmos:
\[
y_i=\log(p_i),\qquad 
\Delta_{\log} \;=\; \mathbb{E}[y_i \mid D_i=1]-\mathbb{E}[y_i \mid D_i=0],
\qquad
\%\text{prima}\approx 100\,(e^{\Delta_{\log}}-1).
\]
Esto se interpreta como “especialidad paga \(x\%\) más” (aprox.).

### Prima condicional por territorio (idea)
Para comparar **dentro de estado** (controlando heterogeneidad territorial), se piensa en:
\[
\log(p_i)=\alpha+\beta D_i+\gamma_{s(i)}+u_i,
\]
donde \(\gamma_{s(i)}\) son efectos fijos por estado. Aquí \(\beta\) es una prima “dentro de territorio”.
""")

# ======================================================
# Upload
# ======================================================
st.sidebar.markdown("## 1) Sube la base (CSV)")
uploaded = st.sidebar.file_uploader("Arrastra tu CSV aquí", type=["csv"])

if uploaded is None:
    st.info("Sube un archivo CSV para comenzar.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"No pude leer el CSV. Error: {e}")
    st.stop()

df = build_variables(raw)

# ======================================================
# Sidebar filters
# ======================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 2) Filtros")

seg = st.sidebar.radio("Segmento", ["Todos", "Especialidad", "Convencional"], index=0)

states = sorted(df["Estado"].dropna().unique().tolist())
default_states = [s for s in ["Chiapas", "Oaxaca", "Veracruz", "Puebla", "Guerrero"] if s in states]
sel_states = st.sidebar.multiselect("Estado", states, default=default_states or states[:5])

price_mode = st.sidebar.selectbox(
    "Variable de precio",
    ["Precio base (p_i)", "Precio base winsorizado (p_iW)"] + [f"Etapa: {c}" for c in ALL_STAGE_COLS],
    index=1,
)
if price_mode.startswith("Etapa: "):
    price_col = price_mode.replace("Etapa: ", "")
else:
    price_col = "p_iW" if "winsorizado" in price_mode else "p_i"

use_log = st.sidebar.toggle("Escala logarítmica en ejes de precio", value=False)
show_points = st.sidebar.toggle("Mostrar puntos (jitter) en boxplots", value=False)

px_series = safe_numeric(df[price_col]) if price_col in df.columns else pd.Series([np.nan] * len(df))
if px_series.notna().any():
    minp, maxp = float(np.nanmin(px_series)), float(np.nanmax(px_series))
    pr_range = st.sidebar.slider("Rango de precio (MXN/kg)", min_value=minp, max_value=maxp, value=(minp, maxp))
else:
    pr_range = None

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
if pr_range is not None and price_col in dff.columns:
    s = safe_numeric(dff[price_col])
    dff = dff[(s.isna()) | ((s >= pr_range[0]) & (s <= pr_range[1]))]

# KPIs
lo, hi = df.attrs.get("winsor_lo", np.nan), df.attrs.get("winsor_hi", np.nan)
s_sel = safe_numeric(dff[price_col]) if price_col in dff.columns else pd.Series(dtype=float)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Observaciones (filtradas)", f"{len(dff):,}")
with k2:
    st.metric("Mediana", f"{np.nanmedian(s_sel):,.2f} MXN/kg" if s_sel.notna().any() else "—")
with k3:
    st.metric("P10–P90", f"{np.nanquantile(s_sel,0.10):,.1f}–{np.nanquantile(s_sel,0.90):,.1f}" if s_sel.notna().sum() >= 5 else "—")
with k4:
    st.metric("Winsor global (para p_i)", f"{lo:,.2f} / {hi:,.2f}" if np.isfinite(lo) and np.isfinite(hi) else "—")

# ======================================================
# Tabs
# ======================================================
st.markdown("---")
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Resumen", "Descriptivas", "Distribuciones", "Boxplots", "Correlaciones", "PCA"]
)

# ======================================================
# TAB 0 — Summary (table + construction + winsor)
# ======================================================
with tab0:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vista rápida (tabla)")
    st.markdown('<div class="muted">Tabla filtrada (primeras 300 filas) — útil para revisar consistencia, faltantes y codificación territorial.</div>', unsafe_allow_html=True)
    st.dataframe(dff.head(300), use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Construcción de precios: \(p_i\) y \(p_i^W\) (lectura metodológica)")
    st.markdown(
        r"""
**(1) Mínimos y máximos por etapa.** La encuesta reporta, para varias etapas/procesos, un **mínimo** y un **máximo**.
Construimos un “precio representativo” por etapa como **punto medio**:
\[
\widehat p=\frac{p_{\min}+p_{\max}}{2},
\]
y si sólo hay uno de los dos valores, usamos el disponible.

**(2) Precio base \(p_i\).** Dada la heterogeneidad (no siempre se reportan todas las etapas),
definimos un precio base por observación con prioridad **especialidad → convencional**:
\[
p_i=
\begin{cases}
\text{primera columna no nula de especialidad}, & \text{si existe}\\
\text{primera columna no nula de convencional}, & \text{si no existe especialidad}\\
\text{NA}, & \text{si no hay precios reportados}.
\end{cases}
\]

**(3) Winsorización \(p_i^W\) (robustez).**
Sea \(q_{0.01}\) el cuantil 1% y \(q_{0.99}\) el cuantil 99% de \(p_i\). Definimos
\[
p_i^{W}=\min\{\max\{p_i,\,q_{0.01}\},\,q_{0.99}\}.
\]
Esto **no elimina** observaciones; reduce la influencia de colas extremas sobre medias, varianzas y regresiones.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # quick descriptive premium
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Prima descriptiva (rápida) para la variable seleccionada")
    prem = premium_descriptive(dff, price_col) if (price_col in dff.columns) else None
    if prem is None:
        st.info("Se requieren ambos segmentos (Especialidad y Convencional) y datos válidos de precio para estimar una prima descriptiva.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Δ (media): Especialidad − Convencional", f"{prem['delta']:,.2f} MXN/kg")
        with c2:
            st.metric("Prima % (semi-log aprox.)", f"{prem['pct']:,.1f}%" if np.isfinite(prem["pct"]) else "—")
        with c3:
            st.metric("Δlog (media log)", f"{prem['dlog']:.3f}" if np.isfinite(prem["dlog"]) else "—")
        st.markdown(
            r"""
**Interpretación.**  
- Si \(\Delta>0\), especialidad paga más en promedio (diferencia en niveles).  
- Si \(\Delta_{\log}>0\), especialidad paga más en términos proporcionales (aprox. \(100(e^{\Delta_{\log}}-1)\%\)).  
- La comparación cambia si activas \(p_i^W\): si la prima persiste bajo winsor, sugiere **robustez**.
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 1 — Descriptives (p_i vs p_iW + by State×Segment)
# ======================================================
with tab1:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Estadísticas descriptivas — \(p_i\) (original)")
        desc_pi = describe_prices(dff, "p_i")
        if desc_pi.empty:
            st.info("No hay datos en p_i con los filtros actuales.")
        else:
            st.dataframe(desc_pi.style.format("{:,.2f}"), use_container_width=True)
        st.markdown(
            '<div class="muted">Lectura: p95–p99 cuantifican cola; si son muy altos respecto a p50, la media puede ser poco representativa.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Estadísticas descriptivas — \(p_i^W\) (winsorizado)")
        desc_piW = describe_prices(dff, "p_iW")
        if desc_piW.empty:
            st.info("No hay datos en p_iW con los filtros actuales.")
        else:
            st.dataframe(desc_piW.style.format("{:,.2f}"), use_container_width=True)
        st.markdown(
            '<div class="muted">Lectura: winsor reduce max/p99; si p50 casi no cambia, el centro es estable y el recorte afecta principalmente la cola.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Descriptivas por Estado × Segmento (para la variable seleccionada)")
    if price_col not in dff.columns:
        st.warning("La variable seleccionada no existe en la base.")
    else:
        tmp = dff.copy()
        tmp["_price"] = safe_numeric(tmp[price_col])
        g = (
            tmp.dropna(subset=["_price"])
            .groupby(["Estado", "Segmento"])["_price"]
            .agg(
                n="count",
                mean="mean",
                median="median",
                std="std",
                p10=lambda x: x.quantile(0.10),
                p90=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )
        g = g.sort_values(["Estado", "Segmento"])
        st.dataframe(
            g.style.format(
                {"mean": "{:,.2f}", "median": "{:,.2f}", "std": "{:,.2f}", "p10": "{:,.2f}", "p90": "{:,.2f}"}
            ),
            use_container_width=True,
            height=560,
        )
        st.markdown(
            r"""
**Interpretación.**
- Compara **medianas** por estado (nivel típico) y la banda \(p10\)–\(p90\) (dispersión).
- Si, dentro de un mismo estado, especialidad está sistemáticamente por encima, sugiere prima **condicional** al territorio.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 2 — Distributions (hist + compare p_i vs p_iW)
# ======================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### Histograma interactivo — {price_col}")
    tmp = dff.dropna(subset=[price_col]).copy()
    tmp["_price"] = safe_numeric(tmp[price_col])

    if tmp.empty:
        st.info("No hay datos para graficar con los filtros actuales.")
    else:
        if use_log:
            tmp = tmp[tmp["_price"] > 0].copy()
            tmp["_x"] = np.log(tmp["_price"])
            x_enc = alt.X("_x:Q", bin=alt.Bin(maxbins=40), title="log(Precio)")
        else:
            tmp["_x"] = tmp["_price"]
            x_enc = alt.X("_x:Q", bin=alt.Bin(maxbins=40), title="Precio (MXN/kg)")

        hist = (
            alt.Chart(tmp)
            .mark_bar(opacity=0.85)
            .encode(
                x=x_enc,
                y=alt.Y("count():Q", title="Frecuencia"),
                color=alt.Color("Segmento:N", title="Segmento"),
                tooltip=[alt.Tooltip("count():Q", title="n")],
            )
            .properties(height=420)
        )
        st.altair_chart(hist.interactive(), use_container_width=True)

        st.markdown(r"""
**Cómo leer esta figura.**
- Si la distribución de **Especialidad** se desplaza a la derecha respecto a **Convencional**, sugiere \(\Delta>0\).
- Una cola derecha muy larga puede inflar \(\mathbb{E}[p]\); por eso conviene comparar también con \(p_i^W\).
- Con log activado, se ve con más nitidez la estructura en colas (valores altos).
""")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Comparación: \(p_i\) vs \(p_i^W\) (misma muestra filtrada)")
    comp = dff.copy()
    comp = comp.assign(p_i=safe_numeric(comp["p_i"]), p_iW=safe_numeric(comp["p_iW"]))
    comp_long = comp.melt(id_vars=["Segmento", "Estado"], value_vars=["p_i", "p_iW"], var_name="Serie", value_name="Precio")
    comp_long = comp_long.dropna(subset=["Precio"])

    if comp_long.empty:
        st.info("No hay datos para comparar p_i vs p_iW con los filtros actuales.")
    else:
        ch = (
            alt.Chart(comp_long)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("Precio:Q", bin=alt.Bin(maxbins=40), title="Precio (MXN/kg)"),
                y=alt.Y("count():Q", title="Frecuencia"),
                color=alt.Color("Serie:N", title="Serie"),
                tooltip=[alt.Tooltip("count():Q", title="n")],
            )
            .properties(height=420)
        )
        st.altair_chart(ch.facet(column=alt.Column("Segmento:N", title="")).resolve_scale(x="independent"), use_container_width=True)

        st.markdown(r"""
**Interpretación.**
- \(p_i^W\) “recorta” la cola extrema (por construcción, por arriba y por abajo).
- Si los histogramas de \(p_i\) y \(p_i^W\) son casi iguales salvo la cola, la robustez es “barata”: estabiliza sin cambiar el centro.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 3 — Boxplots (state × segment)
# ======================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### Boxplot por Estado — {price_col}")
    if price_col not in dff.columns:
        st.warning("La variable seleccionada no existe.")
    else:
        tmp = dff.copy()
        tmp["_price"] = safe_numeric(tmp[price_col])
        tmp = tmp.dropna(subset=["_price"])

        if tmp.empty:
            st.info("No hay datos para boxplot con los filtros actuales.")
        else:
            order = tmp.groupby("Estado")["_price"].median().sort_values().index.tolist()

            if use_log:
                tmp = tmp[tmp["_price"] > 0].copy()
                tmp["_y"] = np.log(tmp["_price"])
                y_enc = alt.Y("_y:Q", title="log(Precio)")
                tooltip_price = alt.Tooltip("_price:Q", format=",.2f", title="Precio (MXN/kg)")
            else:
                tmp["_y"] = tmp["_price"]
                y_enc = alt.Y("_y:Q", title="Precio (MXN/kg)")
                tooltip_price = alt.Tooltip("_price:Q", format=",.2f", title="Precio (MXN/kg)")

            base = alt.Chart(tmp).encode(
                x=alt.X("Estado:N", sort=order, title="Estado"),
                y=y_enc,
                color=alt.Color("Segmento:N", title="Segmento"),
                tooltip=[alt.Tooltip("Estado:N"), alt.Tooltip("Segmento:N"), tooltip_price],
            )

            box = base.mark_boxplot(size=28, opacity=0.85).properties(height=540)
            chart = box

            if show_points:
                pts = base.mark_circle(size=18, opacity=0.18).properties(height=540)
                chart = box + pts

            st.altair_chart(chart.interactive(), use_container_width=True)

            st.markdown(r"""
**Cómo leer esta figura.**
- La **línea central** es la mediana; las **cajas** reflejan el rango intercuartílico (IQR).
- IQR grande sugiere mayor heterogeneidad del mercado local (calidad/proceso/organización/comercialización).
- Si, dentro de estado, especialidad se ubica consistentemente por encima, apoya una prima **condicional al territorio**.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 4 — Correlations (levels vs standardized)
# ======================================================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Correlaciones entre etapas/precios (heatmaps)")
    st.markdown(
        r"""
Sea \(X_j\) una columna de precio por etapa/proceso.  
- **Niveles:** correlación entre \(X_j\) tal cual.  
- **Estandarizadas:** usamos \(Z_j=(X_j-\bar X_j)/s_j\), y correlacionamos \(Z_j\).  

La estandarización elimina escala (MXN/kg) y resalta **co-movimiento**.
        """
    )

    have_cols = [c for c in ALL_STAGE_COLS if c in dff.columns]
    if len(have_cols) < 3:
        st.warning("No hay suficientes columnas de etapa para correlaciones (se requieren al menos 3).")
    else:
        X = dff[have_cols].copy()
        corr_raw = X.corr(min_periods=15)

        Z = (X - X.mean()) / X.std(ddof=1)
        corr_std = Z.corr(min_periods=15)

        def corr_long(C: pd.DataFrame, label: str) -> pd.DataFrame:
            L = C.stack(dropna=False).reset_index()
            L.columns = ["var1", "var2", "corr"]
            L["tipo"] = label
            return L

        L = pd.concat([corr_long(corr_raw, "Niveles"), corr_long(corr_std, "Estandarizadas")], ignore_index=True)
        L["var1"] = pd.Categorical(L["var1"], categories=have_cols, ordered=True)
        L["var2"] = pd.Categorical(L["var2"], categories=have_cols, ordered=True)

        heat = (
            alt.Chart(L)
            .mark_rect()
            .encode(
                x=alt.X("var2:N", title="", sort=have_cols),
                y=alt.Y("var1:N", title="", sort=have_cols),
                color=alt.Color("corr:Q", title="Corr", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
                tooltip=[alt.Tooltip("var1:N"), alt.Tooltip("var2:N"), alt.Tooltip("corr:Q", format=".2f")],
            )
            .properties(height=380)
        )

        text = alt.Chart(L).mark_text(fontSize=10).encode(
            x="var2:N",
            y="var1:N",
            text=alt.Text("corr:Q", format=".2f"),
            color=alt.condition("abs(datum.corr) > 0.65", alt.value("black"), alt.value("rgba(0,0,0,0.55)")),
        )

        charts = (heat + text).facet(column=alt.Column("tipo:N", title=""))
        st.altair_chart(charts, use_container_width=True)

        st.markdown(r"""
**Interpretación.**
- Correlaciones altas (p.ej. \(>0.7\)) sugieren que dos precios se mueven juntos (posible factor común: mercado local, intermediación o condiciones de oferta).
- Si la correlación cambia mucho entre niveles y estandarizadas, la diferencia puede estar dominada por **escala** y no por co-movimiento real.
""")

        show_table = st.toggle("Mostrar tabla de correlación (niveles)", value=False)
        if show_table:
            st.dataframe(corr_raw.style.format("{:.2f}"), use_container_width=True, height=420)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 5 — PCA (biplot)
# ======================================================
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### PCA (biplot) con precios estandarizados")

    st.markdown(
        r"""
**Definición (idea).** Con precios estandarizados \(Z\), PCA busca direcciones \(v_1,v_2\) que maximizan varianza:
\[
v_1=\arg\max_{\|v\|=1}\operatorname{Var}(Zv),\qquad
v_2=\arg\max_{\|v\|=1,\,v\perp v_1}\operatorname{Var}(Zv).
\]
Los **scores** son \(\text{PC}_k=Zv_k\). Las **cargas** (loadings) indican qué variables “empujan” cada componente.
        """
    )

    block = st.radio("Bloque PCA", ["Especialidad", "Convencional", "Todo (etapas)"], horizontal=True)

    if block == "Especialidad":
        cols = [c for c in SPECIAL_COLS if c in dff.columns]
        min_nonmissing = 2
        title = "PCA — Especialidad"
    elif block == "Convencional":
        cols = [c for c in CONV_COLS if c in dff.columns]
        min_nonmissing = 2
        title = "PCA — Convencional"
    else:
        cols = [c for c in ALL_STAGE_COLS if c in dff.columns]
        min_nonmissing = 3
        title = "PCA — Todo (etapas)"

    if len(cols) < min_nonmissing:
        st.warning("No hay suficientes columnas para correr el PCA con este bloque.")
    else:
        scores, loadings, evr = pca_2d(dff[cols], min_nonmissing=min_nonmissing)
        s = scores.copy()
        s["Segmento"] = dff.loc[s.index, "Segmento"].values
        s["Estado"] = dff.loc[s.index, "Estado"].values

        ev = pd.DataFrame({"Componente": ["PC1", "PC2"], "Varianza (%)": [evr[0] * 100, evr[1] * 100]})
        ev_ch = (
            alt.Chart(ev)
            .mark_bar()
            .encode(
                x=alt.X("Componente:N", title=""),
                y=alt.Y("Varianza (%):Q", title="Varianza explicada (%)"),
                tooltip=[alt.Tooltip("Varianza (%):Q", format=".2f")],
            )
            .properties(height=220, title="Varianza explicada (PC1–PC2)")
        )
        st.altair_chart(ev_ch, use_container_width=True)

        # Biplot
        sel = alt.selection_point(fields=["Segmento"], bind="legend")

        pts = (
            alt.Chart(s.reset_index(drop=True))
            .mark_circle(size=60, opacity=0.55)
            .encode(
                x=alt.X("PC1:Q", title="PC1"),
                y=alt.Y("PC2:Q", title="PC2"),
                color=alt.Color("Segmento:N", title="Segmento"),
                tooltip=[
                    alt.Tooltip("PC1:Q", format=".2f"),
                    alt.Tooltip("PC2:Q", format=".2f"),
                    alt.Tooltip("Estado:N"),
                    alt.Tooltip("Segmento:N"),
                ],
            )
            .add_params(sel)
            .transform_filter(sel)
            .properties(height=560, title=title)
        )

        # scale loadings (visual)
        scale = 3.0
        Ld = loadings.copy()
        Ld["x"] = 0.0
        Ld["y"] = 0.0
        Ld["x2"] = Ld["PC1"] * scale
        Ld["y2"] = Ld["PC2"] * scale
        Ld["var"] = Ld.index

        segs = alt.Chart(Ld.reset_index(drop=True)).mark_rule(opacity=0.95, strokeWidth=2.5).encode(
            x="x:Q",
            y="y:Q",
            x2="x2:Q",
            y2="y2:Q",
            tooltip=[alt.Tooltip("var:N"), alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")],
        )
        labels = alt.Chart(Ld.reset_index(drop=True)).mark_text(align="left", dx=6, dy=-6, fontSize=11).encode(
            x="x2:Q", y="y2:Q", text="var:N"
        )

        st.altair_chart((pts + segs + labels).interactive(), use_container_width=True)

        st.markdown(
            r"""
**Cómo leer el biplot.**
- Flechas largas ⇒ variables influyentes (carga grande).
- Flechas alineadas ⇒ co-movimiento positivo; opuestas ⇒ trade-off (correlación negativa).
- Puntos alejados ⇒ perfiles de precios distintos (estructura latente).

**Lectura económica típica.**
- **PC1** suele capturar un factor común de **valorización** (nivel/segmentación/calidad).
- **PC2** suele capturar contraste por **proceso/etapa** (lavado vs honey vs natural; verde vs pergamino).
"""
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Footer helper



