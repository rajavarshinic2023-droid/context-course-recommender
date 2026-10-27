"""Streamlit front-end. Run:  streamlit run app.py"""
import html as _html
import logging

import altair as alt
import pandas as pd
import streamlit as st

from src import analytics
from src.config import get_setting
from src.learning_logic import DIFFICULTIES, FORMATS, GOALS, ValidationError
from src.partner_client import PartnerClient
from src.pipeline import run_recommendation

log = logging.getLogger("app")

st.set_page_config(
    page_title="StudyCompass — Context-Aware Study Resource Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME (dark navy + teal + cyan)
# =============================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg-main:#0B1120; --bg-sidebar:#111A2E;
  --bg-card:#162238; --bg-elevated:#1C2B43;
  --accent-primary:#2DD4BF; --accent-secondary:#38BDF8;
  --text-main:#F1F5F9; --text-secondary:#A7B4C8;
  --border:#2A3A52;
  --success:#34D399; --warning:#FBBF24; --error:#FB7185;
  --radius-sm:8px; --radius-md:12px; --radius-lg:16px;
  color-scheme: dark;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background-color:#0B1120 !important;
  color:#F1F5F9 !important;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
  -webkit-font-smoothing:antialiased;
}
.block-container {
  max-width:1200px !important;
  padding:1.5rem 2rem 3rem !important;
  margin:auto !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility:hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
  background-color:#111A2E !important;
  border-right:1px solid #2A3A52 !important;
}
[data-testid="stSidebar"] * { color:#F1F5F9; }
[data-testid="stSidebar"] .block-container { padding:1.4rem 1rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:0.35rem !important; }
[data-testid="stSidebar"] .stButton > button {
  width:100%; justify-content:flex-start; text-align:left;
  background:transparent !important; border:1px solid transparent !important;
  color:#A7B4C8 !important; font-weight:500 !important;
  font-size:0.88rem !important; padding:0.65rem 0.85rem !important;
  border-radius:8px !important;
  transition:background 0.15s, color 0.15s, border-color 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:rgba(45,212,191,0.06) !important;
  color:#F1F5F9 !important;
  border-color:rgba(45,212,191,0.22) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background:rgba(45,212,191,0.12) !important;
  color:#2DD4BF !important;
  border-color:rgba(45,212,191,0.45) !important;
  font-weight:600 !important;
}

/* Headings */
h1,h2,h3,h4,h5,h6 {
  color:#F1F5F9 !important;
  font-family:'Space Grotesk','Inter',sans-serif !important;
  letter-spacing:-0.02em !important;
}

/* Forms & inputs */
[data-testid="stForm"] {
  background:#162238 !important;
  border:1px solid #2A3A52 !important;
  border-radius:16px !important;
  padding:1.5rem !important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div,
[data-testid="stDateInput"] input, [data-testid="stTimeInput"] input {
  background-color:#1C2B43 !important;
  color:#F1F5F9 !important;
  border:1px solid #2A3A52 !important;
  border-radius:8px !important;
  -webkit-text-fill-color:#F1F5F9 !important;
  font-family:'Inter',sans-serif !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color:#A7B4C8 !important;
  -webkit-text-fill-color:#A7B4C8 !important;
  opacity:0.7 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus,
.stNumberInput input:focus, [data-baseweb="select"] > div:focus-within {
  border-color:#2DD4BF !important;
  box-shadow:0 0 0 3px rgba(45,212,191,0.15) !important;
}
label, [data-testid="stWidgetLabel"] {
  color:#A7B4C8 !important;
  font-weight:500 !important;
  font-size:0.83rem !important;
}

/* Dropdowns */
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background-color:#1C2B43 !important;
  border:1px solid #2A3A52 !important;
}
[role="option"] {
  background-color:#1C2B43 !important;
  color:#F1F5F9 !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
  background-color:rgba(45,212,191,0.15) !important;
  color:#2DD4BF !important;
}
div[data-baseweb="tag"] {
  background:rgba(45,212,191,0.15) !important;
  color:#2DD4BF !important;
  border:1px solid rgba(45,212,191,0.4) !important;
  border-radius:6px !important;
}
div[data-baseweb="tag"] span { color:#2DD4BF !important; }

/* Buttons */
[data-testid="stMain"] .stButton > button,
[data-testid="stMain"] .stFormSubmitButton > button {
  background-color:#1C2B43 !important;
  color:#F1F5F9 !important;
  border:1px solid #2A3A52 !important;
  border-radius:8px !important;
  font-weight:600 !important;
  font-size:0.9rem !important;
  padding:0.55rem 1.05rem !important;
  transition:all 0.15s ease !important;
}
[data-testid="stMain"] .stButton > button:hover,
[data-testid="stMain"] .stFormSubmitButton > button:hover {
  border-color:#2DD4BF !important;
  color:#2DD4BF !important;
}
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] .stFormSubmitButton > button[kind="primary"] {
  background:linear-gradient(135deg,#2DD4BF 0%,#38BDF8 100%) !important;
  color:#0B1120 !important;
  border:none !important;
  font-weight:700 !important;
  box-shadow:0 8px 20px -10px rgba(45,212,191,0.6) !important;
}
[data-testid="stMain"] .stButton > button[kind="primary"]:hover,
[data-testid="stMain"] .stFormSubmitButton > button[kind="primary"]:hover {
  filter:brightness(1.08) !important;
  transform:translateY(-1px) !important;
}
.stLinkButton > a {
  background-color:rgba(45,212,191,0.1) !important;
  color:#2DD4BF !important;
  border:1px solid rgba(45,212,191,0.4) !important;
  border-radius:8px !important;
  font-weight:600 !important;
  text-decoration:none !important;
  padding:0.5rem 1rem !important;
}
.stLinkButton > a:hover {
  background-color:rgba(45,212,191,0.2) !important;
  border-color:#2DD4BF !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background:#162238 !important;
  border:1px solid #2A3A52 !important;
  border-radius:12px !important;
  padding:1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
  color:#A7B4C8 !important; font-size:0.72rem !important;
  text-transform:uppercase !important; letter-spacing:0.06em !important;
  font-weight:600 !important;
}
[data-testid="stMetricValue"] {
  color:#F1F5F9 !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-weight:700 !important;
}

/* Alerts, expanders, progress */
[data-testid="stAlert"] {
  background-color:#162238 !important;
  border:1px solid #2A3A52 !important;
  border-radius:12px !important;
  color:#F1F5F9 !important;
}
[data-testid="stAlert"] p { color:#F1F5F9 !important; }
[data-testid="stExpander"] {
  background:#162238 !important;
  border:1px solid #2A3A52 !important;
  border-radius:12px !important;
}
[data-testid="stExpander"] summary {
  color:#F1F5F9 !important; font-weight:600 !important;
}
.stProgress > div > div > div > div {
  background:linear-gradient(90deg,#2DD4BF 0%,#38BDF8 100%) !important;
  border-radius:999px !important;
}
.stProgress > div > div > div {
  background:rgba(45,212,191,0.12) !important;
  border-radius:999px !important;
}

/* Containers & charts */
[data-testid="stVerticalBlockBorderWrapper"] {
  background:#162238 !important;
  border:1px solid #2A3A52 !important;
  border-radius:12px !important;
}
[data-testid="stArrowVegaLiteChart"],
[data-testid="stPlotlyChart"] {
  background:#162238 !important;
  border:1px solid #2A3A52 !important;
  border-radius:12px !important;
  padding:1rem !important;
}

/* Captions, markdown */
[data-testid="stCaptionContainer"] p { color:#A7B4C8 !important; }
.stMarkdown p, .stMarkdown li, .stMarkdown strong { color:#F1F5F9; }
.stMarkdown a { color:#2DD4BF !important; }
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label {
  color:#F1F5F9 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:#0B1120; }
::-webkit-scrollbar-thumb { background:#2A3A52; border-radius:999px; }
::-webkit-scrollbar-thumb:hover { background:#3A4D6B; }

/* ---- Custom components ---- */
.sc-hero {
  background:linear-gradient(135deg,#111A2E 0%,#162238 55%,#1C2B43 100%);
  border:1px solid #2A3A52;
  border-radius:16px;
  padding:2rem 2.2rem;
  margin-bottom:1.5rem;
  position:relative; overflow:hidden;
}
.sc-hero::before {
  content:""; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,#2DD4BF 0%,#38BDF8 50%,#2DD4BF 100%);
}
.sc-hero-brand {
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem; font-weight:700;
  color:#2DD4BF; letter-spacing:0.15em;
  text-transform:uppercase; margin-bottom:0.8rem;
}
.sc-hero-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:clamp(1.5rem,3vw,2.1rem);
  font-weight:700; color:#F1F5F9;
  letter-spacing:-0.03em; margin:0 0 0.5rem; line-height:1.15;
}
.sc-hero-sub {
  font-size:0.94rem; color:#A7B4C8;
  max-width:640px; line-height:1.55; margin:0;
}

.sc-section-title {
  display:flex; align-items:center; gap:0.55rem;
  font-family:'Space Grotesk',sans-serif;
  font-size:1.1rem; font-weight:700; color:#F1F5F9;
  margin:0.4rem 0 0.9rem; letter-spacing:-0.02em;
}
.sc-section-title::before {
  content:""; width:4px; height:18px; border-radius:4px;
  background:linear-gradient(180deg,#2DD4BF 0%,#38BDF8 100%);
}

.sc-badge {
  display:inline-flex; align-items:center; gap:0.35rem;
  padding:0.24rem 0.68rem; border-radius:999px;
  font-size:0.72rem; font-weight:600; line-height:1.4;
}
.sc-badge-success { background:rgba(52,211,153,0.14); color:#34D399; border:1px solid rgba(52,211,153,0.35); }
.sc-badge-warning { background:rgba(251,191,36,0.14); color:#FBBF24; border:1px solid rgba(251,191,36,0.35); }
.sc-badge-error   { background:rgba(251,113,133,0.14); color:#FB7185; border:1px solid rgba(251,113,133,0.35); }
.sc-badge-info    { background:rgba(56,189,248,0.14); color:#38BDF8; border:1px solid rgba(56,189,248,0.35); }
.sc-badge-neutral { background:rgba(167,180,200,0.08); color:#A7B4C8; border:1px solid rgba(167,180,200,0.25); }

.sc-table-wrap {
  border:1px solid #2A3A52; border-radius:12px;
  overflow-x:auto; background:#162238; margin-bottom:1rem;
}
.sc-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.sc-table th {
  background:#1C2B43; color:#A7B4C8; text-align:left;
  font-weight:600; font-size:0.72rem; letter-spacing:0.06em;
  text-transform:uppercase; padding:0.75rem 1rem;
  border-bottom:1px solid #2A3A52;
}
.sc-table td {
  padding:0.72rem 1rem; color:#F1F5F9;
  border-bottom:1px solid #2A3A52; vertical-align:top;
}
.sc-table tr:last-child td { border-bottom:none; }
.sc-table tr:hover td { background:rgba(45,212,191,0.04); }
.sc-table a { color:#2DD4BF; text-decoration:none; font-weight:600; }
.sc-table a:hover { color:#38BDF8; text-decoration:underline; }

.sc-stat {
  background:#162238; border:1px solid #2A3A52;
  border-radius:12px; padding:1.05rem 1.2rem;
}
.sc-stat-label {
  font-size:0.7rem; color:#A7B4C8;
  text-transform:uppercase; letter-spacing:0.06em;
  font-weight:600; margin-bottom:0.4rem;
}
.sc-stat-value {
  font-family:'Space Grotesk',sans-serif;
  font-size:1.55rem; font-weight:700; color:#F1F5F9;
  letter-spacing:-0.03em; line-height:1.1;
}
.sc-stat-value-accent {
  background:linear-gradient(135deg,#2DD4BF 0%,#38BDF8 100%);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}

.sc-empty {
  text-align:center; padding:3rem 2rem;
  background:#162238; border:1px dashed #2A3A52;
  border-radius:16px;
}
.sc-empty-icon { font-size:2rem; margin-bottom:0.8rem; opacity:0.6; }
.sc-empty-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:1.05rem; font-weight:600; color:#F1F5F9;
  margin:0 0 0.45rem;
}
.sc-empty-text {
  font-size:0.88rem; color:#A7B4C8;
  max-width:440px; margin:0 auto; line-height:1.55;
}

.sc-rec-rank {
  display:inline-flex; align-items:center; justify-content:center;
  min-width:44px; height:44px; padding:0 0.6rem;
  border-radius:10px;
  background:linear-gradient(135deg,rgba(45,212,191,0.18) 0%,rgba(56,189,248,0.18) 100%);
  color:#2DD4BF; font-family:'JetBrains Mono',monospace;
  font-weight:700; font-size:0.95rem;
  border:1px solid rgba(45,212,191,0.45);
}
.sc-rec-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:1.02rem; font-weight:700; color:#F1F5F9;
  line-height:1.35; margin:0 0 0.2rem;
}
.sc-rec-meta { font-size:0.78rem; color:#A7B4C8; margin:0; }
.sc-rec-why {
  margin:0.85rem 0; padding:0.75rem 1rem;
  background:rgba(45,212,191,0.06);
  border-left:3px solid #2DD4BF;
  border-radius:0 8px 8px 0;
  font-size:0.88rem; color:#F1F5F9; line-height:1.55;
}
.sc-rec-why strong { color:#2DD4BF; }

.sc-progress {
  background:rgba(45,212,191,0.12);
  border-radius:999px; height:6px;
  overflow:hidden; margin:0.6rem 0 0.2rem;
}
.sc-progress-fill {
  height:100%;
  background:linear-gradient(90deg,#2DD4BF,#38BDF8);
  border-radius:999px;
}

.sc-grid-3 {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:0.9rem; margin-bottom:1.5rem;
}

@media (max-width:768px) {
  .block-container { padding:1rem 1rem 2rem !important; }
  .sc-hero { padding:1.4rem 1.2rem; }
  .sc-hero-title { font-size:1.4rem; }
  .sc-grid-3 { grid-template-columns:1fr 1fr; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
ORIGIN_LABEL = {
    "live": "Live · DEV Community API",
    "local": "Curated library",
    "partner": "Partner API",
}
CHART_COLORS = ["#2DD4BF", "#38BDF8", "#8B6CE0", "#FBBF24", "#FB7185", "#34D399", "#A7B4C8"]


def esc(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
        return f"{val:.2f}"
    return _html.escape(str(val))


def hero(brand: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="sc-hero">'
        f'<div class="sc-hero-brand">{esc(brand)}</div>'
        f'<h1 class="sc-hero-title">{esc(title)}</h1>'
        f'<p class="sc-hero-sub">{esc(subtitle)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(
        f'<div class="sc-section-title">{esc(text)}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, variant: str = "neutral") -> str:
    return f'<span class="sc-badge sc-badge-{variant}">{esc(text)}</span>'


def stat_grid(items) -> None:
    cards = []
    for label, value, accent in items:
        cls = "sc-stat-value-accent" if accent else "sc-stat-value"
        cards.append(
            f'<div class="sc-stat">'
            f'<div class="sc-stat-label">{esc(label)}</div>'
            f'<div class="{cls}">{esc(value)}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="sc-grid-3">{"".join(cards)}</div>', unsafe_allow_html=True)


def empty_state(title: str, text: str, icon: str = "◇") -> None:
    st.markdown(
        f'<div class="sc-empty">'
        f'<div class="sc-empty-icon">{icon}</div>'
        f'<div class="sc-empty-title">{esc(title)}</div>'
        f'<p class="sc-empty-text">{esc(text)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_table(df: pd.DataFrame, link_columns=None) -> None:
    link_columns = set(link_columns or [])
    html = '<div class="sc-table-wrap"><table class="sc-table"><thead><tr>'
    for col in df.columns:
        html += f"<th>{esc(col)}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            if col in link_columns and pd.notna(val) and str(val).strip():
                html += f'<td><a href="{esc(val)}" target="_blank" rel="noopener noreferrer">Open</a></td>'
            else:
                html += f"<td>{esc(val)}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


def progress_bar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div class="sc-progress">'
        f'<div class="sc-progress-fill" style="width:{pct:.2f}%"></div>'
        f'</div>'
    )


def recommendation_card(rec: dict, result: dict) -> None:
    rank = rec.get("rank", 0)
    score = float(rec.get("score", 0))
    origin = rec.get("origin", "")
    origin_label = ORIGIN_LABEL.get(origin, origin or "—")

    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:0.9rem">'
            f'<div class="sc-rec-rank">{int(rank):02d}</div>'
            f'<div style="flex:1;min-width:0">'
            f'<div class="sc-rec-title">{esc(rec.get("title",""))}</div>'
            f'<p class="sc-rec-meta">{esc(rec.get("type",""))} · {esc(rec.get("source",""))} · {esc(origin_label)}</p>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(progress_bar(score), unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div style="text-align:right;font-family:Space Grotesk;'
                f'font-weight:700;color:#2DD4BF;font-size:1.1rem;line-height:1">'
                f'{score:.1f}<span style="color:#A7B4C8;font-size:0.72rem;font-weight:500"> /100</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="sc-rec-why"><strong>Why this resource:</strong> '
            f'{esc(rec.get("explanation",""))}</div>',
            unsafe_allow_html=True,
        )

        meta = rec.get("metadata") or {}
        bits = []
        if meta.get("difficulty"):
            bits.append(f"Difficulty: {meta['difficulty']}")
        if meta.get("published_at"):
            bits.append(f"Published: {str(meta['published_at'])[:10]}")
        if meta.get("reading_time_min"):
            bits.append(f"Read time: {meta['reading_time_min']} min")
        if bits:
            st.caption(" · ".join(bits))

        if meta.get("description"):
            st.markdown(
                f'<p style="color:#A7B4C8;font-size:0.85rem;line-height:1.55;'
                f'margin-top:0.4rem">{esc(meta["description"][:280])}</p>',
                unsafe_allow_html=True,
            )

        with st.expander("Score breakdown"):
            comps = rec.get("components", {}) or {}
            weights = result.get("weights", {}) or {}
            comp_df = pd.DataFrame({
                "Component": [k.capitalize() for k in comps],
                "Score": [f"{float(v):.1f}" for v in comps.values()],
                "Weight": [f"{float(weights.get(k, 0)):.2f}" for k in comps],
            })
            render_table(comp_df)

        url = rec.get("url")
        if url:
            st.link_button("Open resource →", url, use_container_width=True)


def dark_bar_chart(df, x=None, y=None, height=280, horizontal=False):
    if isinstance(df, pd.Series):
        name = df.name if df.name is not None else "value"
        df = df.reset_index()
        if df.columns[-1] != name:
            df = df.rename(columns={df.columns[-1]: name})
    df = df.copy()
    if x is None:
        x = df.columns[0]
    if y is None:
        nums = df.select_dtypes("number").columns.tolist()
        y = nums[-1] if nums else df.columns[-1]

    label_style = dict(labelColor="#A7B4C8", titleColor="#A7B4C8",
                       labelFont="Inter", labelFontSize=11)
    grid_style = dict(grid=True, gridColor="#2A3A52", gridDash=[3, 3],
                      domainColor="#2A3A52", tickColor="#2A3A52")

    if horizontal:
        base = alt.Chart(df).encode(
            y=alt.Y(f"{x}:N", sort="-x", axis=alt.Axis(**label_style, grid=False)),
            x=alt.X(f"{y}:Q", axis=alt.Axis(**label_style, **grid_style)),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=".2f")],
        )
    else:
        base = alt.Chart(df).encode(
            x=alt.X(f"{x}:N", sort="-y",
                    axis=alt.Axis(**label_style, labelAngle=-30, grid=False)),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(**label_style, **grid_style)),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=".2f")],
        )

    chart = (
        base.mark_bar(
            cornerRadiusTopLeft=6, cornerRadiusTopRight=6,
            cornerRadiusBottomLeft=6, cornerRadiusBottomRight=6,
        )
        .encode(color=alt.Color(f"{x}:N", legend=None,
                                scale=alt.Scale(range=CHART_COLORS)))
        .properties(height=height, width="container", background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(labelFont="Inter", titleFont="Inter")
    )
    st.altair_chart(chart, use_container_width=True)


# =============================================================================
# SESSION
# =============================================================================
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("result", None)

PAGES = [
    "Home",
    "Find Resources",
    "Resource Library",
    "Learning Analytics",
    "AI & API Status",
    "Settings",
]


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.65rem;margin:0 0 1.4rem;'
        'padding-bottom:1.1rem;border-bottom:1px solid #2A3A52">'
        '<div style="width:38px;height:38px;border-radius:10px;'
        'background:linear-gradient(135deg,#2DD4BF,#38BDF8);'
        'display:flex;align-items:center;justify-content:center;font-size:1.15rem;'
        'box-shadow:0 6px 16px -6px rgba(45,212,191,0.5)">🧭</div>'
        '<div style="line-height:1.15">'
        '<div style="font-family:\'Space Grotesk\';font-weight:700;font-size:1.05rem;'
        'color:#F1F5F9;letter-spacing:-0.02em">StudyCompass</div>'
        '<div style="font-size:0.62rem;color:#A7B4C8;letter-spacing:0.08em;'
        'text-transform:uppercase;font-weight:600;margin-top:2px">Context-Aware Learning</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    for p in PAGES:
        if st.button(
            p,
            key=f"nav_{p}",
            use_container_width=True,
            type="primary" if p == st.session_state["page"] else "secondary",
        ):
            st.session_state["page"] = p
            st.rerun()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    _result = st.session_state.get("result")
    if _result:
        _n = len(_result.get("recommendations", []))
        st.markdown(
            f'<div style="padding:0.75rem 0.9rem;border-radius:8px;'
            f'background:rgba(45,212,191,0.08);border:1px solid rgba(45,212,191,0.28)">'
            f'<div style="font-size:0.62rem;color:#2DD4BF;text-transform:uppercase;'
            f'letter-spacing:0.07em;font-weight:700;margin-bottom:0.3rem">Active session</div>'
            f'<div style="font-size:0.88rem;color:#F1F5F9;font-weight:600">'
            f'{_n} recommendation{"s" if _n != 1 else ""}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding:0.75rem 0.9rem;border-radius:8px;'
            'background:rgba(167,180,200,0.06);border:1px solid rgba(167,180,200,0.22)">'
            '<div style="font-size:0.62rem;color:#A7B4C8;text-transform:uppercase;'
            'letter-spacing:0.07em;font-weight:700">No active session</div></div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGES
# =============================================================================
page = st.session_state["page"]
result = st.session_state.get("result")


# ---------------------------------------------------------------- Home
if page == "Home":
    hero(
        "STUDYCOMPASS · CONTEXT-AWARE LEARNING",
        "Your personal learning dashboard.",
        "Find resources that match your topic, level, and learning goals — "
        "ranked by AI with transparent explanations.",
    )

    c1, c2, _ = st.columns([1.2, 1.2, 2])
    if c1.button("Find Learning Resources", type="primary", use_container_width=True):
        st.session_state["page"] = "Find Resources"
        st.rerun()
    if c2.button("Browse Library", use_container_width=True):
        st.session_state["page"] = "Resource Library"
        st.rerun()

    if not result:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        empty_state(
            "No active learning session",
            "Start by searching for resources on your topic. Your recommendations, "
            "analytics, and AI status will appear here.",
            "🎯",
        )
    else:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        section_title("Current session")

        recs = result.get("recommendations", [])
        all_scored = result.get("all_scored", [])
        avg = sum(float(r.get("score", 0)) for r in recs) / len(recs) if recs else 0.0
        live = sum(1 for r in recs if r.get("origin") == "live")
        local = sum(1 for r in recs if r.get("origin") == "local")

        stat_grid([
            ("Recommendations", len(recs), True),
            ("Evaluated", len(all_scored), False),
            ("Avg. score", f"{avg:.1f}", False),
            ("Live", live, False),
            ("Curated", local, False),
        ])

        section_title("AI & retrieval status")
        pz = result.get("personalization", {}) or {}
        rt = result.get("retrieval", {}) or {}
        ranking = result.get("ranking", {}) or {}
        partner = result.get("partner", {}) or {}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Personalization**")
            if pz.get("used_llm"):
                st.markdown(badge(f"✓ {pz.get('status', 'LLM used')}", "success"),
                            unsafe_allow_html=True)
            else:
                st.markdown(badge(f"⚠ {pz.get('status', 'Fallback')}", "warning"),
                            unsafe_allow_html=True)
        with c2:
            st.markdown("**Ranking**")
            if ranking.get("status") == "llm_used":
                st.markdown(badge(f"✓ {ranking.get('engine', 'LLM')}", "success"),
                            unsafe_allow_html=True)
            else:
                st.markdown(badge(f"⚠ {ranking.get('status', 'Unavailable')}", "error"),
                            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Retrieval**")
            if rt.get("fallback_used"):
                st.markdown(badge("⚠ Local fallback", "warning"), unsafe_allow_html=True)
            else:
                st.markdown(badge("✓ Live API", "success"), unsafe_allow_html=True)
        with c2:
            st.markdown("**Partner API**")
            pstatus = str(partner.get("status", "not used")).lower()
            if pstatus in ("ok", "success", "used"):
                st.markdown(badge(f"✓ {partner.get('status')}", "success"),
                            unsafe_allow_html=True)
            else:
                st.markdown(badge(f"· {partner.get('status', 'Not used')}", "neutral"),
                            unsafe_allow_html=True)

        if recs:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            section_title("Top recommendations")
            for rec in recs[:3]:
                recommendation_card(rec, result)


# ---------------------------------------------------------------- Find Resources
elif page == "Find Resources":
    hero(
        "FIND RESOURCES",
        "Search & rank learning resources.",
        "Describe what you're learning and get a ranked shortlist "
        "with transparent scoring.",
    )

    with st.form("search_form"):
        st.markdown("**Your learning context**")
        c1, c2 = st.columns(2)
        topic = c1.text_input(
            "Topic or subject *",
            placeholder="e.g. Python functions, Database normalization",
            max_chars=100,
        )
        difficulty = c2.selectbox("Your current level", DIFFICULTIES)

        c3, c4 = st.columns(2)
        formats = c3.multiselect(
            "Preferred formats *",
            list(FORMATS),
            default=["Articles", "Practice Questions"],
        )
        score = c4.number_input(
            "Previous score (0–100)",
            min_value=0, max_value=100, value=50, step=1,
        )

        goal = st.selectbox("Learning goal", GOALS)
        extra = st.text_area(
            "Additional context (optional)",
            max_chars=500,
            placeholder="e.g. I struggle with recursion and default arguments.",
        )

        submitted = st.form_submit_button(
            "Get recommendations",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            with st.spinner("Personalizing and ranking resources…"):
                st.session_state["result"] = run_recommendation({
                    "topic": topic,
                    "difficulty": difficulty,
                    "preferred_formats": formats,
                    "previous_score": score,
                    "learning_goal": goal,
                    "additional_context": extra,
                })
            result = st.session_state["result"]
            st.success("Recommendations ready.")
        except ValidationError as exc:
            for msg in exc.errors:
                st.error(msg)
            result = None
        except Exception as exc:
            log.error("Pipeline failure: %s", type(exc).__name__)
            st.error("Something went wrong while generating recommendations. "
                     "Please try again.")
            result = None

    if not result:
        empty_state(
            "No recommendations yet",
            "Fill in the form above and submit to get your personalized resource list.",
            "🎯",
        )
    else:
        pz = result.get("personalization", {}) or {}
        ranking = result.get("ranking", {}) or {}
        rt = result.get("retrieval", {}) or {}

        c1, c2, c3 = st.columns(3)
        with c1:
            if pz.get("used_llm"):
                st.markdown(badge(f"Personalization: {pz.get('status', 'LLM')}", "success"),
                            unsafe_allow_html=True)
            else:
                st.markdown(badge(f"Personalization: {pz.get('status', 'Fallback')}", "warning"),
                            unsafe_allow_html=True)
        with c2:
            if ranking.get("status") == "llm_used":
                st.markdown(badge(f"Ranking: {ranking.get('engine', 'LLM')}", "success"),
                            unsafe_allow_html=True)
            else:
                st.markdown(badge(f"Ranking: {ranking.get('status', 'Unavailable')}", "error"),
                            unsafe_allow_html=True)
        with c3:
            if rt.get("fallback_used"):
                st.markdown(badge("Retrieval: Local fallback", "warning"), unsafe_allow_html=True)
            else:
                st.markdown(badge("Retrieval: Live API", "success"), unsafe_allow_html=True)

        ctx = (pz.get("context") or {})
        if ctx:
            section_title("Personalized learning guidance")
            st.markdown(
                f'<div class="sc-stat">'
                f'<div class="sc-stat-label">Interpreted topic</div>'
                f'<div style="font-size:1rem;color:#F1F5F9;font-weight:600;margin-bottom:0.85rem">'
                f'{esc(ctx.get("interpreted_topic",""))}</div>'
                + (
                    f'<p style="margin:0 0 0.35rem;color:#A7B4C8;font-size:0.85rem">'
                    f'<strong style="color:#F1F5F9">Suggested difficulty:</strong> '
                    f'{esc(ctx.get("suggested_difficulty",""))}</p>'
                    if ctx.get("suggested_difficulty") else ""
                )
                + (
                    f'<p style="margin:0;color:#A7B4C8;font-size:0.85rem">'
                    f'<strong style="color:#F1F5F9">Strategy:</strong> '
                    f'{esc(ctx.get("learning_strategy",""))}</p>'
                    if ctx.get("learning_strategy") else ""
                )
                + '</div>',
                unsafe_allow_html=True,
            )

        recs = result.get("recommendations", [])
        if not recs:
            empty_state(
                "No resources matched",
                "Try a broader topic, a different difficulty level, or fewer format filters.",
                "🔎",
            )
        else:
            section_title(f"Ranked recommendations ({len(recs)})")
            for rec in recs:
                recommendation_card(rec, result)


# ---------------------------------------------------------------- Resource Library
elif page == "Resource Library":
    hero(
        "RESOURCE LIBRARY",
        "Explore every evaluated resource.",
        "Search, filter, and sort all resources from your most recent search.",
    )

    if not result:
        empty_state(
            "No resources to display",
            "Generate recommendations first to populate the resource library.",
            "📚",
        )
    else:
        df = analytics.results_to_df(result.get("all_scored", []))
        if df.empty:
            empty_state(
                "No resources found",
                "Adjust your topic, difficulty, or preferred format and try again.",
                "📚",
            )
        else:
            f1, f2, f3, f4 = st.columns(4)
            type_opts = sorted(df["Type"].dropna().unique()) if "Type" in df.columns else []
            origin_opts = sorted(df["Origin"].dropna().unique()) if "Origin" in df.columns else []
            diff_opts = sorted(df["Difficulty"].dropna().unique()) if "Difficulty" in df.columns else []

            types = f1.multiselect("Format", type_opts)
            origins = f2.multiselect("Source type", origin_opts)
            diffs = f3.multiselect("Difficulty", diff_opts) if diff_opts else []
            search = f4.text_input("Search title", placeholder="Type to filter…")

            c1, c2 = st.columns(2)
            only_relevant = c1.checkbox("Only topic-relevant resources", value=True)
            sort_by = c2.selectbox("Sort by", ["Score", "Title", "Type"])

            view = df.copy()
            if types and "Type" in view.columns:
                view = view[view["Type"].isin(types)]
            if origins and "Origin" in view.columns:
                view = view[view["Origin"].isin(origins)]
            if diffs and "Difficulty" in view.columns:
                view = view[view["Difficulty"].isin(diffs)]
            if search and "Title" in view.columns:
                view = view[view["Title"].astype(str).str.contains(search, case=False, regex=False)]
            if only_relevant and "Relevant" in view.columns:
                view = view[view["Relevant"]]

            if sort_by in view.columns:
                view = view.sort_values(sort_by, ascending=(sort_by != "Score"))

            st.caption(f"Showing {len(view)} of {len(df)} evaluated resources")

            display_cols = [c for c in
                            ["Title", "Type", "Source", "Origin", "Score", "Difficulty", "URL"]
                            if c in view.columns]
            render_table(view[display_cols],
                         link_columns=["URL"] if "URL" in display_cols else [])


# ---------------------------------------------------------------- Analytics
elif page == "Learning Analytics":
    hero(
        "LEARNING ANALYTICS",
        "Understand your recommendations.",
        "A transparent view of how your latest resources were scored and where "
        "they came from.",
    )

    if not result:
        empty_state(
            "No analytics available",
            "Generate recommendations first to see analytics about your resource "
            "mix and scores.",
            "📊",
        )
    else:
        m = analytics.summary_metrics(result)
        stat_grid([
            ("Recommended", m["recommended"], True),
            ("Evaluated", m["evaluated"], False),
            ("Avg. score", m["avg_score"], False),
            ("Live results", m["live"], False),
            ("Local results", m["local"], False),
        ])

        a, b = st.columns(2)
        dist = analytics.type_distribution(result)
        if not dist.empty:
            with a:
                st.markdown("**Resource formats**")
                dark_bar_chart(dist, height=280)

        with b:
            st.markdown("**Recommendation scores**")
            rdf = analytics.results_to_df(result.get("recommendations", []))
            if not rdf.empty and "Title" in rdf.columns and "Score" in rdf.columns:
                dark_bar_chart(rdf[["Title", "Score"]], x="Title", y="Score", height=280)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        section_title("Component averages")
        comp_df = analytics.mean_components(result)
        if isinstance(comp_df, pd.Series):
            comp_df = comp_df.reset_index()
            comp_df.columns = ["Component", "Score"]
        elif "index" in comp_df.columns:
            comp_df = comp_df.rename(columns={"index": "Component"})
        if comp_df.shape[1] >= 2:
            dark_bar_chart(
                comp_df.iloc[:, :2],
                x=comp_df.columns[0], y=comp_df.columns[1],
                height=280, horizontal=True,
            )


# ---------------------------------------------------------------- AI & API Status
elif page == "AI & API Status":
    hero(
        "AI & API STATUS",
        "Provider and retrieval status.",
        "What powered your last run — personalization, ranking, and retrieval.",
    )

    section_title("Configuration")
    rows = []
    gemini_ok = bool(get_setting("GEMINI_API_KEY"))
    rows.append(("Gemini API key", "Configured" if gemini_ok else "Not configured"))

    try:
        groq_ok = bool(get_setting("GROQ_API_KEY"))
    except Exception:
        groq_ok = False
    rows.append(("Groq API key", "Configured" if groq_ok else "Not configured"))

    try:
        pc = PartnerClient()
        partner_ok = bool(pc.configured)
    except Exception:
        partner_ok = False
    rows.append(("Partner API URL", "Configured" if partner_ok else "Not configured"))

    render_table(pd.DataFrame(rows, columns=["Provider", "Configuration"]))

    if not result:
        st.info("No run yet in this session. Submit a search to see the outcome.")
    else:
        section_title("Last run outcomes")
        pz = result.get("personalization", {}) or {}
        rt = result.get("retrieval", {}) or {}
        ranking = result.get("ranking", {}) or {}
        partner = result.get("partner", {}) or {}

        run_rows = [
            ("Personalization",
             f"{pz.get('status', 'unknown')} — "
             f"{'LLM used' if pz.get('used_llm') else 'fallback'}"),
            ("Ranking", f"{ranking.get('engine', 'unknown')} — {ranking.get('status', 'unknown')}"),
            ("Live retrieval", rt.get("live_status", "unknown")),
            ("Local fallback", "Used" if rt.get("fallback_used") else "Not used"),
            ("Partner API", partner.get("status", "unknown")),
        ]
        render_table(pd.DataFrame(run_rows, columns=["Component", "Outcome"]))

        st.caption(
            "“Configured” only means the credential exists — it does not confirm "
            "the provider is currently reachable. The last-run outcomes above show "
            "what actually happened."
        )


# ---------------------------------------------------------------- Settings
elif page == "Settings":
    hero(
        "SETTINGS",
        "Preferences & session.",
        "Manage your session and display preferences.",
    )

    section_title("Session")
    st.markdown(
        "Clear the current session to remove recommendations, analytics, and "
        "status data. The next search will start fresh."
    )
    if st.button("Reset session", type="secondary"):
        st.session_state["result"] = None
        st.session_state["page"] = "Home"
        st.success("Session reset.")
        st.rerun()

    section_title("Appearance")
    st.markdown(
        "StudyCompass uses a dark theme designed for extended reading and data "
        "analysis. A light theme is not currently available."
    )

    section_title("About")
    st.markdown(
        "- **StudyCompass** — a context-aware study resource recommender.\n"
        "- Personalization and ranking powered by Gemini, with Groq fallback.\n"
        "- Resource retrieval combines live APIs with a curated library.\n"
        "- All scores and explanations are transparent and traceable."
    )