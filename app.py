"""
Mutation Viewer - companion to the Expression-to-Mutation manuscript.

Predicts somatic mutation status from bulk tumor RNA expression, per TCGA cohort.
This app lets readers explore the data behind the figures with more control.

Run:  streamlit run app.py   (or: uv run streamlit run app.py)
"""

from __future__ import annotations

import streamlit as st

import data
import theme
from pages_ import about, mutation, performance, shap

st.set_page_config(
    page_title="Mutation Viewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject_css()

# ── Branded sidebar header ───────────────────────────────────────────────────
GLYPH = (
    '<span class="glyph"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="#fff" stroke-width="2.4" stroke-linecap="round">'
    '<path d="M3 12c4-8 14-8 18 0M3 12c4 8 14 8 18 0"/></svg></span>'
)
meta = data.bundle_meta()
with st.sidebar:
    theme.md(
        f'<div class="brand"><div class="mark">{GLYPH} Mutation Viewer</div>'
        '<div class="sub">Companion to the manuscript</div>'
        f'<div class="nav-foot">Data bundle <span class="t-mono">{theme.esc(meta["generated"])}</span>'
        f' &middot; sha <span class="t-mono">{theme.esc(meta["sha7"])}</span></div></div>'
    )

# ── Pages ────────────────────────────────────────────────────────────────────
pages = {
    "Explore": [
        st.Page(performance.render, title="Model performance",
                icon=":material/bar_chart:", url_path="performance"),
        st.Page(mutation.render, title="Mutation viewer",
                icon=":material/biotech:", url_path="mutation-viewer"),
        st.Page(shap.render, title="SHAP explorer",
                icon=":material/scatter_plot:", url_path="shap"),
    ],
    "Reference": [
        st.Page(about.render, title="About / Model card",
                icon=":material/info:", url_path="about", default=True),
    ],
}
nav = st.navigation(pages)
nav.run()
