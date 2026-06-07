"""
Shared visual identity + reusable UI components for the Mutation Viewer companion.

Palette inherited from the manuscript figures. Two typefaces:
Source Sans 3 (UI/prose) + IBM Plex Mono (data/coordinates).

The "figure block" is composed from native Streamlit pieces (bordered container +
markdown header/caption/provenance + expander) styled by the CSS below, so plots stay
interactive while matching the high-fidelity mockups.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Sequence

import streamlit as st

# ── Palette (manuscript figures) ──────────────────────────────────────────────
ORANGE = "#DD8D6E"
BEIGE  = "#F0E8D1"
GREEN  = "#558771"
TEAL   = "#82A899"
PURPLE = "#885784"
GOLD   = "#C2AB42"

GREEN_DARK  = "#436b5a"
PURPLE_DARK = "#6e4569"
INK         = "#2A2620"
INK_2       = "#524d44"
MUTED       = "#837c6f"

# Beeswarm diverging scale (SHAP) — matches Figure 2
BEE_INDIGO  = "#5A49C6"
BEE_CREAM   = "#FFF3DB"
BEE_MAGENTA = "#E12A5F"

# ── Glossary — single source of truth, referenced inline via def_chip() ───────
GLOSSARY: dict[str, dict[str, str]] = {
    "predicted probability": {
        "abbr": "",
        "short": "The model's estimated likelihood (0-1) that a tumor carries a somatic "
                 "mutation in the target gene, from its expression profile.",
        "long": "The model's estimated chance, from 0 to 1, that a tumor has a mutation in the "
                "target gene, based only on its RNA expression. Each tumor is scored by a model "
                "that never saw it during training (held-out cross-validation), so the scores are "
                "comparable across genes and cancer types.",
    },
    "prevalence": {
        "abbr": "",
        "short": "The fraction of samples in a cohort that carry a mutation in the target "
                 "gene - the base rate a model must beat.",
        "long": "The fraction of tumors in a cancer type that actually carry a mutation in the "
                "gene. This is the base rate a model has to beat.",
    },
    "AUPRC": {
        "abbr": "area under the PR curve",
        "short": "Area under the precision-recall curve; 0-1, higher is better. Suited to a "
                 "rare positive class.",
        "long": "Area under the precision-recall curve. One score for how well the model ranks "
                "mutated tumors above non-mutated ones, without picking a cutoff. It works well "
                "when mutations are rare. Runs from 0 to 1; higher is better.",
    },
    "normalized AUPRC": {
        "abbr": "(AUPRC - prevalence) / (1 - prevalence)",
        "short": "AUPRC rescaled against the cohort's prevalence baseline, so rare and common "
                 "targets compare on one axis. 0 = random, 1 = perfect.",
        "long": "AUPRC after removing the head start a model gets when a mutation is common: "
                "(AUPRC - prevalence) / (1 - prevalence). It is 0 for random guessing and 1 for a "
                "perfect model, and goes below 0 worse than chance. This is the main score for "
                "comparing genes and cancer types.",
    },
    "SHAP value": {
        "abbr": "",
        "short": "A per-feature contribution to a single prediction. Positive values push "
                 "toward 'mutated'. SHAP describes association, not causation.",
        "long": "How much one gene's expression moved a single prediction. Positive values push "
                "toward 'mutated', negative values push away. SHAP shows association, not cause: an "
                "important gene helps the model predict the mutation, but need not be linked to it "
                "biologically. (Computed with Tree SHAP on a matched XGBoost model.)",
    },
}


def _css() -> str:
    return """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --orange:#DD8D6E; --beige:#F0E8D1; --green:#558771; --teal:#82A899; --purple:#885784; --gold:#C2AB42;
  --green-dark:#436b5a; --green-tint:#e9f0ec; --green-tint-2:#d8e6df;
  --purple-dark:#6e4569; --purple-tint:#f0e8ef; --orange-tint:#f8e7df; --gold-tint:#f4eecf; --teal-tint:#e8efec;
  --bee-indigo:#5A49C6; --bee-cream:#FFF3DB; --bee-magenta:#E12A5F;
  --page:#FBFAF6; --surface:#FFFFFF; --surface-tint:#FAF6EC; --sunk:#F4F1E8;
  --ink:#2A2620; --ink-2:#524d44; --muted:#837c6f; --muted-light:#aaa294;
  --rule:#E7E1D2; --rule-strong:#d8d1bf;
  --shadow-sm:0 1px 2px rgba(60,48,20,0.05);
  --shadow-pop:0 8px 28px rgba(50,40,15,0.13),0 2px 6px rgba(50,40,15,0.07);
  --r-card:7px; --r-ctrl:5px; --r-chip:4px;
  --font-ui:'Source Sans 3',-apple-system,'Segoe UI',sans-serif;
  --font-mono:'IBM Plex Mono','JetBrains Mono',monospace;
  --t-micro:11px;
  --transition:150ms cubic-bezier(0.3,0.1,0.3,1);
}

/* ── Base typography on Streamlit surfaces ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
.stMarkdown, .stMarkdown p, .stMarkdown li,
h1,h2,h3,h4,h5,h6, label, button, input, select, textarea,
[data-testid="stWidgetLabel"], .stSelectbox, .stRadio {
  font-family: var(--font-ui);
}
.stApp { background: var(--page); color: var(--ink); }
.t-mono, .mono { font-family: var(--font-mono) !important; font-feature-settings:"tnum" 1; }

/* tighten default top padding, widen reading column a touch */
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
#MainMenu, footer, [data-testid="stDecoration"] { display:none; }
[data-testid="stHeader"] { background: transparent; }

/* ── Headings / prose primitives ── */
.t-display{ font-size:30px; font-weight:600; letter-spacing:-0.02em; line-height:1.14; color:var(--ink); }
.t-h2{ font-size:21px; font-weight:600; letter-spacing:-0.012em; line-height:1.2; color:var(--ink); }
.t-h3{ font-size:16px; font-weight:600; color:var(--ink); }
.t-sec{ font-size:13.5px; color:var(--ink-2); }
.t-micro{ font-size:var(--t-micro); font-weight:600; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted); }
.lede{ font-size:17px; line-height:1.6; color:var(--ink-2); max-width:66ch; }
.muted{ color:var(--muted); }
.eyebrow{ margin-bottom:6px; }
a { color: var(--green-dark); text-underline-offset:2px; }
a:hover { color: var(--green); }

/* ── Sidebar: brand + native nav ── */
[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--rule); }
/* lift brand (user content) above the auto nav */
[data-testid="stSidebarContent"]{ display:flex !important; flex-direction:column; }
[data-testid="stSidebarContent"] > [data-testid="stSidebarNav"]{ order:2; }
[data-testid="stSidebarContent"] > [data-testid="stSidebarUserContent"]{ order:1; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"]{ padding-top:1rem; }
.brand .nav-foot{ margin-top:8px; }
.brand{ padding:2px 4px 14px; margin-bottom:6px; border-bottom:1px solid var(--rule); }
.brand .mark{ display:flex; align-items:center; gap:9px; font-size:16px; font-weight:700; letter-spacing:-0.02em; color:var(--ink); }
.brand .glyph{ width:22px; height:22px; border-radius:5px; background:var(--green); display:grid; place-items:center; flex:0 0 auto; }
.brand .sub{ font-size:12px; color:var(--muted); margin-top:4px; }
[data-testid="stSidebarNav"] { padding-top:4px; }
[data-testid="stSidebarNav"] ul { gap:1px; }
[data-testid="stSidebarNav"] a { border-radius:var(--r-ctrl); }
[data-testid="stSidebarNav"] a span { font-size:14px; font-weight:500; color:var(--ink-2); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:var(--green-tint); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color:var(--green-dark); font-weight:600; }
.nav-foot{ font-size:12px; color:var(--muted-light); padding:10px 4px 0; line-height:1.7; }
.nav-foot .t-mono{ color:var(--muted); }

/* ── Buttons / download buttons ── */
.stButton>button, .stDownloadButton>button{
  border:1px solid var(--rule-strong); background:var(--surface); color:var(--ink);
  border-radius:var(--r-ctrl); font-size:13.5px; font-weight:500; padding:6px 13px; transition:all var(--transition);
}
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--green); color:var(--green-dark); background:var(--green-tint); }

/* ── Selectbox / radio / inputs ── */
[data-baseweb="select"]>div{ border-radius:var(--r-ctrl); border-color:var(--rule-strong); background:var(--surface); }
[data-testid="stWidgetLabel"] label, .stSelectbox label, .stRadio label{
  font-size:var(--t-micro); font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted);
}

/* segmented control (st.segmented_control / radio horizontal) */
[data-testid="stSegmentedControl"] button[aria-checked="true"]{ background:var(--surface); color:var(--ink); font-weight:600; }

/* ── Bordered container == figure card / card ── */
[data-testid="stVerticalBlockBorderWrapper"]{
  border-color:var(--rule) !important; border-radius:var(--r-card); background:var(--surface);
}

/* ── Expander -> "show data" ── */
[data-testid="stExpander"]{ border:none; }
[data-testid="stExpander"] details{ border:1px solid var(--rule); border-radius:var(--r-ctrl); background:var(--surface); }
[data-testid="stExpander"] summary{ font-size:13px; font-weight:600; color:var(--ink-2); }
[data-testid="stExpander"] summary:hover{ color:var(--green-dark); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{ gap:22px; border-bottom:1px solid var(--rule); }
.stTabs [data-baseweb="tab"]{ font-size:14px; font-weight:500; color:var(--muted); padding:0 0 10px; }
.stTabs [aria-selected="true"]{ color:var(--green-dark); }

/* ── Dataframe ── */
[data-testid="stDataFrame"]{ border:1px solid var(--rule); border-radius:var(--r-ctrl); }

/* ============================================================
   Custom HTML components (emitted via st.markdown)
   ============================================================ */
.page-head{ margin-bottom:26px; }
.page-head .lede{ margin-top:10px; }

/* figure header row (Fig chip + name) */
.fhead{ display:flex; align-items:baseline; gap:9px; min-width:0; }
.fhead .fno{ font-family:var(--font-mono); font-size:11px; font-weight:600; color:var(--green-dark);
  background:var(--green-tint); padding:2px 7px; border-radius:var(--r-chip); flex:0 0 auto; }
.fhead .fname{ font-size:14.5px; font-weight:600; color:var(--ink); }

/* caption */
.fcap{ font-size:13.5px; color:var(--ink-2); line-height:1.55; max-width:80ch; margin-top:2px; }
.fcap .capnum{ font-weight:600; color:var(--ink); }

/* provenance band */
.provenance{ display:flex; align-items:center; flex-wrap:wrap; gap:0 14px; padding:9px 12px; margin-top:10px;
  background:var(--surface-tint); border:1px solid var(--rule); border-radius:var(--r-ctrl);
  font-family:var(--font-mono); font-size:11.5px; color:var(--muted); }
.provenance .pv{ display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
.provenance .pv b{ color:var(--ink-2); font-weight:600; }
.provenance .dot{ width:3px; height:3px; border-radius:50%; background:var(--muted-light); }

/* definition chip */
.def{ border-bottom:1.5px dotted var(--green); cursor:help; font-weight:500; }

/* glossary */
.glossary{ display:grid; gap:0; }
.gterm{ display:grid; grid-template-columns:220px 1fr; gap:24px; padding:15px 0; border-bottom:1px solid var(--rule); align-items:start; }
.gterm:last-child{ border-bottom:none; }
.gterm dt{ font-weight:600; color:var(--ink); }
.gterm dt .abbr{ font-family:var(--font-mono); font-size:12px; color:var(--green-dark); display:block; margin-top:2px; font-weight:500; }
.gterm dd{ color:var(--ink-2); font-size:14px; margin:0; }

/* notices */
.notice{ display:flex; gap:12px; align-items:flex-start; padding:14px 16px; border-radius:var(--r-card);
  border:1px solid var(--rule); background:var(--surface-tint); }
.notice .nico{ width:30px; height:30px; border-radius:7px; flex:0 0 auto; display:grid; place-items:center; background:var(--sunk); color:var(--muted); font-size:16px; }
.notice .ntitle{ font-weight:600; font-size:14px; color:var(--ink); margin-bottom:2px; }
.notice .nbody{ font-size:13.5px; color:var(--ink-2); line-height:1.5; }
.notice.neutral, .notice.empty{ background:var(--sunk); } .notice.neutral .nico, .notice.empty .nico{ background:#ece8db; }
.notice.gold, .notice.warning{ background:var(--gold-tint); border-color:#e7dcab; } .notice.gold .nico, .notice.warning .nico{ background:#ecdf9f; color:#7d6c1b; }
.notice.orange{ background:var(--orange-tint); border-color:#eecdbf; } .notice.orange .nico{ background:#f0c9b6; color:#9c4f2f; }
.notice.purple{ background:var(--purple-tint); border-color:#e3cfe0; } .notice.purple .nico{ background:#e1cadd; color:var(--purple-dark); }

/* pinned causal caveat */
.caveat{ display:flex; gap:12px; align-items:flex-start; padding:13px 16px; border-radius:var(--r-card);
  background:var(--purple-tint); border:1px solid #e0cadd; border-left:3px solid var(--purple); }
.caveat .cico{ color:var(--purple-dark); flex:0 0 auto; font-size:18px; line-height:1.2; }
.caveat .ctext{ font-size:13.5px; color:var(--purple-dark); line-height:1.5; }
.caveat .ctext b{ font-weight:600; }

/* coverage table */
.coverage{ border:1px solid var(--rule); border-radius:var(--r-card); overflow:hidden; background:var(--surface); }
.cov-table{ width:100%; border-collapse:collapse; font-size:13px; }
.cov-table th{ font-size:var(--t-micro); font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted);
  padding:10px 12px; background:var(--sunk); border-bottom:1px solid var(--rule); }
.cov-table th.colh{ text-align:center; }
.cov-table td{ padding:9px 12px; border-bottom:1px solid var(--rule); }
.cov-table tr:last-child td{ border-bottom:none; }
.cov-table .rowh{ font-weight:600; color:var(--ink); white-space:nowrap; }
.cov-table .rowh .sub{ font-weight:400; color:var(--muted); font-size:12px; }
.cov-table td.cell{ text-align:center; }
.cov-cell{ display:inline-flex; align-items:center; justify-content:center; gap:6px; font-family:var(--font-mono);
  font-size:11.5px; font-weight:500; padding:3px 9px; border-radius:20px; min-width:56px; }
.cov-cell .pip{ width:8px; height:8px; border-radius:50%; flex:0 0 auto; }
.cov-cell.full{ background:var(--green-tint-2); color:var(--green-dark); } .cov-cell.full .pip{ background:var(--green); }
.cov-cell.partial{ background:var(--gold-tint); color:#7d6c1b; } .cov-cell.partial .pip{ background:var(--gold); }
.cov-cell.none{ background:var(--sunk); color:var(--muted-light); } .cov-cell.none .pip{ background:transparent; border:1.5px solid var(--muted-light); }

/* misc chips / keyvals / manifest */
.chip{ display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:500; padding:3px 9px; border-radius:20px; background:var(--sunk); color:var(--ink-2); }
.chip .swatch{ width:9px; height:9px; border-radius:50%; }
.kv{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; font-size:13.5px; }
.kv dt{ color:var(--muted); } .kv dd{ color:var(--ink); margin:0; }
.kv dd.mono{ font-family:var(--font-mono); font-size:12.5px; }
.manifest{ font-family:var(--font-mono); font-size:12.5px; line-height:1.9; color:var(--ink-2); }
.manifest .mk{ color:var(--muted); } .manifest .mv{ color:var(--ink); } .manifest .hash{ color:var(--purple-dark); }
.pred-card{ display:grid; grid-template-columns:auto 1fr auto 1fr auto; gap:16px; align-items:center; padding:20px;
  background:var(--surface-tint); border:1px solid var(--rule); border-radius:var(--r-card); text-align:center; }
.pred-card .lbl{ font-size:var(--t-micro); text-transform:uppercase; letter-spacing:.08em; color:var(--muted); font-weight:600; }
.pred-card .val{ font-size:15px; font-weight:600; color:var(--ink); margin-top:4px; }
.pred-card .arr{ color:var(--green); font-size:20px; }
"""


# ── Injection ────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def esc(s: object) -> str:
    return _html.escape(str(s))


# ── Page header ──────────────────────────────────────────────────────────────
def page_head(eyebrow: str, title: str, lede_html: str) -> None:
    md(
        f'<div class="page-head">'
        f'<div class="t-micro eyebrow">{esc(eyebrow)}</div>'
        f'<div class="t-display">{esc(title)}</div>'
        f'<div class="lede">{lede_html}</div>'
        f"</div>"
    )


# ── Inline definition chip (hover tooltip via title attr) ────────────────────
def def_chip(term: str, label: str | None = None) -> str:
    entry = GLOSSARY.get(term.lower()) or GLOSSARY.get(term)
    tip = entry["short"] if entry else ""
    text = label or term
    return f'<span class="def" title="{esc(tip)}">{esc(text)}</span>'


# ── Figure-block pieces (compose inside a bordered st.container) ──────────────
def fig_header(fno: str, name: str) -> str:
    return f'<div class="fhead"><span class="fno">{esc(fno)}</span><span class="fname">{esc(name)}</span></div>'


def fig_caption(num: str, text_html: str) -> None:
    md(f'<div class="fcap"><span class="capnum">{esc(num)}</span> {text_html}</div>')


def provenance(pairs: Sequence[tuple[str, str]]) -> None:
    parts = []
    for i, (label, value) in enumerate(pairs):
        if i:
            parts.append('<span class="dot"></span>')
        parts.append(f'<span class="pv"><b>{esc(label)}</b> {esc(value)}</span>')
    md(f'<div class="provenance">{"".join(parts)}</div>')


# ── Notices / caveat ─────────────────────────────────────────────────────────
def notice(kind: str, title: str, body_html: str, icon: str | None = None) -> None:
    if icon is None:
        icon = {"warning": "⚠", "gold": "⚠", "orange": "⚠",
                "empty": "∅", "neutral": "i"}.get(kind, "i")
    md(
        f'<div class="notice {esc(kind)}"><span class="nico">{esc(icon)}</span>'
        f'<div class="ntext"><div class="ntitle">{esc(title)}</div>'
        f'<div class="nbody">{body_html}</div></div></div>'
    )


def caveat() -> None:
    md(
        '<div class="caveat"><span class="cico">&#9888;</span>'
        '<div class="ctext"><b>SHAP identifies predictive associations, not causal '
        'regulation.</b> An important feature helps the model predict mutation status in this '
        'cohort - it does not imply the feature regulates, or is regulated by, the mutation.'
        "</div></div>"
    )


# ── Glossary block ───────────────────────────────────────────────────────────
def glossary_block() -> None:
    rows = []
    for term, e in GLOSSARY.items():
        label = term[0].upper() + term[1:]
        abbr = f'<span class="abbr">{esc(e["abbr"])}</span>' if e["abbr"] else ""
        rows.append(f"<div class='gterm'><dt>{esc(label)}{abbr}</dt><dd>{esc(e['long'])}</dd></div>")
    md(f'<dl class="glossary">{"".join(rows)}</dl>')


# ── Coverage matrix ──────────────────────────────────────────────────────────
def _cov_cell(state: str, text: str) -> str:
    return f'<span class="cov-cell {state}"><span class="pip"></span>{esc(text)}</span>'


def coverage_table(rows: Iterable[dict]) -> None:
    """rows: dicts with cohort, label, and {perf,variant,shap,bee} each = (state, text)."""
    body = []
    for r in rows:
        sub = f' <span class="sub">{esc(r["label"])}</span>' if r.get("label") else ""
        cells = "".join(
            f'<td class="cell">{_cov_cell(*r[k])}</td>'
            for k in ("perf", "variant", "shap", "bee")
        )
        body.append(f'<tr><td class="rowh">{esc(r["cohort"])}{sub}</td>{cells}</tr>')
    md(
        '<div class="coverage"><table class="cov-table"><thead><tr>'
        '<th style="text-align:left">Cohort</th><th class="colh">Performance</th>'
        '<th class="colh">Variant data</th><th class="colh">SHAP features</th>'
        '<th class="colh">Beeswarm</th></tr></thead><tbody>'
        f'{"".join(body)}</tbody></table></div>'
    )
    md(
        '<div class="row" style="display:flex;gap:16px;margin-top:10px;font-size:12.5px;color:var(--muted)">'
        f'{_cov_cell("full","Available")} {_cov_cell("partial","Partial")} '
        f'{_cov_cell("none","Not yet available")}</div>'
    )
