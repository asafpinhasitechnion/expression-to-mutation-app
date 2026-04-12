"""
Transcript Mutation Viewer — Streamlit app
==========================================

Visualize per-variant predicted probabilities on a transcript model.

Usage
-----
From the repo root:
    streamlit run app.py

Adding new datasets
-------------------
Drop any .parquet file into mutation_viewer/data/ and add an entry to
DATA_CATALOG below.  It will appear automatically as a checkbox in the sidebar.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from transcript_utils import (
    load_gtf,
    build_transcript_model,
    genomic_to_transcript_coord,
    hgvsc_to_tx_coord,
    load_fasta_sequence,
    translate_cds,
    strip_version,
    normalize_chrom,
)

# ── Data catalogue ───────────────────────────────────────────────────────────
# Maps filename (inside mutation_viewer/data/) to the label shown in the UI.
# Files in data/ that are NOT listed here are shown with their stem as the label.

DATA_DIR     = Path(__file__).parent / "data"
SHAP_APP_DIR = Path(__file__).parent / "shap_app"

# ── Style constants ──────────────────────────────────────────────────────────
from constants import (                                   # noqa: E402
    CMAP_ORANGE_GREEN,
    AXIS_LABEL_FS, TICK_FS, TITLE_FS,
    COLORBAR_LABEL_FS, COLORBAR_TICK_FS,
    make_linear_cmap,
)

# Beeswarm colormap matching Figure 2 notebook exactly
CMAP_ORANGE_PURPLE_VIVID = make_linear_cmap(
    ["#5A49C6", "#FFF3DB", "#E12A5F"], name="orange_purple_vivid"
)

DATA_CATALOG: dict[str, str] = {
    "missense_mutations.parquet": "Missense mutations",
    "silent_mutations.parquet":   "Silent mutations",
}


def available_datasets() -> list[tuple[str, str]]:
    """Return (filename, display_label) pairs for every parquet in DATA_DIR."""
    if not DATA_DIR.exists():
        return []
    files = sorted(DATA_DIR.glob("*.parquet"))
    return [
        (f.name, DATA_CATALOG.get(f.name, f.stem.replace("_", " ").title()))
        for f in files
    ]


# ── Column detection ──────────────────────────────────────────────────────────

# For each logical field, ordered list of candidate column names (preferred first)
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "chrom":       ["Chromosome", "Chr", "CHROM", "chrom"],
    "pos":         ["Start_Position", "Start", "POS", "pos", "position"],
    "ref":         ["Reference_Allele", "REF", "ref"],
    "alt":         ["Tumor_Seq_Allele2", "Tumor_Seq_Allele1", "ALT", "alt"],
    "gene":        ["Hugo_Symbol", "Gene", "gene", "gene_name"],
    "transcript":  ["Transcript_ID", "transcript_id", "Feature"],
    "hgvsp":       ["HGVSp_Short", "HGVSp"],
    "hgvsc":       ["HGVSc"],
    "impact":      ["IMPACT", "Impact"],
    "consequence": ["Consequence", "Variant_Classification"],
    "pred_prob":   ["pred_prob", "PredProb", "prediction"],
    "cancer":      ["Cancer", "cancer_type", "Cancer_Type", "project_id"],
    "sample":      ["Tumor_Sample_Barcode", "sample_id", "Sample"],
    "codons":      ["Codons", "Codon_Change"],
}

# Impact → color  (covers both VEP IMPACT and MAF Variant_Classification)
IMPACT_COLORS: dict[str, str] = {
    # VEP IMPACT
    "HIGH":     "#C0392B",
    "MODERATE": "#E67E22",
    "LOW":      "#27AE60",
    "MODIFIER": "#7F8C8D",
    # MAF Variant_Classification
    "Frame_Shift_Del":   "#C0392B",
    "Frame_Shift_Ins":   "#C0392B",
    "Nonsense_Mutation": "#C0392B",
    "Splice_Site":       "#C0392B",
    "Missense_Mutation": "#E67E22",
    "In_Frame_Del":      "#E67E22",
    "In_Frame_Ins":      "#E67E22",
    "Silent":            "#27AE60",
    "RNA":               "#7F8C8D",
    "Intron":            "#7F8C8D",
    "3'UTR":             "#7F8C8D",
    "5'UTR":             "#7F8C8D",
}
DEFAULT_COLOR = "#95A5A6"

# Consequence keyword fallbacks (substring match)
CONSEQUENCE_KEYWORDS: list[tuple[str, str]] = [
    ("stop_gained",   "#C0392B"),
    ("frameshift",    "#C0392B"),
    ("splice",        "#C0392B"),
    ("missense",      "#E67E22"),
    ("inframe",       "#E67E22"),
    ("synonymous",    "#27AE60"),
    ("silent",        "#27AE60"),
]


def detect_col(df: pd.DataFrame, key: str) -> str | None:
    for c in COLUMN_CANDIDATES[key]:
        if c in df.columns:
            return c
    return None


def detect_all_cols(df: pd.DataFrame) -> dict[str, str | None]:
    return {k: detect_col(df, k) for k in COLUMN_CANDIDATES}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading mutation data...")
def load_datasets(filenames: tuple[str, ...]) -> pd.DataFrame:
    """Load and concatenate one or more files from DATA_DIR."""
    frames: list[pd.DataFrame] = []
    for fname in filenames:
        p = DATA_DIR / fname
        if p.suffix == ".parquet":
            frames.append(pd.read_parquet(p))
        else:
            sep = "\t" if (p.suffix in (".tsv", ".gz") or "maf" in p.name.lower()) else ","
            frames.append(pd.read_csv(p, sep=sep, low_memory=False))
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


# ── Variant aggregation ───────────────────────────────────────────────────────

def aggregate_variants(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    metric: str = "mean",
    group_by_cancer: bool = False,
) -> pd.DataFrame:
    """
    Aggregate multiple sample rows for the same genomic variant.

    Variant key: (chrom, pos, ref, alt).
    Result always has '_agg_prob' and '_count' columns.
    """
    key_cols = [cols[k] for k in ("chrom", "pos", "ref", "alt") if cols[k]]

    if not key_cols:
        out = df.copy()
        if cols["pred_prob"]:
            out["_agg_prob"] = out[cols["pred_prob"]]
        else:
            out["_agg_prob"] = np.nan
        out["_count"] = 1
        return out

    group_cols = key_cols[:]
    if group_by_cancer and cols["cancer"] and cols["cancer"] in df.columns:
        group_cols.append(cols["cancer"])

    # Annotation columns: carry over first non-null value per variant
    annot_keys = ["hgvsp", "hgvsc", "impact", "consequence", "gene", "transcript", "codons"]
    annot_cols = [cols[k] for k in annot_keys if cols[k] and cols[k] in df.columns]

    agg_spec: dict[str, object] = {}
    for c in annot_cols:
        agg_spec[c] = "first"
    if cols["pred_prob"] and cols["pred_prob"] in df.columns:
        fn = np.mean if metric == "mean" else np.median
        agg_spec[cols["pred_prob"]] = [fn, "count"]

    agg = df.groupby(group_cols, dropna=False).agg(agg_spec)

    # Flatten MultiIndex columns.
    # Annotation columns use "first" — drop the suffix to preserve original names
    # (e.g. ("HGVSc", "first") → "HGVSc", ("pred_prob", "mean") → "pred_prob_mean").
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            func = str(c[1]) if len(c) > 1 else ""
            if func == "first":
                new_cols.append(str(c[0]))
            else:
                new_cols.append("_".join(str(x) for x in c if x).rstrip("_"))
        else:
            new_cols.append(c)
    agg.columns = new_cols
    agg = agg.reset_index()

    # Normalise aggregated pred_prob column name
    prob_col = cols["pred_prob"]
    if prob_col:
        metric_col  = f"{prob_col}_{metric}"  # e.g. pred_prob_mean
        # pandas agg may name it differently; search for it
        for candidate in [metric_col, f"{prob_col}_<lambda>", f"{prob_col}_fn"]:
            if candidate in agg.columns:
                agg["_agg_prob"] = agg[candidate]
                break
        else:
            # Fallback: find the first float column that isn't a key
            for c in agg.columns:
                if c not in group_cols + annot_cols and agg[c].dtype.kind == "f":
                    agg["_agg_prob"] = agg[c]
                    break
            else:
                agg["_agg_prob"] = np.nan

        count_col = f"{prob_col}_count"
        if count_col in agg.columns:
            agg["_count"] = agg[count_col]
        else:
            agg["_count"] = 1
    else:
        agg["_agg_prob"] = np.nan
        agg["_count"] = 1

    return agg


# ── Variant helpers ───────────────────────────────────────────────────────────

def _variant_color(row: pd.Series, cols: dict[str, str | None]) -> str:
    for col_key in ("impact", "consequence"):
        c = cols[col_key]
        if not c or c not in row.index:
            continue
        val = str(row[c]) if pd.notna(row[c]) else ""
        if not val or val in ("nan", "."):
            continue
        if val in IMPACT_COLORS:
            return IMPACT_COLORS[val]
        val_lower = val.lower()
        for kw, color in CONSEQUENCE_KEYWORDS:
            if kw in val_lower:
                return color
    return DEFAULT_COLOR


def _variant_label(row: pd.Series, cols: dict[str, str | None]) -> str:
    for col_key in ("hgvsp", "hgvsc"):
        c = cols[col_key]
        if c and c in row.index:
            v = str(row[c]) if pd.notna(row[c]) else ""
            if v and v not in ("nan", "."):
                return v
    ref = str(row[cols["ref"]]) if cols["ref"] and cols["ref"] in row.index else ""
    alt = str(row[cols["alt"]]) if cols["alt"] and cols["alt"] in row.index else ""
    return f"{ref}>{alt}" if ref and alt else ""


def _variant_hover(row: pd.Series, cols: dict[str, str | None], metric: str) -> str:
    parts: list[str] = [f"<b>{_variant_label(row, cols)}</b>"]
    if "_tx_coord" in row.index and pd.notna(row["_tx_coord"]):
        parts.append(f"Transcript pos: {int(row['_tx_coord'])}")
    if "_agg_prob" in row.index and pd.notna(row["_agg_prob"]):
        parts.append(f"pred_prob ({metric}): {row['_agg_prob']:.3f}")
    if "_count" in row.index:
        parts.append(f"n samples: {int(row['_count'])}")
    for col_key in ("consequence", "impact", "cancer"):
        c = cols[col_key]
        if c and c in row.index and pd.notna(row[c]):
            parts.append(f"{col_key.capitalize()}: {row[c]}")
    codons_col = cols.get("codons")
    if codons_col and codons_col in row.index and pd.notna(row.get(codons_col)):
        raw = str(row[codons_col])
        if "/" in raw:
            ref_c, alt_c = raw.split("/", 1)
            parts.append(f"Codon: {ref_c.upper()} → {alt_c.upper()}")
        else:
            parts.append(f"Codon: {raw}")
    return "<br>".join(parts)


# ── Plotly figure ─────────────────────────────────────────────────────────────

def build_figure(
    tx_model: dict,
    mut_agg: pd.DataFrame,
    cols: dict[str, str | None],
    metric: str,
    highlight_mask: pd.Series | None = None,
    highlight_label: str | None = None,
) -> go.Figure:
    """
    Two-row plotly figure:
      Row 1 — exon/CDS transcript structure
      Row 2 — mutation scatter (x = transcript coord, size ∝ pred_prob, color = impact)
    """
    tx_len    = tx_model["tx_length"]
    cds_start = tx_model.get("cds_start_tx")
    cds_end   = tx_model.get("cds_end_tx")

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.28, 0.72],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Transcript structure (exons | CDS)", "Mutations"],
    )

    # ── Row 1: transcript structure ───────────────────────────────────────────
    # Intron backbone
    fig.add_trace(
        go.Scatter(
            x=[0, tx_len], y=[0.5, 0.5],
            mode="lines",
            line=dict(color="#444444", width=2),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Exons — drawn as filled polygons via scatter
    offset = 0
    for s, e in tx_model["exons"]:
        ex_len = int(e) - int(s) + 1
        x0, x1 = offset, offset + ex_len
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[0.2, 0.2, 0.8, 0.8, 0.2],
                mode="lines",
                fill="toself",
                fillcolor="#82A899",
                line=dict(color="#558771", width=1),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )
        offset += ex_len

    # CDS overlay (semi-transparent purple)
    if cds_start is not None and cds_end is not None:
        fig.add_trace(
            go.Scatter(
                x=[cds_start, cds_end, cds_end, cds_start, cds_start],
                y=[0.1, 0.1, 0.9, 0.9, 0.1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(136,87,132,0.25)",
                line=dict(color="#885784", width=1),
                name="CDS",
                showlegend=True,
            ),
            row=1, col=1,
        )

    fig.update_yaxes(range=[0, 1], showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    # ── Row 2: mutation scatter — y = pred_prob (0→1 calibrated axis) ──────────
    if not mut_agg.empty and "_tx_coord" in mut_agg.columns:
        plot_df = mut_agg.dropna(subset=["_tx_coord"]).copy()

        if not plot_df.empty:
            probs = (
                plot_df["_agg_prob"].fillna(0.5).values
                if "_agg_prob" in plot_df.columns and plot_df["_agg_prob"].notna().any()
                else np.full(len(plot_df), 0.5)
            )
            # Marker size ∝ sample count (position already encodes probability)
            counts_col = (
                plot_df["_count"].values
                if "_count" in plot_df.columns
                else np.ones(len(plot_df))
            )
            c_max = float(counts_col.max())
            sizes = 6 + 10 * (counts_col / c_max) if c_max > 0 else np.full(len(plot_df), 9.0)

            colors = [_variant_color(row, cols) for _, row in plot_df.iterrows()]
            labels = [_variant_label(row, cols) for _, row in plot_df.iterrows()]
            hovers = [_variant_hover(row, cols, metric) for _, row in plot_df.iterrows()]

            tx_coords = plot_df["_tx_coord"].values.astype(float)
            y_vals    = np.clip(probs, 0.0, 1.0)

            fig.add_trace(
                go.Scatter(
                    x=tx_coords,
                    y=y_vals,
                    mode="markers+text",
                    marker=dict(
                        size=sizes,
                        color=colors,
                        opacity=0.85,
                        line=dict(color="white", width=0.5),
                    ),
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=7),
                    hovertext=hovers,
                    hoverinfo="text",
                    name="variants",
                    showlegend=False,
                ),
                row=2, col=1,
            )

            if highlight_mask is not None:
                highlight_mask = highlight_mask.reindex(plot_df.index, fill_value=False)
                hi_df = plot_df.loc[highlight_mask]
                if not hi_df.empty:
                    hi_probs = (
                        hi_df["_agg_prob"].fillna(0.5).values
                        if "_agg_prob" in hi_df.columns and hi_df["_agg_prob"].notna().any()
                        else np.full(len(hi_df), 0.5)
                    )
                    hi_counts = hi_df["_count"].values if "_count" in hi_df.columns else np.ones(len(hi_df))
                    hi_cmax = float(hi_counts.max()) if len(hi_counts) else 0.0
                    hi_sizes = 10 + 14 * (hi_counts / hi_cmax) if hi_cmax > 0 else np.full(len(hi_df), 14.0)
                    hi_colors = [_variant_color(row, cols) for _, row in hi_df.iterrows()]
                    hi_labels = [_variant_label(row, cols) for _, row in hi_df.iterrows()]
                    hi_hovers = [_variant_hover(row, cols, metric) for _, row in hi_df.iterrows()]

                    fig.add_trace(
                        go.Scatter(
                            x=hi_df["_tx_coord"].values.astype(float),
                            y=np.clip(hi_probs, 0.0, 1.0),
                            mode="markers+text",
                            marker=dict(
                                size=hi_sizes,
                                color=hi_colors,
                                opacity=1.0,
                                symbol="diamond",
                                line=dict(color="black", width=2),
                            ),
                            text=hi_labels,
                            textposition="top center",
                            textfont=dict(size=8, color="black"),
                            hovertext=hi_hovers,
                            hoverinfo="text",
                            name=highlight_label or "flagged",
                            showlegend=bool(highlight_label),
                        ),
                        row=2, col=1,
                    )

            fig.update_yaxes(
                range=[-0.08, 1.12],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1.0"],
                title_text="pred_prob",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.4)",
                zeroline=True,
                zerolinecolor="rgba(150,150,150,0.5)",
                row=2, col=1,
            )

    fig.update_xaxes(title_text="Transcript coordinate (nt)", row=2, col=1)
    fig.update_layout(
        height=580,
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.04, x=0),
    )

    return fig


# ── Aligned sequence viewer ───────────────────────────────────────────────────

# ── Summary panel helpers ─────────────────────────────────────────────────────

_AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}

_TRANSITIONS = {"A>G", "G>A", "C>T", "T>C"}


def _extract_sub(row: pd.Series, cols: dict) -> str | None:
    """Return 'X>Y' nucleotide substitution from HGVSc or ref/alt columns."""
    hc = cols.get("hgvsc")
    if hc and hc in row.index and pd.notna(row[hc]):
        m = re.search(r"([ACGT])>([ACGT])", str(row[hc]).upper())
        if m:
            return f"{m.group(1)}>{m.group(2)}"
    rc, ac = cols.get("ref"), cols.get("alt")
    if rc and ac and rc in row.index and ac in row.index:
        ref = str(row[rc]).strip().upper()
        alt = str(row[ac]).strip().upper()
        if len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT" and ref != alt:
            return f"{ref}>{alt}"
    return None


def _parse_aa_parts(row: pd.Series, cols: dict) -> tuple[str | None, str | None]:
    """Return (ref_aa, alt_aa) as single-letter codes, or (None, None) if unparseable."""
    hc = cols.get("hgvsp")
    if not hc or hc not in row.index:
        return None, None
    val = str(row[hc]) if pd.notna(row.get(hc)) else ""
    if not val or val in ("nan", "."):
        return None, None
    # 3-letter codes: p.Arg248Trp / p.Arg248Arg / p.Arg248Ter
    m = re.match(r"p\.([A-Z][a-z]{2})\d+([A-Z][a-z]{2}|Ter)", val)
    if m:
        return _AA3TO1.get(m.group(1), m.group(1)), _AA3TO1.get(m.group(2), m.group(2))
    # 1-letter codes: p.R248W / p.R248R / p.R248*
    m = re.match(r"p\.([A-Z*])\d+([A-Z*])", val)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _extract_aa_change(row: pd.Series, cols: dict) -> str | None:
    """Return 'X>Y' single-letter amino-acid change, or None (excludes synonymous)."""
    ref_aa, alt_aa = _parse_aa_parts(row, cols)
    if ref_aa is None or alt_aa is None or ref_aa == alt_aa:
        return None
    return f"{ref_aa}>{alt_aa}"


def _build_aa_labels(
    mut_agg: pd.DataFrame,
    cols: dict[str, str | None],
    aa_mode: str,
    include_silent: bool,
) -> pd.Series:
    labels: list[str | None] = []
    for _, row in mut_agg.iterrows():
        ref_aa, alt_aa = _parse_aa_parts(row, cols)
        if ref_aa is None:
            labels.append(None)
            continue
        is_silent = ref_aa == alt_aa
        if is_silent and not include_silent:
            labels.append(None)
            continue
        if aa_mode == "source":
            labels.append(ref_aa)
        elif aa_mode == "target":
            labels.append(alt_aa)
        else:
            labels.append(f"{ref_aa}>{alt_aa}")
    return pd.Series(labels, index=mut_agg.index, dtype="object")


def get_flag_options(
    mut_agg: pd.DataFrame,
    cols: dict[str, str | None],
    aa_mode: str,
    include_silent: bool,
) -> tuple[list[str], list[str], list[str]]:
    aa_options: list[str] = []
    sub_options: list[str] = []
    cancer_options: list[str] = []

    if "_agg_prob" in mut_agg.columns:
        aa_labels = _build_aa_labels(mut_agg, cols, aa_mode, include_silent)
        aa_options = sorted(aa_labels.dropna().astype(str).unique().tolist())

        sub_labels = mut_agg.apply(lambda r: _extract_sub(r, cols), axis=1)
        sub_options = sorted(sub_labels.dropna().astype(str).unique().tolist())

    cancer_col = cols.get("cancer")
    if cancer_col and cancer_col in mut_agg.columns:
        cancer_options = sorted(mut_agg[cancer_col].dropna().astype(str).unique().tolist())

    return aa_options, sub_options, cancer_options


def build_highlight_mask(
    mut_agg: pd.DataFrame,
    cols: dict[str, str | None],
    aa_mode: str,
    include_silent: bool,
    aa_flag: str,
    sub_flag: str,
    cancer_flag: str,
) -> tuple[pd.Series, str | None]:
    mask = pd.Series(False, index=mut_agg.index)
    active: list[str] = []

    if aa_flag != "None":
        aa_labels = _build_aa_labels(mut_agg, cols, aa_mode, include_silent)
        mask |= aa_labels.eq(aa_flag)
        active.append(f"AA {aa_flag}")

    if sub_flag != "None":
        sub_labels = mut_agg.apply(lambda r: _extract_sub(r, cols), axis=1)
        mask |= sub_labels.eq(sub_flag)
        active.append(f"Sub {sub_flag}")

    cancer_col = cols.get("cancer")
    if cancer_flag != "None" and cancer_col and cancer_col in mut_agg.columns:
        mask |= mut_agg[cancer_col].astype(str).eq(cancer_flag)
        active.append(f"Cancer {cancer_flag}")

    label = " | ".join(active) if active else None
    return mask, label


def build_summary_panels(
    mut_agg: pd.DataFrame,
    cols: dict[str, str | None],
    metric: str,
    tx_model: dict | None = None,
    smooth_window: int = 50,
    aa_mode: str = "change",       # "change" | "source" | "target"
    include_silent: bool = True,
) -> go.Figure:
    """
    Four parallel summary panels:
      1. Smoothed mean pred_prob along the transcript coordinate
      2. Mean pred_prob by amino-acid grouping (change / source AA / target AA)
      3. Mean pred_prob by nucleotide substitution type (ts vs tv coloured)
      4. pred_prob distribution per cancer (box + strip; needs group_by_cancer enabled)
    """
    cancer_col = cols.get("cancer")
    has_cancer = bool(cancer_col and cancer_col in mut_agg.columns)

    aa_titles = {"change": "By AA change", "source": "By source AA", "target": "By target AA"}

    fig = make_subplots(
        rows=1, cols=4,
        column_widths=[0.37, 0.17, 0.17, 0.29],
        subplot_titles=[
            f"Smoothed mean pred_prob (window ≈{smooth_window} nt)",
            aa_titles.get(aa_mode, "By AA"),
            "By substitution type",
            "Per cancer" if has_cancer else "Per cancer (enable 'Group by cancer')",
        ],
        horizontal_spacing=0.08,
    )

    COLOR_LINE = "#885784"
    COLOR_FILL = "rgba(136,87,132,0.15)"
    COLOR_DOT  = "rgba(136,87,132,0.25)"
    COLOR_TS   = "#885784"   # transitions — purple
    COLOR_TV   = "#558771"   # transversions — green
    COLOR_BAR  = "#82A899"

    # ── Panel 1: smoothed mean across transcript coordinate ───────────────────
    if "_tx_coord" in mut_agg.columns and "_agg_prob" in mut_agg.columns:
        mapped = mut_agg.dropna(subset=["_tx_coord", "_agg_prob"])
        if not mapped.empty:
            tx_len = tx_model["tx_length"] if tx_model else int(mapped["_tx_coord"].max()) + 1
            all_x  = mapped["_tx_coord"].values.astype(float)
            all_y  = mapped["_agg_prob"].values.astype(float)

            n_bins = 300
            bins = np.linspace(0, tx_len, n_bins + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_means = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = (all_x >= bins[i]) & (all_x < bins[i + 1])
                if mask.any():
                    bin_means[i] = all_y[mask].mean()

            valid = ~np.isnan(bin_means)
            if valid.sum() > 1:
                interp = np.interp(np.arange(n_bins), np.where(valid)[0], bin_means[valid])
                kw = max(3, smooth_window * n_bins // max(tx_len, 1))
                xk = np.arange(-kw, kw + 1, dtype=float)
                kernel = np.exp(-0.5 * (xk / max(kw / 2, 1)) ** 2)
                kernel /= kernel.sum()
                smoothed = np.convolve(interp, kernel, mode="same")
                smoothed[~valid] = np.nan
            else:
                smoothed = bin_means

            fig.add_trace(go.Scatter(
                x=all_x, y=all_y, mode="markers",
                marker=dict(size=3, color=COLOR_DOT),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

            valid_sm = ~np.isnan(smoothed)
            fig.add_trace(go.Scatter(
                x=bin_centers[valid_sm], y=smoothed[valid_sm],
                mode="lines",
                line=dict(color=COLOR_LINE, width=2),
                fill="tozeroy", fillcolor=COLOR_FILL,
                showlegend=False,
                hovertemplate="pos %{x:.0f}<br>mean: %{y:.3f}<extra></extra>",
            ), row=1, col=1)

            if tx_model:
                for xv, clr in [(tx_model.get("cds_start_tx"), "#558771"),
                                 (tx_model.get("cds_end_tx"),   "#DD8D6E")]:
                    if xv is not None:
                        fig.add_shape(type="line", x0=xv, x1=xv, y0=0, y1=1,
                                      line=dict(color=clr, width=1, dash="dash"),
                                      row=1, col=1)

            fig.update_xaxes(title_text="Transcript coord (nt)", row=1, col=1)
            fig.update_yaxes(title_text=f"pred_prob ({metric})", range=[0, 1.05], row=1, col=1)

    # ── Panel 2: mean pred_prob by amino-acid grouping ────────────────────────
    if "_agg_prob" in mut_agg.columns:
        aa_df = mut_agg.copy()
        aa_df["_aa_label"] = _build_aa_labels(mut_agg, cols, aa_mode, include_silent)
        aa_df = aa_df.dropna(subset=["_aa_label", "_agg_prob"])

        if not aa_df.empty:
            aa_grp = (
                aa_df.groupby("_aa_label")["_agg_prob"]
                .agg(mean="mean", count="count")
                .reset_index()
            )
            if len(aa_grp) > 25:
                aa_grp = aa_grp.nlargest(25, "count")
            aa_grp = aa_grp.sort_values("mean")

            fig.add_trace(go.Bar(
                x=aa_grp["mean"], y=aa_grp["_aa_label"],
                orientation="h",
                marker_color=COLOR_BAR,
                customdata=aa_grp["count"],
                hovertemplate="%{y}: mean=%{x:.3f}  (n=%{customdata})<extra></extra>",
                showlegend=False,
            ), row=1, col=2)

            fig.update_xaxes(title_text="mean pred_prob", range=[0, 1], row=1, col=2)
            fig.update_yaxes(tickfont=dict(size=9), row=1, col=2)

    # ── Panel 3: mean pred_prob by nucleotide substitution type ──────────────
    if "_agg_prob" in mut_agg.columns:
        sub_types = mut_agg.apply(lambda r: _extract_sub(r, cols), axis=1)
        sub_df = mut_agg.copy()
        sub_df["_sub_type"] = sub_types
        sub_df = sub_df.dropna(subset=["_sub_type", "_agg_prob"])

        if not sub_df.empty:
            sub_grp = (
                sub_df.groupby("_sub_type")["_agg_prob"]
                .agg(mean="mean", count="count")
                .reset_index()
                .sort_values("_sub_type")
            )
            bar_colors = [
                COLOR_TS if s in _TRANSITIONS else COLOR_TV
                for s in sub_grp["_sub_type"]
            ]
            fig.add_trace(go.Bar(
                x=sub_grp["_sub_type"], y=sub_grp["mean"],
                marker_color=bar_colors,
                customdata=sub_grp["count"],
                hovertemplate="%{x}: mean=%{y:.3f}  (n=%{customdata})<extra></extra>",
                showlegend=False,
            ), row=1, col=3)

            for label, color in [("Transition (ts)", COLOR_TS), ("Transversion (tv)", COLOR_TV)]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=color, symbol="square"),
                    name=label, showlegend=True,
                ))

            fig.update_xaxes(title_text="Substitution", tickangle=45,
                             tickfont=dict(size=9), row=1, col=3)
            fig.update_yaxes(title_text="mean pred_prob", range=[0, 1], row=1, col=3)

    # ── Panel 4: pred_prob distribution per cancer ────────────────────────────
    if has_cancer and "_agg_prob" in mut_agg.columns:
        cancer_df = mut_agg.dropna(subset=[cancer_col, "_agg_prob"])
        if not cancer_df.empty:
            # Sort cancers by median pred_prob
            medians = cancer_df.groupby(cancer_col)["_agg_prob"].median().sort_values()
            fig.add_trace(go.Box(
                x=cancer_df["_agg_prob"].values,
                y=cancer_df[cancer_col].values,
                orientation="h",
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker=dict(size=3, color=COLOR_LINE, opacity=0.45),
                line=dict(color=COLOR_LINE, width=1),
                fillcolor=COLOR_FILL,
                showlegend=False,
                hovertemplate="%{y}<br>pred_prob=%{x:.3f}<extra></extra>",
            ), row=1, col=4)

            fig.update_yaxes(
                categoryorder="array",
                categoryarray=medians.index.tolist(),
                tickfont=dict(size=8),
                row=1, col=4,
            )
            fig.update_xaxes(title_text="pred_prob", range=[0, 1], row=1, col=4)

    # Dynamic height: accommodate cancer labels on panel 4
    n_cancers = 0
    if has_cancer and "_agg_prob" in mut_agg.columns:
        _cdf = mut_agg.dropna(subset=[cancer_col, "_agg_prob"])
        n_cancers = int(_cdf[cancer_col].nunique()) if not _cdf.empty else 0
    fig_height = max(420, min(800, n_cancers * 20 + 160)) if n_cancers else 420

    fig.update_layout(
        height=fig_height,
        margin=dict(l=60, r=20, t=55, b=65),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.25, x=0.60),
    )
    for c in range(1, 5):
        fig.update_xaxes(showgrid=False, row=1, col=c)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)",
                         zeroline=False, row=1, col=c)

    return fig


BASE_COLORS = {"A": "#1A7F37", "T": "#C0392B", "G": "#D4A017", "C": "#1A5276"}


@st.cache_data(show_spinner=False)
def build_aligned_viewer(
    seq: str,
    protein: str | None,
    cds_start: int | None,
    window_start: int,
    window_size: int,
    mut_df: pd.DataFrame,
    cols: dict[str, str | None],
    metric: str,
) -> go.Figure:
    """
    Single-figure aligned viewer with three tracks sharing the same x-axis:
      y = 3.0  — Mutations  (triangle-down markers + labels)
      y = 1.5  — Protein    (one AA per codon, at codon center)
      y = 0.0  — DNA        (one character per nucleotide, color-coded by base)

    Dotted vertical stems connect each mutation marker down to the DNA track.
    Subtle vertical lines mark codon boundaries inside the CDS.
    """
    window_end = min(window_start + window_size, len(seq))
    win_seq    = seq[window_start:window_end]
    positions  = list(range(window_start, window_start + len(win_seq)))

    Y_DNA     = 0.0
    Y_AA      = 1.5
    Y_MUT_MIN = 3.2   # pred_prob = 0.0
    Y_MUT_MAX = 5.2   # pred_prob = 1.0

    # Display mode based on window width
    LETTER_THRESH = 150   # ≤ this: show letters
    BLOCK_THRESH  = 500   # ≤ this: show colored squares; above: hide DNA/AA

    fig = go.Figure()

    # ── DNA track ─────────────────────────────────────────────────────────────
    from collections import defaultdict
    by_color: dict[str, dict] = defaultdict(lambda: {"x": [], "t": []})
    for pos, base in zip(positions, win_seq):
        c = BASE_COLORS.get(base.upper(), "#666666")
        by_color[c]["x"].append(pos)
        by_color[c]["t"].append(base.upper())

    if window_size <= LETTER_THRESH:
        for color, data in by_color.items():
            fig.add_trace(go.Scatter(
                x=data["x"], y=[Y_DNA] * len(data["x"]),
                mode="text", text=data["t"],
                textfont=dict(size=13, color=color, family="Courier New, monospace"),
                showlegend=False, hoverinfo="skip",
            ))
    elif window_size <= BLOCK_THRESH:
        sq_size = max(3, int(600 / window_size))
        for color, data in by_color.items():
            fig.add_trace(go.Scatter(
                x=data["x"], y=[Y_DNA] * len(data["x"]),
                mode="markers",
                marker=dict(symbol="square", size=sq_size, color=color),
                showlegend=False, hoverinfo="skip",
            ))
    # else: window too wide — DNA display hidden, mutations remain

    # ── Protein track ─────────────────────────────────────────────────────────
    if cds_start is not None and protein:
        aa_x: list[float] = []
        aa_t: list[str]   = []
        for pos in positions:
            if pos >= cds_start:
                codon_offset = (pos - cds_start) % 3
                codon_idx    = (pos - cds_start) // 3
                if codon_offset == 1 and codon_idx < len(protein):
                    aa_x.append(pos)
                    aa_t.append(protein[codon_idx])
        if aa_x:
            if window_size <= LETTER_THRESH * 3:   # 1 AA per codon → readable up to 3× DNA threshold
                fig.add_trace(go.Scatter(
                    x=aa_x, y=[Y_AA] * len(aa_x),
                    mode="text", text=aa_t,
                    textfont=dict(size=13, color="#333333", family="Courier New, monospace"),
                    showlegend=False, hoverinfo="skip",
                ))
            elif window_size <= BLOCK_THRESH:
                fig.add_trace(go.Scatter(
                    x=aa_x, y=[Y_AA] * len(aa_x),
                    mode="markers",
                    marker=dict(symbol="square", size=max(3, int(180 / window_size) + 2),
                                color="#885784", opacity=0.6),
                    showlegend=False, hoverinfo="skip",
                ))

    # ── Codon boundary lines (subtle) ─────────────────────────────────────────
    # Accumulated here; mutation stems are added below — both applied in one
    # fig.update_layout(shapes=...) call to avoid the per-call Python overhead
    # of add_vline / add_shape in a loop.
    all_shapes: list[dict] = []
    if cds_start is not None:
        for pos in positions:
            if pos >= cds_start and (pos - cds_start) % 3 == 0:
                all_shapes.append(dict(
                    type="line", xref="x", yref="paper",
                    x0=pos - 0.5, x1=pos - 0.5, y0=0, y1=1,
                    line=dict(color="rgba(180,180,180,0.4)", width=1),
                ))

    # ── Mutation track ────────────────────────────────────────────────────────
    if "_tx_coord" in mut_df.columns:
        in_win = mut_df[
            (mut_df["_tx_coord"] >= window_start) &
            (mut_df["_tx_coord"] <  window_end)
        ].dropna(subset=["_tx_coord"])

        if not in_win.empty:
            probs = (
                in_win["_agg_prob"].fillna(0.5).values
                if "_agg_prob" in in_win.columns
                else np.ones(len(in_win)) * 0.5
            )
            p_min, p_max = float(probs.min()), float(probs.max())

            # ── y position = absolute pred_prob on calibrated axis ───────────
            # 0.0 maps to Y_MUT_MIN, 1.0 maps to Y_MUT_MAX — same scale always
            y_muts = Y_MUT_MIN + np.clip(probs, 0.0, 1.0) * (Y_MUT_MAX - Y_MUT_MIN)

            colors = [_variant_color(row, cols) for _, row in in_win.iterrows()]
            labels = [_variant_label(row, cols) for _, row in in_win.iterrows()]

            # ── enriched hover: codon change + trinucleotide context (fix 3) ──
            codons_col = cols.get("codons")
            hgvsc_col  = cols.get("hgvsc")
            hovers = []
            for _, row in in_win.iterrows():
                parts = [f"<b>{_variant_label(row, cols)}</b>"]
                if "_tx_coord" in row.index and pd.notna(row["_tx_coord"]):
                    parts.append(f"Transcript pos: {int(row['_tx_coord'])}")
                if "_agg_prob" in row.index and pd.notna(row["_agg_prob"]):
                    parts.append(f"pred_prob ({metric}): {row['_agg_prob']:.3f}")
                if "_count" in row.index:
                    parts.append(f"n samples: {int(row['_count'])}")
                for col_key in ("consequence", "impact"):
                    c = cols[col_key]
                    if c and c in row.index and pd.notna(row[c]):
                        parts.append(f"{col_key.capitalize()}: {row[c]}")
                # Codon change
                if codons_col and codons_col in row.index and pd.notna(row.get(codons_col)):
                    raw = str(row[codons_col])
                    if "/" in raw:
                        ref_c, alt_c = raw.split("/", 1)
                        parts.append(f"Codon: {ref_c.upper()} → {alt_c.upper()}")
                    else:
                        parts.append(f"Codon: {raw}")
                # Trinucleotide context from sequence
                if pd.notna(row.get("_tx_coord")):
                    tc = int(row["_tx_coord"])
                    if 0 < tc < len(seq) - 1:
                        trinuc = seq[tc - 1 : tc + 2].upper()
                        # Extract ref>alt from HGVSc (e.g. c.375G>T → G>T)
                        hgvsc_val = str(row[hgvsc_col]) if hgvsc_col and hgvsc_col in row.index else ""
                        m = re.search(r"([ACGT])>([ACGT])", hgvsc_val.upper())
                        if m:
                            context = f"{trinuc[0]}[{m.group(1)}>{m.group(2)}]{trinuc[2]}"
                        else:
                            context = trinuc
                        parts.append(f"Context: {context}")
                hovers.append("<br>".join(parts))

            # Dotted stems — height matches each mutation's y position
            for tx_coord, y_mut in zip(in_win["_tx_coord"].values, y_muts):
                all_shapes.append(dict(
                    type="line", xref="x", yref="y",
                    x0=float(tx_coord), x1=float(tx_coord),
                    y0=Y_DNA + 0.15, y1=float(y_mut) - 0.2,
                    line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
                ))

            fig.add_trace(go.Scatter(
                x=in_win["_tx_coord"].values,
                y=y_muts,
                mode="markers+text",
                marker=dict(
                    symbol="triangle-down", size=12, color=colors,
                    opacity=0.9, line=dict(color="white", width=0.5),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hovers,
                hoverinfo="text",
                showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(
            title="Transcript coordinate (nt)",
            range=[window_start - 0.8, window_start + len(win_seq) - 0.2],
            tickmode="array",
            tickvals=[p for p in positions if p % 10 == 0 or p == positions[0]],
            ticktext=[str(p) for p in positions if p % 10 == 0 or p == positions[0]],
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickvals=(
                [Y_DNA, Y_AA] +
                [Y_MUT_MIN + v * (Y_MUT_MAX - Y_MUT_MIN) for v in [0, 0.25, 0.5, 0.75, 1.0]]
            ),
            ticktext=["DNA", "Protein", "0", "0.25", "0.5", "0.75", "1.0"],
            range=[Y_DNA - 0.7, Y_MUT_MAX + 1.0],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            title=dict(text="pred_prob", font=dict(size=10)),
        ),
        height=450,
        margin=dict(l=80, r=20, t=30, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        shapes=all_shapes,
    )

    return fig


# ── Cached variant mapping (expensive — runs once per unique parameter set) ───

@st.cache_data(show_spinner="Building gene index...")
def build_gene_to_tx(filenames: tuple[str, ...]) -> dict[str, str]:
    """
    For each gene in the combined dataset, pick the transcript with the most
    rows that also exists in the GTF.  Genes with no GTF match are excluded.
    Returns {gene_name: transcript_id}.
    Cached per unique combination of selected files.
    """
    mut_raw = load_datasets(filenames)
    cols    = detect_all_cols(mut_raw)
    gtf_df  = load_gtf()

    gtf_tx_ids = set(gtf_df["transcript_id"].apply(strip_version).unique())
    gene_col   = cols["gene"]
    tx_col     = cols["transcript"]
    gene_to_tx: dict[str, str] = {}

    if not (gene_col and gene_col in mut_raw.columns and tx_col and tx_col in mut_raw.columns):
        return gene_to_tx

    # Pre-strip transcript IDs once to avoid repeated work in the loop
    stripped_tx = mut_raw[tx_col].dropna().apply(strip_version)
    mut_raw = mut_raw.copy()
    mut_raw["_tx_stripped"] = stripped_tx

    for gene, grp in mut_raw.groupby(gene_col, sort=False):
        counts = grp["_tx_stripped"].dropna().value_counts()
        for tx_stripped in counts.index:
            if tx_stripped in gtf_tx_ids:
                original = grp.loc[grp["_tx_stripped"] == tx_stripped, tx_col].iloc[0]
                gene_to_tx[str(gene)] = str(original)
                break

    return gene_to_tx


@st.cache_data(show_spinner="Mapping variants to transcript...")
def get_mapped_variants(
    filenames: tuple[str, ...],
    gene: str,
    transcript_id: str,
    aggregate: bool,
    metric: str,
    group_by_cancer: bool,
    selected_cancers: tuple[str, ...],
) -> tuple[pd.DataFrame, dict, dict | None]:
    """
    Filter, aggregate, and coordinate-map variants for one gene/transcript.

    Returns (mut_agg, cols, tx_model).
    Cached by Streamlit — only reruns when any parameter changes.
    """
    mut_raw = load_datasets(filenames)
    cols    = detect_all_cols(mut_raw)
    gtf_df  = load_gtf()
    tx_model = build_transcript_model(gtf_df, transcript_id)

    # Filter to gene
    gene_col = cols["gene"]
    mut_df = mut_raw.copy()
    if gene_col and gene_col in mut_df.columns:
        mut_df = mut_df[mut_df[gene_col] == gene]

    # Optional cancer filter
    if selected_cancers and cols["cancer"] and cols["cancer"] in mut_df.columns:
        mut_df = mut_df[mut_df[cols["cancer"]].isin(selected_cancers)]

    # Filter by transcript ID
    if cols["transcript"] and cols["transcript"] in mut_df.columns:
        stripped_target = strip_version(transcript_id)
        tx_mask = mut_df[cols["transcript"]].apply(
            lambda x: strip_version(str(x)) == stripped_target if pd.notna(x) else False
        )
        if tx_mask.any():
            mut_df = mut_df[tx_mask]

    # Aggregate
    if aggregate and not mut_df.empty:
        mut_agg = aggregate_variants(mut_df, cols, metric=metric, group_by_cancer=group_by_cancer)
    else:
        mut_agg = mut_df.copy()
        if cols["pred_prob"] and cols["pred_prob"] in mut_agg.columns:
            mut_agg["_agg_prob"] = mut_agg[cols["pred_prob"]]
        else:
            mut_agg["_agg_prob"] = np.nan
        mut_agg["_count"] = 1

    # Map genomic → transcript coordinates
    if tx_model is not None:
        pos_col      = cols["pos"]
        chrom_col    = cols["chrom"]
        tx_chrom     = tx_model["chrom"]
        hgvsc_col    = cols["hgvsc"]
        cds_start_tx = tx_model.get("cds_start_tx")

        def _map(row: pd.Series) -> float:
            if hgvsc_col and hgvsc_col in row.index and pd.notna(row[hgvsc_col]):
                tx_coord = hgvsc_to_tx_coord(str(row[hgvsc_col]), cds_start_tx)
                if tx_coord is not None:
                    return float(tx_coord)
            if not pos_col or pos_col not in row.index:
                return np.nan
            if chrom_col and chrom_col in row.index:
                if normalize_chrom(str(row[chrom_col])) != tx_chrom:
                    return np.nan
            try:
                c = genomic_to_transcript_coord(tx_model, int(row[pos_col]))
                return float(c) if c is not None else np.nan
            except Exception:
                return np.nan

        mut_agg["_tx_coord"] = mut_agg.apply(_map, axis=1)
    else:
        mut_agg["_tx_coord"] = np.nan

    return mut_agg, cols, tx_model


# ── SHAP scatter plot helpers (Plotly, styled like the notebooks) ─────────────

def _mpl_to_plotly_colorscale(cmap, n: int = 12) -> list:
    """Convert a matplotlib colormap to a Plotly colorscale list."""
    return [
        [i / (n - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, _) in enumerate(cmap(np.linspace(0, 1, n)))
    ]


_COLORSCALE_OG = _mpl_to_plotly_colorscale(CMAP_ORANGE_GREEN)

_SCATTER_LAYOUT = dict(
    template="simple_white",
    height=360,
    font=dict(size=9, color="black"),
    margin=dict(l=50, r=10, t=30, b=50),
    yaxis=dict(title=dict(text="AUPRC", font=dict(color="black")), tickfont=dict(color="black"), linecolor="black", tickcolor="black"),
    coloraxis=dict(
        colorscale=_COLORSCALE_OG,
        colorbar=dict(
            title=dict(text="Prevalence", side="right", font=dict(color="black")),
            thickness=12,
            len=0.8,
            tickfont=dict(color="black"),
        ),
    ),
)


def _plot_auprc_vs_rank(metrics: pd.DataFrame, cancer: str) -> go.Figure | None:
    """Interactive AUPRC vs prevalence-rank scatter (Plotly, Figure 1 style)."""
    df = metrics[["auprc_mean", "prevalence_mean"]].dropna()
    if len(df) < 2:
        return None
    df = df.sort_values("prevalence_mean", ascending=False).copy()
    df["rank"] = np.arange(len(df))

    fig = go.Figure(go.Scatter(
        x=df["rank"],
        y=df["auprc_mean"],
        mode="markers",
        marker=dict(
            size=9,
            color=df["prevalence_mean"],
            coloraxis="coloraxis",
            opacity=0.9,
            line=dict(color="black", width=0.4),
        ),
        text=df.index,
        customdata=np.column_stack([df["prevalence_mean"]]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Rank: %{x}<br>"
            "AUPRC: %{y:.3f}<br>"
            "Prevalence: %{customdata[0]:.3f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_SCATTER_LAYOUT,
        title=dict(text=cancer, font=dict(size=10)),
        xaxis=dict(title=dict(text="Rank by prevalence", font=dict(color="black")), tickfont=dict(color="black"), linecolor="black", tickcolor="black"),
    )
    return fig


def _plot_auprc_vs_nfeatures(metrics: pd.DataFrame, cancer: str) -> go.Figure | None:
    """Interactive AUPRC vs #SHAP features scatter (Plotly, Figure 2 panel-E style)."""
    df = metrics[["auprc_mean", "prevalence_mean", "n_shap_features", "mean_abs_shap"]].dropna()
    if len(df) < 2:
        return None

    fig = go.Figure(go.Scatter(
        x=df["n_shap_features"],
        y=df["auprc_mean"],
        mode="markers",
        marker=dict(
            size=10,
            color=df["prevalence_mean"],
            coloraxis="coloraxis",
            opacity=0.75,
            line=dict(color="black", width=0.4),
        ),
        text=df.index,
        customdata=np.column_stack([
            df["prevalence_mean"],
            df["n_shap_features"].astype(int),
            df["mean_abs_shap"],
        ]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "SHAP genes: %{customdata[1]}<br>"
            "AUPRC: %{y:.3f}<br>"
            "Prevalence: %{customdata[0]:.3f}<br>"
            "Mean |SHAP|: %{customdata[2]:.4f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_SCATTER_LAYOUT,
        title=dict(text=cancer, font=dict(size=10)),
        xaxis=dict(title=dict(text="Number of SHAP genes", font=dict(color="black")), tickfont=dict(color="black"), linecolor="black", tickcolor="black"),
    )
    return fig


# ── SHAP analysis helpers ─────────────────────────────────────────────────────

@st.cache_data
def _load_kfold(cancer: str) -> pd.DataFrame:
    return pd.read_csv(SHAP_APP_DIR / cancer / "kfold_summary.csv", index_col=0)


@st.cache_data
def _load_shap_matrix(cancer: str) -> pd.DataFrame:
    return pd.read_csv(
        SHAP_APP_DIR / cancer / "shap_summary_feature_summary_matrix.csv",
        index_col=0,
    )


@st.cache_data
def _load_beeswarm(cancer: str, gene: str) -> pd.DataFrame:
    return pd.read_parquet(SHAP_APP_DIR / cancer / f"beeswarm_{gene}.parquet")


@st.cache_data
def _load_gene_metrics(cancer: str) -> pd.DataFrame:
    """Join kfold metrics with SHAP-derived stats; flag genes with beeswarm files."""
    kfold_df    = _load_kfold(cancer)
    shap_matrix = _load_shap_matrix(cancer)

    shap_stats = (
        shap_matrix.assign(abs_shap=shap_matrix["mean_shap"].abs())
        .groupby("target")["abs_shap"]
        .agg(n_shap_features="count", mean_abs_shap="mean")
    )
    metrics = kfold_df.join(shap_stats, how="left")

    beeswarm_genes = {
        f.stem[len("beeswarm_"):]
        for f in (SHAP_APP_DIR / cancer).glob("beeswarm_*.parquet")
    }
    metrics["has_beeswarm"] = metrics.index.isin(beeswarm_genes)
    return metrics


def render_shap_tab() -> None:
    available_cancers = sorted(
        d.name for d in SHAP_APP_DIR.iterdir() if d.is_dir()
    )
    if not available_cancers:
        st.error(f"No SHAP data found in {SHAP_APP_DIR}")
        return

    with st.sidebar:
        st.header("Cancer")
        cancer = st.selectbox(
            "Cancer type", options=available_cancers, key="shap_cancer",
            label_visibility="collapsed",
        )

    metrics = _load_gene_metrics(cancer)

    # ── Panels 1 & 2: side-by-side scatter plots ──────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AUPRC vs. Prevalence rank")
        fig1 = _plot_auprc_vs_rank(metrics, cancer)
        if fig1 is not None:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Not enough data for this cancer.")

    with col2:
        st.subheader("AUPRC vs. SHAP features")
        fig2 = _plot_auprc_vs_nfeatures(metrics, cancer)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No SHAP data available for this cancer.")

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("SHAP feature importance (beeswarm)")

    genes_with_beeswarm = sorted(metrics.index[metrics["has_beeswarm"]].tolist())
    if not genes_with_beeswarm:
        st.info("No beeswarm data available for this cancer type.")
        return

    selected_gene = st.selectbox(
        "Mutation target gene", options=genes_with_beeswarm, key="shap_gene",
    )

    if selected_gene:
        df_bee    = _load_beeswarm(cancer, selected_gene)
        features  = [c for c in df_bee.columns if not c.startswith("x_") and c != "sample_id"]
        shap_vals = df_bee[features].to_numpy(dtype=float)
        X         = df_bee[[f"x_{f}" for f in features]].to_numpy(dtype=float)

        n_display = st.slider(
            "Max features to display",
            min_value=5,
            max_value=len(features),
            value=min(20, len(features)),
            step=5,
            key="shap_n_display",
        )

        shap.summary_plot(
            shap_vals, X,
            feature_names=features,
            max_display=n_display,
            cmap=CMAP_ORANGE_PURPLE_VIVID,
            show=False,
        )
        fig_bee = plt.gcf()
        # Style colorbar axis to match notebook convention
        if len(fig_bee.axes) > 1:
            cbar_ax = fig_bee.axes[-1]
            cbar_ax.set_ylabel(
                "Gene expression", fontsize=COLORBAR_LABEL_FS,
                rotation=270, labelpad=10,
            )
            cbar_ax.tick_params(labelsize=COLORBAR_TICK_FS)
        st.pyplot(fig_bee, use_container_width=True)
        plt.close("all")


# ── Main Streamlit app ────────────────────────────────────────────────────────

_HOME_IMAGE = Path(__file__).parent / "conceptual_visualization-01.png"

# ── Home tab ──────────────────────────────────────────────────────────────────
# Edit render_home_tab() freely — it is intentionally self-contained so the
# content can be updated without touching anything else in the app.

def render_home_tab() -> None:
    st.markdown(
        """
        ## Expression-to-Mutation Prediction in Cancer

        > *Preprint / publication link: **[coming soon](#)***
        """,
        unsafe_allow_html=False,
    )

    if _HOME_IMAGE.exists():
        st.image(str(_HOME_IMAGE), use_container_width=True)
    else:
        st.warning(f"Visualization image not found: {_HOME_IMAGE}")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            ### About the study

            Somatic mutations are central to cancer biology — they drive tumor evolution and
            shape the transcriptional state of cells. Yet genomic and transcriptomic data are
            often collected separately, and mutation detection from expression data remains
            largely unexplored.

            This work asks a simple question: **do expression profiles encode enough information
            to recover the underlying mutational state?**  We directly model the
            expression-to-mutation (E2M) relationship across large tumor cohorts from TCGA,
            treating the transcriptome as a functional readout of somatic alterations.

            Beyond prediction, interpreting the models recovers known pathway relationships,
            highlights which genes are most affected by each mutation across cancer types, and
            reveals less-characterized regulatory connections — providing a systematic map of
            how mutations shape transcriptional programs in human tumors.
            """
        )

    with col2:
        st.markdown(
            """
            ### Methods in brief

            **Multitask neural network** — a shared-encoder architecture jointly predicts the
            mutation status of all frequently mutated genes for a given cancer type, enabling
            efficient cross-mutation representation learning.

            **XGBoost + SHAP** — per-gene classifiers are trained on the expression features
            most predictive of each mutation, and SHAP values quantify the contribution of
            individual genes to each prediction, making the models interpretable.

            Models are evaluated in cross-validation on TCGA and on independent external
            cohorts, including bulk tumor datasets and single-cell data from cancer cell lines.
            """
        )

    st.markdown("---")

    st.markdown(
        """
        ### What's in this app

        Given the scale of results — spanning 30 cancer types, hundreds of mutation targets,
        and thousands of expression features — this app provides an interactive interface to
        explore them directly.

        | Tab | Contents |
        |-----|----------|
        | **Mutation Viewer** | Per-gene predicted mutation probabilities mapped onto the transcript model. Browse variants, filter by cancer type, and inspect positional hotspots. |
        | **SHAP Analysis** | Per-cancer prediction performance and SHAP-based feature importance. Explore which expression genes drive mutation predictions for each target. |
        """
    )


def render_mutation_tab() -> None:

    # ── Sidebar — dataset selector ────────────────────────────────────────────
    catalog = available_datasets()
    with st.sidebar:
        st.header("Datasets")
        if not catalog:
            st.error(f"No parquet files found in {DATA_DIR}")
            st.stop()

        selected_files: list[str] = []
        for fname, label in catalog:
            if st.checkbox(label, value=True):
                selected_files.append(fname)

        if not selected_files:
            st.warning("Select at least one dataset.")
            st.stop()

    try:
        mut_raw = load_datasets(tuple(selected_files))
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return

    cols = detect_all_cols(mut_raw)
    gene_to_tx = build_gene_to_tx(tuple(selected_files))
    gene_options = sorted(gene_to_tx.keys())

    with st.sidebar:
        # ── Gene selector ─────────────────────────────────────────────────────
        st.header("Gene")
        selected_gene = st.selectbox("Gene", options=[""] + gene_options)
        selected_tx_id = gene_to_tx.get(selected_gene) if selected_gene else None
        if selected_tx_id:
            st.caption(f"Transcript: {strip_version(selected_tx_id)}")

        # ── Aggregation ───────────────────────────────────────────────────────
        st.header("Aggregation")
        aggregate     = st.toggle("Aggregate variants", value=True)
        metric        = st.radio("Metric", ["mean", "median"], horizontal=True)

        # ── Cancer filter ─────────────────────────────────────────────────────
        cancer_col = cols["cancer"]
        group_by_cancer = False
        selected_cancers: list[str] = []
        if cancer_col and cancer_col in mut_raw.columns:
            st.header("Cancer")
            group_by_cancer = st.toggle("Group by cancer", value=False)
            all_cancers = sorted(mut_raw[cancer_col].dropna().unique().tolist())
            selected_cancers = st.multiselect("Filter cancer(s)", options=all_cancers)

    if not selected_gene or not selected_tx_id:
        st.info("Select a gene to continue.")
        return

    # ── Filter, aggregate, and map variants (cached per unique parameter set) ──
    mut_agg, cols, tx_model = get_mapped_variants(
        filenames=tuple(selected_files),
        gene=selected_gene,
        transcript_id=selected_tx_id,
        aggregate=aggregate,
        metric=metric,
        group_by_cancer=group_by_cancer,
        selected_cancers=tuple(sorted(selected_cancers)),
    )

    if tx_model is None:
        st.error(f"Transcript {selected_tx_id} not found in GTF.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_variants = len(mut_agg)
    n_mapped   = int(mut_agg["_tx_coord"].notna().sum()) if "_tx_coord" in mut_agg.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gene",        selected_gene)
    m2.metric("Transcript",  strip_version(selected_tx_id))
    m3.metric("Length",      f"{tx_model['tx_length']:,} nt")
    m4.metric("Variants",    f"{n_variants} ({n_mapped} mapped)")

    if n_variants - n_mapped > 0:
        st.caption(
            f"{n_variants - n_mapped} variants unmapped "
            "(intronic, wrong chromosome, or missing position)."
        )

    # ── Debug expander ────────────────────────────────────────────────────────
    with st.expander("🔍 Debug info (open if variants show 0 mapped)", expanded=(n_mapped == 0)):
        st.markdown("**Detected columns**")
        st.json({k: v for k, v in cols.items() if v is not None})

        st.markdown("**Transcript model**")
        st.json({
            "chrom":        tx_model.get("chrom"),
            "strand":       tx_model.get("strand"),
            "tx_length":    tx_model.get("tx_length"),
            "cds_start_tx": tx_model.get("cds_start_tx"),
            "cds_end_tx":   tx_model.get("cds_end_tx"),
            "n_exons":      len(tx_model.get("exons", [])),
        })

        st.markdown("**Sample rows from mut_agg** (key columns)")
        debug_cols = []
        for key in ("chrom", "pos", "hgvsc", "hgvsp", "consequence", "pred_prob"):
            c = cols[key]
            if c and c in mut_agg.columns:
                debug_cols.append(c)
        if "_tx_coord" in mut_agg.columns:
            debug_cols.append("_tx_coord")
        st.dataframe(mut_agg[debug_cols].head(10), use_container_width=True)

        st.markdown("**HGVSc parsing test** (first 5 rows)")
        hgvsc_col = cols["hgvsc"]
        if hgvsc_col and hgvsc_col in mut_agg.columns:
            for _, row in mut_agg.head(5).iterrows():
                val = str(row[hgvsc_col]) if pd.notna(row.get(hgvsc_col)) else "NaN"
                result = hgvsc_to_tx_coord(val, tx_model.get("cds_start_tx"))
                st.text(f"  HGVSc={val!r}  →  tx_coord={result}")
        else:
            st.warning(f"No HGVSc column found (tried: {COLUMN_CANDIDATES['hgvsc']})")

    # ── Main figure ───────────────────────────────────────────────────────────
    # ── Summary panels (smoothed line | AA | substitution | cancer) ──────────
    with st.sidebar:
        st.header("Summary panels")
        smooth_window = st.slider(
            "Smoothing window (nt)", min_value=10, max_value=300, value=50, step=10,
            help="Gaussian smoothing bandwidth for the transcript mean plot",
        )
        aa_mode = st.radio(
            "AA grouping",
            options=["change", "source", "target"],
            format_func={"change": "AA change (R>W)", "source": "Source AA (R>)", "target": "Target AA (>W)"}.get,
            horizontal=True,
        )
        include_silent = st.checkbox(
            "Include silent/synonymous", value=True,
            help="When checked, synonymous variants (ref AA = alt AA) are included in the AA panel",
        )
    aa_flag_options, sub_flag_options, cancer_flag_options = get_flag_options(
        mut_agg, cols, aa_mode, include_silent,
    )
    st.markdown("**Flag mutations in upper panel**")
    fc1, fc2, fc3 = st.columns(3)
    aa_flag = fc1.selectbox(
        "AA change/source/target",
        options=["None"] + aa_flag_options,
        key=f"flag_aa_{selected_tx_id}_{aa_mode}",
    )
    sub_flag = fc2.selectbox(
        "Substitution type",
        options=["None"] + sub_flag_options,
        key=f"flag_sub_{selected_tx_id}_{aa_mode}",
    )
    cancer_flag = fc3.selectbox(
        "Cancer type",
        options=["None"] + cancer_flag_options,
        key=f"flag_cancer_{selected_tx_id}_{aa_mode}",
    )
    highlight_mask, highlight_label = build_highlight_mask(
        mut_agg, cols, aa_mode, include_silent, aa_flag, sub_flag, cancer_flag,
    )
    if highlight_label:
        st.caption(f"Flagging {int(highlight_mask.sum())} mutation(s): {highlight_label}")
    fig = build_figure(
        tx_model, mut_agg, cols, metric,
        highlight_mask=highlight_mask,
        highlight_label=highlight_label,
    )
    st.plotly_chart(fig, use_container_width=True)
    fig_summary = build_summary_panels(
        mut_agg, cols, metric,
        tx_model=tx_model,
        smooth_window=smooth_window,
        aa_mode=aa_mode,
        include_silent=include_silent,
    )
    st.plotly_chart(fig_summary, use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    with st.expander("Legend", expanded=True):
        st.markdown("**Color = functional impact**")
        legend_entries = {
            "HIGH / frameshift / nonsense": "#C0392B",
            "MODERATE / missense / in-frame": "#E67E22",
            "LOW / silent": "#27AE60",
            "MODIFIER / non-coding": "#7F8C8D",
            "Unknown": "#95A5A6",
        }
        cols_leg = st.columns(len(legend_entries))
        for i, (label, color) in enumerate(legend_entries.items()):
            cols_leg[i].markdown(
                f'<span style="color:{color}; font-size:20px;">●</span> {label}',
                unsafe_allow_html=True,
            )
        st.markdown("**Marker size** ∝ sample count (y-position encodes probability)")

    # ── Variant table ─────────────────────────────────────────────────────────
    with st.expander("Variant table", expanded=False):
        show_cols = ["_tx_coord", "_agg_prob", "_count"]
        for key in ("hgvsp", "hgvsc", "consequence", "impact", "chrom", "pos", "ref", "alt", "cancer"):
            c = cols[key]
            if c and c in mut_agg.columns:
                show_cols.append(c)
        show_cols = [c for c in show_cols if c in mut_agg.columns]
        st.dataframe(
            mut_agg[show_cols].sort_values("_tx_coord").reset_index(drop=True),
            use_container_width=True,
        )

    # ── Aligned sequence viewer ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Aligned sequence viewer")
    st.caption(
        "DNA · Protein · Mutations on a shared axis. "
        "Loads from pre-filtered reference data — typically under a second."
    )

    seq_col1, seq_col2 = st.columns([1, 3])
    load_seq = seq_col1.button("Load / refresh sequence")

    if load_seq or st.session_state.get("_seq_loaded_for") == selected_tx_id:
        seq = load_fasta_sequence(selected_tx_id)
        st.session_state["_seq_loaded_for"] = selected_tx_id

        if seq is None:
            st.error(f"Sequence for {selected_tx_id} not found in FASTA.")
        else:
            cds_s = tx_model.get("cds_start_tx")
            cds_e = tx_model.get("cds_end_tx")
            protein = translate_cds(seq, cds_s, cds_e) if (cds_s is not None and cds_e is not None) else None

            tx_len = tx_model["tx_length"]

            # Default window: center on first mapped mutation, or start of transcript
            mapped_coords = (
                mut_agg["_tx_coord"].dropna().values
                if "_tx_coord" in mut_agg.columns
                else np.array([])
            )
            default_center = int(np.median(mapped_coords)) if len(mapped_coords) else 0

            ctrl1, ctrl2 = st.columns([3, 1])

            # Per-transcript session-state keys
            size_key      = f"_wsize_{selected_tx_id}"
            start_key     = f"_wstart_{selected_tx_id}"
            center_key    = f"_wcenter_{selected_tx_id}"
            prev_size_key = f"_wprevsize_{selected_tx_id}"

            window_size = ctrl2.select_slider(
                "Window size (nt)",
                options=[50, 80, 100, 120, 150, 200, 300, 500, 800],
                value=st.session_state.get(size_key, 100),
                key=size_key,
            )

            # Initialise center on first visit for this transcript
            if center_key not in st.session_state:
                st.session_state[center_key] = default_center

            # When window size changes, recompute start to preserve stored center
            if st.session_state.get(prev_size_key) != window_size:
                c = st.session_state[center_key]
                st.session_state[start_key] = max(0, min(c - window_size // 2, tx_len - window_size))
            st.session_state[prev_size_key] = window_size

            window_start = ctrl1.slider(
                "Window start (transcript coord)",
                min_value=0,
                max_value=max(0, tx_len - window_size),
                step=10,
                key=start_key,
            )
            # Keep center up-to-date as the user moves the start slider
            st.session_state[center_key] = window_start + window_size // 2

            fig_seq = build_aligned_viewer(
                seq, protein, cds_s,
                window_start, window_size,
                mut_agg, cols, metric,
            )
            st.plotly_chart(fig_seq, use_container_width=True)

            n_in_window = (
                ((mut_agg["_tx_coord"] >= window_start) &
                 (mut_agg["_tx_coord"] < window_start + window_size))
                .sum()
                if "_tx_coord" in mut_agg.columns else 0
            )
            st.caption(
                f"Showing nt {window_start}–{window_start + window_size} "
                f"| {n_in_window} mutation(s) in window "
                + (f"| CDS starts at nt {cds_s}" if cds_s is not None else "| no CDS annotated")
            )


def main() -> None:
    st.set_page_config(page_title="Mutation Viewer", layout="wide")

    with st.sidebar:
        view = st.radio(
            "View",
            ["Home", "Mutation Viewer", "SHAP Analysis"],
            label_visibility="collapsed",
        )
        st.markdown("---")

    if view == "Home":
        render_home_tab()
    elif view == "Mutation Viewer":
        st.title("Transcript Mutation Viewer")
        render_mutation_tab()
    else:
        st.title("SHAP Feature Analysis")
        render_shap_tab()


if __name__ == "__main__":
    main()
