"""Mutation viewer: map observed MC3 variants onto a selected transcript."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import data
import theme
from transcript_utils import (
    build_transcript_model,
    genomic_to_transcript_coord,
    normalize_chrom,
    strip_version,
)


CLASS_COLORS = {
    "Missense_Mutation": theme.ORANGE,
    "Silent": theme.GREEN,
    "missense": theme.ORANGE,
    "silent": theme.GREEN,
}


def _prepare_variants(frame: pd.DataFrame, aggregate: bool, metric: str) -> pd.DataFrame:
    frame = frame.copy()
    if not aggregate:
        frame["_agg_prob"] = frame["pred_prob"].astype(float)
        frame["_count"] = 1
        return frame
    keys = [
        "Cancer", "Hugo_Symbol", "Chromosome", "Start_Position", "Reference_Allele",
        "Tumor_Seq_Allele2", "Variant_Classification", "Amino_Acid_Change",
    ]
    keys = [column for column in keys if column in frame]
    grouped = frame.groupby(keys, dropna=False)["pred_prob"]
    values = grouped.mean() if metric == "Mean" else grouped.median()
    out = values.rename("_agg_prob").reset_index()
    out["_count"] = grouped.size().to_numpy()
    return out


def _map_variants(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    frame = frame.copy()
    tx_chrom = normalize_chrom(str(model["chrom"]))
    on_chrom = frame["Chromosome"].astype(str).map(normalize_chrom).eq(tx_chrom)
    positions = pd.to_numeric(frame["Start_Position"], errors="coerce")

    # Resolve each genomic position to a transcript coordinate once, not per row.
    coord_by_pos: dict[int, float] = {}
    for pos in positions.where(on_chrom).dropna().astype(int).unique():
        try:
            coordinate = genomic_to_transcript_coord(model, int(pos))
        except (TypeError, ValueError):
            coordinate = None
        coord_by_pos[int(pos)] = float(coordinate) if coordinate is not None else np.nan

    coords = positions.map(lambda pos: coord_by_pos.get(int(pos)) if pd.notna(pos) else np.nan)
    frame["_tx_coord"] = coords.where(on_chrom)
    return frame


def _transcript_segments(model: dict) -> list[tuple[int, int]]:
    segments = []
    offset = 0
    for start, end in model["exons"]:
        length = int(end) - int(start) + 1
        segments.append((offset, offset + length))
        offset += length
    return segments


def _transcript_plot(frame: pd.DataFrame, model: dict, gene: str, transcript: str) -> go.Figure:
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.20, 0.80],
    )
    for start, end in _transcript_segments(model):
        figure.add_shape(
            type="rect", x0=start, x1=end, y0=0.28, y1=0.72,
            fillcolor=theme.BEIGE, line=dict(color="#BFB293", width=0.7), row=1, col=1,
        )
    if model.get("cds_start_tx") is not None:
        figure.add_shape(
            type="rect", x0=model["cds_start_tx"], x1=model["cds_end_tx"], y0=0.18, y1=0.82,
            fillcolor=theme.TEAL, opacity=0.82, line=dict(color=theme.GREEN_DARK, width=0.6),
            row=1, col=1,
        )
    figure.add_trace(go.Scatter(
        x=[0, model["tx_length"]], y=[0.5, 0.5], mode="lines",
        line=dict(color=theme.MUTED, width=1), hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    mapped = frame.dropna(subset=["_tx_coord", "_agg_prob"]).copy()
    for mutation_class, group in mapped.groupby("Variant_Classification", dropna=False):
        label = str(mutation_class)
        color = CLASS_COLORS.get(label, theme.PURPLE)
        x_values = group["_tx_coord"].to_numpy()
        y_values = group["_agg_prob"].to_numpy()
        for x_value, y_value in zip(x_values, y_values):
            figure.add_shape(
                type="line", x0=x_value, x1=x_value, y0=0, y1=y_value,
                line=dict(color=color, width=0.75), opacity=0.45, row=2, col=1,
            )
        amino = group.get("Amino_Acid_Change", pd.Series("", index=group.index)).fillna("")
        custom = np.column_stack([
            group["Chromosome"].astype(str), group["Start_Position"].astype(str),
            group["Reference_Allele"].astype(str), group["Tumor_Seq_Allele2"].astype(str),
            amino.astype(str), group["_count"].astype(int),
        ])
        figure.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="markers", name=label.replace("_", " "),
            marker=dict(
                color=color, size=np.clip(7 + 2.2 * np.sqrt(group["_count"]), 8, 20),
                line=dict(color=theme.INK, width=0.45), opacity=0.88,
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[4]}</b><br>Transcript coordinate: %{x:.0f}<br>"
                "Genomic: chr%{customdata[0]}:%{customdata[1]} %{customdata[2]}>%{customdata[3]}<br>"
                "Predicted probability: %{y:.3f}<br>Samples: %{customdata[5]}<extra></extra>"
            ),
        ), row=2, col=1)

    figure.update_yaxes(visible=False, range=[0, 1], row=1, col=1)
    figure.update_yaxes(title="Predicted probability", range=[0, 1.02], gridcolor="#ECE8DE", row=2, col=1)
    figure.update_xaxes(title="Transcript coordinate (nt)", range=[0, model["tx_length"]], row=2, col=1)
    figure.update_layout(
        height=590, margin=dict(l=25, r=25, t=45, b=45),
        title=dict(text=f"{gene} · {strip_version(transcript)} · {model['strand']} strand", x=0.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=theme.INK, size=12),
        legend=dict(orientation="h", y=1.02, x=0.55),
    )
    return figure


def _summary_plot(frame: pd.DataFrame) -> go.Figure:
    mapped = frame.dropna(subset=["_tx_coord", "_agg_prob"]).copy()
    figure = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Along the transcript", "Amino-acid changes", "Nucleotide substitutions", "By cohort"),
        vertical_spacing=0.22, horizontal_spacing=0.16,
    )
    if not mapped.empty:
        bins = min(30, max(5, int(np.sqrt(len(mapped)))))
        mapped["_bin"] = pd.cut(mapped["_tx_coord"], bins=bins)
        position = mapped.groupby("_bin", observed=True).agg(
            coordinate=("_tx_coord", "mean"), probability=("_agg_prob", "mean"), n=("_count", "sum")
        )
        figure.add_trace(go.Scatter(
            x=position["coordinate"], y=position["probability"], mode="lines+markers",
            line=dict(color=theme.GREEN, width=2), marker=dict(size=6), showlegend=False,
            hovertemplate="Coordinate: %{x:.0f}<br>Mean probability: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

        aa = mapped.loc[mapped["Amino_Acid_Change"].notna()].groupby("Amino_Acid_Change").agg(
            probability=("_agg_prob", "mean"), n=("_count", "sum")
        ).nlargest(12, "n").sort_values("probability")
        figure.add_trace(go.Bar(
            x=aa["probability"], y=aa.index, orientation="h", marker_color=theme.ORANGE,
            showlegend=False, customdata=aa["n"],
            hovertemplate="%{y}<br>Mean probability: %{x:.3f}<br>Samples: %{customdata}<extra></extra>",
        ), row=1, col=2)

        mapped["_substitution"] = (
            mapped["Reference_Allele"].astype(str) + ">" + mapped["Tumor_Seq_Allele2"].astype(str)
        )
        substitutions = mapped.groupby("_substitution").agg(
            probability=("_agg_prob", "mean"), n=("_count", "sum")
        ).sort_values("probability")
        figure.add_trace(go.Bar(
            x=substitutions.index, y=substitutions["probability"], marker_color=theme.PURPLE,
            showlegend=False, customdata=substitutions["n"],
            hovertemplate="%{x}<br>Mean probability: %{y:.3f}<br>Samples: %{customdata}<extra></extra>",
        ), row=2, col=1)

        cohorts = mapped.groupby("Cancer").agg(
            probability=("_agg_prob", "mean"), n=("_count", "sum")
        ).sort_values("probability")
        figure.add_trace(go.Bar(
            x=cohorts["probability"], y=cohorts.index, orientation="h", marker_color=theme.TEAL,
            showlegend=False, customdata=cohorts["n"],
            hovertemplate="%{y}<br>Mean probability: %{x:.3f}<br>Samples: %{customdata}<extra></extra>",
        ), row=2, col=2)

    figure.update_yaxes(range=[0, 1], title="Mean probability", row=1, col=1)
    figure.update_xaxes(range=[0, 1], title="Mean probability", row=1, col=2)
    figure.update_yaxes(range=[0, 1], title="Mean probability", row=2, col=1)
    figure.update_xaxes(range=[0, 1], title="Mean probability", row=2, col=2)
    figure.update_layout(
        height=720, margin=dict(l=25, r=25, t=55, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=theme.INK, size=11),
    )
    figure.update_xaxes(gridcolor="#ECE8DE")
    figure.update_yaxes(gridcolor="#ECE8DE")
    return figure


def render() -> None:
    theme.page_head(
        "The core",
        "Mutation viewer",
        "Place the mutations seen in real tumors along a gene's transcript, and read the model's "
        "predicted probability for each one. Probabilities use a fixed 0 to 1 scale.",
    )

    catalog = data.available_datasets()
    if not catalog:
        theme.notice("empty", "No variant data", "No mutation parquet files are present in this bundle.")
        return
    if data.load_gtf().empty:
        theme.notice(
            "warning", "Transcript reference unavailable",
            "The variant tables are present, but <code>gtf_filtered.parquet</code> has not been built. "
            "Run <code>python Figure_Scripts/mutation_viewer/build_bundle.py</code>.",
        )
        return

    labels = {filename: label for filename, label in catalog}
    selected_files = st.multiselect(
        "Mutation datasets", list(labels), default=list(labels), format_func=lambda value: labels[value]
    )
    if not selected_files:
        theme.notice("empty", "Select a dataset", "Choose at least one mutation dataset to continue.")
        return
    raw = data.load_variants(tuple(selected_files))
    cohorts = data.cohorts_in(raw)
    if not cohorts:
        theme.notice("empty", "No cohorts found", "The selected datasets contain no cohort labels.")
        return

    # Open on a clean, recognizable example (the manuscript's flagship) rather than the
    # alphabetical first, which is often a sparse or passenger gene.
    default_cohort = "BRCA" if "BRCA" in cohorts else raw["Cancer"].astype(str).value_counts().idxmax()
    selection = st.columns([1.2, 1.1, 1.7])
    cohort = selection[0].selectbox(
        "TCGA cohort", cohorts,
        index=cohorts.index(default_cohort) if default_cohort in cohorts else 0,
        format_func=lambda value: f"{value} - {data.cohort_label(value)}",
    )
    genes = sorted(raw.loc[raw["Cancer"].astype(str).eq(cohort), "Hugo_Symbol"].dropna().astype(str).unique())
    if "TP53" in genes:
        default_gene = "TP53"
    else:
        metrics = data.gene_metrics(cohort)
        ranked = metrics[metrics["gene"].isin(genes)].sort_values("norm_auprc", ascending=False) if not metrics.empty else metrics
        default_gene = ranked["gene"].iloc[0] if not ranked.empty else genes[0]
    gene = selection[1].selectbox("Mutation target", genes, index=genes.index(default_gene))
    transcripts = data.transcripts_for_gene(gene)
    if transcripts.empty:
        theme.notice("warning", "Missing transcript annotation", f"No GENCODE transcript is available for {theme.esc(gene)}.")
        return
    transcript_options = transcripts["transcript_id"].astype(str).tolist()
    transcript = selection[2].selectbox(
        "Transcript", transcript_options,
        format_func=lambda value: (
            f"{strip_version(value)} · CDS {int(transcripts.loc[transcripts['transcript_id'].eq(value), 'CDS'].iloc[0]):,} nt"
        ),
    )

    filtered = raw.loc[raw["Cancer"].astype(str).eq(cohort) & raw["Hugo_Symbol"].astype(str).eq(gene)].copy()
    available_classes = sorted(filtered["Variant_Classification"].dropna().astype(str).unique())
    controls = st.columns([1.4, 0.9, 1.1])
    selected_classes = controls[0].multiselect("Mutation classes", available_classes, default=available_classes)
    aggregate = controls[1].toggle("Aggregate repeated variants", value=True)
    metric = controls[2].radio("Aggregation", ["Mean", "Median"], horizontal=True)
    filtered = filtered.loc[filtered["Variant_Classification"].astype(str).isin(selected_classes)].copy()
    if filtered.empty:
        theme.notice("empty", "No variants for this selection", "Change the dataset or mutation-class filters.")
        return

    prepared = _prepare_variants(filtered, aggregate, metric)
    model = build_transcript_model(data.load_gtf(), transcript)
    if model is None:
        theme.notice("warning", "Missing transcript model", "The selected transcript could not be reconstructed from the GTF.")
        return
    mapped = _map_variants(prepared, model)
    mapped_count = int(mapped["_tx_coord"].notna().sum())
    sample_count = int(filtered["sample4"].nunique()) if "sample4" in filtered else len(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tumor samples", f"{sample_count:,}")
    c2.metric("Variant observations", f"{len(filtered):,}")
    c3.metric("Displayed variants", f"{len(mapped):,}")
    c4.metric("Mapped to transcript", f"{mapped_count:,} / {len(mapped):,}")

    if mapped_count < len(mapped):
        theme.notice(
            "warning", "Some variants are not on the plot",
            f"{len(mapped) - mapped_count:,} variants fall outside this transcript's exons, or sit on a "
            "different chromosome. They are still in the table below.",
        )
    if sample_count < 20:
        theme.notice(
            "warning", "Few samples",
            f"Only {sample_count} tumors carry these mutations, so read the positions with caution.",
        )

    meta = data.bundle_meta()
    with st.container(border=True):
        theme.md(theme.fig_header("01", "Transcript-level mutation probabilities"))
        st.plotly_chart(_transcript_plot(mapped, model, gene, transcript), width="stretch")
        theme.fig_caption(
            "1",
            "The track at the top shows the transcript's exons, with the coding region in teal. Each "
            "marker is a real mutation placed at its position on the transcript. Height is the "
            f"{theme.def_chip('predicted probability')}; size is the number of tumors that carry it.",
        )
        theme.provenance([
            ("Cohort", cohort), ("Target", gene), ("Transcript", strip_version(transcript)),
            ("Aggregation", metric.lower() if aggregate else "none"), ("Bundle", meta["sha7"]),
        ])

    with st.container(border=True):
        theme.md(theme.fig_header("02", "Variant summaries"))
        st.plotly_chart(_summary_plot(mapped), width="stretch")
        theme.fig_caption(
            "2", "The same filtered variants, summarized four ways. Every probability axis runs from 0 to 1."
        )
        theme.provenance([("Cohort", cohort), ("Target", gene), ("Rows", str(len(mapped)))])

    export_columns = [
        "Cancer", "Hugo_Symbol", "Chromosome", "Start_Position", "Reference_Allele",
        "Tumor_Seq_Allele2", "Variant_Classification", "Amino_Acid_Change", "_tx_coord",
        "_agg_prob", "_count",
    ]
    export = mapped[[column for column in export_columns if column in mapped]].rename(columns={
        "_tx_coord": "transcript_coordinate", "_agg_prob": "predicted_probability",
        "_count": "sample_count",
    }).sort_values(["transcript_coordinate", "predicted_probability"], na_position="last")
    with st.expander("Show and download variant data"):
        st.dataframe(export, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV", export.to_csv(index=False).encode("utf-8"),
            file_name=f"{cohort}_{gene}_{strip_version(transcript)}_variants.csv", mime="text/csv",
        )
