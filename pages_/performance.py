"""Model performance: compare mutation targets within one TCGA cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme


METRICS = {
    "AUPRC": ("auprc_mean", "AUPRC", theme.GREEN),
    "Prevalence": ("prevalence_mean", "Mutation prevalence", theme.ORANGE),
    "Normalized AUPRC": ("norm_auprc", "Normalized AUPRC", theme.PURPLE),
    "SHAP features": ("n_shap_features", "Features with non-zero mean SHAP", theme.TEAL),
}


def _rank_plot(frame: pd.DataFrame, metric_label: str, top_n: int) -> go.Figure:
    column, axis_label, color = METRICS[metric_label]
    plot = frame.dropna(subset=[column]).nlargest(top_n, column).sort_values(column)
    figure = go.Figure()
    if metric_label == "AUPRC":
        figure.add_bar(
            x=plot["prevalence_mean"], y=plot["gene"], orientation="h",
            name="Prevalence baseline", marker_color=theme.BEIGE,
            marker_line_color="#D8CDAF", marker_line_width=0.5,
            hovertemplate="<b>%{y}</b><br>Prevalence: %{x:.3f}<extra></extra>",
        )
    figure.add_bar(
        x=plot[column], y=plot["gene"], orientation="h", name=axis_label,
        marker_color=color, opacity=0.9,
        customdata=np.column_stack([
            plot["auprc_mean"], plot["prevalence_mean"], plot["norm_auprc"],
            plot["n_shap_features"],
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>AUPRC: %{customdata[0]:.3f}<br>"
            "Prevalence: %{customdata[1]:.3f}<br>Normalized AUPRC: %{customdata[2]:.3f}<br>"
            "SHAP features: %{customdata[3]:.0f}<extra></extra>"
        ),
    )
    if metric_label == "Normalized AUPRC":
        figure.add_vline(x=0, line_color=theme.MUTED, line_width=1)
    figure.update_layout(
        barmode="overlay" if metric_label == "AUPRC" else "relative",
        height=max(430, 23 * len(plot) + 110),
        margin=dict(l=20, r=20, t=25, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=theme.INK, size=12),
        xaxis=dict(title=axis_label, gridcolor="#ECE8DE", zeroline=False),
        yaxis=dict(title=None, tickfont=dict(family="IBM Plex Mono", size=11)),
        legend=dict(orientation="h", y=1.03, x=0, title=None),
    )
    return figure


def _lift_plot(frame: pd.DataFrame) -> go.Figure:
    plot = frame.dropna(subset=["auprc_mean", "prevalence_mean", "norm_auprc"]).copy()
    plot = plot.sort_values("prevalence_mean", ascending=False).reset_index(drop=True)
    plot["prevalence_rank"] = np.arange(1, len(plot) + 1)
    figure = go.Figure(go.Scatter(
        x=plot["prevalence_rank"], y=plot["auprc_mean"], mode="markers",
        text=plot["gene"],
        marker=dict(
            size=np.clip(7 + 18 * np.maximum(plot["norm_auprc"], 0), 7, 20),
            color=plot["norm_auprc"], colorscale=[[0, theme.ORANGE], [0.5, theme.BEIGE], [1, theme.GREEN]],
            cmin=-0.2, cmax=1, colorbar=dict(title="Normalized<br>AUPRC", thickness=12),
            line=dict(color=theme.INK, width=0.35), opacity=0.86,
        ),
        customdata=np.column_stack([
            plot["prevalence_mean"], plot["norm_auprc"], plot["n_shap_features"]
        ]),
        hovertemplate=(
            "<b>%{text}</b><br>Prevalence rank: %{x:.0f}<br>Prevalence: %{customdata[0]:.3f}<br>"
            "AUPRC: %{y:.3f}<br>Normalized AUPRC: %{customdata[1]:.3f}<br>"
            "SHAP features: %{customdata[2]:.0f}"
            "<extra></extra>"
        ),
    ))
    figure.update_layout(
        height=500, margin=dict(l=20, r=30, t=25, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=theme.INK, size=12),
        xaxis=dict(
            title="Rank by mutation prevalence (most prevalent first)", gridcolor="#ECE8DE",
            dtick=max(1, len(plot) // 10),
        ),
        yaxis=dict(title="AUPRC", gridcolor="#ECE8DE", range=[0, None]),
    )
    return figure


def render() -> None:
    theme.page_head(
        "Which models work",
        "Model performance",
        "Compare mutation targets <b>within one cancer cohort</b>. AUPRC is shown together "
        "with its mutation-prevalence baseline so predictive lift is not confused with target frequency.",
    )

    cohorts = list(data.shap_cohorts())
    if not cohorts:
        theme.notice("empty", "No performance data", "No cohort metric files are present in this bundle.")
        return

    controls = st.columns([1.2, 1.4, 0.8])
    cohort = controls[0].selectbox(
        "TCGA cohort", cohorts, format_func=lambda value: f"{value} - {data.cohort_label(value)}"
    )
    metric_label = controls[1].radio(
        "Rank genes by", list(METRICS), index=2, horizontal=True
    )
    metrics = data.gene_metrics(cohort)
    if metrics.empty:
        theme.notice("empty", "No metrics for this cohort", "The selected cohort has no readable metric table.")
        return
    max_top = min(60, len(metrics))
    top_n = controls[2].number_input("Genes shown", min_value=5, max_value=max_top, value=min(25, max_top), step=5)

    best = metrics.loc[metrics["norm_auprc"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Targets evaluated", f"{len(metrics):,}")
    c2.metric("Best target", str(best["gene"]))
    c3.metric("Best normalized AUPRC", f"{best['norm_auprc']:.3f}")
    c4.metric("Targets with beeswarms", f"{len(data.beeswarm_genes(cohort)):,}")

    meta = data.bundle_meta()
    with st.container(border=True):
        theme.md(theme.fig_header("01", f"{cohort} target ranking"))
        st.plotly_chart(_rank_plot(metrics, metric_label, int(top_n)), width="stretch")
        theme.fig_caption(
            "1",
            f"Genes ranked by {theme.def_chip(metric_label)}. For AUPRC, the beige bar is the "
            "mutation-prevalence baseline and the green bar is held-out model performance.",
        )
        theme.provenance([
            ("Cohort", cohort), ("Targets", str(len(metrics))),
            ("Bundle", meta["generated"]), ("Threshold", f"normalized AUPRC > {data.SHAP_NORM_AUPRC_THRESHOLD}"),
        ])

    with st.container(border=True):
        theme.md(theme.fig_header("02", "Performance by prevalence rank"))
        st.plotly_chart(_lift_plot(metrics), width="stretch")
        theme.fig_caption(
            "2",
            "Each point is one mutation target, ordered from most to least prevalent. Marker color "
            "and size encode normalized AUPRC; hover text reports the underlying prevalence.",
        )
        theme.provenance([("Cohort", cohort), ("Cross-validation", "5-fold"), ("Bundle", meta["sha7"])])

    display_columns = [
        "gene", "auprc_mean", "auprc_std", "prevalence_mean", "norm_auprc", "n_shap_features"
    ]
    table = metrics[[column for column in display_columns if column in metrics]].sort_values(
        "norm_auprc", ascending=False
    )
    with st.expander("Show and download performance data"):
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV", table.to_csv(index=False).encode("utf-8"),
            file_name=f"{cohort}_mutation_model_performance.csv", mime="text/csv",
        )
