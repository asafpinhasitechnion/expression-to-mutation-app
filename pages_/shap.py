"""SHAP explorer: target-level feature summaries and per-sample beeswarms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme


def _target_features(cohort: str, target: str) -> pd.DataFrame:
    matrix = data.load_shap_matrix(cohort)
    if matrix.empty:
        return pd.DataFrame()
    selected = matrix.loc[
        matrix["target_gene"].astype(str).eq(str(target)), ["feature", "mean_shap"]
    ].copy()
    if selected.empty:
        return selected
    selected = selected.groupby("feature", as_index=False)["mean_shap"].mean()
    selected["abs_mean_shap"] = selected["mean_shap"].abs()
    return selected.sort_values("abs_mean_shap", ascending=False)


def _feature_plot(features: pd.DataFrame, top_n: int) -> go.Figure:
    plot = features.head(top_n).sort_values("mean_shap")
    colors = np.where(plot["mean_shap"] >= 0, theme.PURPLE, theme.ORANGE)
    figure = go.Figure(go.Bar(
        x=plot["mean_shap"], y=plot["feature"], orientation="h", marker_color=colors,
        customdata=plot["abs_mean_shap"],
        hovertemplate="<b>%{y}</b><br>Mean SHAP: %{x:.4f}<br>|Mean SHAP|: %{customdata:.4f}<extra></extra>",
    ))
    figure.add_vline(x=0, line_color=theme.MUTED, line_width=1)
    figure.update_layout(
        height=max(430, 23 * len(plot) + 100), margin=dict(l=20, r=20, t=25, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=theme.INK, size=12),
        xaxis=dict(title="Mean SHAP value", gridcolor="#ECE8DE", zeroline=False),
        yaxis=dict(title=None, tickfont=dict(family="IBM Plex Mono", size=10)),
    )
    return figure


def _relative_expression(values: pd.Series) -> pd.Series:
    """Scale one feature's expression to SHAP-style robust relative values."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.nan, index=numeric.index, dtype=float)

    color_min, color_max = np.nanpercentile(finite, [5, 95])
    if color_min == color_max:
        color_min, color_max = np.nanpercentile(finite, [1, 99])
    if color_min == color_max:
        color_min, color_max = float(finite.min()), float(finite.max())
    if color_min == color_max:
        relative = pd.Series(0.5, index=numeric.index, dtype=float)
        relative[numeric.isna()] = np.nan
        return relative

    return ((numeric.clip(color_min, color_max) - color_min) / (color_max - color_min)).clip(0, 1)


def _beeswarm_plot(frame: pd.DataFrame, max_features: int) -> tuple[go.Figure, pd.DataFrame]:
    feature_columns = [
        column for column in frame.columns
        if column != "sample_id" and not column.startswith("x_") and f"x_{column}" in frame.columns
    ]
    if not feature_columns:
        return go.Figure(), pd.DataFrame()
    importance = pd.Series(
        {feature: frame[feature].abs().mean() for feature in feature_columns}, name="mean_abs_shap"
    ).sort_values(ascending=False)
    selected = importance.head(max_features).index.tolist()
    selected = list(reversed(selected))

    figure = go.Figure()
    rng = np.random.default_rng(42)
    long_rows = []
    for y_index, feature in enumerate(selected):
        shap_values = pd.to_numeric(frame[feature], errors="coerce")
        expression = pd.to_numeric(frame[f"x_{feature}"], errors="coerce")
        relative_expression = _relative_expression(expression)
        valid = shap_values.notna() & expression.notna()
        jitter = rng.uniform(-0.28, 0.28, int(valid.sum()))
        samples = frame.loc[valid, "sample_id"].astype(str) if "sample_id" in frame else pd.Series("", index=frame.index[valid])
        figure.add_trace(go.Scattergl(
            x=shap_values.loc[valid], y=y_index + jitter, mode="markers", showlegend=False,
            marker=dict(
                size=5.5, color=relative_expression.loc[valid], coloraxis="coloraxis", opacity=0.72,
                line=dict(width=0),
            ),
            customdata=np.column_stack([
                samples,
                expression.loc[valid],
                relative_expression.loc[valid],
            ]),
            hovertemplate=(
                f"<b>{feature}</b><br>Sample: %{{customdata[0]}}<br>SHAP: %{{x:.4f}}<br>"
                "Expression count: %{customdata[1]:.3f}<br>"
                "Relative expression: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))
        long_rows.append(pd.DataFrame({
            "sample_id": samples.to_numpy(), "feature": feature,
            "shap_value": shap_values.loc[valid].to_numpy(),
            "expression_value": expression.loc[valid].to_numpy(),
            "relative_expression": relative_expression.loc[valid].to_numpy(),
        }))
    figure.add_vline(x=0, line_color=theme.MUTED, line_width=1)
    figure.update_layout(
        height=max(500, 31 * len(selected) + 110), margin=dict(l=20, r=25, t=25, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", color=theme.INK, size=12),
        xaxis=dict(title="SHAP value (impact on prediction)", gridcolor="#ECE8DE", zeroline=False),
        yaxis=dict(
            title=None, tickmode="array", tickvals=list(range(len(selected))), ticktext=selected,
            tickfont=dict(family="IBM Plex Mono", size=10),
        ),
        coloraxis=dict(
            colorscale=[[0, theme.BEE_INDIGO], [0.5, theme.BEE_CREAM], [1, theme.BEE_MAGENTA]],
            cmin=0, cmax=1,
            colorbar=dict(
                title="Relative expression",
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Low", "High"],
                thickness=12,
                len=0.72,
            ),
        ),
    )
    return figure, pd.concat(long_rows, ignore_index=True)


def render() -> None:
    theme.page_head(
        "Feature attribution",
        "SHAP explorer",
        "See which genes' expression the model leans on when predicting a mutation. Where it is "
        "available, you can also look at the values for every individual tumor.",
    )
    theme.caveat()

    cohorts = list(data.shap_cohorts())
    if not cohorts:
        theme.notice("empty", "No SHAP data", "No cohort SHAP directories are present in this bundle.")
        return

    selectors = st.columns([1.2, 1.2, 0.7])
    cohort = selectors[0].selectbox(
        "TCGA cohort", cohorts, format_func=lambda value: f"{value} - {data.cohort_label(value)}"
    )
    targets = data.shap_targets(cohort)
    if not targets:
        theme.notice("empty", "No SHAP targets", "The selected cohort has no readable target-level SHAP table.")
        return
    beeswarm_available = set(data.beeswarm_genes(cohort))
    default_index = next((index for index, gene in enumerate(targets) if gene in beeswarm_available), 0)
    target = selectors[1].selectbox(
        "Mutation target", targets, index=default_index,
        format_func=lambda gene: f"{gene}{' - per-sample' if gene in beeswarm_available else ''}",
    )
    top_n = selectors[2].number_input("Features shown", min_value=5, max_value=40, value=20, step=5)

    features = _target_features(cohort, target)
    metrics = data.load_kfold(cohort)
    target_metrics = metrics.loc[target] if target in metrics.index else pd.Series(dtype=float)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUPRC", f"{target_metrics.get('auprc_mean', np.nan):.3f}")
    c2.metric("Prevalence", f"{target_metrics.get('prevalence_mean', np.nan):.3f}")
    c3.metric("Normalized AUPRC", f"{target_metrics.get('norm_auprc', np.nan):.3f}")
    c4.metric("SHAP features", f"{len(features):,}")

    meta = data.bundle_meta()
    with st.container(border=True):
        theme.md(theme.fig_header("01", "Target-level feature summary"))
        st.plotly_chart(_feature_plot(features, int(top_n)), width="stretch")
        theme.fig_caption(
            "1",
            f"Genes ranked by how much their expression moves the prediction on average "
            f"({theme.def_chip('SHAP value')}). Purple pushes toward 'mutated', orange pushes away.",
        )
        theme.provenance([
            ("Cohort", cohort), ("Target", target), ("Features", str(len(features))),
            ("Bundle", meta["sha7"]),
        ])

    beeswarm = data.load_beeswarm(cohort, target)
    if beeswarm.empty:
        theme.notice(
            "empty", "No per-sample view for this target",
            "The summary above still applies. Pick a target marked <b>per-sample</b> to see values "
            "for individual tumors.",
        )
    else:
        beeswarm_figure, beeswarm_long = _beeswarm_plot(beeswarm, int(top_n))
        with st.container(border=True):
            theme.md(theme.fig_header("02", "Per-sample SHAP beeswarm"))
            st.plotly_chart(beeswarm_figure, width="stretch")
            theme.fig_caption(
                "2",
                "Each point is one tumor. Left-to-right position is how much that gene's expression "
                "moved the prediction; color is expression relative to other tumors for the same gene "
                "(5th-95th percentiles define Low and High). The up-and-down spread only keeps points "
                "from overlapping.",
            )
            theme.provenance([
                ("Cohort", cohort), ("Target", target),
                ("Samples", str(beeswarm["sample_id"].nunique() if "sample_id" in beeswarm else len(beeswarm))),
                ("Bundle", meta["generated"]),
            ])
        with st.expander("Show and download per-sample SHAP data"):
            st.dataframe(beeswarm_long, width="stretch", hide_index=True)
            st.download_button(
                "Download beeswarm CSV", beeswarm_long.to_csv(index=False).encode("utf-8"),
                file_name=f"{cohort}_{target}_shap_beeswarm.csv", mime="text/csv",
            )

    with st.expander("Show and download feature-summary data"):
        st.dataframe(features, width="stretch", hide_index=True)
        st.download_button(
            "Download feature summary CSV", features.to_csv(index=False).encode("utf-8"),
            file_name=f"{cohort}_{target}_shap_feature_summary.csv", mime="text/csv",
        )
