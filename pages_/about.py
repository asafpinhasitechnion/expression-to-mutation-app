"""About / model card."""

from __future__ import annotations

import streamlit as st

import data
import theme
from theme import esc, md


def render() -> None:
    meta = data.bundle_meta()

    theme.page_head(
        "About",
        "Expression to Mutation",
        "Mutations leave marks on a tumor's gene expression. This study asks whether expression "
        "alone is enough to tell which genes are mutated.",
    )

    md(
        '<p class="t-sec" style="max-width:76ch">We trained a separate model for each TCGA cancer '
        "type. For every commonly mutated gene, the model predicts whether a tumor carries a "
        "mutation in that gene, using only its RNA expression. This app lets you explore the "
        "results behind the manuscript.</p>"
    )

    md(
        '<div class="pred-card">'
        '<div><div class="lbl">Input</div><div class="val">Tumor RNA expression</div></div>'
        '<div class="arr">&#8594;</div>'
        '<div><div class="lbl">Model</div><div class="val">Cancer-specific predictor</div></div>'
        '<div class="arr">&#8594;</div>'
        '<div><div class="lbl">Output</div><div class="val">Mutation probability</div></div>'
        "</div>"
    )

    st.markdown("### Explore the results")
    st.markdown(
        "- **Model performance:** see how well the model predicts each gene, by cancer type.\n"
        "- **Mutation viewer:** see real mutations laid out along a gene's transcript.\n"
        "- **SHAP explorer:** see which genes' expression drives each prediction."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Variant cohorts", len(meta["variant_cohorts"]))
    c2.metric("SHAP cohorts", len(meta["shap_cohorts"]))
    c3.metric("Data generated", meta["generated"])

    theme.caveat()

    with st.expander("Definitions"):
        theme.glossary_block()

    with st.expander("Coverage and data provenance"):
        theme.coverage_table(data.coverage_rows())
        excluded = ", ".join(meta["excluded"]) or "none"
        st.markdown("#### Bundle details")
        md(
            '<dl class="kv">'
            f'<dt>Generated</dt><dd class="mono">{esc(meta["generated"])}</dd>'
            f'<dt>Bundle SHA</dt><dd class="mono">{esc(meta["sha7"])}</dd>'
            f'<dt>Variant threshold</dt><dd>Normalized AUPRC &gt; {esc(meta["norm_threshold"])}</dd>'
            f'<dt>Excluded</dt><dd class="mono">{esc(excluded)}</dd>'
            '<dt>Variants</dt><dd>MC3 somatic mutation events</dd>'
            '<dt>Probabilities</dt><dd>Held-out multitask neural-network predictions</dd>'
            "</dl>"
        )
