"""
Data-access layer for the Mutation Viewer companion.

Single source of truth = the frozen, hashed `bundle_manifest.json` plus the parquet/CSV
files it describes. Everything the UI calls about coverage, cohorts, genes, transcripts and
provenance is derived here, so it stays correct as the bundle grows.
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import streamlit as st

VIEWER_DIR = Path(__file__).resolve().parent
DATA_DIR = VIEWER_DIR / "data"
SHAP_DIR = VIEWER_DIR / "shap_bundle"
SHAP_INDEX_PATH = SHAP_DIR / "index.json"
MANIFEST_PATH = VIEWER_DIR / "bundle_manifest.json"
GTF_PATH = VIEWER_DIR / "gtf_filtered.parquet"
SEQ_PATH = VIEWER_DIR / "sequences_filtered.parquet"

# SHAP interpretation used a looser gate than the variant analysis (Methods).
VARIANT_NORM_AUPRC_THRESHOLD = 0.1
SHAP_NORM_AUPRC_THRESHOLD = 0.05

DATA_CATALOG = {
    "missense_mutations.parquet": "TCGA - missense variants (MC3)",
    "silent_mutations.parquet": "TCGA - silent variants (MC3)",
}

# TCGA study abbreviations -> readable cohort names (subset that appears in the bundle).
COHORT_LABELS = {
    "ACC": "Adrenocortical carcinoma", "BLCA": "Bladder urothelial", "BRCA": "Breast",
    "CESC": "Cervical", "CHOL": "Cholangiocarcinoma", "COAD": "Colon adeno.",
    "DLBC": "Lymphoma (DLBCL)", "ESCA": "Esophageal", "GBM": "Glioblastoma",
    "HNSC": "Head & neck", "KICH": "Kidney chromophobe", "KIRC": "Kidney clear-cell",
    "KIRP": "Kidney papillary", "LGG": "Lower-grade glioma", "LIHC": "Liver",
    "LUAD": "Lung adeno.", "LUSC": "Lung squamous", "MESO": "Mesothelioma",
    "OV": "Ovarian", "PAAD": "Pancreatic", "PCPG": "Paraganglioma", "PRAD": "Prostate",
    "READ": "Rectum adeno.", "SARC": "Sarcoma", "SKCM": "Melanoma", "STAD": "Stomach",
    "TGCT": "Testicular germ cell", "THCA": "Thyroid", "THYM": "Thymoma",
    "UCEC": "Uterine endometrial", "UCS": "Uterine carcinosarcoma", "UVM": "Uveal melanoma",
}


def cohort_label(code: str) -> str:
    return COHORT_LABELS.get(str(code).upper(), str(code))


# ── Manifest / bundle metadata ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def bundle_sha7() -> str:
    if not MANIFEST_PATH.exists():
        return "unknown"
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()[:7]


def bundle_meta() -> dict:
    m = load_manifest()
    generated = (m.get("generated_at_utc") or "")[:10] or "unknown"
    variants = m.get("variants", {})
    scope = m.get("analysis_scope", {})
    validation = m.get("validation", {})
    return {
        "generated": generated,
        "sha7": bundle_sha7(),
        "n_sources": len(m.get("sources", {})),
        "variant_cohorts": scope.get("variant_cohorts", []),
        "shap_cohorts": sorted(validation.get("shap_cohorts", {}).keys()),
        "excluded": scope.get("excluded_by_default", []),
        "norm_threshold": variants.get("normalized_auprc_threshold", VARIANT_NORM_AUPRC_THRESHOLD),
        "sources": m.get("sources", {}),
        "variant_files": variants.get("files", []),
    }


# ── Datasets / variants ──────────────────────────────────────────────────────
def available_datasets() -> list[tuple[str, str]]:
    if not DATA_DIR.exists():
        return []
    files = sorted(DATA_DIR.glob("*.parquet"))
    return [(f.name, DATA_CATALOG.get(f.name, f.stem.replace("_", " ").title())) for f in files]


@st.cache_data(show_spinner="Loading variants...")
def load_variants(filenames: tuple[str, ...]) -> pd.DataFrame:
    frames = [pd.read_parquet(DATA_DIR / f) for f in filenames if (DATA_DIR / f).exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def cohorts_in(df: pd.DataFrame) -> list[str]:
    if df.empty or "Cancer" not in df:
        return []
    return sorted(df["Cancer"].dropna().astype(str).unique())


def genes_in(df: pd.DataFrame, cohort: str | None = None) -> list[str]:
    if df.empty or "Hugo_Symbol" not in df:
        return []
    sub = df if cohort is None else df[df["Cancer"].astype(str) == cohort]
    return sorted(sub["Hugo_Symbol"].dropna().astype(str).unique())


@st.cache_data(show_spinner="Loading transcript reference...")
def load_gtf() -> pd.DataFrame:
    if not GTF_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(GTF_PATH)


@st.cache_data(show_spinner=False)
def transcripts_for_gene(gene: str) -> pd.DataFrame:
    """Return transcripts ranked by coding span, then exon span."""
    gtf = load_gtf()
    if gtf.empty:
        return pd.DataFrame()
    rows = gtf.loc[gtf["gene_name"].astype(str).eq(str(gene))].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["length"] = rows["end"] - rows["start"] + 1
    ranked = (
        rows.pivot_table(
            index=["transcript_id", "transcript_name", "chrom", "strand"],
            columns="feature",
            values="length",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for column in ("CDS", "exon"):
        if column not in ranked:
            ranked[column] = 0
    return ranked.sort_values(
        ["CDS", "exon", "transcript_id"], ascending=[False, False, True]
    ).reset_index(drop=True)


# ── SHAP bundle ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_shap_index() -> dict:
    if not SHAP_INDEX_PATH.exists():
        return {}
    return json.loads(SHAP_INDEX_PATH.read_text(encoding="utf-8"))


def _shap_cohort_info(cohort: str) -> dict:
    return load_shap_index().get("cohorts", {}).get(str(cohort), {})


@lru_cache(maxsize=1)
def shap_cohorts() -> tuple[str, ...]:
    return tuple(sorted(load_shap_index().get("cohorts", {})))


@st.cache_data(show_spinner=False)
def load_kfold(cohort: str) -> pd.DataFrame:
    """Per-gene CV metrics for a cohort: index=gene, cols incl auprc_mean, prevalence_mean."""
    relative_path = _shap_cohort_info(cohort).get("metrics_file")
    if not relative_path:
        return pd.DataFrame()
    path = SHAP_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "gene" not in df.columns:
        return pd.DataFrame()
    df = df.set_index("gene")
    df.index = df.index.astype(str)
    if {"auprc_mean", "prevalence_mean"}.issubset(df.columns):
        prev = df["prevalence_mean"].clip(upper=0.999999)
        df["norm_auprc"] = (df["auprc_mean"] - prev) / (1.0 - prev)
    return df


@st.cache_data(show_spinner=False)
def load_shap_matrix(cohort: str) -> pd.DataFrame:
    relative_path = _shap_cohort_info(cohort).get("features_file")
    if not relative_path:
        return pd.DataFrame()
    path = SHAP_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def shap_targets(cohort: str) -> list[str]:
    info = _shap_cohort_info(cohort)
    beeswarm_targets = list(info.get("beeswarm_targets", []))
    if info.get("shap_targets", 0) == len(beeswarm_targets):
        return beeswarm_targets
    matrix = load_shap_matrix(cohort)
    if matrix.empty:
        return []
    return sorted(matrix["target_gene"].dropna().astype(str).unique())


def beeswarm_genes(cohort: str) -> list[str]:
    return list(_shap_cohort_info(cohort).get("beeswarm_targets", []))


@st.cache_data(show_spinner=False)
def load_beeswarm(cohort: str, gene: str) -> pd.DataFrame:
    info = _shap_cohort_info(cohort)
    relative_path = info.get("archive")
    if not relative_path or gene not in set(info.get("beeswarm_targets", [])):
        return pd.DataFrame()
    archive_path = SHAP_DIR / relative_path
    if not archive_path.exists():
        return pd.DataFrame()
    member = f"beeswarm_{gene}.parquet"
    with ZipFile(archive_path) as archive:
        try:
            payload = archive.read(member)
        except KeyError:
            return pd.DataFrame()
    return pd.read_parquet(BytesIO(payload))


@st.cache_data(show_spinner=False)
def shap_feature_counts(cohort: str) -> pd.Series:
    """Per-target count of expression features with non-zero mean SHAP."""
    mat = load_shap_matrix(cohort)
    if mat.empty:
        return pd.Series(dtype=int)
    return mat.groupby("target_gene").size().astype(int)


@st.cache_data(show_spinner=False)
def gene_metrics(cohort: str) -> pd.DataFrame:
    """Performance table for the model-performance page: genes within a cohort.

    Columns: gene, auprc_mean, prevalence_mean, norm_auprc, n_shap_features.
    """
    kf = load_kfold(cohort)
    if kf.empty:
        return pd.DataFrame()
    out = kf.reset_index().rename(columns={kf.index.name or "index": "gene"})
    if "gene" not in out.columns:
        out = out.rename(columns={out.columns[0]: "gene"})
    counts = shap_feature_counts(cohort)
    out["n_shap_features"] = out["gene"].map(counts).fillna(0).astype(int)
    return out


# ── Coverage (derived from manifest validation) ──────────────────────────────
@st.cache_data(show_spinner=False)
def coverage_rows() -> list[dict]:
    m = load_manifest()
    validation = m.get("validation", {}).get("shap_cohorts", {})
    variant_cohorts = {c["cohort"]: c for c in m.get("variants", {}).get("cohorts", [])}
    all_cohorts = sorted(set(validation) | set(variant_cohorts))
    # variant cohorts first
    all_cohorts.sort(key=lambda c: (c not in variant_cohorts, c))

    rows = []
    for c in all_cohorts:
        v = validation.get(c, {})
        mt = v.get("metric_targets", 0)
        bee = v.get("beeswarm_targets", 0)
        shap_n = v.get("shap_targets", 0)
        vc = variant_cohorts.get(c, {})
        sel = vc.get("selected_genes", 0)
        rows.append({
            "cohort": c,
            "label": cohort_label(c),
            "perf": ("full", f"{mt} targets") if mt else ("none", "none"),
            "variant": ("full", f"{sel} genes") if sel else ("none", "none"),
            "shap": ("full", f"{shap_n} targets") if shap_n else ("none", "none"),
            "bee": ("partial", f"{bee} targets") if bee else ("none", "none"),
        })
    return rows
