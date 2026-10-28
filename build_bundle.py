"""Build the Mutation Viewer data bundle from the E2M-analyses outputs.

Reads, under --analysis-root:
- Runs/TCGA/output/<COHORT>/cv/metrics.csv and oof_probabilities.csv (multitask network, out-of-fold)
- Runs/TCGA/output/<COHORT>/shap/<TARGET>/feature_summary.csv and sample_shap_top_features.csv.gz
- Runs/TCGA/data/mutation_events/<COHORT>_mc3.txt.gz (MC3 events, GRCh37)
- Figures/Figure3/figure3_helpers.py, so the variants match Figure 3

Writes data/, shap_bundle/, gtf_filtered.parquet and bundle_manifest.json next to this script.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pandas as pd

VIEWER_DIR = Path(__file__).resolve().parent
DATA_DIR = VIEWER_DIR / "data"
SHAP_DIR = VIEWER_DIR / "shap_bundle"
GTF_OUTPUT = VIEWER_DIR / "gtf_filtered.parquet"
MANIFEST_PATH = VIEWER_DIR / "bundle_manifest.json"

NOT_COHORTS = {"all", "tmb"}  # pan-cancer model and TMB regression
VARIANT_EXCLUDED = ["all", "UVM"]  # as in Figure 3
VARIANT_COLUMNS = {
    "chr": "Chromosome", "start": "Start_Position", "end": "End_Position",
    "reference": "Reference_Allele", "alt": "Tumor_Seq_Allele2",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, frame: pd.DataFrame | None = None) -> dict:
    record = {"path": path.relative_to(VIEWER_DIR).as_posix(), "bytes": path.stat().st_size, "sha256": sha256sum(path)}
    if frame is not None:
        record.update({"rows": len(frame), "columns": frame.columns.tolist()})
    return record


def cohort_names(results_dir: Path) -> list[str]:
    return sorted(p.name for p in results_dir.iterdir()
                  if p.is_dir() and p.name.lower() not in NOT_COHORTS and (p / "cv" / "metrics.csv").exists())


def read_metrics(results_dir: Path, cohort: str) -> pd.DataFrame:
    """Pooled out-of-fold metrics of the evaluable targets."""
    metrics = pd.read_csv(results_dir / cohort / "cv" / "metrics.csv")
    metrics = metrics[metrics["evaluable"].astype(bool)].rename(columns={"target": "gene"})
    return metrics[["gene", "n_samples", "n_positive", "prevalence", "auprc", "normalized_auprc", "roc_auc"]]


# ── Variants ─────────────────────────────────────────────────────────────────
def build_variant_data(analysis_root: Path, results_dir: Path, cohorts: list[str], threshold: float) -> dict:
    sys.path[:0] = [str(analysis_root / "Figures" / "Figure3"), str(analysis_root / "Figures")]
    from figure3_helpers import build_missense_only_df, build_silent_only_df, load_full_mutation_maf

    event_dir = analysis_root / "Runs" / "TCGA" / "data" / "mutation_events"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    silent_frames, missense_frames, summary = [], [], []
    for cohort in cohorts:
        metrics = read_metrics(results_dir, cohort)
        genes = metrics.loc[metrics["normalized_auprc"] > threshold].sort_values("normalized_auprc", ascending=False)["gene"].tolist()
        probabilities = pd.read_csv(results_dir / cohort / "cv" / "oof_probabilities.csv", index_col=0)
        genes = [g for g in genes if g in probabilities.columns]
        pred_long = (probabilities[genes].rename_axis(index="sample", columns="gene").stack()
                     .rename("pred_prob").reset_index().assign(Cancer=cohort))
        print(f"{cohort}: {len(genes)} genes", flush=True)
        if not genes:
            summary.append({"cohort": cohort, "selected_genes": 0, "silent_rows": 0, "missense_rows": 0})
            continue
        event_path = event_dir / f"{cohort}_mc3.txt.gz"
        maf = load_full_mutation_maf(event_path, pass_only=True).assign(source_file=event_path.name)
        silent = build_silent_only_df(maf, pred_long).rename(columns=VARIANT_COLUMNS)
        missense = build_missense_only_df(maf, pred_long).rename(columns=VARIANT_COLUMNS)
        silent_frames.append(silent)
        missense_frames.append(missense)
        summary.append({"cohort": cohort, "selected_genes": len(genes), "silent_rows": len(silent),
                        "missense_rows": len(missense)})

    files = []
    for name, frames in (("silent_mutations.parquet", silent_frames), ("missense_mutations.parquet", missense_frames)):
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(DATA_DIR / name, index=False)
        files.append(file_record(DATA_DIR / name, frame))
    return {"normalized_auprc_threshold": threshold, "cohorts": summary, "files": files}


# ── SHAP ─────────────────────────────────────────────────────────────────────
def beeswarm_table(path: Path) -> pd.DataFrame:
    """Per-sample SHAP of a target's top features, one column per feature plus x_<feature> for its expression
    (as Figures/Figure2 load_beeswarm_table)."""
    long = pd.read_csv(path)
    features = long["feature"].drop_duplicates().tolist()  # SHAP rank order
    shap = long.pivot(index="sample_id", columns="feature", values="shap_value")[features]
    expression = long.pivot(index="sample_id", columns="feature", values="expression_value")[features].add_prefix("x_")
    frame = pd.concat([shap, expression], axis=1).astype("float32").reset_index()  # float32 as written by the run
    frame.columns.name = None
    return frame


def build_shap_bundle(results_dir: Path, cohorts: list[str]) -> dict:
    if SHAP_DIR.exists():
        shutil.rmtree(SHAP_DIR)
    (SHAP_DIR / "cohorts").mkdir(parents=True)
    (SHAP_DIR / "beeswarms").mkdir()
    index = {"version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cohorts": {}}
    for cohort in cohorts:
        metrics = read_metrics(results_dir, cohort)
        metrics_path = SHAP_DIR / "cohorts" / f"{cohort}_metrics.parquet"
        metrics.to_parquet(metrics_path, index=False)

        target_dirs = sorted(p for p in (results_dir / cohort / "shap").glob("*") if (p / "feature_summary.csv").exists())
        features = []
        archive_path = SHAP_DIR / "beeswarms" / f"{cohort}.zip"
        beeswarm_targets = []
        with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for target_dir in target_dirs:
                summary = pd.read_csv(target_dir / "feature_summary.csv")
                features.append(summary.loc[summary["mean_abs_shap"] > 0, ["target", "feature", "mean_abs_shap", "mean_shap"]])
                sample_path = target_dir / "sample_shap_top_features.csv.gz"
                if sample_path.exists():
                    buffer = BytesIO()
                    beeswarm_table(sample_path).to_parquet(buffer, index=False)
                    archive.writestr(f"beeswarm_{target_dir.name}.parquet", buffer.getvalue())
                    beeswarm_targets.append(target_dir.name)
        features = (pd.concat(features, ignore_index=True) if features
                    else pd.DataFrame(columns=["target", "feature", "mean_abs_shap", "mean_shap"]))
        features = features.rename(columns={"target": "target_gene"}).assign(cohort=cohort)
        features = features[["cohort", "target_gene", "feature", "mean_abs_shap", "mean_shap"]]
        features_path = SHAP_DIR / "cohorts" / f"{cohort}_features.parquet"
        features.to_parquet(features_path, index=False)
        if not beeswarm_targets:
            archive_path.unlink()

        index["cohorts"][cohort] = {
            "metrics_file": f"cohorts/{metrics_path.name}",
            "features_file": f"cohorts/{features_path.name}",
            "metric_targets": int(metrics["gene"].nunique()),
            "shap_targets": int(features["target_gene"].nunique()),
            "shap_rows": len(features),
            "archive": f"beeswarms/{archive_path.name}" if beeswarm_targets else None,
            "beeswarm_targets": beeswarm_targets,
            "beeswarm_target_count": len(beeswarm_targets),
            "archive_bytes": archive_path.stat().st_size if beeswarm_targets else 0,
            "archive_sha256": sha256sum(archive_path) if beeswarm_targets else None,
        }
        print(f"{cohort}: {len(target_dirs)} SHAP targets", flush=True)
    (SHAP_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    cohorts_info = {c: {k: v[k] for k in ("metric_targets", "shap_targets", "shap_rows", "beeswarm_target_count",
                                          "archive_bytes", "archive_sha256")}
                    for c, v in index["cohorts"].items()}
    return {
        "index": "shap_bundle/index.json",
        "cohort_count": len(cohorts),
        "metric_target_count": sum(v["metric_targets"] for v in cohorts_info.values()),
        "shap_target_count": sum(v["shap_targets"] for v in cohorts_info.values()),
        "beeswarm_target_count": sum(v["beeswarm_target_count"] for v in cohorts_info.values()),
        "cohorts": cohorts_info,
    }


# ── Transcript reference ─────────────────────────────────────────────────────
def build_reference_data(gtf_path: Path) -> dict:
    genes = set()
    for path in sorted(DATA_DIR.glob("*.parquet")):
        genes.update(pd.read_parquet(path, columns=["Hugo_Symbol"])["Hugo_Symbol"].dropna().astype(str))
    rows = []
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in {"transcript", "exon", "CDS"}:
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]+)"', parts[8]))
            if attrs.get("gene_name") not in genes or not attrs.get("transcript_id"):
                continue
            rows.append({
                "feature": parts[2], "chrom": parts[0], "start": int(parts[3]), "end": int(parts[4]),
                "strand": parts[6], "gene_name": attrs["gene_name"], "gene_id": attrs.get("gene_id", ""),
                "transcript_id": attrs["transcript_id"], "transcript_name": attrs.get("transcript_name", ""),
            })
    gtf = pd.DataFrame(rows)
    gtf.to_parquet(GTF_OUTPUT, index=False)
    return {"genes": len(genes), "genes_in_gtf": int(gtf["gene_name"].nunique()),
            "transcripts_in_gtf": int(gtf["transcript_id"].nunique()), "files": [file_record(GTF_OUTPUT, gtf)]}


# ── Validation ───────────────────────────────────────────────────────────────
def validate_bundle(results_dir: Path, shap_info: dict) -> dict:
    """Every variant's probability must equal the current out-of-fold probability."""
    results = {"variant_files": {}, "shap_cohorts": {}}
    for name in ("silent_mutations.parquet", "missense_mutations.parquet"):
        frame = pd.read_parquet(DATA_DIR / name)
        checked = mismatches = 0
        for cohort, group in frame.groupby("Cancer"):
            probabilities = pd.read_csv(results_dir / str(cohort) / "cv" / "oof_probabilities.csv", index_col=0)
            probabilities.index = probabilities.index.str[:15]  # TCGA-XX-XXXX-01, as sample4
            lookup = group[["sample4", "Hugo_Symbol", "pred_prob"]].drop_duplicates()
            current = probabilities.stack().reindex(pd.MultiIndex.from_frame(lookup[["sample4", "Hugo_Symbol"]]))
            mismatches += int((abs(lookup["pred_prob"].to_numpy() - current.to_numpy()) > 1e-12).sum())
            checked += len(lookup)
        if mismatches:
            raise ValueError(f"{name}: {mismatches}/{checked} probabilities differ from the current results")
        results["variant_files"][name] = {"rows": len(frame), "probabilities_checked": checked}
    for cohort, values in shap_info["cohorts"].items():
        results["shap_cohorts"][cohort] = {
            "metric_targets": values["metric_targets"], "shap_targets": values["shap_targets"],
            "shap_rows": values["shap_rows"], "beeswarm_targets": values["beeswarm_target_count"],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, required=True, help="E2M-analyses checkout")
    parser.add_argument("--gtf", type=Path, help="GENCODE v23lift37 (GRCh37, as MC3) annotation GTF; "
                        "default Runs/TCGA/data/annotation/gencode.v23lift37.annotation.gtf.gz")
    parser.add_argument("--normalized-auprc-threshold", type=float, default=0.1, help="variant genes, as Figure 3")
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    results_dir = analysis_root / "Runs" / "TCGA" / "output"
    gtf = args.gtf or analysis_root / "Runs" / "TCGA" / "data" / "annotation" / "gencode.v23lift37.annotation.gtf.gz"
    cohorts = cohort_names(results_dir)
    variant_cohorts = [c for c in cohorts if c not in VARIANT_EXCLUDED]
    commit = (analysis_root / ".git" / "HEAD").read_text().strip()
    if commit.startswith("ref: "):
        ref = analysis_root / ".git" / commit[5:]
        commit = ref.read_text().strip() if ref.exists() else commit

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "analysis_repository": "github.com/asafpinhasitechnion/E2M-analyses",
            "analysis_commit": commit,
            "results_dir": "Runs/TCGA/output",
            "mc3_dir": "Runs/TCGA/data/mutation_events",
            "gtf": gtf.name,
        },
        "analysis_scope": {
            "variant_cohorts": variant_cohorts,
            "excluded_by_default": VARIANT_EXCLUDED,
            "reason": "Variant data matches the Figure 3 mutation-effect analysis (cancer-specific models, UVM excluded).",
        },
    }
    manifest["variants"] = build_variant_data(analysis_root, results_dir, variant_cohorts, args.normalized_auprc_threshold)
    manifest["shap"] = build_shap_bundle(results_dir, cohorts)
    manifest["reference"] = build_reference_data(gtf)
    manifest["validation"] = validate_bundle(results_dir, manifest["shap"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Bundle complete: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
