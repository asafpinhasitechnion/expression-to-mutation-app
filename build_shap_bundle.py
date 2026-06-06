"""Build the indexed SHAP bundle used by the manuscript viewer.

Target-level metrics and feature summaries come from the canonical lean result bundle.
Per-sample beeswarms are optional and are stored as one uncompressed ZIP archive per
cohort. Parquet already provides compression, so ZIP_STORED avoids wasted CPU while
allowing the app to read only the selected target into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pandas as pd


VIEWER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIEWER_DIR.parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results" / "TCGA_results" / "Lean_multitask_nn"
DEFAULT_OUTPUT_DIR = VIEWER_DIR / "shap_bundle"
EXCLUDED_RESULT_DIRS = {"all", "shap_beeswarm", "tmb_prediction"}


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_names(results_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in results_dir.iterdir()
        if path.is_dir()
        and path.name.lower() not in EXCLUDED_RESULT_DIRS
        and (path / "kfold_prediction" / "summary.csv").exists()
    )


def read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0).rename_axis("gene").reset_index()
    frame["gene"] = frame["gene"].astype(str)
    if {"auprc_mean", "prevalence_mean"}.issubset(frame.columns):
        prevalence = frame["prevalence_mean"].clip(upper=0.999999)
        frame["norm_auprc"] = (frame["auprc_mean"] - prevalence) / (1.0 - prevalence)
    return frame


def read_features(path: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {}
    if "target" in frame.columns and "target_gene" not in frame.columns:
        rename["target"] = "target_gene"
    frame = frame.rename(columns=rename)
    required = {"target_gene", "feature", "mean_shap"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "mean_abs_shap" not in frame.columns:
        frame["mean_abs_shap"] = frame["mean_shap"].abs()
    frame["cohort"] = cohort
    return frame[["cohort", "target_gene", "feature", "mean_abs_shap", "mean_shap"]]


def _archive_genes(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        return sorted(
            Path(name).stem.removeprefix("beeswarm_")
            for name in archive.namelist()
            if Path(name).name.startswith("beeswarm_") and name.endswith(".parquet")
        )


def _copy_archive(source: Path, destination: Path) -> list[str]:
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return _archive_genes(destination)


def _write_archive(files: list[Path], destination: Path, cohort: str) -> list[str]:
    genes = []
    with ZipFile(destination, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(files):
            prefix = f"{cohort}_beeswarm_"
            if path.name.startswith(prefix):
                gene = path.stem.removeprefix(prefix)
            else:
                gene = path.stem.removeprefix("beeswarm_")
            archive.write(path, arcname=f"beeswarm_{gene}.parquet")
            genes.append(gene)
    return sorted(genes)


def package_beeswarms(source: Path, destination_dir: Path, cohorts: list[str]) -> dict[str, dict]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for old_archive in destination_dir.glob("*.zip"):
        old_archive.unlink()

    packaged: dict[str, dict] = {}
    for cohort in cohorts:
        destination = destination_dir / f"{cohort}.zip"
        archive_candidates = [
            source / f"{cohort}.zip",
            source / f"{cohort}_beeswarms.zip",
            source / f"{cohort}_shap_beeswarms.zip",
        ]
        archive_source = next((path for path in archive_candidates if path.exists()), None)
        cohort_dir = source / cohort / "shap"
        if not cohort_dir.exists():
            cohort_dir = source / cohort

        if archive_source is not None:
            genes = _copy_archive(archive_source, destination)
        else:
            files = sorted(cohort_dir.glob("beeswarm_*.parquet")) if cohort_dir.exists() else []
            if not files:
                files = sorted(source.glob(f"{cohort}_beeswarm_*.parquet"))
            if not files:
                continue
            genes = _write_archive(files, destination, cohort)

        packaged[cohort] = {
            "archive": f"beeswarms/{destination.name}",
            "beeswarm_targets": genes,
            "beeswarm_target_count": len(genes),
            "archive_bytes": destination.stat().st_size,
            "archive_sha256": sha256sum(destination),
        }
    return packaged


def add_archive_only_summaries(
    features: pd.DataFrame,
    cohort: str,
    archive_path: Path,
    archived_genes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Derive target summaries when a valid beeswarm is absent from the lean table."""
    summarized = set(features["target_gene"].astype(str)) if not features.empty else set()
    missing = sorted(set(archived_genes).difference(summarized))
    if not missing:
        return features, []

    additions = []
    with ZipFile(archive_path) as archive:
        for gene in missing:
            frame = pd.read_parquet(BytesIO(archive.read(f"beeswarm_{gene}.parquet")))
            feature_columns = [
                column
                for column in frame.columns
                if column != "sample_id"
                and not column.startswith("x_")
                and f"x_{column}" in frame.columns
            ]
            if not feature_columns:
                continue
            values = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
            summary = pd.DataFrame({
                "cohort": cohort,
                "target_gene": gene,
                "feature": feature_columns,
                "mean_abs_shap": values.abs().mean(axis=0).to_numpy(),
                "mean_shap": values.mean(axis=0).to_numpy(),
            })
            additions.append(summary)
    if not additions:
        return features, []
    return pd.concat([features, *additions], ignore_index=True), missing


def build_shap_bundle(results_dir: Path, beeswarm_source: Path, output_dir: Path) -> dict:
    results_dir = results_dir.resolve()
    beeswarm_source = beeswarm_source.resolve()
    output_dir = output_dir.resolve()
    cohorts_dir = output_dir / "cohorts"
    beeswarms_dir = output_dir / "beeswarms"
    cohorts_dir.mkdir(parents=True, exist_ok=True)
    for old_file in cohorts_dir.glob("*.parquet"):
        old_file.unlink()

    cohorts = cohort_names(results_dir)
    beeswarms = package_beeswarms(beeswarm_source, beeswarms_dir, cohorts)
    source_index_path = beeswarm_source / "index.json"
    source_description = str(beeswarm_source)
    if source_index_path.exists():
        source_description = json.loads(source_index_path.read_text(encoding="utf-8")).get(
            "source", source_description
        )
    index = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_source": portable_path(results_dir),
        "beeswarm_source": source_description,
        "storage": "One metrics parquet, one feature-summary parquet, and at most one beeswarm ZIP per cohort.",
        "cohorts": {},
    }

    for cohort in cohorts:
        summary_path = results_dir / cohort / "kfold_prediction" / "summary.csv"
        features_path = results_dir / cohort / "shap" / "shap_feature_summary_long.csv"
        metrics = read_metrics(summary_path)
        features = read_features(features_path, cohort) if features_path.exists() else pd.DataFrame(
            columns=["cohort", "target_gene", "feature", "mean_abs_shap", "mean_shap"]
        )
        beeswarm_info = beeswarms.get(cohort, {})
        derived_targets = []
        if beeswarm_info:
            features, derived_targets = add_archive_only_summaries(
                features,
                cohort,
                output_dir / beeswarm_info["archive"],
                beeswarm_info["beeswarm_targets"],
            )
        metrics_output = cohorts_dir / f"{cohort}_metrics.parquet"
        features_output = cohorts_dir / f"{cohort}_features.parquet"
        metrics.to_parquet(metrics_output, index=False)
        features.to_parquet(features_output, index=False)

        details = {
            "metrics_file": f"cohorts/{metrics_output.name}",
            "features_file": f"cohorts/{features_output.name}",
            "metric_targets": int(metrics["gene"].nunique()),
            "shap_targets": int(features["target_gene"].nunique()) if not features.empty else 0,
            "shap_rows": len(features),
            "metrics_bytes": metrics_output.stat().st_size,
            "features_bytes": features_output.stat().st_size,
            "derived_summary_targets": derived_targets,
        }
        details.update(beeswarms.get(cohort, {
            "archive": None,
            "beeswarm_targets": [],
            "beeswarm_target_count": 0,
            "archive_bytes": 0,
            "archive_sha256": None,
        }))
        index["cohorts"][cohort] = details

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    manifest_cohorts = {
        cohort: {
            key: values[key]
            for key in (
                "metric_targets",
                "shap_targets",
                "shap_rows",
                "beeswarm_target_count",
                "archive_bytes",
                "archive_sha256",
                "derived_summary_targets",
            )
        }
        for cohort, values in index["cohorts"].items()
    }
    return {
        "format": "indexed_cohort_shap_bundle_v1",
        "index": str(index_path.relative_to(VIEWER_DIR)),
        "beeswarm_source": source_description,
        "cohort_count": len(cohorts),
        "metric_target_count": sum(v["metric_targets"] for v in index["cohorts"].values()),
        "shap_target_count": sum(v["shap_targets"] for v in index["cohorts"].values()),
        "beeswarm_target_count": sum(v["beeswarm_target_count"] for v in index["cohorts"].values()),
        "cohorts": manifest_cohorts,
    }


def update_viewer_manifest(info: dict, results_dir: Path, beeswarm_source: Path) -> None:
    manifest_path = VIEWER_DIR / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("sources", {})["shap_summary_results"] = str(results_dir.resolve())
    manifest["sources"]["shap_beeswarms"] = info["beeswarm_source"]
    manifest["shap"] = info
    manifest.setdefault("validation", {})["shap_cohorts"] = {
        cohort: {
            "metric_targets": values["metric_targets"],
            "shap_targets": values["shap_targets"],
            "shap_rows": values["shap_rows"],
            "beeswarm_targets": values["beeswarm_target_count"],
        }
        for cohort, values in info["cohorts"].items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--beeswarm-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-manifest-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = build_shap_bundle(args.results_dir, args.beeswarm_source, args.output_dir)
    if not args.no_manifest_update:
        update_viewer_manifest(info, args.results_dir, args.beeswarm_source)
    print(
        f"SHAP bundle complete: {info['cohort_count']} cohorts, "
        f"{info['shap_target_count']:,} target summaries, "
        f"{info['beeswarm_target_count']:,} per-sample beeswarms"
    )


if __name__ == "__main__":
    main()
