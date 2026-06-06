"""Build the manuscript mutation-viewer data bundle from current E2M results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


VIEWER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIEWER_DIR.parents[1]
FIGURE_SCRIPT_DIR = PROJECT_ROOT / "Figure_Scripts"
if str(FIGURE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPT_DIR))

from figure3_helpers import (  # noqa: E402
    build_missense_only_df,
    build_silent_only_df,
    load_full_mutation_maf,
    normalize_sample4,
)
from build_shap_bundle import build_shap_bundle  # noqa: E402


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results" / "TCGA_results" / "Lean_multitask_nn"
DEFAULT_MC3_DIR = (
    PROJECT_ROOT.parents[1]
    / "Data"
    / "RNA"
    / "TCGA"
    / "Xena"
    / "tcga_xena_mutations_data"
)
DEFAULT_GTF = PROJECT_ROOT.parent / "input" / "gencode.v23lift37.annotation.gtf.gz"
DEFAULT_FASTA = PROJECT_ROOT.parent / "input" / "gencode.v23lift37.transcripts.fa.gz"

VARIANT_COLUMNS = {
    "chr": "Chromosome",
    "start": "Start_Position",
    "end": "End_Position",
    "reference": "Reference_Allele",
    "alt": "Tumor_Seq_Allele2",
}


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


def cohort_names(results_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted({name.upper() for name in requested})
    excluded = {"all", "shap_beeswarm", "tmb_prediction"}
    names = []
    for path in results_dir.iterdir():
        if path.is_dir() and path.name.lower() not in excluded:
            if (path / "kfold_prediction" / "summary.csv").exists():
                names.append(path.name)
    # Figure 3 excludes UVM because its small cohort was not used in that analysis.
    return sorted(name for name in names if name.upper() != "UVM")


def selected_genes(summary_path: Path, threshold: float) -> list[str]:
    summary = pd.read_csv(summary_path, index_col=0)
    score = (summary["auprc_mean"] - summary["prevalence_mean"]) / (
        1.0 - summary["prevalence_mean"]
    )
    return score.loc[score > threshold].sort_values(ascending=False).index.astype(str).tolist()


def prediction_long(probability_path: Path, genes: list[str], cancer: str) -> pd.DataFrame:
    probabilities = pd.read_csv(probability_path, index_col=0)
    genes = [gene for gene in genes if gene in probabilities.columns]
    if not genes:
        return pd.DataFrame(columns=["Cancer", "sample", "gene", "pred_prob"])
    out = (
        probabilities.loc[:, genes]
        .rename_axis(index="sample", columns="gene")
        .stack()
        .rename("pred_prob")
        .reset_index()
    )
    out.insert(0, "Cancer", cancer)
    return out


def standardize_variant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={key: value for key, value in VARIANT_COLUMNS.items() if key in frame})
    if "Tumor_Sample_Barcode" not in frame and "sample4" in frame:
        frame["Tumor_Sample_Barcode"] = frame["sample4"]
    return frame


def build_variant_data(
    results_dir: Path,
    mc3_dir: Path,
    cohorts: list[str],
    threshold: float,
    data_dir: Path,
) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    silent_frames = []
    missense_frames = []
    cohort_summary = []

    for index, cancer in enumerate(cohorts, start=1):
        cohort_dir = results_dir / cancer
        summary_path = cohort_dir / "kfold_prediction" / "summary.csv"
        probability_path = (
            cohort_dir
            / "kfold_prediction"
            / "combined_predictions"
            / "probabilities.csv"
        )
        event_path = mc3_dir / f"{cancer}_mc3.txt.gz"
        missing = [path for path in (summary_path, probability_path, event_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing inputs for {cancer}: {missing}")

        genes = selected_genes(summary_path, threshold)
        pred_long = prediction_long(probability_path, genes, cancer)
        print(
            f"[{index}/{len(cohorts)}] {cancer}: "
            f"{len(genes)} selected genes, {len(pred_long):,} prediction rows",
            flush=True,
        )
        if pred_long.empty:
            cohort_summary.append(
                {"cohort": cancer, "selected_genes": 0, "silent_rows": 0, "missense_rows": 0}
            )
            continue

        maf = load_full_mutation_maf(event_path, pass_only=True).assign(
            source_file=event_path.name,
        )
        silent = standardize_variant_columns(build_silent_only_df(maf, pred_long))
        missense = standardize_variant_columns(build_missense_only_df(maf, pred_long))
        silent_frames.append(silent)
        missense_frames.append(missense)
        cohort_summary.append(
            {
                "cohort": cancer,
                "selected_genes": len(genes),
                "silent_rows": len(silent),
                "missense_rows": len(missense),
            }
        )

    silent_all = pd.concat(silent_frames, ignore_index=True) if silent_frames else pd.DataFrame()
    missense_all = pd.concat(missense_frames, ignore_index=True) if missense_frames else pd.DataFrame()
    silent_path = data_dir / "silent_mutations.parquet"
    missense_path = data_dir / "missense_mutations.parquet"
    silent_all.to_parquet(silent_path, index=False)
    missense_all.to_parquet(missense_path, index=False)

    return {
        "normalized_auprc_threshold": threshold,
        "cohorts": cohort_summary,
        "files": [file_record(silent_path, silent_all), file_record(missense_path, missense_all)],
    }


def parse_gtf_attributes(raw: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+) "([^"]+)"', raw))


def collect_viewer_genes(data_dir: Path) -> set[str]:
    genes: set[str] = set()
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["Hugo_Symbol"])
        genes.update(frame["Hugo_Symbol"].dropna().astype(str))
    return genes


def build_reference_data(gtf_path: Path, fasta_path: Path, data_dir: Path) -> dict:
    genes = collect_viewer_genes(data_dir)
    rows = []
    transcript_ids: set[str] = set()
    keep_features = {"transcript", "exon", "CDS"}

    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in keep_features:
                continue
            attrs = parse_gtf_attributes(parts[8])
            if attrs.get("gene_name") not in genes:
                continue
            transcript_id = attrs.get("transcript_id", "")
            if not transcript_id:
                continue
            transcript_ids.add(transcript_id.split(".")[0])
            rows.append(
                {
                    "feature": parts[2],
                    "chrom": parts[0],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "gene_name": attrs.get("gene_name", ""),
                    "gene_id": attrs.get("gene_id", ""),
                    "transcript_id": transcript_id,
                    "transcript_name": attrs.get("transcript_name", ""),
                }
            )

    gtf_frame = pd.DataFrame(rows)
    sequences: dict[str, str] = {}
    current_id = None
    current_sequence: list[str] = []
    with gzip.open(fasta_path, "rt") as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_sequence)
                candidate = line[1:].split("|")[0].split(".")[0]
                current_id = candidate if candidate in transcript_ids else None
                current_sequence = []
            elif current_id is not None:
                current_sequence.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_sequence)

    sequence_frame = pd.DataFrame(
        {"transcript_id": list(sequences), "sequence": list(sequences.values())}
    )
    gtf_output = VIEWER_DIR / "gtf_filtered.parquet"
    sequence_output = VIEWER_DIR / "sequences_filtered.parquet"
    gtf_frame.to_parquet(gtf_output, index=False)
    sequence_frame.to_parquet(sequence_output, index=False)
    return {
        "genes": len(genes),
        "transcripts_in_gtf": int(gtf_frame["transcript_id"].nunique()),
        "transcript_sequences": len(sequence_frame),
        "files": [file_record(gtf_output, gtf_frame), file_record(sequence_output, sequence_frame)],
    }


def reset_directory(path: Path) -> None:
    def remove_readonly(function, target, _exc_info):
        os.chmod(target, stat.S_IWRITE)
        function(target)

    if path.exists():
        shutil.rmtree(path, onerror=remove_readonly)
    path.mkdir(parents=True, exist_ok=True)


def file_record(path: Path, frame: pd.DataFrame | None = None) -> dict:
    record = {
        "path": path.relative_to(VIEWER_DIR).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256sum(path),
    }
    if frame is not None:
        record.update({"rows": len(frame), "columns": frame.columns.tolist()})
    return record


def existing_shap_bundle_info(shap_dir: Path) -> dict:
    index_path = shap_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "No existing SHAP bundle found. Supply --shap-source with the complete server export."
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cohorts = index.get("cohorts", {})
    cohort_summary = {
        cohort: {
            key: values.get(key)
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
        for cohort, values in cohorts.items()
    }
    return {
        "format": "indexed_cohort_shap_bundle_v1",
        "index": str(index_path.relative_to(VIEWER_DIR)),
        "beeswarm_source": index.get("beeswarm_source"),
        "cohort_count": len(cohorts),
        "metric_target_count": sum(v.get("metric_targets", 0) for v in cohorts.values()),
        "shap_target_count": sum(v.get("shap_targets", 0) for v in cohorts.values()),
        "beeswarm_target_count": sum(v.get("beeswarm_target_count", 0) for v in cohorts.values()),
        "cohorts": cohort_summary,
    }


def validate_bundle(results_dir: Path, data_dir: Path, shap_dir: Path) -> dict:
    results = {"variant_files": {}, "shap_cohorts": {}}
    for name in ("silent_mutations.parquet", "missense_mutations.parquet"):
        path = data_dir / name
        frame = pd.read_parquet(path)
        required = {
            "Cancer",
            "sample4",
            "Hugo_Symbol",
            "pred_prob",
            "Chromosome",
            "Start_Position",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        mismatches = 0
        checked = 0
        for cancer, group in frame.groupby("Cancer"):
            probability_path = (
                results_dir
                / str(cancer)
                / "kfold_prediction"
                / "combined_predictions"
                / "probabilities.csv"
            )
            probabilities = pd.read_csv(probability_path, index_col=0)
            probabilities.index = normalize_sample4(pd.Series(probabilities.index)).to_numpy()
            lookup = group[["sample4", "Hugo_Symbol", "pred_prob"]].drop_duplicates()
            current = []
            for row in lookup.itertuples(index=False):
                value = probabilities.at[row.sample4, row.Hugo_Symbol]
                current.append(value)
            delta = (lookup["pred_prob"].to_numpy() - pd.Series(current).to_numpy())
            checked += len(delta)
            mismatches += int((abs(delta) > 1e-12).sum())
        if mismatches:
            raise ValueError(f"{name}: {mismatches}/{checked} probabilities differ from current results")
        results["variant_files"][name] = {"rows": len(frame), "probabilities_checked": checked}

    shap_index_path = shap_dir / "index.json"
    if shap_index_path.exists():
        shap_index = json.loads(shap_index_path.read_text(encoding="utf-8"))
        for cohort, values in shap_index.get("cohorts", {}).items():
            results["shap_cohorts"][cohort] = {
                "metric_targets": values.get("metric_targets", 0),
                "shap_targets": values.get("shap_targets", 0),
                "shap_rows": values.get("shap_rows", 0),
                "beeswarm_targets": values.get("beeswarm_target_count", 0),
            }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--mc3-dir", type=Path, default=DEFAULT_MC3_DIR)
    parser.add_argument("--shap-source", type=Path)
    parser.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--normalized-auprc-threshold", type=float, default=0.1)
    parser.add_argument("--cohorts", nargs="+")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = VIEWER_DIR / "data"
    shap_dir = VIEWER_DIR / "shap_bundle"
    cohorts = cohort_names(args.results_dir, args.cohorts)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": ".",
        "sources": {
            "results_dir": portable_path(args.results_dir),
            "mc3_dir": portable_path(args.mc3_dir),
            "gtf": portable_path(args.gtf),
            "fasta": portable_path(args.fasta),
        },
        "analysis_scope": {
            "variant_cohorts": cohorts,
            "excluded_by_default": ["all", "UVM"],
            "reason": "Matches the maintained Figure 3 mutation-effect analysis scope.",
        },
    }

    if not args.skip_variants:
        manifest["variants"] = build_variant_data(
            args.results_dir,
            args.mc3_dir,
            cohorts,
            args.normalized_auprc_threshold,
            data_dir,
        )
    if not args.skip_shap:
        if args.shap_source is None:
            manifest["shap"] = existing_shap_bundle_info(shap_dir)
        else:
            manifest["sources"]["shap_source"] = portable_path(args.shap_source)
            manifest["shap"] = build_shap_bundle(args.results_dir, args.shap_source, shap_dir)
    if not args.skip_reference:
        manifest["reference"] = build_reference_data(args.gtf, args.fasta, data_dir)

    manifest["validation"] = validate_bundle(args.results_dir, data_dir, shap_dir)
    manifest_path = VIEWER_DIR / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Bundle complete: {manifest_path}")


if __name__ == "__main__":
    main()
