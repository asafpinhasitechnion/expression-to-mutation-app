"""
preprocess_reference.py
=======================
One-time script: filter the full Gencode GTF and FASTA down to only the
transcripts present in data/*.parquet, then save as fast parquet files that
the app can load in seconds.

Configuration
-------------
Set the two environment variables before running, or edit the fallback paths
in the CONFIG block below:

    set GENCODE_GTF=C:/path/to/gencode.v23.annotation.gtf.gz
    set GENCODE_FASTA=C:/path/to/gencode.v23.transcripts.fa.gz

Run from Figure_Scripts_Final/:
    python mutation_viewer/preprocess_reference.py

Outputs (written next to this script):
    mutation_viewer/gtf_filtered.parquet       -- transcript/exon/CDS rows
    mutation_viewer/sequences_filtered.parquet -- transcript_id -> sequence
"""

from __future__ import annotations

import gzip
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

# Read from environment variables; fall back to the paths used during initial
# preprocessing.  Edit the fallback strings if you run this on a new machine.
GTF_PATH = Path(os.environ.get(
    "GENCODE_GTF",
    r"C:\Users\KerenYlab.MEDICINE\OneDrive - Technion\Asaf"
    r"\Expression_to_Mutation\input\gencode.v23.annotation.gtf.gz",
))
FASTA_PATH = Path(os.environ.get(
    "GENCODE_FASTA",
    r"C:\Users\KerenYlab.MEDICINE\OneDrive - Technion\Asaf"
    r"\Expression_to_Mutation\input\gencode.v23.transcripts.fa.gz",
))

DATA_DIR     = SCRIPT_DIR / "data"
OUT_GTF_PATH = SCRIPT_DIR / "gtf_filtered.parquet"
OUT_SEQ_PATH = SCRIPT_DIR / "sequences_filtered.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_version(eid: str) -> str:
    return eid.split(".")[0] if isinstance(eid, str) else str(eid)


def collect_transcript_ids() -> set[str]:
    """Return version-stripped transcript IDs from every parquet in data/."""
    ids: set[str] = set()
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"  WARNING: no parquet files found in {DATA_DIR}", file=sys.stderr)
        return ids
    for path in parquet_files:
        df = pd.read_parquet(path)
        for col in ("Transcript_ID", "Feature"):
            if col in df.columns:
                ids.update(strip_version(t) for t in df[col].dropna().unique())
    print(f"  Collected {len(ids)} unique transcript IDs from {len(parquet_files)} file(s).")
    return ids


# ── GTF filtering ─────────────────────────────────────────────────────────────

def filter_gtf(target_ids: set[str]) -> pd.DataFrame:
    print(f"  Parsing GTF: {GTF_PATH.name} ...")
    attr_re      = re.compile(r'(\w+) "([^"]+)"')
    keep_features = {"transcript", "exon", "CDS"}
    rows: list[dict] = []
    matched = 0

    with gzip.open(GTF_PATH, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in keep_features:
                continue
            attrs = dict(attr_re.findall(parts[8]))
            tx_id = strip_version(attrs.get("transcript_id", ""))
            if tx_id not in target_ids:
                continue
            rows.append({
                "feature":         parts[2],
                "chrom":           parts[0],
                "start":           int(parts[3]),
                "end":             int(parts[4]),
                "strand":          parts[6],
                "gene_name":       attrs.get("gene_name", ""),
                "gene_id":         attrs.get("gene_id", ""),
                "transcript_id":   attrs.get("transcript_id", ""),
                "transcript_name": attrs.get("transcript_name", ""),
            })
            matched += 1

    df = pd.DataFrame(rows)
    print(f"  GTF: kept {matched} rows for {df['transcript_id'].nunique()} transcripts.")
    return df


# ── FASTA filtering ───────────────────────────────────────────────────────────

def filter_fasta(target_ids: set[str]) -> pd.DataFrame:
    print(f"  Scanning FASTA: {FASTA_PATH.name} ...")
    records: dict[str, str] = {}
    current_id: str | None  = None
    current_seq: list[str]  = []

    with gzip.open(FASTA_PATH, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                # Flush previous record if it was a target
                if current_id in target_ids:
                    records[current_id] = "".join(current_seq)
                tid = strip_version(line[1:].split("|")[0])
                current_id  = tid if tid in target_ids else None
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)

    # Flush last record
    if current_id in target_ids:
        records[current_id] = "".join(current_seq)

    found = len(records)
    missing = len(target_ids) - found
    print(f"  FASTA: extracted {found} sequences ({missing} transcript IDs had no matching sequence).")

    return pd.DataFrame(
        {"transcript_id": list(records.keys()), "sequence": list(records.values())}
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for path in (GTF_PATH, FASTA_PATH):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("Step 1/3 — collecting transcript IDs from parquet files …")
    target_ids = collect_transcript_ids()

    print("Step 2/3 — filtering GTF …")
    gtf_df = filter_gtf(target_ids)
    gtf_df.to_parquet(OUT_GTF_PATH, index=False)
    size_kb = OUT_GTF_PATH.stat().st_size // 1024
    print(f"  Saved: {OUT_GTF_PATH.name}  ({size_kb} KB)")

    print("Step 3/3 - filtering FASTA ...")
    seq_df = filter_fasta(target_ids)
    seq_df.to_parquet(OUT_SEQ_PATH, index=False)
    size_kb = OUT_SEQ_PATH.stat().st_size // 1024
    print(f"  Saved: {OUT_SEQ_PATH.name}  ({size_kb} KB)")

    print("\nDone. Update transcript_utils.py to load from these parquet files.")


if __name__ == "__main__":
    main()
