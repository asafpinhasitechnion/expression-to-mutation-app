"""
Transcript utilities for mutation_viewer:
  - GTF loading (from pre-filtered parquet)
  - Genomic → transcript coordinate mapping
  - Sequence loading (from pre-filtered parquet)
  - CDS translation
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Reference file paths (pre-processed, relative to this file) ──────────────

_HERE = Path(__file__).parent
GTF_PARQUET_PATH      = _HERE / "gtf_filtered.parquet"
SEQUENCES_PARQUET_PATH = _HERE / "sequences_filtered.parquet"

# ── Codon table ───────────────────────────────────────────────────────────────

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ── ID helpers ────────────────────────────────────────────────────────────────

def strip_version(eid: str) -> str:
    """ENST00000123.4  →  ENST00000123"""
    if isinstance(eid, str):
        return eid.split(".")[0]
    return str(eid)


def normalize_chrom(c: str) -> str:
    """Ensure chromosome name has 'chr' prefix to match GTF."""
    c = str(c)
    return c if c.startswith("chr") else "chr" + c


# ── GTF loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading reference data...")
def load_gtf() -> pd.DataFrame:
    """
    Load pre-filtered GTF parquet.

    Columns: feature, chrom, start, end, strand,
             gene_name, gene_id, transcript_id, transcript_name
    """
    return pd.read_parquet(GTF_PARQUET_PATH)


# ── Transcript queries ────────────────────────────────────────────────────────

def get_gene_transcripts(gtf_df: pd.DataFrame, gene_name: str) -> pd.DataFrame:
    """Return unique transcripts for a gene (columns: transcript_id, transcript_name, chrom, strand)."""
    mask = (gtf_df["gene_name"] == gene_name) & (gtf_df["feature"] == "transcript")
    return gtf_df.loc[mask, ["transcript_id", "transcript_name", "chrom", "strand"]].drop_duplicates()


# ── Transcript model ──────────────────────────────────────────────────────────

def build_transcript_model(gtf_df: pd.DataFrame, transcript_id: str) -> dict | None:
    """
    Build exon/CDS coordinate model for a transcript.

    Returns dict with keys:
      exons         – list of (genomic_start, genomic_end), in transcript order
      cds_regions   – same but for CDS features
      strand        – '+' or '-'
      chrom         – chromosome string (e.g. 'chr17')
      tx_length     – total transcript length (sum of exon lengths)
      cds_start_tx  – transcript-coord (0-based) start of CDS, or None
      cds_end_tx    – transcript-coord (exclusive) end of CDS, or None
    """
    stripped = strip_version(transcript_id)
    mask = gtf_df["transcript_id"].apply(strip_version) == stripped
    tx_df = gtf_df.loc[mask]
    if tx_df.empty:
        return None

    strand = tx_df["strand"].iloc[0]
    chrom  = tx_df["chrom"].iloc[0]

    exon_df = tx_df.loc[tx_df["feature"] == "exon", ["start", "end"]].drop_duplicates()
    cds_df  = tx_df.loc[tx_df["feature"] == "CDS",  ["start", "end"]].drop_duplicates()

    ascending = (strand == "+")
    exons      = exon_df.sort_values("start", ascending=ascending).values.tolist()
    cds_regions = cds_df.sort_values("start", ascending=ascending).values.tolist()

    tx_length = sum(int(e) - int(s) + 1 for s, e in exons)

    # Map CDS genomic boundaries to transcript coordinates
    cds_start_tx = cds_end_tx = None
    if cds_regions:
        all_starts = [int(s) for s, _ in cds_regions]
        all_ends   = [int(e) for _, e in cds_regions]
        if strand == "+":
            cds_start_tx = _genomic_to_tx(exons, strand, min(all_starts))
            cds_end_tx   = _genomic_to_tx(exons, strand, max(all_ends))
        else:
            cds_start_tx = _genomic_to_tx(exons, strand, max(all_ends))
            cds_end_tx   = _genomic_to_tx(exons, strand, min(all_starts))

        if cds_start_tx is not None and cds_end_tx is not None:
            lo = min(cds_start_tx, cds_end_tx)
            hi = max(cds_start_tx, cds_end_tx) + 1
            cds_start_tx, cds_end_tx = lo, hi

    return {
        "exons":        exons,
        "cds_regions":  cds_regions,
        "strand":       strand,
        "chrom":        chrom,
        "tx_length":    tx_length,
        "cds_start_tx": cds_start_tx,
        "cds_end_tx":   cds_end_tx,
    }


# ── Coordinate mapping ────────────────────────────────────────────────────────

def _genomic_to_tx(
    exons_sorted: list[list[int]],
    strand: str,
    pos: int,
) -> int | None:
    """
    Map a genomic 1-based position to a 0-based transcript coordinate.
    Returns None if the position falls in an intron or outside the transcript.
    """
    offset = 0
    for s, e in exons_sorted:
        s, e = int(s), int(e)
        if strand == "+":
            if s <= pos <= e:
                return offset + (pos - s)
            if pos < s:
                return None          # intronic / upstream
        else:
            if s <= pos <= e:
                return offset + (e - pos)
            if pos > e:
                return None          # intronic / upstream (on - strand)
        offset += e - s + 1
    return None


def genomic_to_transcript_coord(tx_model: dict, pos: int) -> int | None:
    """Public wrapper: map a genomic position to transcript coordinate."""
    return _genomic_to_tx(tx_model["exons"], tx_model["strand"], pos)


# ── FASTA sequence loading ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading sequences...")
def _load_sequences_df() -> pd.DataFrame:
    """Load the pre-filtered sequences parquet (cached once per session)."""
    return pd.read_parquet(SEQUENCES_PARQUET_PATH).set_index("transcript_id")


def load_fasta_sequence(transcript_id: str) -> str | None:
    """Return the transcript sequence, or None if not found."""
    target = strip_version(transcript_id)
    df = _load_sequences_df()
    if target in df.index:
        return str(df.at[target, "sequence"])
    return None


# ── HGVSc → transcript coordinate ────────────────────────────────────────────

_HGVSC_CODING_RE = re.compile(r"^(\d+)([\+\-]\d+)?")


def hgvsc_to_tx_coord(hgvsc: str, cds_start_tx: int | None) -> int | None:
    """
    Parse a cDNA HGVSc string and return a 0-based transcript coordinate.

    Handled cases:
      c.842G>A     → cds_start_tx + 841   (exonic substitution)
      c.842+5del   → None                 (intronic)
      c.-12G>A     → None                 (5'UTR)
      c.*45G>A     → None                 (3'UTR / downstream)

    Also handles transcript-prefixed forms: NM_000546.5:c.842G>A
    """
    if not isinstance(hgvsc, str):
        return None
    if ":" in hgvsc:
        hgvsc = hgvsc.split(":", 1)[1]
    if not hgvsc.startswith("c."):
        return None
    body = hgvsc[2:]
    if body[:1] in ("-", "*", "?"):
        return None          # UTR / non-coding
    m = _HGVSC_CODING_RE.match(body)
    if not m:
        return None
    if m.group(2):           # intronic offset (+N or -N after cds position)
        return None
    if cds_start_tx is None:
        return None
    cds_pos = int(m.group(1))   # 1-based CDS position
    return cds_start_tx + cds_pos - 1  # 0-based transcript coordinate


# ── Translation ───────────────────────────────────────────────────────────────

def translate_cds(seq: str, cds_start: int, cds_end: int) -> str:
    """Translate CDS region of a transcript sequence to a protein string."""
    cds_seq = seq[cds_start:cds_end].upper()
    protein: list[str] = []
    for i in range(0, len(cds_seq) - 2, 3):
        aa = CODON_TABLE.get(cds_seq[i : i + 3], "?")
        protein.append(aa)
        if aa == "*":
            break
    return "".join(protein)
