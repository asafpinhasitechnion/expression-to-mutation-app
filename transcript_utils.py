"""Transcript utilities for the mutation viewer: genomic to transcript coordinate mapping."""

from __future__ import annotations

import pandas as pd

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
