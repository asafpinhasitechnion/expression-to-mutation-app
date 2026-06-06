#!/usr/bin/env bash
set -euo pipefail

# Package every per-target beeswarm produced by the current SHAP workflow.
# Parquet files are stored without a second compression pass. The viewer opens only
# the selected member, so thousands of SHAP files do not need to be copied or loaded.

SRC_ROOT="${1:-/storage/md_keren/asafpi/Expression_to_Mutation/Clean_TCGA_E2M_prediction/output/multitask_nn}"
DST_ROOT="${2:-/storage/md_keren/asafpi/Expression_to_Mutation/Clean_TCGA_E2M_prediction/output/mutation_viewer_shap_bundle}"

mkdir -p "$DST_ROOT"

python - "$SRC_ROOT" "$DST_ROOT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)

index = {
    "version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(source),
    "cohorts": {},
}

for cohort_dir in sorted(path for path in source.iterdir() if path.is_dir()):
    shap_dir = cohort_dir / "shap"
    files = sorted(shap_dir.glob("beeswarm_*.parquet"))
    if not files:
        continue

    archive_path = destination / f"{cohort_dir.name}.zip"
    temporary_path = destination / f".{cohort_dir.name}.zip.tmp"
    with ZipFile(temporary_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    os.replace(temporary_path, archive_path)

    genes = [path.stem[len("beeswarm_"):] for path in files]
    index["cohorts"][cohort_dir.name] = {
        "archive": archive_path.name,
        "beeswarm_target_count": len(genes),
        "beeswarm_targets": genes,
        "archive_bytes": archive_path.stat().st_size,
    }
    print(f"{cohort_dir.name}: {len(genes)} targets -> {archive_path}", flush=True)

(destination / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
print(f"Done: {len(index['cohorts'])} cohorts -> {destination}")
PY
