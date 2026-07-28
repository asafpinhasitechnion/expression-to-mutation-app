# Mutation Viewer

This is the maintained manuscript application. It uses the current mutation-prediction
bundle under `Results/TCGA_results/Lean_multitask_nn_log` and the same MC3 event-level
inputs and normalized-AUPRC threshold used by `Figure_Scripts/Figure3.ipynb`.

Parquet and cohort ZIP assets are stored directly in the repository, not in Git LFS.

The application contains four functional pages:

- About / Model Card: study scope, definitions, coverage, provenance, and interpretation caveats.
- Model Performance: genes ranked within a selected cohort.
- Mutation Viewer: observed variants mapped to transcript coordinates with fixed-scale probabilities.
- SHAP Explorer: target-level feature summaries and per-sample beeswarms for every
  target produced by the current SHAP workflow.

## Build the data bundle

From the `Final_E2M` project root:

```powershell
python Figure_Scripts/mutation_viewer/build_bundle.py --results-dir Results/TCGA_results/Lean_multitask_nn_log
```

This rebuilds mutation and reference data while preserving the existing indexed SHAP
bundle. SHAP archives are replaced only when an explicit `--shap-source` is supplied.

The builder creates:

- `data/silent_mutations.parquet`
- `data/missense_mutations.parquet`
- `shap_bundle/cohorts/<COHORT>_metrics.parquet`
- `shap_bundle/cohorts/<COHORT>_features.parquet`
- `shap_bundle/beeswarms/<COHORT>.zip`
- `shap_bundle/index.json`
- `gtf_filtered.parquet`
- `sequences_filtered.parquet`
- `bundle_manifest.json`

By default, mutation-viewer variants match Figure 3 scope: 31 cancer-specific cohorts
except UVM, with target genes passing normalized AUPRC > 0.1. The `all` model is excluded
because the transcript analysis uses cancer-specific probabilities.

The target-level summaries come from the canonical lean results. Per-sample Parquet files
are kept in one uncompressed ZIP per cohort, and the application reads only the selected
target. This avoids thousands of loose files and does not recompress Parquet data.

## Refresh all per-sample SHAP data

On the cluster, from `Clean_TCGA_E2M_prediction/submit`:

```bash
./export_all_shap_beeswarms.sh \
  ../output/multitask_nn \
  ../output/mutation_viewer_shap_bundle
```

Copy the resulting directory locally, then rebuild:

```powershell
python Figure_Scripts/mutation_viewer/build_shap_bundle.py `
  --beeswarm-source "<COPIED mutation_viewer_shap_bundle>"
```

The exporter discovers every `beeswarm_*.parquet`; no cohort/gene list is maintained by
hand. The pan-cancer `all` archive is not included in the viewer, which presents the 32
cancer-specific models.

## Run the application

```powershell
python -m pip install -r Figure_Scripts/mutation_viewer/requirements.txt
streamlit run Figure_Scripts/mutation_viewer/app.py
```

Do not manually replace individual viewer files. Re-run the builder so the manifest and
probability validation stay synchronized with the analysis results.
