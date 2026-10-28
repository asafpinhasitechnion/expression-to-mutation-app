# Mutation Viewer

Companion app for the Expression-to-Mutation manuscript (Pinhasi & Yizhak). It shows the results of
the per-cohort TCGA models that predict somatic mutation status from tumor RNA expression.

Pages:

- **About / Model card:** study scope, definitions, coverage and data provenance.
- **Model performance:** genes ranked within a cohort by AUPRC, prevalence or normalized AUPRC.
- **Mutation viewer:** observed MC3 missense and silent variants placed on a transcript, with each
  tumor's predicted score for the gene.
- **SHAP explorer:** feature importance per target, and per-sample SHAP beeswarms of its top 20 features.

## Data

Everything the app reads is built from the analysis repository,
[E2M-analyses](https://github.com/asafpinhasitechnion/E2M-analyses), and stored in this repository:

| File | Contents | Source in E2M-analyses |
|---|---|---|
| `data/missense_mutations.parquet`, `data/silent_mutations.parquet` | Variants with the predicted score of the tumor for the gene; genes with normalized AUPRC > 0.1, all cohorts except UVM (as Figure 3) | `Runs/TCGA/output/<COHORT>/cv/oof_probabilities.csv`, `Runs/TCGA/data/mutation_events/<COHORT>_mc3.txt.gz` |
| `shap_bundle/cohorts/<COHORT>_metrics.parquet` | Pooled out-of-fold metrics of the evaluable targets | `Runs/TCGA/output/<COHORT>/cv/metrics.csv` |
| `shap_bundle/cohorts/<COHORT>_features.parquet` | Mean and mean absolute SHAP of every feature with non-zero SHAP | `Runs/TCGA/output/<COHORT>/shap/<TARGET>/feature_summary.csv` |
| `shap_bundle/beeswarms/<COHORT>.zip` | One parquet per target: SHAP and expression of its top 20 features in every sample | `Runs/TCGA/output/<COHORT>/shap/<TARGET>/sample_shap_top_features.csv.gz` |
| `gtf_filtered.parquet` | Transcripts, exons and CDS of the variant genes (GENCODE v23lift37, GRCh37 as MC3) | `gencode.v23lift37.annotation.gtf.gz` |
| `bundle_manifest.json`, `shap_bundle/index.json` | Analysis commit, row counts, checksums, coverage | |

Scores are from the multitask network, out-of-fold over 5-fold cross-validation. SHAP is computed
for targets with normalized AUPRC > 0.05, with Tree SHAP on a per-target XGBoost model.

## Rebuild the data

After the TCGA run and SHAP outputs change, from this directory:

```powershell
python -m pip install -r requirements.txt matplotlib seaborn scipy
python build_bundle.py --analysis-root D:\Asaf\E2M-analyses
```

The builder uses the Figure 3 helpers of E2M-analyses for the variant join, so the variants match
Figure 3. It checks every variant's score against the current out-of-fold predictions, and writes
the analysis commit into `bundle_manifest.json`. The GTF defaults to
`Runs/TCGA/data/annotation/gencode.v23lift37.annotation.gtf.gz` (download from GENCODE release 23,
GRCh37 mapping); `--gtf` overrides it. Rebuild rather than replacing single files, so the manifest
stays in step with the data.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```
