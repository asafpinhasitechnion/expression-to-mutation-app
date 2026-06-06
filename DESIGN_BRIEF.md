# Design brief — Mutation Viewer (manuscript companion)

A conceptual design brief. Produce a **visual + interaction design direction** (layout,
hierarchy, color, typography, component treatments, empty/notice states). Mockups in
HTML/CSS are welcome as a reference — they will be **translated into a Streamlit app**, so
keep the direction expressible within Streamlit idioms (see Constraints). Do **not** write
application logic; focus on the design.

---

## 1. What this is

A companion web app for a scientific manuscript. The study trains models that predict
**somatic mutation status from bulk tumor RNA expression**, independently within each TCGA
cancer cohort. The app lets readers of the paper explore the underlying data with more
control than static figures allow.

This is a **secondary companion, not core functionality.** The bar is: simple, credible,
legible, and obviously trustworthy — not flashy or feature-maximal.

## 2. Audience & intent

- **Audience:** manuscript readers, peer reviewers, and other researchers. Scientifically
  literate; not necessarily app-savvy.
- **Intent:** let an interested reader (a) understand what the model does, (b) look up a
  specific cohort / gene / variant, and (c) download the data behind any view.
- **Design priority order:** trust and clarity > navigability > visual polish > feature
  count. When in doubt, make it calmer and more legible, not richer.

## 3. Tone & design principles

- **Scientific credibility.** Reads like a well-made figure in a journal: restrained,
  precise, generous whitespace, strong typographic hierarchy. Avoid dashboard/SaaS gloss,
  gradients-for-decoration, and playful microcopy.
- **Every figure carries its provenance.** Each plot should have an obvious, consistent
  affordance to (a) reveal/download the underlying table and (b) see what cohort / target /
  threshold / date it came from.
- **Definitions are always one hover/click away.** Key terms (predicted probability,
  prevalence, AUPRC, normalized AUPRC, SHAP value) live in one glossary and are referenced
  inline, never re-explained ad hoc.
- **Graceful degradation is a first-class state, not an error.** Coverage is partial (see
  §6). "No SHAP for this selection," "variant unmapped," "missing transcript annotation,"
  and "limited sample size" must each have a calm, designed notice state — never a blank
  area or a stack trace.
- **A pinned caveat near every SHAP view:** SHAP identifies predictive *associations*, not
  causal regulation.

## 4. Visual identity (inherit from the manuscript — do not invent a new palette)

Harmonize with the paper's figures so the app feels like part of the publication.

**Palette (hex):**
- Orange `#DD8D6E`, Beige `#F0E8D1`, Green `#558771`, Teal `#82A899`, Purple `#885784`,
  Gold `#C2AB42`.
- Diverging scales used in plots: Orange→Beige→Green and Orange→Beige→Purple. A vivid
  beeswarm scale runs Indigo `#5A49C6` → Cream `#FFF3DB` → Magenta `#E12A5F`.
- Treat beige as the warm neutral / surface tint; green and purple as primary accents;
  orange as the warm highlight. Keep large surfaces near-white with restrained accent use.

**Typography:** sans-serif, manuscript-scale and quiet. The figures use a small type ramp
(title 10 / axis 9 / tick 8). The app UI can be a touch larger for screen legibility, but
keep the same restrained spirit — clear hierarchy through weight and spacing, not size
explosions or many competing colors.

**Plots are Plotly** (already built) — your job is the *frame around them* (containers,
captions, controls, spacing), and to define how a plot, its caption, its provenance line,
and its data-download affordance compose into one consistent "figure block."

## 5. Information architecture (≈5 areas)

Design a clean primary navigation across these. For each, define the page layout, where
controls/selectors live, and how results + provenance + download compose.

1. **About / Model card** — study description in plain language; what the model predicts;
   the glossary of definitions; a **coverage matrix** (which cohorts/targets have variant
   data vs SHAP data); provenance block (data-generation date, source files, AUPRC
   threshold, excluded cohorts); the causal-association disclaimer. This is the trust page.
2. **Model performance** — comparison plots across cohorts/targets: AUPRC, mutation
   prevalence, normalized AUPRC, and number of SHAP features. The "which models work" entry
   point.
3. **Mutation viewer** (the core) — pick dataset → cohort → gene → transcript; variants
   mapped onto transcript exon/CDS coordinates; a transcript-level plot with predicted
   probability on a **fixed 0–1 scale**; filters by cancer type and mutation class; mean/
   median aggregation of repeated variants; summary panels by transcript position, amino-
   acid substitution, nucleotide substitution, and cohort. Per variant the data shows:
   position, ref/alt allele, consequence, amino-acid change, predicted probability, cohort,
   sample count.
4. **SHAP explorer** — per cohort + target feature-importance results; per-sample beeswarm
   plots where available; explicit notice where SHAP is absent.
5. **Cross-cutting: data + definitions** — a consistent "show / download data" pattern under
   every figure, and a reachable glossary. Design these as reusable components, not one-offs.

## 6. Data realities the design must absorb

- The app reads a **frozen, hashed, dated data bundle** with a manifest. All "meta" content
  (coverage, dates, sources, threshold) is derived from that manifest — design these blocks
  as **data-driven and self-updating**, not hand-authored copy.
- **Coverage is partial and uneven.** Right now variant data is essentially one cohort
  (BRCA, a handful of genes); SHAP covers ~15 cohorts with only a few beeswarm targets each.
  The design must make "what's available" obvious up front (the coverage matrix) and must
  make selecting an unavailable combination feel handled, not broken. Build for coverage
  growing over time without redesign.

## 7. Constraints (keep the direction translatable to Streamlit)

- Final implementation is **Streamlit (Python)**, deployed on Streamlit Community Cloud.
  Favor patterns Streamlit supports well: a sidebar or top nav, single-column or simple
  multi-column layouts, cards/containers, expanders for "show data," tabs/segmented
  controls, selectboxes/radios for selection, popovers/tooltips for definitions. Light
  custom CSS is fine; assume **no bespoke JavaScript framework**.
- Prefer a small, repeatable set of components (a "figure block," a "notice," a "definition
  chip," a "provenance line," a "coverage cell") over many unique screens.

## 8. What to deliver

1. An overall **visual direction**: color usage rules, type ramp, spacing, surface/elevation
   treatment, accent strategy — grounded in §4.
2. **Layout patterns** for each of the 5 areas in §5 (wireframe-level is fine), showing where
   navigation, selectors, figures, provenance, and downloads sit.
3. The **reusable component set** from §7, including all four notice/empty states from §3.
4. How **definitions and the causal disclaimer** surface inline.
5. Optional HTML/CSS mockups of the key screens (About, Mutation viewer, SHAP explorer) as a
   visual reference for translation — but the conceptual direction is the primary deliverable.
