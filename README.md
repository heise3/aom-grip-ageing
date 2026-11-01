# AOM eligibility and low grip strength: code and aggregate results

Study: Weakness that BMI does not see: grip strength and long-term prognosis among older adults who qualify for anti-obesity medication in 20 countries.

This archive contains research source code, frozen method documents and selected aggregate result/plot tables. It contains no participant-level data, HRS/RAND data or HRS analysis script, data credentials, or internal bootstrap checkpoint files. The numerical results are the locked QCv2 and v0.8.1 aggregate results; no analysis was rerun to prepare the archive.

## Layout and reproduction boundary

- analysis_source: historical primary/QC/posthoc and v0.7/v0.8.1 researcher-run source.
- aggregate_results: reviewed table/figure summaries with retained suppression.
- figure_source: native matplotlib rendering scripts.
- protocols: locked original plan and amendments; later additions are post hoc.
- resources: scientific schema/configuration templates and aggregate reference tables.

The historical scripts are archived source, not a newly tested portable one-click workflow. Scientific functions were not edited. Local machine paths have been replaced by ${PROJECT_ROOT}, ${LOCAL_HOME} and ${DATA_VOLUME}; set the project/input layout explicitly in a licensed researcher environment. Inspect CODE_PROVENANCE.csv for original and distributed hashes. Resource path binding and machine file-mtime/version checks require site-specific configuration. Do not remove scientific integrity gates to make a new dataset version run.

The published-data inputs must be obtained separately from each provider under the relevant agreements. Do not redistribute the underlying surveys. Current provider rules may restrict AI access and require independent researcher execution. The assistant did not execute any archived participant-processing code during this submission-preparation task.

Pinned extension dependencies in the original receipt are numpy 2.5.4, pandas 3.1.0, patsy 1.0.3, statsmodels 0.15.0 and scipy 1.18.1; bootstrap 500 and seeds are recorded in config_TEMPLATE.json. Main models are unweighted cloglog with interval terms and person-cluster covariance; descriptive estimates retain survey weights. Three prognosis cohorts are CHARLS/MHAS/SHARE; ELSA/KLoSA remain descriptive with their stated limitations. HRS is outside this archive.

The selected FI screen resource contains only concept/selection/equivalence flags, not small item-cell statistics. The SHARE model reference is restricted to the same-sample main-FI comparator; use the supplied aggregate result tables for published outputs. The full historical runner resource layout must be reconstructed from licensed inputs before running; archived files have not been substituted into a new scientific run.

## Rights and disclosure

Original code is licensed under MIT; original aggregate outputs, research protocols and documentation are licensed under CC BY4.0. Underlying provider data and third-party materials are excluded from these licenses. The authors expressly confirmed both the license selection and permission to release the selected summaries under their accepted provider agreements. See LICENSE and LICENSE-DATA-CC-BY-4.0.md.

Version v1.0.0 is the public code/aggregate release, not a new statistical analysis. Cite the released snapshot at https://github.com/heise3/aom-grip-ageing/releases/tag/v1.0.0 . A DOI will be linked separately only after a registered archival record is publicly verified; a reserved draft DOI is not treated as a publication.

Suppression is part of the display: <10 and complementary withheld values remain as released. MHAS 262 weighted-description /270 prognosis denominators, including 8 missing weights, are the explicit approved presentation exception. The archive is not an independent certification of all provider-specific permissions.


## Citation and scope

Use CITATION.cff for the software citation. The manuscript itself has not been published by this repository. Code development, study ideas and figure development involved the authors; AI tools assisted with code, aggregate checks, drafting/editing and figure preparation. The v0.7/v0.8.1 extension runs were researcher-executed and only aggregate returns were reviewed by the assistant. No new participant-processing or HRS code was executed to publish this repository.
