# A v0.8 prewriting numeric extension — registered2026-10-31

All R1–R5 are post hoc, added after reviewing v0.7. No primary estimand, covariate, outcome, cutpoint, QC rule, pooling rule or standard population is changed. This is not a result-blind preregistration. Participant execution is restricted to the independent researcher; the assistant reads aggregate returns only. HRS/RAND files and scripts must not be opened or run.

## R1 same-sample SHARE comparison

Reconstruct exactly the previously locked36-item main FI; use the same selection screen, scoring,≥80% nonmissing rule and frozen wave panel. Do not add items or relax the≥30-item requirement. Restrict the primary composite panel to the FI-complete baseline-ADL0 full-covariate sample. Expected independent n5787 and events2569. Refit the FI model only to verify the returned coefficient/clusterSE/CI; stop if mismatch exceeds1e−6. Fit the no-FI reference on exactly the same people and intervals, same categorical interval/country terms and full primary baseline covariates, cloglog link, person-cluster robust SE, no survey weights.

Define relative log-HR change as100×(β_noFI−β_FI)/β_noFI. A positive value is descriptive relative reduction on the log-HR scale, not a mediated fraction or causal effect. Do not clamp the value/CI to0–100 and do not call the FI model a lower bound. At|β_noFI|≤1e−8 the relative statistic is undefined; retain HR outputs but do not manufacture a finite ratio.500 paired participant bootstrap replicates, seed20261031; same person resampling frequencies feed both models. Use frequency-weighted coefficient fits on the identical fixed design matrices; original point estimates retain person-cluster robust95%CIs. Refit both coefficients each replicate. Percentile95%CI for relative change only if≥450 valid pairs; report failures/types without replacing failed replicates. This follows the earlier computational acceptance rule. Aggregate checkpoints store replicate coefficients/statistics only, no person IDs, rows or fitted objects.

## R2 matched prognosis-domain flags

For CHARLS206, MHAS270 and SHARE5791 frozen primary complete-case people, calculate each inherited definition's survey-weighted flagged/100 and design-based logit95%CI, reusing the original weights, full-design domain linearisation and single-PSU handling. Retain primary model sample and all covariates. Weight exclusions follow v0.7: do not set missing/nonpositive weights to1. Report model n, positive-weight n and excluded-weight n separately. If positive-weight n differs from primary n, keep the numeric weighted subset estimate explicitly labelled; mark population_match_status=WEIGHT_VALID_SUBSET_DIFFERS. Do not claim it is exactly the same population or render the matched-population main Figure4a without an author decision. This is a scope check, not a search for a favourable result.

Nine-year flagged/unflagged risks, paired delta-method RD and IPCW AUC increment/500-bootstrapCI are extracted unchanged from the checked v0.7 aggregates; no prediction/risk/primary refits for this task. Bind the reconstructed complete-case n, event counts, low-grip count and frozen input size/mtime metadata; stop on disagreement. Output a single nine-row R2 table with all metrics/CIs and explicit denominators. Never insert all-candidate prevalence into the prognosis-domain cell.

## R3–R5

R3 extracts existing SHARE FI summaries and FI-per0.1 mortality validity model. Its descriptive denominator is all eligible≥65 baseline candidates with valid FI; the death model is not restricted to ADL0. R4 consolidates dated original/post hoc plans and unavailable states; report-only additions are identified. R5 uses three prognosis cohorts in main Figure4a, with original risk/AUC panels; the five-cohort≥65 candidate description becomes a separate supplementary vector figure. Rendering uses aggregates only and must wait for R2 matched-domain values. No new clinical or predictive superiority inference.

Small cells: inherit the v0.7 reviewed provider-specific rules, including ELSA3 and complementary suppression of main-FI sex denominators. No blanket<10 rule is reintroduced. No ELSA participant execution is needed in this add-on. CHARLS actual registration and KLoSA output-terms caveats remain.

Output new versions only. Preserve v0.6/v0.7/QCv2 and all existing result files. No manuscript or cover letter is drafted in this numeric task. Do not run B or HRS, or upgrade releases.
