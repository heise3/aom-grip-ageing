# A v0.7 post hoc extensions — author decisions recorded 2026-10-31

All changes below were requested after review of v0.6 and are **post hoc**. They do not overwrite the v0.6/QCv2 protocol, data, standard population or results. Source/DUA gates apply separately.

## T2 author replacement

Same construction and scoring rules, cohort-specific available items. Eligible deficits must be health-related, increase with age, have prevalence1%–80%, and missingness≤10% in the cohort sample. Main FI requires≥30 items; insufficient items are reported without relaxing the standard. Binary deficits0/1; ordered categories evenly spaced0–1. At least80% of selected items must be nonmissing per person; otherwise FI missing. Do not impose an extra≥30 observed-item condition.

Main FI excludes the five ADL outcome domains, grip, **all help/assistance items**. FI-noMob additionally excludes mobility/strength items such as walking, lifting, climbing, stooping. FI-common uses only items common and equivalent in all five cohorts; report item count, including when<30. No unverified field-name match is called equivalent. No FI version uses both a component item and its summary count/scale.

Models: frozen main association model plus FI per0.1; repeat for three FI versions. Tertiles are cohort-specific, not a0.25 threshold. FI validity: mean/median/P99/max, five-year age means, sex differences, FI-per0.1 mortality HR. KLoSA only death analyses; ELSA only if its required death-status gate passes. REML plus modified HK pooling unchanged.

Implementation decisions, fixed before FI screening: selection denominator is all baseline eligible candidates with valid grip/BMI/sex aged≥65, without requiring ADL=0 or model complete cases; selection is unweighted. Missingness is the fraction of unmapped/missing codes. Prevalence means score>0 for binary and ordered deficits; both this prevalence and mean severity are reported. Age increase means positive unadjusted linear slope of the item score against continuous age among nonmissing values, with no P-value criterion and no requirement for every noisy five-year bin to be strictly monotone. Item screening uses no follow-up outcomes or grip-effect estimates. Candidate grid and source coding must be locked before computing these screens. No extra items will be added after seeing a failing item count.

Tertiles use baseline valid-FI eligible participants and empirical1/3,2/3 quantiles; boundaries and ties are reported. Tied scores stay together. If boundaries coincide or a stratum lacks support, report unavailable; do not split identical FI values by row order. FI-noMob applies the≥30 standard; FI-common has the author's explicit<30 exception. Common-item equivalence must be documented before intersecting item screens.

## T3 approved discrimination

Nine-year cumulative/dynamic IPCW AUC, apparent sample performance, not Harrell C or external validation. Same complete-case people for base and augmented models; base age+sex+BMI+six-disease count+categorical interval, SHARE also retains the country structural term. Add only one grip definition at a time. The primary adjusted association model remains age+sex+education+BMI+smoking+diabetes+six-disease count+interval (+SHARE country), person-cluster robust SE, no survey weights. Prevalence is survey-weighted in all eligible candidates; prognosis is baseline ADL=0. These denominators are not interchangeable.

Use inherited nominal-wave endpoint times for IPCW and outcome ascertainment. At9 years, cases have observed composite endpoint≤9. Controls are observed event-free beyond9, including event-free administrative records exactly at9 (known free of the endpoint through that visit). Use the marginal censoring KM left limit G(9-) for this closed-visit convention; do not consume administrative censoring at9 before weighting its known event-free controls. Case weights use G(T-). Assumes independent censoring. SHARE model-standardized9-year risk retains constant within-interval hazard interpolation. Neither AUC nor risks are exact-date outcomes.

500 paired participant bootstrap replicates, same resample for base and augmented predictions. Refit requested prediction models, recompute censoring weights in each bootstrap. Percentile95%CI if≥450 valid replicates (same computational acceptance threshold as earlier500-replicate post hoc work); otherwise report the point estimate and unavailableCI with failure fraction/reasons. No favorable-replicate selection. No data-dependent model/covariate changes.

## T4 approved landmark

Only first-follow-up observed event-free eligible participants; analysis starts in subsequent intervals. Original baseline covariates, exposure, categorical interval structure and censoring/death-alive-conflict rules retained. No death calendar dates read or used. This replaces the previous early/undatable-death-exclusion sensitivity in v0.7 reporting; preserve historic results. The estimand is conditional on surviving event-free to first follow-up, not the original baseline estimand.

## T1, T5–T7

T1 changes publication suppression to documented provider obligations, QCv2 only. Audit of the original fitmodel source identified its <10-event screen as a project cell-disclosure gate (the source explicitly calls it cell suppression), not a rank/convergence diagnosis. Remove that project screen under T1; keep actual estimability checks (zero-event exposure arm, rank deficiency, nonconvergence, invalid SE) and the original formula/estimator/covariance. An early v0.7 precheck retaining the old disclosure gate and excluding horizon-end controls is quarantined as implementation-only, not a valid unavailable scientific estimate. No AUC estimate/bootstrap had been produced before the endpoint convention was corrected. T5 ELSA G.3 fails post-W6 mortality coverage; no fourth prognostic cohort added. T6 is release-specific metadata feasibility first; do not upgrade source releases silently. T7 HRS is write-only and never run by the assistant. Current MHAS policy requires independent researcher execution. No manuscripts or journal submission actions in this task.
