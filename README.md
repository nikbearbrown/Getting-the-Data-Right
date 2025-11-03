# CGetting the Data Right: Balance and Stratification for Analysis, ML, and Causal Inference

> **Thesis:** Most downstream failures in analytics and modeling trace back to upstream imbalance. Balance and stratification are the “design” layer of data science: the disciplined preparation that makes estimation, testing, prediction, and causal claims trustworthy.

---

## 1. Why Balance Comes Before Modeling

Data scientists spend immense effort tuning models but comparatively little designing the comparison their model will learn from. Whether you are estimating a treatment effect, testing a hypothesis, training a classifier, or forecasting, your conclusions only make sense relative to **who is being compared to whom under which conditions**. If important covariates differ systematically across the groups you compare, you are measuring a blend of signal and confounding.

**Balance** is the condition where the joint distribution of relevant covariates is comparable across the groups or folds that will be contrasted (e.g., treated vs. control; train vs. test; positive vs. negative; time windows). **Stratification** is the practice of dividing the data into well-defined, homogeneous subgroups (strata) within which comparisons are fair and aggregation is valid.

**Design-first workflow (DFW):**

1. Define estimand/target; 2) Specify covariates that must be comparable; 3) Create strata or achieve balance (matching/weighting/stratification); 4) Diagnose balance; 5) Lock the design; 6) Only then analyze, test, or model.

---

## 2. Core Concepts and Vocabulary

* **Covariates (X):** Pre-exposure, pre-outcome features that may confound a comparison or shift model behavior.
* **Balance:** Similarity of covariate distributions between comparison units. Frequently summarized by standardized mean differences (SMD), empirical CDF overlap, and propensity-score diagnostics.
* **Positivity/Overlap:** Every covariate pattern should have nonzero probability for each group/action. Violations show up as non-overlapping supports; they cause extrapolation and fragile inference.
* **Strata:** Non-overlapping, collectively exhaustive bins (e.g., age bands, risk deciles, propensity-score quintiles) within which units are comparable.
* **Estimands:** ATE, ATT, ATC in causal inference; class-conditional error rates and calibration in ML; mean/median differences under well-defined populations in hypothesis testing.
* **Design vs. Analysis:** Design is model-agnostic preparation to ensure fair comparisons; analysis consumes the designed data to estimate/learn/test.

---

## 3. Stratification: Building Comparable Micro-Worlds

Stratification partitions a heterogeneous dataset into slices where units are more comparable. Within each stratum, you compare like with like and then aggregate.

### 3.1 What to Stratify On

* **Direct drivers of exposure or outcome:** age, baseline risk, prior utilization, seasonality, region, device type, channel, historic propensity.
* **Design variables:** cohort entry date, data source, site, instrument (A/B test variant, clinic, branch).

### 3.2 Granularity, Monotonicity, and Sufficiency

* **Granularity:** Too coarse → residual imbalance; too fine → sparse strata and variance blow-up. Start with 5–10 bins for continuous variables (quantiles) and expand only where diagnostics suggest.
* **Monotonicity:** Choose stratification variables with monotone relationships to exposure/outcome when possible (e.g., risk scores), enabling stable aggregation.
* **Sufficiency:** If a low-dimensional summary (e.g., a risk or propensity score) balances many features, stratify on that rather than on every feature separately.

### 3.3 Stratified Estimation Templates

* **Difference in means within strata:** (\hat{\Delta} = \sum_s w_s(\bar{Y}*{1s} - \bar{Y}*{0s})), with weights (w_s) proportional to stratum size or to target population.
* **Stratified tests:** Cochran–Mantel–Haenszel for categorical outcomes; stratified t-tests or rank tests for continuous outcomes; stratified log-rank for survival.
* **ML with stratified CV:** Build folds that preserve the joint distribution of key covariates and target prevalence (e.g., stratified k-fold by class and site).

### 3.4 Practical Checklist

* Pick 3–8 high-impact covariates and/or a balancing score.
* Cut into quantile-based bins; ensure minimum cell size (e.g., ≥50 per cell for modeling, ≥10 per cell for simple tests).
* Verify per-stratum SMDs and prevalence; merge or split strata until balanced.
* Freeze bin edges; materialize stratum IDs for all downstream steps and reporting.

---

## 4. Quantifying Balance

### 4.1 Standardized Mean Difference (SMD)

For covariate (X):
[
\text{SMD} = \frac{\bar{X}_1 - \bar{X}_0}{\sqrt{(S_1^2 + S_0^2)/2}}
]
Interpretation thresholds (absolute SMD): <0.10 good; 0.10–0.20 small; >0.20 problematic.

### 4.2 Distributional Diagnostics

* **Overlaid density/CDF plots** between groups.
* **Love plot** of absolute SMDs pre- vs post-design.
* **Overlap plots** of a balancing score (e.g., propensity) to visualize positivity.

### 4.3 Balance Is About X, Not Y

All diagnostics must be performed **without using the outcome** (leakage inflates apparent balance and biases conclusions). The design is chosen solely on (X) and exposure/treatment indicators.

---

## 5. Three Routes to Balance

### 5.1 Stratification (by covariates or balancing scores)

* **When to use:** Interpretability, governance, transparent reporting; when business logic requires subgroup fairness or per-stratum SLAs.
* **Pros:** Simple, robust, easy to explain; natural for reporting and deployment.
* **Cons:** Coarsening can leave within-stratum imbalance unless strata are sufficiently fine.

### 5.2 Matching

* **Idea:** Pair or group similar units across groups using distances (e.g., Mahalanobis, cosine, learned embeddings) often with calipers on a balancing score.
* **Use cases:** Causal ATT analyses; panel construction; nearest-neighbor cohort creation; offline uplift experiments.
* **Tradeoffs:** Discards data that cannot be fairly matched (which is good for bias, but may increase variance). Consider 1:k matching for precision.

### 5.3 Weighting

* **Idea:** Reweight observations (e.g., inverse probability/propensity, overlap weights, entropy balancing) so the weighted covariate distribution aligns with a target population.
* **Use cases:** Estimating ATE/ATT with full-sample efficiency; domain adaptation for ML (shift from training to deployment mix); survey calibration.
* **Tradeoffs:** Unstable weights under weak overlap; requires stabilization, truncation, or overlap weighting to control variance.

**Tip:** Start with stratification on a good balancing score; escalate to matching for interpretability or to weighting for efficiency; verify via the same balance diagnostics.

---

## 6. Balancing Scores: Propensity and Risk

A **balancing score** is any function of covariates that, if equal across groups, implies balance of the covariates themselves. Two especially useful scores:

* **Propensity score** (e(X)=P(A=1\mid X)). Use for exposure/treatment comparisons, domain shift (e.g., channel adoption), missingness mechanisms (e.g., selection models).
* **Baseline risk score** (prognostic or outcome score) (r(X)=E[Y\mid X]) computed **without** using exposure; align risk across groups to reduce residual confounding and improve power.

**Best practice:** Combine both—match/stratify/weight on the logit propensity with calipers **and** check balance on a baseline risk score.

---

## 7. Balance for Hypothesis Testing

### 7.1 Classical Two-Group Tests

If balance holds (by design or luck), classical parametric tests (t-test, ANCOVA) have valid Type I error. Under imbalance, variance inflation/deflation and bias occur. Fix by:

* **Stratified testing:** Conduct within-stratum tests; aggregate (e.g., inverse-variance weights).
* **Covariate-adjusted tests:** ANCOVA or regression with pre-registered covariates, but only after verifying linearity/functional-form assumptions.

### 7.2 Multiple Groups and Factorial Settings

* Create cells by crossing key covariates with group labels; ensure each cell has adequate size. Use block or stratified permutation tests for robust inference.

### 7.3 Small Samples

* Prefer exact or permutation tests within strata; match tightly to reduce model dependence, report effect sizes with confidence intervals, not just p-values.

---

## 8. Balance for Machine Learning

### 8.1 Train/Test and Cross-Validation Design

* **Stratified splits:** Preserve target prevalence and key covariate distributions (e.g., geography, device, site) across folds. Use grouped stratification when leakage risk exists (e.g., by user, household, store, or time block).
* **Temporal stratification:** Respect time order; create contiguous time folds that align covariates (seasonality, promotions). Avoid future information in features.

### 8.2 Class Imbalance vs. Covariate Imbalance

* **Class imbalance** (target prevalence) is not the same as **covariate imbalance**. SMOTE/thresholding address prevalence; they do not restore covariate comparability between positives and negatives. Diagnose both.

### 8.3 Domain Shift and Reweighting

* Estimate selection/assignment propensities for the **deployment** population vs. training data; use importance weights to train models that target the deployment mix. Always cap/truncate weights and validate calibration in the target domain.

### 8.4 Calibration and Fairness Across Strata

* Evaluate metrics by stratum: AUC/PR, Brier, calibration slope/intercept, group TPR/FPR. If a small set of strata shows drift, retrain with stratified objectives or group DRO; if many do, revisit design (weights/strata) rather than patching with thresholds alone.

### 8.5 Leakage and Balance

* Many leakage modes are hidden imbalance: target-correlated proxies differ across groups/folds. Add **leakage audits** to the balance checklist (no post-outcome features; align event windows; deduplicate entities across folds).

---

## 9. Balance for Causal Modeling

### 9.1 Roadmap

1. Define estimand (ATE/ATT/ATC). 2) Choose covariates needed for ignorability. 3) Estimate propensity (and optionally risk) scores. 4) Achieve balance (stratify/match/weight with calipers or truncation). 5) Diagnose balance (SMDs, overlap). 6) Lock design. 7) Estimate effects with design-aware methods (paired analyses, weighted regression, doubly robust estimators). 8) Sensitivity analysis for unmeasured confounding.

### 9.2 Propensity Stratification

* Bin logit(e(X)) into K (e.g., 5–10) strata; confirm within-stratum SMDs ≤0.1 for all covariates; estimate stratum-specific contrasts and aggregate with population weights.

### 9.3 Matching

* Nearest neighbor with calipers on logit propensity (e.g., 0.2 SD of logit score to start); consider 1:k for power. Use optimal matching if order dependence is a concern. Analyze as paired/clustered data; report loss of units and change in covariate distributions.

### 9.4 Weighting

* **IPTW/ATE:** weights = A/e(X) + (1−A)/(1−e(X)). Stabilize by multiplying by marginal treatment probabilities; truncate extreme weights (e.g., at 1st/99th percentile).
* **Overlap weights:** weights ∝ e(X)(1−e(X)), which downweight tails and target the region of common support; often yields excellent balance and finite variance.
* **Entropy balancing:** directly solve for weights that match covariate moments to a target.

### 9.5 Doubly Robust Estimation

* Combine outcome regression with propensity weights (AIPW) for consistency if either model is correct; still requires balance diagnostics and positivity.

### 9.6 Sensitivity Analysis

* Quantify how large an unmeasured confounder must be to explain away the effect (e.g., Rosenbaum Γ, E-value). Report the tipping point transparently.

---

## 10. Implementation Patterns

### 10.1 Python Snippets (sketches)

**Stratified splits for classification**

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# y: labels, X: features, group_id: e.g., user or store
for train_idx, test_idx in sgkf.split(X, y, groups=group_id):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    # train model and evaluate per-stratum metrics
```

**Propensity estimation and overlap plot**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

cl = LogisticRegression(max_iter=1000)
cl.fit(X_covariates, treatment)
ps = cl.predict_proba(X_covariates)[:,1]
logit_ps = np.log(ps/(1-ps))

# caliper rule-of-thumb: 0.2 * sd(logit_ps)
caliper = 0.2 * np.std(logit_ps)
```

**Compute SMDs**

```python
def smd(x_t, x_c):
    m1, m0 = np.mean(x_t), np.mean(x_c)
    s1, s0 = np.var(x_t, ddof=1), np.var(x_c, ddof=1)
    return (m1 - m0) / np.sqrt((s1 + s0) / 2)
```

**Overlap/ATE weights (truncated)**

```python
eps = 1e-6
w_ate = treatment/np.clip(ps, eps, 1-eps) + (1-treatment)/np.clip(1-ps, eps, 1-eps)
w_ate = np.clip(w_ate, np.quantile(w_ate, 0.01), np.quantile(w_ate, 0.99))
```

### 10.2 R Snippets (sketches)

**SMD with `tableone`-style logic**

```r
smd <- function(x, g){
  x1 <- x[g==1]; x0 <- x[g==0]
  (mean(x1) - mean(x0)) / sqrt((var(x1) + var(x0))/2)
}
```

**Propensity stratification**

```r
ps <- predict(glm(A ~ ., data=df, family=binomial()), type="response")
q <- quantile(ps, probs = seq(0,1,length.out=6))
df$strata <- cut(ps, breaks = unique(q), include.lowest = TRUE)
```

**Overlap weights (ATE on overlap population)**

```r
w <- ps*(1-ps)
fit <- glm(Y ~ A, data=df, weights=w)
summary(fit)
```

> *Note:* In production, prefer robust libraries (e.g., `MatchIt`, `WeightIt`, `survey` in R; `econml`, `causalml`, `sklearn` + custom scaffolding in Python) and implement balance dashboards.

---

## 11. Time, Panels, and Event Studies

Balance has a temporal dimension:

* **Event alignment:** Construct windows (e.g., −30 to +30 days around an index date) and stratify by pre-trend risk or exposure intensity so that treated and control units are on similar trajectories.
* **Panel matching:** Match on lagged outcomes and covariates; verify pre-trend parallelism within matched sets.
* **Rolling deployments:** In ML A/Bs and feature flags, stratify by rollout wave and geography; compare only within concurrent windows.

---

## 12. Diagnostics-as-Documentation

Treat balance diagnostics as audit artifacts:

* A one-page **Design Report**: estimand, covariates, balance method, loss of units, overlap plots, Love plot, stratum definitions, final SMDs.
* A **Reproducible Design Object**: fixed random seeds, frozen bin edges/PS model, match assignments/weights, codebook.
* **Governance hooks**: thresholds (e.g., all absolute SMDs < 0.1), automatic fail gates in CI/CD before releasing analyses or models.

---

## 13. Pitfalls and Antipatterns

* **Balancing on post-outcome features:** induces severe bias (target leakage disguised as balance).
* **Ignoring positivity:** extreme propensities → unstable weights and speculative extrapolation; trim or restrict to common support.
* **Using p-values for balance:** sample-size dependent and misleading; use SMDs and visual overlap.
* **Over-stratification:** many tiny cells cause variance blow-up; keep cells large enough for stable estimates and model fitting.
* **Design drift:** “peeking” at outcomes to tweak design; freeze design before analysis.
* **Single-score absolutism:** PS alone may hide residual imbalance; add risk score checks and covariate-level SMDs.

---

## 14. Case Miniatures (Patterns You Can Reuse)

1. **Uplift Modeling (Marketing):** Stratify by baseline conversion risk deciles to ensure treatment lifts are not driven by risk mix; report per-decile uplift and global effect with decile weights.
2. **Credit Risk (Regulatory):** Group by geography × channel × time-band; train within-stratum or reweight to deployment mix; monitor calibration drift by stratum monthly.
3. **Healthcare Intervention:** Propensity-score matching with calipers + risk-score balance checks; report ATT with paired robust SEs; include a Γ sensitivity figure.
4. **Product A/B with Staggered Rollout:** Block by launch wave and device; analyze within blocks; aggregate via inverse-variance weights; add pre-period checks.

---

## 15. A Design-First Checklist (Tear-out)

* **Target**: What estimand or deployment population?
* **Covariates**: Which features must be comparable? (Pre-registered.)
* **Method**: Stratify / Match / Weight (why this choice?).
* **Positivity**: Overlap plots; trimming rule.
* **Diagnostics**: Absolute SMDs, Love plot, per-stratum summaries.
* **Lock**: Freeze seeds, bin edges, PS/risk models, inclusion criteria.
* **Analysis**: Model/test respecting the design (paired/clustered/weighted/stratified).
* **Sensitivity**: Unmeasured confounding or domain shift robustness.
* **Report**: Design report + reproducible object.

---

## 16. Conclusion: Design Is the Quiet Superpower

Balance and stratification convert messy observational data into structured, credible comparisons and reliable generalization targets. They are not afterthoughts; they are the foundation. Put design first, and downstream modeling—be it causal, predictive, or inferential—becomes simpler, more stable, and far more defensible.

---

### Appendix A. Minimum Viable Design (MVD) for Busy Teams

1. Choose 6–10 covariates (or a risk/propensity score) and define the population.
2. Stratify into 5–10 bins; ensure ≥1,000 records per bin (or your domain-justified minimum).
3. Verify absolute SMDs <0.1 in each bin and overall (weighted).
4. Lock design; export (id, stratum_id, weight, matched_set_id if any).
5. Train/test/evaluate with stratified CV; report by stratum.

### Appendix B. Quick Reference: Which Tool When?

* **Fair reporting, clear governance** → Stratification.
* **Interpretability, ATT** → Matching with calipers.
* **Efficiency, ATE, domain adaptation** → Weighting (overlap/entropy).
* **All of the above with model safeguard** → Doubly robust + sensitivity analysis.
