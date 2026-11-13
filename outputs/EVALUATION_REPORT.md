# Evaluation report (auto-generated)

## 1. Renewal-risk prediction

Evaluated on the decision cohort against the outcome that would occur with NO action.

| item | AUROC | AUPRC |
|---|---|---|
| risk model (HGB) | 0.736 | 0.665 |
| ceiling: true probabilities | 0.803 | 0.740 |
| chance | 0.500 | 0.431 |


The ceiling row scores the *true* renewal probabilities: no model can beat it, because each renewal is still a coin flip. AUROC near the ceiling means the risk model has extracted what the features contain.

## 2. Causal / counterfactual estimates vs ground truth

ATE error = |estimated - true average effect|. PEHE = RMS error of the per-tenant effect (lower is better; compare with `pehe_if_constant_ate`, the error you would make by giving everyone the true average effect). Spearman / uplift ratio measure whether the *ranking* of tenants by effect is right (0 = random, 1 = perfect).

| estimator | mean_ate_abs_error | mean_pehe | pehe_if_constant_ate | mean_spearman | mean_uplift_ratio | top1_agreement | mean_value_loss_per_tenant |
|---|---|---|---|---|---|---|---|
| naive | 0.128 | 0.147 | 0.072 | 0.000 | -0.005 | 0.302 | 1426.178 |
| s_learner | 0.029 | 0.070 | 0.072 | 0.293 | 0.463 | 0.542 | 555.673 |
| t_learner | 0.025 | 0.114 | 0.072 | 0.235 | 0.423 | 0.469 | 566.149 |
| dr_learner | 0.012 | 0.068 | 0.072 | 0.197 | 0.423 | 0.478 | 689.556 |
| econml_dr | 0.016 | 0.069 | 0.072 | 0.239 | 0.455 | 0.516 | 656.732 |

## 3. Decision policies (estimator = dr_learner)

Net value = P(renew) x retained rent - intervention cost, evaluated with the TRUE probabilities. Non-constraint-aware policies go through the execution layer, which blocks rule violations and anything over budget. 'Oracle' = best possible feasible, within-budget plan.

| policy | net_value_per_tenant | ci_lo | ci_hi | uplift_vs_no_action_per_tenant | regret_vs_oracle_per_tenant | pct_of_oracle_gain | spend_pct_budget | proposals_violating_rules_pct | blocked_by_budget_pct |
|---|---|---|---|---|---|---|---|---|---|
| No action | 10821.8 | 10620.9 | 11010.6 | 0.0 | 1074.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| Random feasible | 10951.5 | 10707.5 | 11134.3 | 129.7 | 944.5 | 12.1 | 100.0 | 0.0 | 38.6 |
| B1 Rule-based | 11126.9 | 10942.4 | 11347.9 | 305.1 | 769.1 | 28.4 | 100.0 | 6.8 | 55.7 |
| B2 Risk-only alert | 11050.5 | 10809.5 | 11272.0 | 228.7 | 845.5 | 21.3 | 35.0 | 0.0 | 0.0 |
| B3 Risk + rules | 11145.7 | 10927.8 | 11350.8 | 323.9 | 750.4 | 30.1 | 100.0 | 6.3 | 13.5 |
| A Causal (no constraints) | 11404.2 | 11203.3 | 11608.4 | 582.4 | 491.8 | 54.2 | 100.0 | 19.8 | 12.1 |
| B Causal + constraints | 11493.5 | 11276.5 | 11712.3 | 671.7 | 402.5 | 62.5 | 99.9 | 0.0 | 0.0 |
| Oracle (true effects) | 11896.0 | 11667.7 | 12102.2 | 1074.2 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 |


Paired bootstrap comparisons:

| comparison | mean_gain_per_tenant | ci_lo | ci_hi | significant |
|---|---|---|---|---|
| B Causal + constraints  minus  B3 Risk + rules | 347.9 | 289.6 | 404.4 | True |
| B Causal + constraints  minus  A Causal (no constraints) | 89.3 | 58.1 | 120.8 | True |
| B Causal + constraints  minus  B1 Rule-based | 366.6 | 310.1 | 424.9 | True |
| A Causal (no constraints)  minus  B3 Risk + rules | 258.5 | 200.4 | 311.8 | True |


Policy B by estimator:

| item | pct_of_oracle_gain (policy B) | regret_per_tenant |
|---|---|---|
| naive | 0.00 | 1074.21 |
| s_learner | 61.43 | 414.37 |
| t_learner | 62.20 | 406.02 |
| dr_learner | 62.53 | 402.49 |
| econml_dr | 61.40 | 414.66 |

## 4. Can we trust the causal estimates on logged data? (checks that work on REAL data)

Effect of each action on P(renew), estimated three ways, next to the hidden truth:

| item | truth | naive | aipw | aipw_lo | aipw_hi | rct | rct_lo | rct_hi |
|---|---|---|---|---|---|---|---|---|
| service_priority | +0.041 | -0.114 | +0.037 | +0.008 | +0.067 | +0.047 | -0.017 | +0.111 |
| rent_concession | +0.056 | -0.114 | +0.036 | -0.001 | +0.073 | +0.053 | -0.012 | +0.119 |
| proactive_outreach | +0.030 | -0.102 | +0.019 | -0.012 | +0.050 | +0.088 | +0.025 | +0.152 |


**Overlap** (every action must have a realistic chance of being chosen for every kind of tenant):

| arm | n | e_min | e_p1 | e_median | e_max | share_e_below_0.05 | ess_ratio |
|---|---|---|---|---|---|---|---|
| no_action | 5124 | 0.001 | 0.028 | 0.511 | 0.978 | 0.021 | 0.443 |
| service_priority | 1968 | 0.011 | 0.027 | 0.155 | 0.966 | 0.050 | 0.543 |
| rent_concession | 1526 | 0.001 | 0.008 | 0.115 | 0.805 | 0.205 | 0.207 |
| proactive_outreach | 1607 | 0.007 | 0.023 | 0.152 | 0.569 | 0.069 | 0.665 |


**Refutations** - placebo (shuffled actions -> effect ~0), random extra covariate (estimate unchanged), 80% subsets (stable):

| arm | ate_original | placebo_mean | placebo_pass(|mean|<2*se_orig) | random_cause_ate | random_cause_pass(|diff|<se) | subset_mean | subset_sd | subset_pass(orig within 2sd) |
|---|---|---|---|---|---|---|---|---|
| service_priority | 0.037 | 0.005 | True | 0.037 | True | 0.037 | 0.007 | True |
| rent_concession | 0.036 | 0.004 | True | 0.037 | True | 0.034 | 0.007 | True |
| proactive_outreach | 0.019 | 0.013 | True | 0.019 | True | 0.024 | 0.011 | True |


**E-values** - strength of an unmeasured confounder needed to explain away the effect:

| item | risk_ratio | e_value_point | e_value_ci_bound |
|---|---|---|---|
| service_priority | 1.065 | 1.327 | 1.129 |
| rent_concession | 1.062 | 1.319 | 1.000 |
| proactive_outreach | 1.033 | 1.219 | 1.000 |

## 5. Off-policy evaluation (estimating a policy's value from logs only)

TRUE = oracle value; IPS / SNIPS / DR / direct use only the logged actions and outcomes. Use this on real PMS history, where no oracle exists.

| policy | truth | ips | snips | direct | dr |
|---|---|---|---|---|---|
| No action | 10969 | 11053 | 11023 | 11059 | 11107 |
| B1 Rule-based | 12032 | 11857 | 12037 | 11831 | 12048 |
| B3 Risk + rules | 11667 | 12023 | 11666 | 11432 | 11725 |
| Causal, feasible (cross-fit) | 11870 | 12218 | 11815 | 12310 | 11661 |
| Oracle, feasible | 12475 | 12521 | 12537 | 11745 | 12499 |
