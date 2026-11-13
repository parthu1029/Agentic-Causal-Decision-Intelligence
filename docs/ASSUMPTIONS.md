# Assumptions baked into the synthetic world (document these in your report)

Everything below is an **assumption, not a finding**. Slide 15 says "documented assumptions" - this is that document.
Change any of them in `config.yaml` / `src/pmsdi/config.py` / `src/pmsdi/data/synthetic_pms.py`, then re-run the pipeline.
`scripts/05_sensitivity.py` re-runs the entire evaluation under different values so you can show the conclusions are not an artefact of one choice.

| Assumption | Value | Where | Basis |
|---|---|---|---|
| Mean renewal probability with no action | 58 % | `target_renewal_rate` | placeholder - replace with a published multifamily renewal benchmark and cite it |
| Hidden tenant concern mix | maintenance 26 %, price 20 %, communication 14 %, relocation 12 %, none 28 % | `DEFAULT_CALIB["concern_mix"]` | placeholder; optionally calibrated from AHS "reasons for moving" |
| Effect of each action on the *logit* of renewal, by concern | see `TAU` matrix | `synthetic_pms.py` | **no public source** - encodes "an action helps when it addresses the real concern" |
| Effect modifiers | service x1.0 if an open request exists else x0.35; concession scales with rent above market; | `tau_logit` | assumption |
| Overall effect size | `effect_scale = 1.0` (true average uplift: service +4 pp, concession +6 pp, outreach +3 pp) | config | swept 0.5x-1.5x in the sensitivity analysis |
| Past managers' behaviour | actions chosen from observed risk signals (late payments, open requests, sentiment, rent gap, satisfaction) via a softmax, 15 % exploration, 15 % of tenants randomised | `logging_policy_probs`, `logging_epsilon`, `rct_fraction` | assumption; produces *confounding by indication* |
| Unmeasured confounding | 0 by default | `hidden_confounding` | swept 0-2 in the sensitivity analysis |
| Costs (relative to monthly rent) | service 0.25, concession 0.50, outreach 0.05 | `cost_rel` | assumption; currency independent. Costs are paid up front whether or not the tenant renews |
| Retained-rent proxy V | rent x (12 + 1) = next lease term + one month of turnover cost avoided | `retained_value` | simplification (no discounting, vacancy length, re-letting probability) |
| Policy rules at decision time | service only with an open request; concession only with zero arrears and <= 2 late payments | `feasibility` | stand-in for company policy |
| Portfolio budget | 6 % of one month's total rent roll | `budget_pct_rent_roll` | chosen so that it binds (unconstrained oracle plan would spend ~13 %) |
| Risk baselines | trained on all history rows regardless of action taken | `fit_risk_model` | "renewal risk under past practice" |

## What the synthetic data can and cannot tell you
* It CAN tell you whether an estimator recovers effects **when the generator's assumptions hold**, and how it degrades when they are violated (sensitivity script).
* It CANNOT tell you the true effect of a concession in a real building. Say this explicitly in the review.
* Individual-level effects are only weakly identifiable (see `pehe` vs `pehe_if_constant_ate` in the report). The defensible claim is about
  *policy value*, not about the precision of any single tenant's estimate.
