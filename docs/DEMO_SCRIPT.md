# 12-minute live demo (data to show: the synthetic "current" cohort of 3,000 tenants whose leases end in 30-90 days)

Setup before the review: `python scripts/run_all.py`, then `streamlit run app/dashboard.py`. Press **Reset demo** in the sidebar before you start.
Demo tenants are in the "Demo tenant" drop-down (written by `scripts/07_pick_demo_tenants.py`; with seed 42 they are #12140, #12483, #14083, #12081, #14717, #12425 - ids may differ on another machine/library version).

1. **Problem (1 min).** Open *Tenant decision*, top-risk tenant. "A risk score says who may leave; we answer *what to do*."
2. **Maintenance-driven tenant (3 min).** Pick it, **Run agentic analysis**. Walk the trace (Trigger -> ... -> Explain), then *Causal estimations* (renewal probability under each action),
   *Financial trade-offs* (uplift, cost, net value), the recommendation text and the evidence tabs. Switch on **Evaluator mode** in the sidebar: the hidden concern and true probabilities appear - "this is only possible because the PMS is synthetic".
3. **Policy constraints (1 min).** Pick *Concession blocked by policy*: the concession is greyed out with the reason; try to execute it anyway - the guardrail refuses.
4. **Human in the loop (1 min).** Approve (or change the action), then **Simulate renewal outcome**. Open *Audit and monitoring*: recommendation, approval (note recommended vs approved), execution, outcome, predicted vs realised.
5. **Portfolio (1 min).** *Portfolio plan*: drag the budget slider; the MILP re-allocates; evaluator mode shows true vs expected gain.
6. **Why causal (2 min).** *Evidence and evaluation* -> *Logged-data checks*: the "association vs causation" figure (naive says actions hurt; adjusted estimate matches the truth and the randomised pilot).
   Then *Decision policies*: 5 policies, value, regret, rule violations.
7. **Trust (2 min).** Overlap + balance + refutations + off-policy evaluation tabs, then *Sensitivity*: "what if the effect is half as big / past managers knew more than we can see?"
8. **What-if (1 min, optional).** *What-if explorer*: set open requests to 0 and watch the recommendation change.

Key sentences: "Everything is synthetic; the effects are assumptions; we show the method recovers them and how it degrades." / "Individual estimates are noisy; the claim is about policy value." / "Constraints and the budget are enforced by the optimiser and again by the execution guardrail."
