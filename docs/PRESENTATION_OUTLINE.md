# PRESENTATION_OUTLINE.md
## Technical Presentation — 15 Slides

No fake model-performance graphs appear on any slide. Where a slide would normally show empirical results, it shows the honest engineering-verification status instead (Slide 12) or a labeled "NOT AVAILABLE" (Slide 9's evaluation portion).

---

**Slide 1 — Title**
"Machine Learning-Based Prediction of Packet Loss in Computer Networks." Names, course, date.

**Slide 2 — Problem Statement**
"Can current network conditions predict packet loss in the *next* interval?" One sentence. No numbers yet.

**Slide 3 — Motivation**
Packet loss is usually detected reactively. Proactive prediction could enable traffic shaping, QoS adjustment, or early warning before degradation is felt by users.

**Slide 4 — Proposed Solution**
One diagram: Network conditions → Feature validation → ML model → Predicted next-interval loss → Risk level. State plainly: regression, not classification, is the primary formulation.

**Slide 5 — Network Architecture**
The dumbbell topology diagram (`docs/ARCHITECTURE.md` Section 1). Explain: only the middle link is shaped; that's the bottleneck; loss emerges from real queueing, never injected.

**Slide 6 — Data Collection**
`iperf3` + `ping` + `tc` sampling loop, one row per interval, leakage-safe target construction. One line: "no real dataset exists yet — environment blocker" (state it here, early, not buried).

**Slide 7 — Features + Target**
15 features (list the 5 config + 10 measured categories, not all 15 names), one target: next-interval packet loss. Emphasize: target is structurally absent from the feature list.

**Slide 8 — ML Pipeline**
Naive baseline + Linear Regression + Decision Tree + Random Forest + Gradient Boosting. One preprocessing contract, fit on train only.

**Slide 9 — Leakage Prevention + Evaluation**
Four leakage safeguards (structural, construction-time, independent validator, inference-time). Group-aware split (chronological/random-group, never row-level). Evaluation results: **NOT AVAILABLE — no real data**; say so on the slide itself, don't skip it.

**Slide 10 — Inference Engine**
Feature contract enforcement, model loading with production/fixture separation, one `predict()` API. No confidence score invented.

**Slide 11 — Streamlit Dashboard**
Screenshot (from a real run) showing DEMONSTRATION mode banner and a test-fixture prediction, clearly labeled. Point out the CURRENT OBSERVATION vs. PREDICTED NEXT INTERVAL split.

**Slide 12 — Testing / Engineering Results**
223/223 automated tests passing, 2 live-Mininet tests correctly skipped. End-to-end fixture integration test. System health check: PASS. Software inference-latency benchmark (median ~12.6ms, test-fixture model, single laptop) — labeled explicitly as a software benchmark, not network performance.

**Slide 13 — Current Limitation**
One slide, stated plainly, no hedging: "No real Mininet experiments were executed. No production model, no real MAE/RMSE/R², no real predictions exist." Give the reason in one line (WSL2 environment blocker).

**Slide 14 — Future Work**
Resolve the environment blocker → run the prepared pilot sweep → train on real data → populate real results. Then: calibrated prediction intervals, live metrics integration.

**Slide 15 — Conclusion**
A complete, tested, leakage-safe software system for network-conditions-to-packet-loss prediction. Engineering validated (223 tests); empirical validation is the explicit next step, not yet claimed.

---

**Presenter note**: Slides 9 and 13 are the two places evaluators will probe hardest. Do not soften either — state "NOT AVAILABLE" and the reason plainly, then move immediately to what *was* verified. A confident, direct admission reads as more rigorous than a vague or evasive one.
