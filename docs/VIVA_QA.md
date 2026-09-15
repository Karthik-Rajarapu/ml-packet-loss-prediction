# VIVA_QA.md
## Viva Preparation — Questions and Answers

Answers are grounded in what was actually built and tested. Where a question implies an empirical result that doesn't exist, the answer says so directly — see Q "What are your actual experimental results?" at the end, which is the single most important answer in this document to get right.

---

## Networking

**1. What is packet loss?**
The percentage of packets sent that never arrive at the receiver, typically because a network device (a router/switch queue) drops them when it cannot forward them fast enough.

**2. What causes packet loss?**
Primarily congestion: when offered traffic exceeds a link's capacity for long enough, its buffer fills and new arrivals are tail-dropped. Also possible: link errors, wireless interference, hardware faults — this project focuses specifically on congestion-driven loss, since that's what the Mininet dumbbell topology is designed to produce genuinely.

**3. What is a bottleneck link?**
The link in a path with the lowest available capacity relative to offered load — the point where queueing and loss actually occur. In our topology it's the single shaped `s1<->s2` link; every other link is fast and unshaped by design, so the bottleneck's behavior is isolated and unambiguous.

**4. Why use a dumbbell topology?**
It's the standard topology for demonstrating congestion in networking coursework: multiple senders share one bottleneck link to receivers, so contention and queueing are directly observable and controllable via one link's parameters.

**5. What is RTT?**
Round-trip time — the time for a packet to travel to a destination and its acknowledgment/reply to return. We measure it via `ping`.

**6. What is jitter?**
Variation in latency between consecutive packets. We use the standard deviation of RTT samples within a sampling interval as a protocol-agnostic jitter proxy.

**7. What is bandwidth utilization?**
The fraction of a link's capacity actually being used: `throughput / configured bandwidth × 100`. Note it can exceed 100% momentarily under above-capacity offered load — we deliberately don't cap it at 100 in our validation, since that would reject legitimate data.

**8. What is queueing?**
The buffering of packets at a network device when they arrive faster than they can be forwarded. Queue depth and occupancy directly relate to both latency (longer queue = longer wait) and loss (full queue = drops).

**9. Why use Mininet?**
It runs a real Linux kernel network stack (real TCP/IP, real queueing, real namespaces) rather than a discrete-event simulation, so packet loss is a genuinely emergent property of the network — not injected. It's also the most recognized network-emulation tool in academic networking courses and light enough to run on a single laptop, unlike ns-3's steeper setup cost. Full comparison: `docs/PHASE_1_NETWORK_EXPERIMENT.md`.

**10. How does iperf3 help?**
It generates real, measurable traffic (TCP or UDP) between hosts and reports throughput and, in UDP mode, exact packet counts and loss percentage directly in its JSON output — more reliable than inferring loss from TCP retransmission behavior.

## Machine Learning

**11. Why use regression?**
Packet loss is a continuous percentage, not a discrete category — regression is the natural formulation. We layer an optional LOW/MODERATE/HIGH classification on top for interpretability, but it's derived from the regression output, never a replacement for it.

**12. What exactly is the target variable?**
`target_next_packet_loss_pct` — the packet-loss percentage measured in the interval immediately following the row's own interval, within the same experiment.

**13. Why predict the next interval?**
Because a system that already knows the current interval's loss doesn't need a prediction — the value of forecasting is in knowing what's *about* to happen, before it happens, so a proactive response (e.g. traffic shaping) is possible.

**14. Why is data leakage dangerous?**
It lets a model implicitly "see" information it wouldn't have at real prediction time, producing misleadingly good test performance that collapses in real deployment. For a temporal task like this one, the most dangerous form is accidentally including future-interval information as a feature.

**15. How is temporal leakage prevented?**
The target is constructed once, centrally, by a function that groups rows by experiment and orders by interval before shifting — so a target is always the *next* row's value in the *same* experiment, never adjacent-in-storage-but-wrong-experiment data. An independent validator then re-derives every target from scratch and confirms it matches.

**16. Why use group-aware splitting?**
Consecutive intervals within one experiment are highly autocorrelated (same configuration, evolving queue state). A row-level random split would put adjacent rows from the same run's trajectory on both sides of the train/test boundary, letting the model "peek" at the local pattern it's being tested on. We split on whole experiment groups instead — either chronologically or via a random group hold-out — with an explicit zero-overlap check.

**17. Why use a naive baseline?**
To prove the ML models learn something real. A model that can't beat "next interval's loss will equal this interval's loss" isn't demonstrating any predictive skill about network dynamics, regardless of how good its R² looks in isolation.

**18. Why Linear Regression?**
As a simple, fast, interpretable reference point — it tests whether the relationship between features and target is largely linear before reaching for more complex models.

**19. Why Decision Tree?**
Captures nonlinear threshold effects (e.g. "loss rises sharply once queue occupancy crosses X") and is directly visualizable/explainable, useful for a viva.

**20. Why Random Forest?**
An ensemble of trees that typically generalizes better than a single tree, robust to noisy measurements, and provides a native feature-importance ranking.

**21. Why Gradient Boosting?**
Often the strongest tabular-data performer among classical ML methods, and a natural point of comparison against Random Forest for the same "tree ensemble" family. XGBoost specifically was not added, since it was only planned if results justified the added complexity — and no real results have existed yet to make that case.

**22. What is MAE?**
Mean Absolute Error — the average absolute difference between predicted and actual packet loss, in percentage points. Easy to explain to a non-technical audience: "on average, we're off by X%."

**23. What is RMSE?**
Root Mean Squared Error — like MAE but penalizes large errors more heavily (squares them before averaging, then takes the square root). More sensitive to occasional large misses.

**24. What is R²?**
The proportion of variance in the target explained by the model, relative to always predicting the mean. R²=1 is a perfect fit; R²=0 means no better than predicting the mean; R² can go negative if the model is worse than that.

**25. What does a negative R² mean?**
The model performs *worse* than simply predicting the average target value every time — a red flag that the model has failed to learn a useful relationship, or is badly overfit/misconfigured. We would report this honestly rather than hide it, per the project's evaluation methodology.

**26. How is feature importance calculated?**
Two ways: native impurity-based importance (how much each feature reduces prediction error across a tree ensemble's splits), and permutation importance (how much a metric degrades when one feature's values are randomly shuffled in the held-out test set). Both are computed from the trained pipeline directly, never recomputed with separate logic.

**27. Does feature importance imply causation?**
No. We describe it explicitly as "predictive importance" — a feature can be highly predictive of packet loss without causing it (e.g. it could be a proxy for an underlying congestion state). This project never makes a causal claim from feature importance.

## System

**28. Why separate inference from Streamlit?**
So the prediction logic (validation, preprocessing, model call, risk classification) is reusable, independently testable, and never duplicated inside UI code. The dashboard is presentation-only — it calls one `InferenceEngine.predict()` function and renders the result.

**29. How is the model artifact loaded?**
As a `joblib`-serialized `sklearn.Pipeline` (preprocessing + model together) plus a JSON metadata sidecar. The loader verifies the artifact and metadata both exist, checks the metadata's feature/target contract matches the current schema, and — critically — checks whether the artifact is marked as a test fixture or production model, refusing to load it through the wrong path if there's a mismatch.

**30. How is feature ordering guaranteed?**
The `ColumnTransformer` used in preprocessing selects columns by name, not position, so it's actually order-independent by construction — but the inference engine still builds its feature frame in one canonical order anyway, as a self-evident contract and a guard against a future refactor that might rely on positional ordering.

**31. What happens if the model is missing?**
The inference engine raises a specific error with an exact, informative message; the CLI and dashboard both catch it and display that message plainly. The system never silently substitutes a test-fixture model for a missing production one — that's enforced structurally, not just as a convention.

**32. Why is there a demonstration mode?**
Because no production model currently exists (Phase 5 is blocked), but the rest of the system — dashboard, inference engine, risk layer — still needs to be shown working. Demonstration mode uses an explicitly, persistently labeled test-fixture model so nobody could mistake its output for a real result.

**33. How would live monitoring be integrated?**
The inference engine's public interface is a plain function taking a `dict` of current metrics and returning a typed result — a future live-metrics collector (reading from a running Mininet experiment or another monitoring source) would only need to produce that same dict shape and call the same function; no dashboard or inference code would need to change. This is a documented design property, not yet an implemented integration.

**34. How would you deploy this system?**
Out of scope for this project by design (explicitly excluded from every phase), but the natural next step would be running the Streamlit app behind a standard web server/reverse proxy, with the trained model artifact and its metadata versioned alongside the code — no architectural blocker exists, it simply wasn't attempted.

**35. What are the current limitations?**
No real network dataset or production model exists (the central limitation). No lag/rolling feature engineering. No browser-based UI verification (only startup/import verification). The reported inference-latency numbers are a synthetic-model, single-laptop software benchmark, not a network or production performance claim. Full list: `docs/RESULTS.md`, `docs/FINAL_REPORT.md` Section 22.

---

## The Critical Question

**"What are your actual experimental results?"**

> The software pipeline, inference engine, dashboard, leakage protection, and integration were implemented and tested — 223 automated tests pass, including a full end-to-end fixture pipeline and adversarial leakage tests. However, live Mininet experiments could not be executed in the available environment, so we do not claim real network empirical metrics such as MAE, RMSE, or R². Those results remain the explicit next step once the environment is available, and every part of the system is already built and tested to produce them with no further code changes required.
