# Machine Learning-Based Prediction of Packet Loss in Computer Networks

**Status: draft.** Empirical sections are explicitly marked NOT AVAILABLE where real network data was not obtainable — see the Limitations section. This draft should not be submitted to a venue or cited as containing validated results without first completing Section 9 (Results) against real data.

## Abstract

Packet loss in computer networks is predominantly driven by queueing dynamics under congestion. We investigate whether current network telemetry (RTT, jitter, throughput, utilization, queue occupancy, retransmissions, current loss) can predict next-interval packet loss via supervised regression. We design a Mininet-based dumbbell-topology experiment framework with explicit data-leakage prevention, a group-aware evaluation methodology, and a comparison of a naive temporal baseline against four regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting). We implement and thoroughly test the full pipeline, including a production-style inference engine and a demonstration dashboard (280 automated tests passing). Due to an unresolved local WSL2/Mininet environment constraint, real network experiments could not be executed during this work; we report the engineering methodology and test-verified system behavior, and explicitly mark empirical accuracy results as unavailable pending real-data collection.

**Keywords**: packet loss prediction, network telemetry, regression, Mininet, data leakage, time-series evaluation, congestion.

## 1. Introduction

Reactive packet-loss handling — detecting loss after it has already degraded a connection — is the default operational posture in most networks. We ask whether a lightweight, interpretable ML pipeline trained on locally observable telemetry can forecast loss one interval ahead, enabling proactive responses.

## 2. Related Problem Context

Packet loss under congestion is classically explained by queueing theory: once arrival rate exceeds service rate for long enough, a finite buffer overflows and tail-drops packets. Rather than deriving a closed-form model, we treat the arrival-rate/service-rate/queue-depth/delay interaction as a supervised learning problem, using controlled experiments to generate ground truth. (A full literature review of prior ML-based packet-loss-prediction work was out of scope for this implementation-focused project and is noted as future work rather than fabricated here — see Limitations.)

## 3. Proposed Method

A one-step-ahead regression formulation: at cutoff `T`, predict packet loss observed during `[T, T+1]` using only information available at or before `T`. Evaluated against a naive persistence baseline (predicted = current loss) to establish whether any model learns real dynamics beyond "loss doesn't change."

## 4. Network Experiment Design

A Mininet dumbbell topology (left hosts → shaped bottleneck → right hosts) is used specifically because it produces **real, emergent** packet loss from genuine Linux kernel queueing under real `iperf3` traffic, rather than a value from a discrete-event simulator or an injected `netem loss%` parameter (which would itself be a leakage risk if used as the primary mechanism — see Section 6). Offered load is swept as a bandwidth-relative factor (e.g. 0.7×/1.3×) specifically so the dataset spans both below- and above-capacity conditions.

## 5. Feature Engineering

15 raw per-interval features: 5 configured link conditions (bandwidth, delay, queue depth, traffic type, flow count) and 10 measured conditions (RTT, jitter, throughput, utilization, packet rate, queue backlog, retransmissions, packets sent/received, current loss). No lag or rolling-window features are used — the model sees only the current interval's own measurements plus static configuration, a deliberate scope boundary given no real data has existed to validate more elaborate features against.

## 6. Prediction Formulation and Leakage Prevention

Target: `packet_loss_pct` at interval `T+1`, constructed by a single, centrally-defined, group- and order-aware shift operation, independently re-verified by a leakage auditor that re-derives every target from its true future value and confirms no target is ever drawn across an experiment boundary. The target column is structurally absent from the feature set — there is no code path by which it could be selected as an input, by construction rather than by convention alone.

## 7. ML Models

Naive persistence baseline; Linear Regression (with feature standardization); Decision Tree Regressor; Random Forest Regressor; Gradient Boosting Regressor. All models share one preprocessing contract (median imputation; one-hot encoding for the single categorical feature) fit strictly on training data only. Fixed random seed (42) throughout for reproducibility.

## 8. Experimental Methodology

Group-aware evaluation: either chronological (train on earlier experiments, test on later ones) or random-group (fixed-seed random hold-out of entire experiment groups) — never a row-level split, since consecutive intervals within one experiment are highly autocorrelated and a row-level split would leak local trajectory information across the train/test boundary. Metrics: MAE, RMSE, R², plus residual-distribution statistics (given packet loss's typically zero-inflated distribution).

## 9. Results

**NOT AVAILABLE.** No real Mininet experiment was executed during this work — the required WSL2/Linux environment was not available on the development machine (Windows features `Microsoft-Windows-Subsystem-Linux`/`VirtualMachinePlatform` remained disabled; `mn`/`iperf3`/`tc`/`ovs-vsctl` were never installed). Consequently there is no real dataset, no production-trained model, and no real MAE/RMSE/R² to report. What *was* verified: the complete software pipeline (280 automated tests), a full synthetic-fixture integration test exercising every stage from raw CSV through inference and risk classification, and a software-only inference-latency measurement (median 12.6 ms per prediction, test-fixture model, single laptop) that is explicitly not a network-performance or accuracy claim.

## 10. Limitations

No empirical accuracy results exist (Section 9). No lag/rolling feature engineering was implemented or evaluated. No literature review of prior packet-loss-prediction approaches was conducted as part of this implementation-focused effort — any such comparison in a future revision requires citing sources actually consulted, not references added for form. The reported latency benchmark reflects one synthetic model on one laptop, not production or network conditions.

## 11. Future Work

Resolve the environment blocker; execute the already-prepared pilot sweep (`configs/pilot_sweep.json`); train and evaluate the existing pipeline unmodified against the resulting real data; populate this section and Section 9 with genuine results; extend to calibrated prediction intervals and, if real-data volume justifies it, lag/rolling features.

## 12. Conclusion

We present a complete, tested, leakage-safe methodology and implementation for next-interval packet-loss regression, validated at the software-engineering level (280 passing tests, a full end-to-end fixture integration test, and a working inference/dashboard system) but not yet at the empirical level, pending resolution of a local network-experiment environment constraint. The system is designed to produce real results with zero code changes once real data is available.

## References

- Lantz, B., Heller, B., McKeown, N. "A Network in a Laptop: Rapid Prototyping for Software-Defined Networks." *Hotnets*, 2010.
- ESnet. iperf3. https://github.com/esnet/iperf.
- Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12, 2011.

*No further references are included. A proper literature survey of prior packet-loss-prediction and network-telemetry ML work should be added before this draft is treated as submission-ready for a research venue — citations should be added only for sources actually read, not to pad this section.*
