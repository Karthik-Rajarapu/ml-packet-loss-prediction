# DEMO_GUIDE.md
## How to Present This Project

This guide sequences a live technical demonstration. Follow the flow; every command below was actually run and verified during development (see `docs/RESULTS.md` Section A) — nothing here requires you to fake output.

**Before you start**: state up front, once, clearly: *"No real Mininet experiments have been executed in this environment, so everything you'll see involving a trained model uses either the naive baseline or an explicitly labeled test fixture — not real network performance."* Say this before anyone asks, not defensively after.

---

## 1. Explain the Problem (1–2 min)

"Packet loss in a network is usually detected only after it happens. We're asking: can we predict the packet loss in the *next* interval from what we can observe *right now* — RTT, jitter, throughput, queue depth, current loss — before it occurs?"

## 2. Show the Network Topology

Open `docs/ARCHITECTURE.md` Section 1 (or draw it live):
```
h1 --\                                  /-- h3
      s1 ====[ shaped bottleneck ]==== s2
h2 --/                                  \-- h4
```
"Only the middle link is bandwidth/delay/queue-shaped — that's the bottleneck. We push more traffic than it can carry so packet loss happens for real, from real queue overflow, not because we told it to."

## 3. Explain the Future-Interval Target

"At time T, our features describe what's happening now or earlier. The label — the thing we're predicting — is the packet loss measured in interval T+1, which by definition doesn't exist yet at prediction time."

## 4. Explain Why Leakage Prevention Matters

"If any feature accidentally contained information from T+1, the model would look artificially perfect in testing and then fail in the real world, because that information wouldn't be available at actual prediction time. We check this four separate ways — show `docs/METHODOLOGY.md` Section 10 if asked for detail." Optionally run:
```bash
py -3 -m pytest tests/test_inference_leakage.py -v
```
and point out the leakage tests passing.

## 5. Show the Feature Pipeline

Open `src/network/schema.py` — point at `FEATURE_COLUMNS` and `TARGET_COLUMN`. "This one list is the single source of truth. Every other part of the system — training, the saved model, the inference engine, even the dashboard's input form — reads from this exact list. We verified that end-to-end in one test."

## 6. Show the ML Model Pipeline

```bash
py -3 -m pytest tests/test_train_pipeline.py tests/test_end_to_end_integration.py -v
```
"This trains all five models — the naive baseline plus four regressors — on a synthetic fixture, end to end, and proves the whole chain works. It does **not** prove real-world accuracy; there's no real data yet."

## 7. Show the Inference Engine

```bash
python3 scripts/predict_packet_loss.py --demo --baseline
python3 scripts/predict_packet_loss.py --demo --use-test-fixture
```
Point out the second command's banner: **"TEST FIXTURE DEMONSTRATION — NOT REAL NETWORK PERFORMANCE."** "That label isn't decoration — the code structurally refuses to let a test model be loaded as if it were production, and vice versa."

## 8. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```
Open the browser to `http://localhost:8501`. You land on **Home** — three workflow cards (Upload, Train & Validate, Predict) and a "Get Started" button. Click it to go to **Upload Data**.

## 9. Demonstrate the Workflow (Test-Fixture Mode)

Walk the sidebar in order: **Upload Data** (upload a small CSV — even one with everyday column names like `rtt`/`loss`/`throughput`; point out the column-mapping section auto-detecting them, with low-confidence guesses flagged for confirmation, never silently accepted) → **Prepare Dataset** (click "Prepare Dataset," show the green checklist and the real, computed zero-loss/non-zero-loss split) → **Train Model** (click "Train Models," show the real MAE/RMSE/R² comparison table and which model was recommended and why) → **Predict**, either the "Manual Prediction" tab (defaults are pre-filled with clearly synthetic example values) or "Batch Prediction" with a small CSV. Show the CURRENT OBSERVATION vs. PREDICTED NEXT INTERVAL split, the risk badge, and — since no production model exists on this machine — the demonstration-mode warning banner directly above the result. Finish on **History** to show the session table, and optionally record a made-up "actual outcome" to show the actual-vs-predicted chart appearing.

## 10. State Plainly That This Is Not Empirical

Close the live-model portion with exactly this: *"Everything you just saw uses a small decision tree trained on a handful of made-up rows, or the simple 'loss doesn't change' baseline. We built and tested the entire pipeline — data validation, leakage protection, five models, an inference engine, this dashboard — but we do not have, and are not claiming, real network prediction accuracy. That requires Mininet experiments we couldn't run in this environment."* Then, if useful, show `docs/RESULTS.md` Section B on screen — the explicit "NOT AVAILABLE" table is more convincing of rigor than trying to talk around the gap.

---

## Quick Command Reference

```bash
py -3 -m pytest tests/ -v                        # 223 passed, 2 skipped
python3 scripts/system_check.py                    # software health check
python3 scripts/smoke_test.py                        # Mininet environment check (will show DRY-RUN on this machine)
python3 scripts/predict_packet_loss.py --help          # CLI options
streamlit run app.py                                      # dashboard
```
