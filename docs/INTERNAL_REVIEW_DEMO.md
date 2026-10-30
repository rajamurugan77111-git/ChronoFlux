# Internal Review Demo — ChronoFlux v7

## Goal
Prove three things clearly:

1. **Detection works:** live traffic → current attack/behaviour.
2. **Forecasting genuinely works:** ordered T1–T4 history → future T5/T6/T7.
3. **Forecast happens before a later observed transition:** prediction time → lead time → actual event.

## Demo sequence

1. Open ChronoFlux Overview.
2. Show Windows backend online.
3. Show Ubuntu sensor connected and its recording-session ID.
4. Generate normal lab traffic and show source/destination IP, ports, protocol, packet rate and device inventory.
5. Show T1–T4 filling with real chronological windows.
6. Run only authorized controlled suspicious behaviour in the isolated lab.
7. Show three detection signals: ML category, anomaly score, signatures.
8. Show the real fusion formula and component scores.
9. Show current ChronoFlux behaviour stage + risk + MITRE-style interpretation.
10. Open Forecasting and show session count, sequence count, train/validation/test split policy and model qualification.
11. If evidence is insufficient, explicitly show **INSUFFICIENT EVIDENCE** rather than a fabricated T5.
12. After enough independent sessions, train/refresh the sequence model.
13. Show accepted T5/T6/T7 predictions and confidence.
14. Repeat an independently recorded controlled sequence and verify a genuine stage transition.
15. Show forecast confusion matrix, Macro F1 and measured transition lead time.
16. Open Evidence Chain and verify SHA-256 chain integrity.
17. Open Offline Analysis and demonstrate benchmark attack-category classification.

## Jury answer — what produces T5?

> “ChronoFlux uses a sequence-based Random Forest approach. Random Forest itself is not time-aware. We build one ordered feature representation from T1 to T4 and train separate Random Forest horizon classifiers for T5, T6 and T7 using actually observed future states from genuine chronological recording sessions.”

## Jury answer — how do you prevent temporal leakage?

> “Each Ubuntu run has a recording-session ID. Entire sessions are ordered by time and assigned to train, validation or test partitions. We do not place overlapping windows from one session across those partitions.”
