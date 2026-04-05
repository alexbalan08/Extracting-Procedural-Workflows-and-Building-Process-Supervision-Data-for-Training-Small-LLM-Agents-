# Extraction Pipeline Experiments — 10 Test Procedures

Model: gpt-5.4-mini, temperature=0, seed=42
Few-shot: 3 examples always in prompt
Structural checker: always on
Minor action naming differences ignored

---

## Baseline (no RAG, no checker, 1 attempt)

- Clean passes: 2/10
- Action F1: 0.926 (FN=4, FP=5)
- Gateway F1: 0.710 (FN=7, FP=2)

Missing: "request handled" (1808351684), "MyEndEvent" (1162283938), 2 actions in 1881121390
Hallucinated: "check_the_critical_level_again" (1881121390), 2 extra checks (1239633710), "WF" (1160550961)
All merge gateways missed. Procedure 676185712 modeled as sequential instead of split.

---

## Checker only (no RAG, max_attempts=2)

- Clean passes: 2/10
- Action F1: 0.917 (FN=5, FP=5)
- Gateway F1: 0.750 (FN=6, FP=2)

Checker fixed missing "MyEndEvent" (1162283938) but introduced a new FN in 676185712 (missing "Place in the Outbox").
No improvement in clean count vs baseline. Loop in 1988807363 still broken.

---

## RAG k=1, checker v2, max_attempts=2

- Clean passes: 4/10
- Action F1: 0.942 (FN=3, FP=4)
- Gateway F1: 0.788 (FN=5, FP=2)

RAG fixes the loop in 1988807363 and gateway structure in 2074496137.
Action FN drops from 4-5 to 3. Two new clean passes vs checker_only.
Still missing merge gateways in 2033265575 and reversed home loan conditions.

---

## RAG k=1, checker v2, max_attempts=3

- Clean passes: 5/10 
- Action F1: 0.942 (FN=3, FP=4)
- Gateway F1: 0.800 (FN=6, FP=0)

Best config. Gateway FP drops to 0 — the 3rd attempt eliminates all hallucinated gateways.
Only config to correctly extract 2033265575 (dual loan process) with all 4 gateways and correct branch conditions.


---

## RAG k=2, checker v2, max_attempts=2

- Clean passes: 4/10 
- Action F1: 0.935 (FN=2, FP=6)
- Gateway F1: 0.750 (FN=6, FP=2)

Lowest action FN (2) but highest action FP (6) — k=2 retrieval causes hallucination.
Procedure 2074496137 gets 3 duplicate actions (hallucinated second round of check+backorder+reserve).
Wrong gateway types in 1160550961 (AND instead of XOR).

---

## RAG k=2, checker v2, max_attempts=3

- Clean passes: 3/10 
- Action F1: 0.935 (FN=3, FP=5)
- Gateway F1: 0.839 (FN=5, FP=0)

Regressions vs k=1: 2074496137 develops a loop error, 2033265575 loses merge gateways and reverses conditions.
Gateway FP=0 (same as k1_c3) but fewer clean passes due to structural flow errors.

---

## Summary


Key findings:
- RAG k=1 is the sweet spot — k=2 causes confabulation and regressions
- The checker alone does not improve clean count vs baseline
- RAG drives improvement by providing structural priors for loops and gateways
- Gateway FN (missing merge gateways) is the dominant error across all configs (5-7 per run)
- 4 procedures are unsolved in all configs: 1808351684, 1881121390, 1239633710, 676185712

