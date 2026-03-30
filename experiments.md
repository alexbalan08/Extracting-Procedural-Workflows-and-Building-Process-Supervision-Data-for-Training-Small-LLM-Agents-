# Some experiments on the first 10 procedures to test if rag always on is benefical or not

## Configuration
- Model: gpt-5.4-mini
- Max attempts for slef-refine: 2 and 3 tested
- RAG: different settings, on demand by tool calling, or always on 
- Structural checker: always on
- Few-shot examples always in the promt also picked carefully

---

**Run 1 — LLM Checker v1 (rag not activated)**

| file_index | Attempt | Remaining Issues | Verdict |
|------------|---------|-----------------|---------|
| 1808351684 | 2 | 2 — explicit end path not modeled | False positive |
| 1881121390 | 2 | 4 — naming inconsistency, wrong predecessors | Legitimate |
| 1162283938 | 2 | 2 — missing MyEndEvent action | Legitimate |
| 2074496137 | 2 | 2 — missing explicit termination path | False positive |
| 1988807363 | 2 | 2 — re-verification not modeled as loop | Debatable |
| 1239633710 | 2 | 2 — extra invented decision point | Legitimate |
| 676185712 | 2 | 8 — missing data objects | Legitimate (secondary) |
| 1160550961 | 2 | 3 — two NCIIC checks collapsed into one | Legitimate |
| 2033265575 | 2 | 3 — checker says no real issue | False positive |
| 802124898 | 1 | — | Clean |

4/10 false positives. The checker was over-flagging termination paths that weren't actually wrong.

---

**Run 2 — LLM Checker v2 (revised prompt, rag not activated)**

| file_index | Attempt | Remaining Issues | Verdict |
|------------|---------|-----------------|---------|
| 1808351684 | 2 | 4 — verbose non-issues, no real structural problem | False positive |
| 1881121390 | 1 | — | Clean |
| 1162283938 | 2 | — | Clean |
| 2074496137 | 2 | — | Clean |
| 1988807363 | 2 | 2 — clarification loop not modeled correctly | Legitimate |
| 1239633710 | 2 | 2 — repeated check not modeled | Legitimate |
| 676185712 | 2 | 1 — repeated "Look up Vendor Number" step missing | Legitimate |
| 1160550961 | 2 | 3 — multiple structural mismatches (NCIIC checks) | Legitimate |
| 2033265575 | 2 | 2 — **inverted gateway logic, low/high liability reversed** | Legitimate — critical |
| 802124898 | 1 | — | Clean |

Down to 1/10 false positives. More importantly, the checker caught 5 real errors including the liability inversion in 2033265575 — the kind of semantic mistake the structural checker can't see at all. 1808351684 is still a problem.

---

**Run 3 — Checker v2 + RAG always on (k=1), max_attempts=2**

| file_index | Attempt | Remaining Issues | Verdict |
|------------|---------|-----------------|---------|
| 1808351684 | 2 | 3 — gateway routing inconsistent with source text | Legitimate |
| 1881121390 | 2 | 3 — "delivery has arrived" modeled as action, extra gateway not in text | Legitimate |
| 1162283938 | 1 | — | Clean |
| 2074496137 | 2 | — | Clean |
| 1988807363 | 1 | — | Clean — resolved in 1 attempt vs 2 previously |
| 1239633710 | 2 | 2 — repeated check still missing | Legitimate |
| 676185712 | 1 | — | Clean — resolved in 1 attempt vs 2 previously |
| 1160550961 | 2 | 2 — NCIIC branch distinctions not reflected | Legitimate |
| 2033265575 | 2 | 2 — **inverted gateway logic persists** | Legitimate and bad error  |
| 802124898 | 1 | — | Clean |

**False positives: 0/10**
**Clean passes: 5/10 (up from 3/10 in run 2)**

- RAG always-on doubled clean passes compared to run 1. We still have issue for 2033265575, but honestly I have issue for this myself to follow the original looping flow
- Procedures 5 and 7 now pass in 1 attempt (in v2 was 2) 
- 0 false positives — checker v2 + RAG always is the best configuration so far


---

**Run 4 — Checker v2 + RAG always on (k=2) + max_attempts=3**

| file_index | Attempt | Remaining Issues | Verdict |
|------------|---------|-----------------|---------|
| 1808351684 | 3 | 1 — "Forward it" action modeling discrepancy | Legitimate |
| 1881121390 | 1 | — | Clean — resolved in 1 attempt (was 2 in run 3) |
| 1162283938 | 1 | — | Clean |
| 2074496137 | 1 | — | Clean — resolved in 1 attempt (was 2 in run 3) |
| 1988807363 | 3 | 1 — clarification loop not fully modeled | Legitimate |
| 1239633710 | 3 | 1 — repeated check still missing | Legitimate |
| 676185712  | 2 | — | Clean |
| 1160550961 | 3 | 1 — PBOC check modeled as action not condition | Legitimate |
| 2033265575 | 3 | — | Clean — **inverted logic fixed on 3rd attempt this is super big and important** |
| 802124898  | 1 | — | Clean |

**False positives: 0/10**
**Clean passes: 6/10**

- The critical inverted logic error on 2033265575 is now fixed with the 3rd attempt, k=2 helped 1881121390 and 2074496137 resolve in 1 attempt (rather than 2)

**Best config so far: checker v2 + RAG always on k=2 + max_attempts=3**

---

**Run 5 — Checker v2 + RAG always on (k=2) + max_attempts=2** *(ablation: isolates effect of 3rd attempt vs k=2)*

| file_index | Attempt | Remaining Issues | Verdict |
|------------|---------|-----------------|---------|
| 1808351684 | 1 | — | Clean — **first clean pass across all 5 runs, was failing even in run 4 with 3 attempts** |
| 1881121390 | 1 | — | Clean — resolved in 1 attempt (consistent with run 4) |
| 1162283938 | 1 | — | Clean |
| 2074496137 | 1 | — | Clean — resolved in 1 attempt (consistent with run 4) |
| 1988807363 | 2 | checker text says "loop is present, no issue there" / "No structural issues found" | False positive — remaining_issues field populated but checker's own text says no real issue |
| 1239633710 | 2 | 1 — Yes branch routed back to place an order, creating a loop not in source text | Legitimate |
| 676185712  | 2 | — | Clean |
| 1160550961 | 2 | 2 — gateway exclusivity mismatches with text (NCIIC and PBOC branches) | Legitimate |
| 2033265575 | 2 | 3 — **inverted liability conditions persist** | Legitimate — confirms 3rd attempt in run 4 was what fixed this |
| 802124898  | 1 | — | Clean |

**False positives: 1/10**
**Clean passes: 6/10**

- 1808351684 finally passes clean — k=2 resolved the gateway routing issue that persisted across all prior runs
- 2033265575 still fails: direct confirmation that the 3rd attempt (run 4) is what fixes the inverted logic, not k=2 alone
- 1988807363 is a false positive: the checker flagged it but its own reasoning confirmed no structural issues
- **Ablation conclusion**: k=1→k=2 (run 3→run 5) adds 1 clean pass (5→6) but introduces 1 false positive; adding the 3rd attempt (run 5→run 4) eliminates that false positive and fixes 2033265575, keeping 6 clean passes — the 3rd attempt improves reliability without changing the clean count
