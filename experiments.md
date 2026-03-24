# Some experiments on the first 10 procedures to test if rag always on is benefical or not

## Configuration
- Model: gpt-5.4-mini
- Max attempts for slef-refine: 2
- RAG: different settings, on demand by tool calling, or always on 
- Structural checker: always on
- Few-shot examples always in the promt also picked b

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

**Run 3 — Checker v2 + RAG always on (k=1)**

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

**Conclusion is that always-on RAG (k=1) is the best setup.** 
