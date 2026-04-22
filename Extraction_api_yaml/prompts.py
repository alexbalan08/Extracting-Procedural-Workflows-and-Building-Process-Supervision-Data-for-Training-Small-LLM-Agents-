#All few shot examples are taken directly from extracted_train.json ground truth and i picked them carefully so the
#cover all possible ways of gateways or even sequential
#we have the RAG implemented anyways if more context is needed

#YAML version of the prompts — same examples as JSON but output is YAML format
#hypothesis: YAML is shorter, fewer brackets, easier for the model to produce correctly


SYSTEM_PROMPT = """You are an expert procedural workflow analyst. Given a natural-language procedure description, extract a fully structured workflow. It will be used to train a PRM and agent for planning tasks.

Output two parts in order:
1. A <reasoning> block where you trace the procedure step by step.
2. A ```yaml block containing the workflow object.

─── WORKFLOW SCHEMA ───────────────────────────────────────────────────────────

actions       : list of action objects, one per step/activity:
  id          : str  – snake_case, unique (duplicate names get _2, _3 suffix)
  name        : str  – EXACT wording from the procedure text (do not paraphrase)
  predecessors: list[str] – action IDs (or "start") that immediately precede this action
  successors  : list[str] – action IDs or gateway IDs that immediately follow
  postconditions: ["{id}_done"]

gateways      : list of gateway objects (only when the flow branches or merges):
  id          : str  – "gateway_{type}_{index}" (e.g. "gateway_xor_3")
  type        : "exclusive" | "parallel" | "inclusive"
  role        : "split" | "merge" | "join_split" | "pass_through"
  incoming_from: list[str] – action/gateway IDs or "start" feeding into this gateway
  branches    : list of { next: str|null, condition?: str }
               `next` is null when a branch leads directly to process end.
               `condition` must be SHORT (1-4 words) using wording from the text
               (e.g. "Yes"/"No", "Approved"/"Rejected"). Do NOT use full sentences.

─── CORE RULES ────────────────────────────────────────────────────────────────

MERGE before multi-predecessor actions:
  Any action reachable from more than one branch MUST have a merge gateway as
  its sole predecessor. Branches point to the gateway, not the action directly.
  This is the most common mistake — always check Step 2 for this.

LOOP detection:
  Signals: "repeat", "check again", "if still X", "once more", "until X".
  Model the re-evaluated step as ONE action entered via a merge gateway whose
  `incoming_from` includes "start" and the loop-back branch. A loop MUST have
  a named exit condition — if none is clear, prefer a single pass.
  NOT a loop: restated text with identical outcomes and no state change
  between iterations (common in PAGED).

PARALLEL fan-out (rare):
  Unconditional "X, also Y" → list both in action A's `successors` directly,
  no gateway. Use gateway_and_<N> only when the text says "in parallel",
  "simultaneously", or requires synchronization.

─── ID CONVENTIONS ────────────────────────────────────────────────────────────
• Action IDs: action name → lowercase → non-alphanumeric to underscore → strip edges
• Gateway IDs: "gateway_xor_<N>" / "gateway_and_<N>" / "gateway_or_<N>" (0-based, document order)
• "start" is used as a predecessor only — never create a "start" action
• Never create "end" or "terminate" actions — model termination with empty successors

─── OUTPUT FORMAT ─────────────────────────────────────────────────────────────

<reasoning>
Step 0 — Scan for signals: loop? parallel fan-out? multi-branch convergence?
Step 1 — Identify actions with their IDs.
Step 2 — Trace predecessors and successors.
Step 3 — Identify gateways (splits, merges, loop-backs).
</reasoning>

```yaml
file_index: <value from the prompt>
actions:
  - id: ...
    name: ...
    ...
gateways:
  - id: ...
    ...
```
"""

#i mannuyally selected 3 example which cover and and also or brances and linear as well
#they needed to be reatively short so we dont confuse the model with too much context.
#some of thhem can get extremely big because of the reaosning traces so selecting manually is important

#Example simple linear procedure  1310881958
_EX1_PROCEDURE = (
    "For the HR Representative, the first step is to create a recruitment vacancy in NGA.net. "
    "After creating the vacancy, the next step is to manage external advertising. "
    "Once the external advertising is managed, the process comes to an end."
)

_EX1_OUTPUT = """<reasoning>
Step 0 — Signals: no loop, no fan-out, no convergence.
Step 1 — Actions:
  - "Create Recruitment Vacancy in NGA.net" → create_recruitment_vacancy_in_nganet
  - "Manage External Advertising" → manage_external_advertising
Step 2 — Flow:
  create_recruitment_vacancy_in_nganet: start → manage_external_advertising
  manage_external_advertising → (end)
Step 3 — Gateways: none, purely sequential.
</reasoning>

```yaml
file_index: 1310881958
actions:
  - id: create_recruitment_vacancy_in_nganet
    name: Create Recruitment Vacancy in NGA.net
    predecessors:
      - start
    successors:
      - manage_external_advertising
    postconditions:
      - create_recruitment_vacancy_in_nganet_done
  - id: manage_external_advertising
    name: Manage External Advertising
    predecessors:
      - create_recruitment_vacancy_in_nganet
    successors: []
    postconditions:
      - manage_external_advertising_done
gateways: []
```"""

#example with XOR gateway with early termination file_index 862270781
#also tests the redundant-restatement pattern: "we repeat" does NOT mean loop here
_EX2_PROCEDURE = (
    "To begin the process in Grenoble, the customer enters the store and then decides what they want. "
    "After that, we check whether it is a custom order or an in-store purchase. "
    "If it is a custom order, the customer fills out a customer invoice and the process ends. "
    "If it is an in-store purchase, the process also ends. "
    "We repeat the same check for custom order or in-store purchase, and if it is a custom order, "
    "the customer fills out a customer invoice and the process ends. "
    "If it is an in-store purchase, the process ends as well."
)

_EX2_OUTPUT = """<reasoning>
Step 0 — Signals: "we repeat" looks like a loop signal, but both iterations
  have identical outcomes with no state change between them → redundant
  restatement, NOT a cycle. No fan-out. No convergence (branches terminate
  independently).
Step 1 — Actions:
  - "Customer Enters Store" → customer_enters_store
  - "Customer Decides What They Want" → customer_decides_what_they_want
  - "Fill Out Customer Invoice" → fill_out_customer_invoice
Step 2 — Flow:
  customer_enters_store: start → customer_decides_what_they_want
  customer_decides_what_they_want → gateway_xor_2
  fill_out_customer_invoice: gateway_xor_2 → (end)
Step 3 — Gateways:
  gateway_xor_2 (split): Custom Order → fill_out_customer_invoice
                         In-Store Purchase → null (end)
</reasoning>

```yaml
file_index: 862270781
actions:
  - id: customer_enters_store
    name: Customer Enters Store
    predecessors:
      - start
    successors:
      - customer_decides_what_they_want
    postconditions:
      - customer_enters_store_done
  - id: customer_decides_what_they_want
    name: Customer Decides What They Want
    predecessors:
      - customer_enters_store
    successors:
      - gateway_xor_2
    postconditions:
      - customer_decides_what_they_want_done
  - id: fill_out_customer_invoice
    name: Fill Out Customer Invoice
    predecessors:
      - gateway_xor_2
    successors: []
    postconditions:
      - fill_out_customer_invoice_done
gateways:
  - id: gateway_xor_2
    type: exclusive
    role: split
    incoming_from:
      - customer_decides_what_they_want
    branches:
      - next: fill_out_customer_invoice
        condition: Custom Order
      - next: null
        condition: In-Store Purchase
```"""

#example 3 with OR gateway — split and merge (file_index 1735666188)
#covers the multi-branch convergence pattern with an inclusive merge before end
_EX3_PROCEDURE = (
    "To start, we need to check the application for completeness. After that, we should sort the applications. "
    "During the step of checking the application for completeness, it is important to consider all applications "
    "in the batch. When sorting the applications, we assume that two piles will be created: one for complete "
    "applications and one for incomplete applications. If an application is complete, we proceed to process it. "
    "However, if an application is incomplete, we discard it. This process is repeated for all complete and "
    "incomplete applications until the end."
)

_EX3_OUTPUT = """<reasoning>
Step 0 — Signals: "repeated... until the end" describes batch iteration over
  a collection, not a re-evaluation of the same instance → single pass.
  No fan-out. Convergence YES — both branches end at the same point, merge needed.
Step 1 — Actions:
  - "Check Application for Completeness" → check_application_for_completeness
  - "Sort Applications" → sort_applications
  - "Process Complete Applications" → process_complete_applications
  - "Discard Applications" → discard_applications
Step 2 — Flow:
  check_application_for_completeness: start → sort_applications
  sort_applications → gateway_or_3
  process_complete_applications: gateway_or_3 → gateway_or_6
  discard_applications: gateway_or_3 → gateway_or_6
  gateway_or_6 → (end)
Step 3 — Gateways:
  gateway_or_3 (split): Complete → process, Incomplete → discard
  gateway_or_6 (merge): both branches converge before end
</reasoning>

```yaml
file_index: 1735666188
actions:
  - id: check_application_for_completeness
    name: Check Application for Completeness
    predecessors:
      - start
    successors:
      - sort_applications
    postconditions:
      - check_application_for_completeness_done
  - id: sort_applications
    name: Sort Applications
    predecessors:
      - check_application_for_completeness
    successors:
      - gateway_or_3
    postconditions:
      - sort_applications_done
  - id: process_complete_applications
    name: Process Complete Applications
    predecessors:
      - gateway_or_3
    successors:
      - gateway_or_6
    postconditions:
      - process_complete_applications_done
  - id: discard_applications
    name: Discard Applications
    predecessors:
      - gateway_or_3
    successors:
      - gateway_or_6
    postconditions:
      - discard_applications_done
gateways:
  - id: gateway_or_3
    type: inclusive
    role: split
    incoming_from:
      - sort_applications
    branches:
      - next: process_complete_applications
        condition: Complete
      - next: discard_applications
        condition: Incomplete
  - id: gateway_or_6
    type: inclusive
    role: merge
    incoming_from:
      - process_complete_applications
      - discard_applications
    branches:
      - next: null
```"""

#example 4 XOR loop with exit condition (artificial)
#targets the loop-detection failure mode: "again" / "repeat" caused sequential
#duplication instead of a cycle back via merge gateway. most valuable artificial
#example — no real PAGED example in the selected set shows this pattern cleanly.
_EX4_PROCEDURE = (
    "To process the request, first review the submission. "
    "If changes are needed, send it back to the author for revision, then review "
    "the submission again. If no changes are needed, approve the request and the "
    "process ends."
)

_EX4_OUTPUT = """<reasoning>
Step 0 — Signals: loop YES — "review the submission again" after revision.
  review_the_submission is re-entered (single action, cycle back).
  Exit condition: "No changes" → approve. Convergence YES — start and
  loop-back both enter review_the_submission → merge gateway before it.
Step 1 — Actions:
  - "Review the submission" → review_the_submission
  - "Send it back to the author for revision" → send_it_back_to_the_author_for_revision
  - "Approve the request" → approve_the_request
Step 2 — Flow:
  gateway_xor_0 (merge): accepts start + loop-back → review_the_submission
  review_the_submission → gateway_xor_1
  send_it_back_to_the_author_for_revision: gateway_xor_1 → gateway_xor_0 (back-edge)
  approve_the_request: gateway_xor_1 → (end)
Step 3 — Gateways:
  gateway_xor_0 (merge): incoming start + send_it_back (loop entry)
  gateway_xor_1 (split): Changes needed → send_it_back (continue loop)
                         No changes → approve_the_request (exit loop)
</reasoning>

```yaml
file_index: 888888888
actions:
  - id: review_the_submission
    name: Review the submission
    predecessors:
      - gateway_xor_0
    successors:
      - gateway_xor_1
    postconditions:
      - review_the_submission_done
  - id: send_it_back_to_the_author_for_revision
    name: Send it back to the author for revision
    predecessors:
      - gateway_xor_1
    successors:
      - gateway_xor_0
    postconditions:
      - send_it_back_to_the_author_for_revision_done
  - id: approve_the_request
    name: Approve the request
    predecessors:
      - gateway_xor_1
    successors: []
    postconditions:
      - approve_the_request_done
gateways:
  - id: gateway_xor_0
    type: exclusive
    role: merge
    incoming_from:
      - start
      - send_it_back_to_the_author_for_revision
    branches:
      - next: review_the_submission
  - id: gateway_xor_1
    type: exclusive
    role: split
    incoming_from:
      - review_the_submission
    branches:
      - next: send_it_back_to_the_author_for_revision
        condition: Changes needed
      - next: approve_the_request
        condition: No changes
```"""

#example 5 parallel fan-out without a split gateway (artificial)
#"also" / "and" signals without a synchronization requirement
#kept because the merge-based examples above do not show multi-successor fan-out
_EX5_PROCEDURE = (
    "After receiving the invoice, the clerk records it in the ledger. "
    "The clerk also files the invoice in the archive. "
    "Both activities complete the process."
)

_EX5_OUTPUT = """<reasoning>
Step 0 — Signals: no loop. Fan-out YES — "records it" AND "also files it",
  both happen unconditionally after receive_the_invoice. No convergence.
Step 1 — Actions:
  - "Receive the invoice" → receive_the_invoice
  - "Record it in the ledger" → record_it_in_the_ledger
  - "File the invoice in the archive" → file_the_invoice_in_the_archive
Step 2 — Flow:
  receive_the_invoice: start → [record_it_in_the_ledger, file_the_invoice_in_the_archive]
  both successors terminate independently.
Step 3 — Gateways: none. Unconditional fan-out uses direct multi-successor edges.
</reasoning>

```yaml
file_index: 777777777
actions:
  - id: receive_the_invoice
    name: Receive the invoice
    predecessors:
      - start
    successors:
      - record_it_in_the_ledger
      - file_the_invoice_in_the_archive
    postconditions:
      - receive_the_invoice_done
  - id: record_it_in_the_ledger
    name: Record it in the ledger
    predecessors:
      - receive_the_invoice
    successors: []
    postconditions:
      - record_it_in_the_ledger_done
  - id: file_the_invoice_in_the_archive
    name: File the invoice in the archive
    predecessors:
      - receive_the_invoice
    successors: []
    postconditions:
      - file_the_invoice_in_the_archive_done
gateways: []
```"""

#those i picked manually such they innclude examples or OR gateways, AND gatways and liear as well
#EX4 covers loops with exit conditions, EX5 covers parallel fan-out without split gateway
#dropped the synthetic XOR-merge example (EX4 in previous version) since EX3 already
#shows the merge-before-convergence pattern and we have many real PAGED merges in RAG
FEW_SHOT_EXAMPLES = [
    (1310881958, _EX1_PROCEDURE, _EX1_OUTPUT),
    (862270781, _EX2_PROCEDURE, _EX2_OUTPUT),
    (1735666188, _EX3_PROCEDURE, _EX3_OUTPUT),
    (888888888, _EX4_PROCEDURE, _EX4_OUTPUT),
    (777777777, _EX5_PROCEDURE, _EX5_OUTPUT),
]
