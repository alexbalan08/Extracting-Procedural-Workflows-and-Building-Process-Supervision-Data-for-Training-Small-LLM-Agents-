

#All few shot examples are taken directly from extracted_train.json ground truth and i picked them carefully so the
#cover all possible ways of gateways or even sequential
#we have the RAG implemented anyways if more context is needed


#!!the prompts were refined using Claude Code!!


SYSTEM_PROMPT = """You are an expert procedural workflow analyst. Given a natural-language procedure description, extract a fully structured workflow. It will be used to train a PRM and agent for planning tasks.

Output two parts in order:
1. A <reasoning> block where you trace the procedure step by step.
2. A ```json block containing the workflow object.

─── WORKFLOW SCHEMA ───────────────────────────────────────────────────────────

actions       : list of action objects, one per step/activity:
  id          : str  – snake_case, unique (duplicate names get _2, _3 suffix)
  name        : str  – EXACT wording from the procedure text (do not paraphrase or rephrase)
  predecessors: list[str] – action IDs (or "start") that immediately precede this action
  successors  : list[str] – action IDs or gateway IDs that immediately follow
  postconditions: ["{id}_done"]

gateways      : list of gateway objects (only when the flow branches or merges):
  id          : str  – "gateway_{type}_{index}" (e.g. "gateway_xor_3")
  type        : "exclusive" | "parallel" | "inclusive"
  role        : "split" | "merge" | "join_split" | "pass_through"
  incoming_from: list[str] – action/gateway IDs or "start" feeding into this gateway
  branches    : list of { next: str|null, condition?: str }
               (next is null when a branch leads directly to process end)
               condition must be SHORT (1-4 words) and use wording from the procedure text.
               Do NOT expand into full sentences — write "Yes"/"No", "Approved"/"Rejected", etc.

─── MERGE GATEWAY RULE (critical — most common mistake) ───────────────────────
• Whenever two or more branches converge back to the same action, you MUST add
  a merge gateway (role: "merge") immediately before that action.
• Signal: an action that can be reached from more than one branch → it needs a
  merge gateway as its sole predecessor; the branches point to the gateway, not
  directly to the action.
• Do NOT connect multiple branch endpoints directly to the same action — always
  route through a merge gateway first.
• This applies to XOR, AND, and OR merges equally.

─── ID CONVENTIONS ────────────────────────────────────────────────────────────
• Action IDs: EXACT action name → lowercase → replace spaces/non-alphanumeric with underscore → strip edges
• Gateway IDs: "gateway_xor_<N>", "gateway_and_<N>", "gateway_or_<N>" (0-based, document order)
• Use "start" as predecessor for the very first action(s) — never create a "start" action node
• Do NOT create "end", "end_process", or "terminate" actions — model termination with empty successors list
• Do NOT split a single mentioned activity into multiple actions

─── OUTPUT FORMAT ─────────────────────────────────────────────────────────────

<reasoning>
Step 1 — Identify actions: list each distinct activity and its ID.
Step 2 — Trace the flow: identify predecessors and successors for each action.
Step 3 — Identify gateways: note any branching (XOR/AND/OR) or merging points.
</reasoning>

```json
{
  "file_index": <value from the prompt>,
  "actions": [...],
  "gateways": [...]
}
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
Step 1 — Identify actions:
  - "Create Recruitment Vacancy in NGA.net" → id: create_recruitment_vacancy_in_nganet
  - "Manage External Advertising" → id: manage_external_advertising

Step 2 — Trace the flow:
  create_recruitment_vacancy_in_nganet: start → manage_external_advertising
  manage_external_advertising: create_recruitment_vacancy_in_nganet → (end)

Step 3 — Identify gateways:
  None — purely sequential.
</reasoning>

```json
{
  "file_index": 1310881958,
  "actions": [
    {
      "id": "create_recruitment_vacancy_in_nganet",
      "name": "Create Recruitment Vacancy in NGA.net",

      "predecessors": ["start"],
      "successors": ["manage_external_advertising"],
      "postconditions": ["create_recruitment_vacancy_in_nganet_done"]
    },
    {
      "id": "manage_external_advertising",
      "name": "Manage External Advertising",

      "predecessors": ["create_recruitment_vacancy_in_nganet"],
      "successors": [],
      "postconditions": ["manage_external_advertising_done"]
    }
  ],
  "gateways": []
}
```"""

#example with XOR gateway with early termination file_index 862270781
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
Step 1 — Identify actions:
  - "Customer Enters Store" → id: customer_enters_store
  - "Customer Decides What They Want" → id: customer_decides_what_they_want
  - "Fill Out Customer Invoice" → id: fill_out_customer_invoice

Step 2 — Trace the flow:
  customer_enters_store: start → customer_decides_what_they_want
  customer_decides_what_they_want: customer_enters_store → gateway_xor_2
  fill_out_customer_invoice: gateway_xor_2 → (end)

Step 3 — Identify gateways:
  gateway_xor_2: exclusive split after customer_decides_what_they_want
    - "Custom Order" → fill_out_customer_invoice
    - "In-Store Purchase" → null (process ends immediately)
</reasoning>

```json
{
  "file_index": 862270781,
  "actions": [
    {
      "id": "customer_enters_store",
      "name": "Customer Enters Store",

      "predecessors": ["start"],
      "successors": ["customer_decides_what_they_want"],
      "postconditions": ["customer_enters_store_done"]
    },
    {
      "id": "customer_decides_what_they_want",
      "name": "Customer Decides What They Want",

      "predecessors": ["customer_enters_store"],
      "successors": ["gateway_xor_2"],
      "postconditions": ["customer_decides_what_they_want_done"]
    },
    {
      "id": "fill_out_customer_invoice",
      "name": "Fill Out Customer Invoice",

      "predecessors": ["gateway_xor_2"],
      "successors": [],
      "postconditions": ["fill_out_customer_invoice_done"]
    }
  ],
  "gateways": [
    {
      "id": "gateway_xor_2",
      "type": "exclusive",
      "role": "split",
      "incoming_from": ["customer_decides_what_they_want"],
      "branches": [
        {"next": "fill_out_customer_invoice", "condition": "Custom Order"},
        {"next": null, "condition": "In-Store Purchase"}
      ]
    }
  ]
}
```"""

#example 3 with OR gateway — split and merge (file_index 1735666188)
_EX3_PROCEDURE = (
    "To start, we need to check the application for completeness. After that, we should sort the applications. "
    "During the step of checking the application for completeness, it is important to consider all applications "
    "in the batch. When sorting the applications, we assume that two piles will be created: one for complete "
    "applications and one for incomplete applications. If an application is complete, we proceed to process it. "
    "However, if an application is incomplete, we discard it. This process is repeated for all complete and "
    "incomplete applications until the end."
)

_EX3_OUTPUT = """<reasoning>
Step 1 — Identify actions:
  - "Check Application for Completeness" → id: check_application_for_completeness
  - "Sort Applications" → id: sort_applications
  - "Process Complete Applications" → id: process_complete_applications
  - "Discard Applications" → id: discard_applications

Step 2 — Trace the flow:
  check_application_for_completeness: start → sort_applications
  sort_applications → gateway_or_3 (inclusive split)
  process_complete_applications: gateway_or_3 → gateway_or_6
  discard_applications: gateway_or_3 → gateway_or_6
  gateway_or_6 merges both branches → (end)

Step 3 — Identify gateways:
  gateway_or_3: inclusive split after sort_applications
    - "Complete" → process_complete_applications
    - "Incomplete" → discard_applications
  gateway_or_6: inclusive merge before end
    - incoming from both process_complete_applications and discard_applications
</reasoning>

```json
{
  "file_index": 1735666188,
  "actions": [
    {
      "id": "check_application_for_completeness",
      "name": "Check Application for Completeness",

      "predecessors": ["start"],
      "successors": ["sort_applications"],
      "postconditions": ["check_application_for_completeness_done"]
    },
    {
      "id": "sort_applications",
      "name": "Sort Applications",

      "predecessors": ["check_application_for_completeness"],
      "successors": ["gateway_or_3"],
      "postconditions": ["sort_applications_done"]
    },
    {
      "id": "process_complete_applications",
      "name": "Process Complete Applications",

      "predecessors": ["gateway_or_3"],
      "successors": ["gateway_or_6"],
      "postconditions": ["process_complete_applications_done"]
    },
    {
      "id": "discard_applications",
      "name": "Discard Applications",

      "predecessors": ["gateway_or_3"],
      "successors": ["gateway_or_6"],
      "postconditions": ["discard_applications_done"]
    }
  ],
  "gateways": [
    {
      "id": "gateway_or_3",
      "type": "inclusive",
      "role": "split",
      "incoming_from": ["sort_applications"],
      "branches": [
        {"next": "process_complete_applications", "condition": "Complete"},
        {"next": "discard_applications", "condition": "Incomplete"}
      ]
    },
    {
      "id": "gateway_or_6",
      "type": "inclusive",
      "role": "merge",
      "incoming_from": ["process_complete_applications", "discard_applications"],
      "branches": [
        {"next": null}
      ]
    }
  ]
}
```"""

#example 4 XOR split where BOTH branches where merge gateway is required
#this is the most commonly missed pattern
_EX4_PROCEDURE = (
    "To process the loan request, first assess the risk. "
    "If the risk is low, approve the loan. "
    "If the risk is high, decline the loan. "
    "After the decision is made, record the outcome in the system and the process ends."
)#see after the both if brnahces we need to merge and record the outcome
#this will be to mitigate the model on getting more FN on merges gateways
#so i will push a fourth procedure as example
_EX4_OUTPUT = """<reasoning>
Step 1 — Identify actions:
  - "Assess the risk" → id: assess_the_risk
  - "Approve the loan" → id: approve_the_loan
  - "Decline the loan" → id: decline_the_loan
  - "Record the outcome in the system" → id: record_the_outcome_in_the_system

Step 2 — Trace the flow:
  assess_the_risk: start → gateway_xor_1
  approve_the_loan: gateway_xor_1 → gateway_xor_3
  decline_the_loan: gateway_xor_1 → gateway_xor_3
  record_the_outcome_in_the_system: gateway_xor_3 → (end)

Step 3 — Identify gateways:
  gateway_xor_1: exclusive split after assess_the_risk
    - "Risk is low" → approve_the_loan
    - "Risk is high" → decline_the_loan
  gateway_xor_3: exclusive merge — both approve and decline converge here before record_outcome
    IMPORTANT: "record the outcome" is reachable from both branches, so a merge gateway
    is required. Do NOT connect approve/decline directly to record_outcome.
</reasoning>

```json
{
  "file_index": 999999999,
  "actions": [
    {
      "id": "assess_the_risk",
      "name": "Assess the risk",

      "predecessors": ["start"],
      "successors": ["gateway_xor_1"],
      "postconditions": ["assess_the_risk_done"]
    },
    {
      "id": "approve_the_loan",
      "name": "Approve the loan",

      "predecessors": ["gateway_xor_1"],
      "successors": ["gateway_xor_3"],
      "postconditions": ["approve_the_loan_done"]
    },
    {
      "id": "decline_the_loan",
      "name": "Decline the loan",

      "predecessors": ["gateway_xor_1"],
      "successors": ["gateway_xor_3"],
      "postconditions": ["decline_the_loan_done"]
    },
    {
      "id": "record_the_outcome_in_the_system",
      "name": "Record the outcome in the system",

      "predecessors": ["gateway_xor_3"],
      "successors": [],
      "postconditions": ["record_the_outcome_in_the_system_done"]
    }
  ],
  "gateways": [
    {
      "id": "gateway_xor_1",
      "type": "exclusive",
      "role": "split",
      "incoming_from": ["assess_the_risk"],
      "branches": [
        {"next": "approve_the_loan", "condition": "Risk is low"},
        {"next": "decline_the_loan", "condition": "Risk is high"}
      ]
    },
    {
      "id": "gateway_xor_3",
      "type": "exclusive",
      "role": "merge",
      "incoming_from": ["approve_the_loan", "decline_the_loan"],
      "branches": [
        {"next": "record_the_outcome_in_the_system"}
      ]
    }
  ]
}
```"""

#those i picked manually such they innclude examples or OR gateways, AND gatways and liear as well
FEW_SHOT_EXAMPLES = [
    (1310881958, _EX1_PROCEDURE, _EX1_OUTPUT),
    (862270781, _EX2_PROCEDURE, _EX2_OUTPUT),
    (1735666188, _EX3_PROCEDURE, _EX3_OUTPUT),
    (999999999, _EX4_PROCEDURE, _EX4_OUTPUT),
]
