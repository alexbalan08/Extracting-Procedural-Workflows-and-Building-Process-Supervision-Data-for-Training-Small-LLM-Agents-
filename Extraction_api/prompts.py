"""
System prompt and 3-shot examples for GPT-4o workflow extraction.
All few-shot examples are taken directly from extracted_train.json ground truth.
"""

SYSTEM_PROMPT = """You are an expert procedural workflow analyst. Given a natural-language procedure description, extract a fully structured workflow.

Output two parts in order:
1. A <reasoning> block where you trace the procedure step by step.
2. A ```json block containing the workflow object.

─── WORKFLOW SCHEMA ───────────────────────────────────────────────────────────

actions       : list of action objects, one per step/activity:
  id          : str  – snake_case, unique (duplicate names get _2, _3 suffix)
  name        : str  – EXACT wording from the procedure text (do not paraphrase or rephrase)
  actor       : str|null – which actor performs it
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

execution_states : list of state snapshots covering every reachable path:
  completed_actions: list[str] – ordered list of action IDs completed so far
  conditions_met   : list[str] – edge conditions satisfied on this path (only add AFTER gateway is traversed)
  available_next   : list[str] – action IDs that can execute next
  can_terminate    : bool      – present and true only when the process may end here

─── ID CONVENTIONS ────────────────────────────────────────────────────────────
• Action IDs: EXACT action name → lowercase → replace spaces/non-alphanumeric with underscore → strip edges
• Gateway IDs: "gateway_xor_<N>", "gateway_and_<N>", "gateway_or_<N>" (0-based, document order)
• Use "start" as predecessor for the very first action(s) — never create a "start" action node
• Do NOT create "end", "end_process", or "terminate" actions — use can_terminate=true in execution_states
• Do NOT split a single mentioned activity into multiple actions

─── EXECUTION STATES RULES ──────────────────────────────────────────────────
• Each XOR branch produces a SEPARATE chain of states
• For AND splits: list ALL parallel actions in available_next simultaneously
• conditions_met is cumulative along a path — empty before the gateway that tests it
• The initial state (completed_actions=[]) must always have conditions_met=[]

─── OUTPUT FORMAT ─────────────────────────────────────────────────────────────

<reasoning>
Step 1 — Identify actions: list each distinct activity, its actor, and ID.
Step 2 — Trace the flow: identify predecessors and successors for each action.
Step 3 — Identify gateways: note any branching (XOR/AND/OR) or merging points.
Step 4 — Enumerate execution states: trace every reachable path from start to termination.
</reasoning>

```json
{
  "file_index": <value from the prompt>,
  "actions": [...],
  "gateways": [...],
  "execution_states": [...]
}
```
"""

# ── Few-shot examples ─────────────────────────────────────────────────────────
# Each tuple: (file_index, procedure_text, expected_output)
# All examples taken directly from extracted_train.json ground truth.

# Example 1: simple linear procedure, no gateways (file_index 1310881958)
_EX1_PROCEDURE = (
    "For the HR Representative, the first step is to create a recruitment vacancy in NGA.net. "
    "After creating the vacancy, the next step is to manage external advertising. "
    "Once the external advertising is managed, the process comes to an end."
)

_EX1_OUTPUT = """<reasoning>
Step 1 — Identify actions:
  - "Create Recruitment Vacancy in NGA.net" → id: create_recruitment_vacancy_in_nganet, actor: HR Representative
  - "Manage External Advertising" → id: manage_external_advertising, actor: HR Representative

Step 2 — Trace the flow:
  create_recruitment_vacancy_in_nganet: start → manage_external_advertising
  manage_external_advertising: create_recruitment_vacancy_in_nganet → (end)

Step 3 — Identify gateways:
  None — purely sequential.

Step 4 — Enumerate execution states:
  State 0: nothing done → next: [create_recruitment_vacancy_in_nganet]
  State 1: [create_recruitment_vacancy_in_nganet] done → next: [manage_external_advertising]
  State 2: both done → can_terminate = true
</reasoning>

```json
{
  "file_index": 1310881958,
  "actions": [
    {
      "id": "create_recruitment_vacancy_in_nganet",
      "name": "Create Recruitment Vacancy in NGA.net",
      "actor": "HR Representative",
      "predecessors": ["start"],
      "successors": ["manage_external_advertising"],
      "postconditions": ["create_recruitment_vacancy_in_nganet_done"]
    },
    {
      "id": "manage_external_advertising",
      "name": "Manage External Advertising",
      "actor": "HR Representative",
      "predecessors": ["create_recruitment_vacancy_in_nganet"],
      "successors": [],
      "postconditions": ["manage_external_advertising_done"]
    }
  ],
  "gateways": [],
  "execution_states": [
    {"completed_actions": [], "conditions_met": [], "available_next": ["create_recruitment_vacancy_in_nganet"]},
    {"completed_actions": ["create_recruitment_vacancy_in_nganet"], "conditions_met": [], "available_next": ["manage_external_advertising"]},
    {"completed_actions": ["create_recruitment_vacancy_in_nganet", "manage_external_advertising"], "conditions_met": [], "available_next": [], "can_terminate": true}
  ]
}
```"""

# Example 2: exclusive (XOR) gateway with early termination (file_index 862270781)
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
  - "Customer Enters Store" → id: customer_enters_store, actor: Grenoble
  - "Customer Decides What They Want" → id: customer_decides_what_they_want, actor: Grenoble
  - "Fill Out Customer Invoice" → id: fill_out_customer_invoice, actor: Grenoble

Step 2 — Trace the flow:
  customer_enters_store: start → customer_decides_what_they_want
  customer_decides_what_they_want: customer_enters_store → gateway_xor_2
  fill_out_customer_invoice: gateway_xor_2 → (end)

Step 3 — Identify gateways:
  gateway_xor_2: exclusive split after customer_decides_what_they_want
    - "Custom Order" → fill_out_customer_invoice
    - "In-Store Purchase" → null (process ends immediately)

Step 4 — Enumerate execution states:
  Path A (Custom Order): enter → decide → fill invoice → terminate
  Path B (In-Store Purchase): enter → decide → terminate immediately
</reasoning>

```json
{
  "file_index": 862270781,
  "actions": [
    {
      "id": "customer_enters_store",
      "name": "Customer Enters Store",
      "actor": "Grenoble",
      "predecessors": ["start"],
      "successors": ["customer_decides_what_they_want"],
      "postconditions": ["customer_enters_store_done"]
    },
    {
      "id": "customer_decides_what_they_want",
      "name": "Customer Decides What They Want",
      "actor": "Grenoble",
      "predecessors": ["customer_enters_store"],
      "successors": ["gateway_xor_2"],
      "postconditions": ["customer_decides_what_they_want_done"]
    },
    {
      "id": "fill_out_customer_invoice",
      "name": "Fill Out Customer Invoice",
      "actor": "Grenoble",
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
      ],
      "actor": "Grenoble"
    }
  ],
  "execution_states": [
    {"completed_actions": [], "conditions_met": [], "available_next": ["customer_enters_store"]},
    {"completed_actions": ["customer_enters_store"], "conditions_met": [], "available_next": ["customer_decides_what_they_want"]},
    {"completed_actions": ["customer_enters_store", "customer_decides_what_they_want"], "conditions_met": ["Custom Order"], "available_next": ["fill_out_customer_invoice"]},
    {"completed_actions": ["customer_enters_store", "customer_decides_what_they_want", "fill_out_customer_invoice"], "conditions_met": ["Custom Order"], "available_next": [], "can_terminate": true},
    {"completed_actions": ["customer_enters_store", "customer_decides_what_they_want"], "conditions_met": ["In-Store Purchase"], "available_next": [], "can_terminate": true}
  ]
}
```"""

# Example 3: inclusive (OR) gateway — split and merge (file_index 1735666188)
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
  - "Check Application for Completeness" → id: check_application_for_completeness, actor: null
  - "Sort Applications" → id: sort_applications, actor: null
  - "Process Complete Applications" → id: process_complete_applications, actor: null
  - "Discard Applications" → id: discard_applications, actor: null

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

Step 4 — Enumerate execution states:
  Path A (Complete only): check → sort → process → terminate
  Path B (Incomplete only): check → sort → discard → terminate
  Path C (Both): check → sort → process → discard → terminate
  Note: inclusive (OR) gateway means one, the other, or BOTH branches can be taken.
</reasoning>

```json
{
  "file_index": 1735666188,
  "actions": [
    {
      "id": "check_application_for_completeness",
      "name": "Check Application for Completeness",
      "actor": null,
      "predecessors": ["start"],
      "successors": ["sort_applications"],
      "postconditions": ["check_application_for_completeness_done"]
    },
    {
      "id": "sort_applications",
      "name": "Sort Applications",
      "actor": null,
      "predecessors": ["check_application_for_completeness"],
      "successors": ["gateway_or_3"],
      "postconditions": ["sort_applications_done"]
    },
    {
      "id": "process_complete_applications",
      "name": "Process Complete Applications",
      "actor": null,
      "predecessors": ["gateway_or_3"],
      "successors": ["gateway_or_6"],
      "postconditions": ["process_complete_applications_done"]
    },
    {
      "id": "discard_applications",
      "name": "Discard Applications",
      "actor": null,
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
  ],
  "execution_states": [
    {"completed_actions": [], "conditions_met": [], "available_next": ["check_application_for_completeness"]},
    {"completed_actions": ["check_application_for_completeness"], "conditions_met": [], "available_next": ["sort_applications"]},
    {"completed_actions": ["check_application_for_completeness", "sort_applications"], "conditions_met": ["Complete"], "available_next": ["process_complete_applications"]},
    {"completed_actions": ["check_application_for_completeness", "sort_applications", "process_complete_applications"], "conditions_met": ["Complete"], "available_next": [], "can_terminate": true},
    {"completed_actions": ["check_application_for_completeness", "sort_applications"], "conditions_met": ["Incomplete"], "available_next": ["discard_applications"]},
    {"completed_actions": ["check_application_for_completeness", "sort_applications", "discard_applications"], "conditions_met": ["Incomplete"], "available_next": [], "can_terminate": true},
    {"completed_actions": ["check_application_for_completeness", "sort_applications", "process_complete_applications"], "conditions_met": ["Complete", "Incomplete"], "available_next": ["discard_applications"]},
    {"completed_actions": ["check_application_for_completeness", "sort_applications", "process_complete_applications", "discard_applications"], "conditions_met": ["Complete", "Incomplete"], "available_next": [], "can_terminate": true}
  ]
}
```"""

# exported list: (file_index, procedure_text, expected_output)
FEW_SHOT_EXAMPLES = [
    (1310881958, _EX1_PROCEDURE, _EX1_OUTPUT),
    (862270781, _EX2_PROCEDURE, _EX2_OUTPUT),
    (1735666188, _EX3_PROCEDURE, _EX3_OUTPUT),
]
