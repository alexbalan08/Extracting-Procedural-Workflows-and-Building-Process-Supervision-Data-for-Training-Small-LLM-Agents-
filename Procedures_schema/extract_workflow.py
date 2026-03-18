import json
import re
from pathlib import Path

from graph_builder import build_graph, build_rid_to_id, extract_actions, extract_gateways
from path_enumeration import enumerate_paths, build_execution_states

project_root = Path(__file__).parent.parent
processed_dir = project_root / 'Data' / 'Processed'


def clean_action_name(name: str) -> str:
    """Remove artifacts from BPMN node names (trailing punctuation, leading numbering)."""
    # strip trailing semicolons, dots, spaces
    name = re.sub(r'[\s;.]+$', '', name)
    # strip leading numbering like ".1 " or "1. "
    name = re.sub(r'^\.?\d+\.?\s+', '', name)
    return name.strip()


def make_id(name: str) -> str:
    """Convert cleaned action name to snake_case ID."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def clean_workflow(workflow: dict) -> dict:
    """Clean action names and regenerate IDs/postconditions consistently."""
    actions = workflow.get("actions", [])

    # build mapping: old_id → new_id
    id_map = {}
    for action in actions:
        old_id = action["id"]
        cleaned_name = clean_action_name(action["name"])
        new_id = make_id(cleaned_name)
        id_map[old_id] = new_id
        # update action fields
        action["name"] = cleaned_name
        action["id"] = new_id
        action["postconditions"] = [f"{new_id}_done"]

    # only remap if any IDs actually changed
    if all(k == v for k, v in id_map.items()):
        return workflow

    def remap(val):
        if isinstance(val, str):
            return id_map.get(val, val)
        if isinstance(val, list):
            return [remap(v) for v in val]
        return val

    # update predecessors/successors in actions
    for action in actions:
        action["predecessors"] = remap(action["predecessors"])
        action["successors"] = remap(action["successors"])

    # update gateway references
    for gw in workflow.get("gateways", []):
        gw["incoming_from"] = remap(gw["incoming_from"])
        for branch in gw.get("branches", []):
            if branch.get("next"):
                branch["next"] = remap(branch["next"])

    # update execution states
    for state in workflow.get("execution_states", []):
        state["completed_actions"] = remap(state["completed_actions"])
        state["available_next"] = remap(state.get("available_next", []))

    return workflow


def extract_workflow(record):
    """Main method where we basically build the graph with the nodes and edges we extracted"""
    nodes, outgoing, incoming = build_graph(record)

    rid_to_id = build_rid_to_id(nodes)

    actions = extract_actions(nodes, outgoing, incoming, rid_to_id)
    gateways = extract_gateways(nodes, outgoing, incoming, rid_to_id)

    #find all start nodes and enumerate execution paths
    start_rids = [rid for rid, n in nodes.items() if n['type'] == 'StartNode']
    unique_paths = enumerate_paths(nodes, outgoing, incoming, rid_to_id, start_rids)
    execution_states = build_execution_states(unique_paths)

    raw_workflow = {
        "actions": actions,
        "gateways": gateways,
        "execution_states": execution_states,
    }

    # clean artifacts from BPMN node names (trailing ;;, leading .1, etc.)
    cleaned = clean_workflow(raw_workflow)

    return {
        "file_index": record['file_index'],
        "procedure_text": record['paragraph'],
        "workflow": cleaned,
    }


#for the final extraction on all dataset, then i will split again into test/train
if __name__ == '__main__':
    input_path = processed_dir / 'merged_dataset.json'
    output_path = processed_dir / 'extracted_workflows.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)


    results = []

    for i, record in enumerate(data):
        workflow = extract_workflow(record)
        results.append(workflow)

        # Print progress every 10 records
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total} records")

    print(f"Writing {total} workflows to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done!")
    
