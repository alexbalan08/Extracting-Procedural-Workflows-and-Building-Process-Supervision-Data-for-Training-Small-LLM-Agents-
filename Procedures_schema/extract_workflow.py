import json
from pathlib import Path

from graph_builder import build_graph, build_rid_to_id, extract_actions, extract_gateways
from path_enumeration import enumerate_paths, build_execution_states

project_root = Path(__file__).parent.parent
processed_dir = project_root / 'Data' / 'Processed'
output_dir = Path(__file__).parent


def extract_workflow(record):
    """Main method where we basically build the graph with the nodes and edges we extracted"""
    nodes, outgoing, incoming = build_graph(record)

    rid_to_id = build_rid_to_id(nodes)

    #we retrieve all unique actors
    actors = list(dict.fromkeys(n['agent'] for n in record['step_nodes'] if n['agent'].strip()))

    actions = extract_actions(nodes, outgoing, incoming, rid_to_id)
    gateways = extract_gateways(nodes, outgoing, incoming, rid_to_id)

    #find all start nodes and enumerate execution paths
    start_rids = [rid for rid, n in nodes.items() if n['type'] == 'StartNode']
    unique_paths = enumerate_paths(nodes, outgoing, incoming, rid_to_id, start_rids)
    execution_states = build_execution_states(unique_paths)

    workflow = {
        "file_index": record['file_index'],
        "procedure_text": record['paragraph'],
        "workflow": {
            "actors": actors,
            "actions": actions,
            "gateways": gateways,
            "execution_states": execution_states,
        }
    }

    return workflow


#this is for testing purpose to validate manually
with open(processed_dir / 'merged_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

samples = [extract_workflow(data[i]) for i in range(20)]

output_path = output_dir / 'workflow_samples.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

for i, s in enumerate(samples):
    w = s['workflow']
    print(f"Record {i}: {len(w['actions'])} actions, {len(w['gateways'])} gateways, actors: {w['actors']}")

print(f"\nSaved to {output_path}")
