"""
Converts extracted workflow JSON to the graph format expected by path_enumeration.py,
then runs the deterministic path enumeration to build execution states for PRM training.

The extracted workflow uses: actions (id, name, successors) + gateways (id, type, branches).
The path_enumeration uses:  nodes dict + outgoing/incoming adjacency lists.
This module bridges the two.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Procedures_schema.path_enumeration import enumerate_paths, build_execution_states


# map extracted type strings back to the internal type codes used by path_enumeration
_GATEWAY_TYPE_MAP = {
    'exclusive': 'XOR',
    'parallel': 'AND',
    'inclusive': 'OR',
}

_START_RID = '__start__'


def workflow_to_graph(workflow):
    """
    Convert extracted workflow JSON to the (nodes, outgoing, incoming, rid_to_id, start_rids)
    tuple that enumerate_paths expects.

    Key decisions:
    - Actions become Activity nodes; rid == action id (strings work as rids just fine)
    - Gateways become XOR/AND/OR nodes
    - A virtual StartNode feeds into all actions that list "start" as predecessor
    - Gateway branches with next=null get a virtual unnamed EndNode (signals path termination)
    """
    actions = workflow.get('actions', [])
    gateways = workflow.get('gateways', [])

    nodes = {}
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    rid_to_id = {}

    # virtual start node — no text, just a dispatch point
    nodes[_START_RID] = {'type': 'StartNode', 'NodeText': '', 'agent': ''}

    # action nodes
    for a in actions:
        nodes[a['id']] = {
            'type': 'Activity',
            'NodeText': a.get('name', ''),
            'agent': a.get('actor') or '',
        }
        rid_to_id[a['id']] = a['id']

    # gateway nodes
    for g in gateways:
        nodes[g['id']] = {
            'type': _GATEWAY_TYPE_MAP.get(g['type'], 'XOR'),
            'NodeText': '',
            'agent': '',
        }

    # edges: virtual start → first actions
    for a in actions:
        if 'start' in a.get('predecessors', []):
            outgoing[_START_RID].append((a['id'], ''))
            incoming[a['id']].append((_START_RID, ''))

    # edges: action → successors (no condition on these edges — conditions live on gateway branches)
    for a in actions:
        for s in a.get('successors', []):
            if s in nodes:
                outgoing[a['id']].append((s, ''))
                incoming[s].append((a['id'], ''))

    # edges: gateway → branches
    end_counter = 0
    for g in gateways:
        for b in g.get('branches', []):
            nxt = b.get('next')
            cond = b.get('condition', '') or ''
            if nxt and nxt in nodes:
                outgoing[g['id']].append((nxt, cond))
                incoming[nxt].append((g['id'], cond))
            elif nxt is None:
                # null branch = process terminates here; create a nameless EndNode
                end_rid = f'__end_{end_counter}__'
                end_counter += 1
                nodes[end_rid] = {'type': 'EndNode', 'NodeText': '', 'agent': ''}
                outgoing[g['id']].append((end_rid, cond))
                incoming[end_rid].append((g['id'], cond))

    return nodes, outgoing, incoming, rid_to_id, [_START_RID]


def build_traces(workflow):
    """
    Build execution states for an extracted workflow using deterministic path enumeration.
    Returns a list of execution state dicts (completed_actions, conditions_met, available_next).
    Returns [] if workflow is None or produces no paths.
    """
    if not workflow:
        return []
    try:
        nodes, outgoing, incoming, rid_to_id, start_rids = workflow_to_graph(workflow)
        paths = enumerate_paths(nodes, outgoing, incoming, rid_to_id, start_rids)
        return build_execution_states(paths)
    except Exception as e:
        #don't crash the whole pipeline if trace building fails for one procedure
        print(f"  Warning: trace building failed — {e}")
        return []
