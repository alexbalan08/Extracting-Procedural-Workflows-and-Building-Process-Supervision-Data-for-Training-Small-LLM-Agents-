import json
from pathlib import Path
from collections import defaultdict

from graph_builder import (
    build_graph, build_rid_to_id, make_gateway_id,
    ACTIONABLE_TYPES, GATEWAY_TYPE_MAP
)
from path_enumeration import enumerate_paths, build_execution_states

project_root = Path(__file__).parent.parent
processed_dir = project_root / 'Data' / 'Processed'
output_dir = Path(__file__).parent


def normalize(text):
    #i will normalize the original actions to match the normalized extracted ones to avoid fuzzy matching
    return ' '.join(text.strip().lower().split()).rstrip(';')


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_metrics(ground_truth, extracted):
    if not ground_truth and not extracted:
        return 1.0, 1.0, 1.0
    matched = ground_truth & extracted
    precision = len(matched) / len(extracted) if extracted else 0.0
    recall = len(matched) / len(ground_truth) if ground_truth else 0.0
    return precision, recall, f1_score(precision, recall)


def _build_gt_from_raw(raw_record):
    """Rebuild ground truth IDs and relations from raw BPMN data
    using the same ID generation logic as graph_builder."""
    nodes, outgoing, incoming = build_graph(raw_record)
    rid_to_id = build_rid_to_id(nodes)

    #build gateway rid -> gateway_id mapping
    rid_to_gateway_id = {}
    for rid, node in nodes.items():
        if node['type'] in ('XOR', 'AND', 'OR'):
            rid_to_gateway_id[rid] = make_gateway_id(nodes, rid, node)

    
    def schema_id(rid):
        if rid in rid_to_id:
            return rid_to_id[rid]
        if rid in rid_to_gateway_id:
            return rid_to_gateway_id[rid]
        node = nodes.get(rid)
        if node and node['type'] == 'StartNode' and not node['NodeText'].strip():
            return "start"
        return None

    return nodes, outgoing, incoming, rid_to_id, rid_to_gateway_id, schema_id


def validate_record(raw_record, extracted_workflow):

    w = extracted_workflow['workflow']
    metrics = {}

    nodes, outgoing, incoming, rid_to_id, rid_to_gateway_id, schema_id = _build_gt_from_raw(raw_record)

    #ACTION EXTRACTION: F1 on normalized action names 
    gt_action_names = set()
    for rid, aid in rid_to_id.items():
        gt_action_names.add(normalize(nodes[rid]['NodeText']))

    ext_action_names = set()
    for a in w['actions']:
        ext_action_names.add(normalize(a['name']))

    p, r, f = set_metrics(gt_action_names, ext_action_names)
    metrics['actions'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_action_names), 'ext_count': len(ext_action_names),
        'missing': sorted(gt_action_names - ext_action_names),
        'extra': sorted(ext_action_names - gt_action_names),
    }

    #GATEWAY EXTRACTION: count, type (or, and or xor) accuracy, role accuracy
    gt_gw_types = defaultdict(int)
    gt_gw_roles = {}
    for rid, gid in rid_to_gateway_id.items():
        gtype = GATEWAY_TYPE_MAP[nodes[rid]['type']]
        gt_gw_types[gtype] += 1
        #compute role from raw in/out degree
        in_deg = len(incoming.get(rid, []))
        out_deg = len(outgoing.get(rid, []))
        if in_deg <= 1 and out_deg > 1:
            gt_gw_roles[gid] = "split"
        elif in_deg > 1 and out_deg <= 1:
            gt_gw_roles[gid] = "merge"
        elif in_deg > 1 and out_deg > 1:
            gt_gw_roles[gid] = "join_split"
        else:
            gt_gw_roles[gid] = "pass_through"

    ext_gw_types = defaultdict(int)
    ext_gw_roles = {}
    for g in w['gateways']:
        ext_gw_types[g['type']] += 1
        ext_gw_roles[g['id']] = g['role']

    gt_count = len(rid_to_gateway_id)
    ext_count = len(w['gateways'])
    all_types = set(gt_gw_types) | set(ext_gw_types)
    type_matches = sum(min(gt_gw_types[t], ext_gw_types[t]) for t in all_types)
    type_total = max(gt_count, ext_count)
    type_accuracy = type_matches / type_total if type_total > 0 else 1.0

    #role accuracy: match gateways by index order and compare roles
    gt_gw_ordered = [rid_to_gateway_id[rid] for rid in rid_to_gateway_id]
    ext_gw_ordered = [g['id'] for g in w['gateways']]
    role_matches = 0
    role_total = min(len(gt_gw_ordered), len(ext_gw_ordered))
    for i in range(role_total):
        gt_role = gt_gw_roles.get(gt_gw_ordered[i])
        ext_role = ext_gw_roles.get(ext_gw_ordered[i])
        if gt_role == ext_role:
            role_matches += 1
    role_accuracy = role_matches / role_total if role_total > 0 else 1.0

    metrics['gateways'] = {
        'count_match': gt_count == ext_count,
        'gt_count': gt_count, 'ext_count': ext_count,
        'type_accuracy': type_accuracy,
        'role_accuracy': role_accuracy,
        'gt_types': dict(gt_gw_types), 'ext_types': dict(ext_gw_types),
    }

    #EDGES
    #Action relations: (action_id, successor_id) and (predecessor_id, action_id)
    #ground truth: from SequenceFlow based on their IDs

    #gt = grownd truth 
    #ext = my extraction
    gt_action_successors = set()
    gt_action_predecessors = set()
    for rid in rid_to_id:
        action_id = rid_to_id[rid]
        #successors  direct outgoing edges from this action node
        for tgt, cond in outgoing.get(rid, []):
            tgt_sid = schema_id(tgt)
            if tgt_sid:
                gt_action_successors.add((action_id, tgt_sid))
        #predecessors  direct incoming edges to this action node
        for src, cond in incoming.get(rid, []):
            src_sid = schema_id(src)
            if src_sid:
                gt_action_predecessors.add((src_sid, action_id))

    ext_action_successors = set()
    ext_action_predecessors = set()
    for a in w['actions']:
        aid = a['id']
        for s in a['successors']:
            ext_action_successors.add((aid, s))
        for p_id in a['predecessors']:
            ext_action_predecessors.add((p_id, aid))

    p, r, f = set_metrics(gt_action_successors, ext_action_successors)
    metrics['action_successors'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_action_successors), 'ext_count': len(ext_action_successors),
        'missing': sorted(gt_action_successors - ext_action_successors)[:5],
        'extra': sorted(ext_action_successors - gt_action_successors)[:5],
    }

    p, r, f = set_metrics(gt_action_predecessors, ext_action_predecessors)
    metrics['action_predecessors'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_action_predecessors), 'ext_count': len(ext_action_predecessors),
        'missing': sorted(gt_action_predecessors - ext_action_predecessors)[:5],
        'extra': sorted(ext_action_predecessors - gt_action_predecessors)[:5],
    }

    #Gateway relations: (gateway_id, next_id) and (incoming_id, gateway_id)
    gt_gw_next = set()
    gt_gw_incoming = set()
    for rid, gid in rid_to_gateway_id.items():
        #outgoing from gateway -> branches
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue
            tgt_sid = schema_id(tgt)
            if tgt_sid:
                gt_gw_next.add((gid, tgt_sid))
            elif tgt_node['type'] == 'EndNode':
                #terminal branch: next is None
                gt_gw_next.add((gid, None))

        #incoming to gateway
        for src, cond in incoming.get(rid, []):
            src_sid = schema_id(src)
            if src_sid:
                gt_gw_incoming.add((src_sid, gid))

    ext_gw_next = set()
    ext_gw_incoming = set()
    for g in w['gateways']:
        gid = g['id']
        for branch in g['branches']:
            ext_gw_next.add((gid, branch['next']))
        for inc in g['incoming_from']:
            ext_gw_incoming.add((inc, gid))

    p, r, f = set_metrics(gt_gw_next, ext_gw_next)
    metrics['gateway_branches_next'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_gw_next), 'ext_count': len(ext_gw_next),
        'missing': sorted(gt_gw_next - ext_gw_next, key=str)[:5],
        'extra': sorted(ext_gw_next - gt_gw_next, key=str)[:5],
    }

    p, r, f = set_metrics(gt_gw_incoming, ext_gw_incoming)
    metrics['gateway_incoming'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_gw_incoming), 'ext_count': len(ext_gw_incoming),
        'missing': sorted(gt_gw_incoming - ext_gw_incoming)[:5],
        'extra': sorted(ext_gw_incoming - gt_gw_incoming)[:5],
    }

    #GATEAWAY branch tuple accuracy  (gateway_id, next_id, condition_norm) 
    gt_branch_tuples = set()
    for rid, gid in rid_to_gateway_id.items():
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue
            tgt_sid = schema_id(tgt)
            if tgt_node['type'] == 'EndNode' and tgt not in rid_to_id:
                tgt_sid = None
            cond_norm = cond.strip() if cond.strip() else None
            gt_branch_tuples.add((gid, tgt_sid, cond_norm))

    ext_branch_tuples = set()
    for g in w['gateways']:
        gid = g['id']
        for branch in g['branches']:
            cond_norm = branch.get('condition')
            ext_branch_tuples.add((gid, branch['next'], cond_norm))

    p, r, f = set_metrics(gt_branch_tuples, ext_branch_tuples)
    metrics['branch_tuples'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_branch_tuples), 'ext_count': len(ext_branch_tuples),
        'missing': sorted(gt_branch_tuples - ext_branch_tuples, key=str)[:5],
        'extra': sorted(ext_branch_tuples - gt_branch_tuples, key=str)[:5],
    }

    #number of outgoing GT branches for gateway i
    #number of extracted branches for gateway i
    gt_gw_ordered_rids = list(rid_to_gateway_id.keys())
    branch_count_matches = 0
    total_compared = min(len(gt_gw_ordered_rids), len(w['gateways']))
    for i in range(total_compared):
        gt_rid = gt_gw_ordered_rids[i]
        gt_branch_count = len(outgoing.get(gt_rid, []))
        ext_branch_count = len(w['gateways'][i]['branches'])
        if gt_branch_count == ext_branch_count:
            branch_count_matches += 1

    metrics['branch_counts'] = {
        'accuracy': branch_count_matches / total_compared if total_compared > 0 else 1.0,
        'total_compared': total_compared,
    }

    #EXECUTION STATES
    start_rids = [rid for rid, node in nodes.items() if node['type'] == 'StartNode']
    gt_unique_paths = enumerate_paths(nodes, outgoing, incoming, rid_to_id, start_rids)
    gt_execution_states = build_execution_states(gt_unique_paths)
    ext_execution_states = w.get('execution_states', [])

    def state_key(state):
        completed = tuple(state.get('completed_actions', []))
        available = tuple(sorted(set(state.get('available_next', []))))
        can_terminate = bool(state.get('can_terminate', False))
        return (completed, available, can_terminate)

    gt_state_keys = {state_key(s) for s in gt_execution_states}
    ext_state_keys = {state_key(s) for s in ext_execution_states}
    p, r, f = set_metrics(gt_state_keys, ext_state_keys)

    action_ids = {a['id'] for a in w['actions']}
    ext_completed_sets = set()
    completed_counts = defaultdict(int)
    unknown_completed = set()
    unknown_available = set()

    for state in ext_execution_states:
        completed = tuple(state.get('completed_actions', []))
        available = list(state.get('available_next', []))
        completed_counts[completed] += 1
        ext_completed_sets.add(completed)

        for aid in completed:
            if aid not in action_ids:
                unknown_completed.add(aid)
        for aid in available:
            if aid not in action_ids:
                unknown_available.add(aid)

    duplicate_completed_states = sorted(
        [list(comp) for comp, c in completed_counts.items() if c > 1],
        key=str
    )[:5]

    missing_parent_prefix = []
    for comp in sorted(ext_completed_sets, key=str):
        if len(comp) <= 1:
            continue
        if tuple(comp[:-1]) not in ext_completed_sets:
            missing_parent_prefix.append(list(comp))
    missing_parent_prefix = missing_parent_prefix[:5]

    metrics['execution_states'] = {
        'precision': p, 'recall': r, 'f1': f,
        'gt_count': len(gt_state_keys), 'ext_count': len(ext_state_keys),
        'missing': sorted(gt_state_keys - ext_state_keys, key=str)[:5],
        'extra': sorted(ext_state_keys - gt_state_keys, key=str)[:5],
        'unknown_completed_actions': sorted(unknown_completed)[:5],
        'unknown_available_actions': sorted(unknown_available)[:5],
        'duplicate_completed_states': duplicate_completed_states,
        'missing_parent_prefix': missing_parent_prefix,
    }

    return metrics


def print_metrics(all_metrics):
    """Print averaged metrics and list records with any score under 1.0."""
    print("=" * 80)
    print("EXTRACTION VALIDATION REPORT")
    print("=" * 80)

    #collect scores per metric and track imperfect records
    metric_keys = [
        ('action_f1',          'Action F1',             lambda m: m['actions']['f1']),
        ('gateway_type_acc',   'Gateway type accuracy',  lambda m: m['gateways']['type_accuracy']),
        ('gateway_role_acc',   'Gateway role accuracy',  lambda m: m['gateways']['role_accuracy']),
        ('action_succ_f1',     'Action successor F1',    lambda m: m['action_successors']['f1']),
        ('action_pred_f1',     'Action predecessor F1',  lambda m: m['action_predecessors']['f1']),
        ('gw_next_f1',         'Gateway next F1',        lambda m: m['gateway_branches_next']['f1']),
        ('gw_incoming_f1',     'Gateway incoming F1',    lambda m: m['gateway_incoming']['f1']),
        ('branch_tuple_f1',    'Branch tuple F1',        lambda m: m['branch_tuples']['f1']),
        ('branch_count_acc',   'Branch count accuracy',  lambda m: m['branch_counts']['accuracy']),
        ('exec_states_f1',     'Execution states F1',    lambda m: m['execution_states']['f1']),
    ]

    avg = defaultdict(list)
    #file_index -> list of (metric_name, score) where score < 1.0
    imperfect = defaultdict(list)

    for m in all_metrics:
        file_idx = m.get('file_index', '?')
        for key, label, extractor in metric_keys:
            score = extractor(m)
            avg[key].append(score)
            if score < 1.0:
                imperfect[file_idx].append((label, score))

    n = len(all_metrics)
    print(f"\nAVERAGES (over {n} records):")
    for key, label, _ in metric_keys:
        print(f"  {label + ':':<25s} {sum(avg[key])/n:.4f}")

    print(f"\n{'=' * 80}")
    if imperfect:
        print(f"RECORDS WITH SCORE < 1.0  ({len(imperfect)} records):")
        for file_idx, issues in sorted(imperfect.items(), key=str):
            issues_str = ', '.join(f"{label}={score:.2f}" for label, score in issues)
            print(f"  file_index={file_idx}:  {issues_str}")
    else:
        print("All records scored 1.0 on every metric.")
    print("=" * 80)


if __name__ == '__main__':
 
    with open(processed_dir / 'merged_dataset.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    #for extraction
    with open(processed_dir / 'extracted_test.json', 'r', encoding='utf-8') as f:
        extracted = json.load(f)

    #match by file_index
    raw_by_index = {r['file_index']: r for r in raw_data}

    all_metrics = []
    for ext in extracted:
        file_idx = ext['file_index']
        raw = raw_by_index.get(file_idx)
        m = validate_record(raw, ext)
        m['file_index'] = file_idx
        all_metrics.append(m)

    print_metrics(all_metrics)
