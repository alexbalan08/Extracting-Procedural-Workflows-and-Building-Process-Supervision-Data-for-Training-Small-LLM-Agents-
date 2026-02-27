import json
from pathlib import Path
from collections import defaultdict

from graph_builder import (
    build_graph, build_rid_to_id, make_gateway_id,
    ACTIONABLE_TYPES, GATEWAY_TYPE_MAP
)

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
    #ground truth: from SequenceFlow, map each edge to schema IDs
    gt_action_successors = set()
    gt_action_predecessors = set()
    for rid in rid_to_id:
        action_id = rid_to_id[rid]
        #successors: direct outgoing edges from this action node
        for tgt, cond in outgoing.get(rid, []):
            tgt_sid = schema_id(tgt)
            if tgt_sid:
                gt_action_successors.add((action_id, tgt_sid))
        #predecessors: direct incoming edges to this action node
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
        'extra': sorted(ext_action_successors - ext_action_successors)[:5],
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

    #--- 4. Gateway branch accuracy (gateway_id, next_id, condition_norm) ---
    gt_branch_tuples = set()
    for rid, gid in rid_to_gateway_id.items():
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue
            tgt_sid = schema_id(tgt)
            if tgt_node['type'] == 'EndNode':
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

    #per-gateway branch count match
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

    return metrics


def print_metrics(all_metrics):
    """Print per-record and averaged metrics."""
    print("=" * 80)
    print("EXTRACTION VALIDATION REPORT")
    print("=" * 80)

    avg = defaultdict(list)

    for i, m in enumerate(all_metrics):
        print(f"\n--- Record {i} (file_index: {m.get('file_index', '?')}) ---")

        a = m['actions']
        print(f"  Actions:           P={a['precision']:.2f}  R={a['recall']:.2f}  F1={a['f1']:.2f}  (GT={a['gt_count']}, Ext={a['ext_count']})")
        if a['missing']:
            print(f"                     Missing: {a['missing']}")
        if a['extra']:
            print(f"                     Extra:   {a['extra']}")
        avg['action_f1'].append(a['f1'])

        g = m['gateways']
        print(f"  Gateways:          count_match={g['count_match']}  type_acc={g['type_accuracy']:.2f}  role_acc={g['role_accuracy']:.2f}  (GT={g['gt_count']}, Ext={g['ext_count']})")
        avg['gateway_type_acc'].append(g['type_accuracy'])
        avg['gateway_role_acc'].append(g['role_accuracy'])

        s = m['action_successors']
        print(f"  Action successors: P={s['precision']:.2f}  R={s['recall']:.2f}  F1={s['f1']:.2f}  (GT={s['gt_count']}, Ext={s['ext_count']})")
        if s['missing']:
            print(f"                     Missing: {s['missing']}")
        avg['action_succ_f1'].append(s['f1'])

        p = m['action_predecessors']
        print(f"  Action predecess:  P={p['precision']:.2f}  R={p['recall']:.2f}  F1={p['f1']:.2f}  (GT={p['gt_count']}, Ext={p['ext_count']})")
        if p['missing']:
            print(f"                     Missing: {p['missing']}")
        avg['action_pred_f1'].append(p['f1'])

        gn = m['gateway_branches_next']
        print(f"  Gateway next:      P={gn['precision']:.2f}  R={gn['recall']:.2f}  F1={gn['f1']:.2f}  (GT={gn['gt_count']}, Ext={gn['ext_count']})")
        if gn['missing']:
            print(f"                     Missing: {gn['missing']}")
        avg['gw_next_f1'].append(gn['f1'])

        gi = m['gateway_incoming']
        print(f"  Gateway incoming:  P={gi['precision']:.2f}  R={gi['recall']:.2f}  F1={gi['f1']:.2f}  (GT={gi['gt_count']}, Ext={gi['ext_count']})")
        if gi['missing']:
            print(f"                     Missing: {gi['missing']}")
        avg['gw_incoming_f1'].append(gi['f1'])

        bt = m['branch_tuples']
        print(f"  Branch tuples:     P={bt['precision']:.2f}  R={bt['recall']:.2f}  F1={bt['f1']:.2f}  (GT={bt['gt_count']}, Ext={bt['ext_count']})")
        if bt['missing']:
            print(f"                     Missing: {bt['missing']}")
        avg['branch_tuple_f1'].append(bt['f1'])

        bc = m['branch_counts']
        print(f"  Branch counts:     accuracy={bc['accuracy']:.2f}  (compared={bc['total_compared']})")
        avg['branch_count_acc'].append(bc['accuracy'])

    n = len(all_metrics)
    print(f"\n{'=' * 80}")
    print(f"AVERAGES (over {n} records):")
    print(f"  Action F1:              {sum(avg['action_f1'])/n:.2f}")
    print(f"  Gateway type accuracy:  {sum(avg['gateway_type_acc'])/n:.2f}")
    print(f"  Gateway role accuracy:  {sum(avg['gateway_role_acc'])/n:.2f}")
    print(f"  Action successor F1:    {sum(avg['action_succ_f1'])/n:.2f}")
    print(f"  Action predecessor F1:  {sum(avg['action_pred_f1'])/n:.2f}")
    print(f"  Gateway next F1:        {sum(avg['gw_next_f1'])/n:.2f}")
    print(f"  Gateway incoming F1:    {sum(avg['gw_incoming_f1'])/n:.2f}")
    print(f"  Branch tuple F1:        {sum(avg['branch_tuple_f1'])/n:.2f}")
    print(f"  Branch count accuracy:  {sum(avg['branch_count_acc'])/n:.2f}")
    print("=" * 80)


if __name__ == '__main__':
    #load raw data
    with open(processed_dir / 'merged_train.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    #load extracted workflows
    with open(output_dir / 'workflow_samples.json', 'r', encoding='utf-8') as f:
        extracted = json.load(f)

    #match by file_index
    raw_by_index = {r['file_index']: r for r in raw_data}

    all_metrics = []
    for ext in extracted:
        file_idx = ext['file_index']
        raw = raw_by_index.get(file_idx)
        if not raw:
            print(f"WARNING: no raw record found for file_index {file_idx}")
            continue
        m = validate_record(raw, ext)
        m['file_index'] = file_idx
        all_metrics.append(m)

    print_metrics(all_metrics)
