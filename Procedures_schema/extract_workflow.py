import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
processed_dir = project_root / 'Data' / 'Processed'
output_dir = Path(__file__).parent


def build_graph(record):
    """To build the graph it's first useful to have a dictionary with all the nodes extracted, their IDs,
    their, role (for example action, or XOR), and in some cases, the node's text"""
    nodes = {}

    for n in record['step_nodes']:
        nodes[n['resourceId']] = n

    
    outgoing = defaultdict(list)
    incoming = defaultdict(list)


    
    #"SequenceFlow" has tis stored and also we have the src and trg to determine the direction of the edge
    for edge in record['SequenceFlow']:
        outgoing[edge['src']].append((edge['tgt'], edge['condition']))
        incoming[edge['tgt']].append((edge['src'], edge['condition']))
        #this will bassicaly be an adjacent list with the pre-condition as well in the cases where there s one

    return nodes, outgoing, incoming



def make_action_id(node_text, seen_ids):
    """This is just a normalization for the actions texts.
    The idea is to lower case everything, remove spaces and replace with _
    Remove 's """

    if not node_text.strip():
        return "unnamed_action"
    
    action_id = node_text.strip().lower()
    action_id = action_id.replace(' ', '_').replace("'", "")
    action_id = ''.join(c for c in action_id if c.isalnum() or c == '_')

    #deduplicate
    #each action will have an unique id which and also we will prevent same action with the same name
    #like order drink and then if we encouter order drink again we make it order_drink_2
    base = action_id
    counter = 2
    while action_id in seen_ids:
        #here we avoid the duplicate
        action_id = f"{base}_{counter}"
        counter += 1

    seen_ids.add(action_id)
    return action_id


def extract_workflow(record):
    """Main method where we basically build the graph with the nodes and edges we extracted
    We will first map the ugly numerical ids to action readable ids which are simple to read by humans"""
    nodes, outgoing, incoming = build_graph(record)

    #gateway-to-gateway edge labels (e.g. "Funds Reserved", "Order Accepted") are conditions/states
    #not actions. They stay as gateway branch conditions and are NOT converted to action nodes.
    #only real Activity nodes from the BPMN data become actions.

    #"sid-A9CAFE2A-85FF-4974-81FD-F086D4281922": "must_found_more_funds"
    #for readability reasons
    seen_ids = set()
    rid_to_id = {}

    #we also include StartNodes and EndNodes that have text (e.g. "Loan application received",
    #"Order declined") since these represent meaningful actions/events in the workflow
    #without this, ~1500 start events and ~2300 end events would be lost
    actionable_types = {'Activity', 'StartNode', 'EndNode'}
    for rid, node in nodes.items():
        if node['type'] in actionable_types and node['NodeText'].strip():
            rid_to_id[rid] = make_action_id(node['NodeText'], seen_ids)

    #we retrieve all unique actors
    actors = list(dict.fromkeys(n['agent'] for n in record['step_nodes'] if n['agent'].strip()))

    #we build actions from activity nodes AND from start/end nodes that have text
    actions = []
    for rid, node in nodes.items():
        if node['type'] not in actionable_types or not node['NodeText'].strip():
            continue

        action_id = rid_to_id[rid]

        #we need to find what comes before so basiclaly everything that points TO this action node
        #and we append what comes before depending on its node type
        predecessors = []
        for src, cond in incoming.get(rid, []):
            src_node = nodes.get(src)
            if not src_node:
                continue
            #if the source is an actionable node with text, reference its action id
            if src_node['type'] in actionable_types and src_node['NodeText'].strip():
                predecessors.append(rid_to_id[src])
            elif src_node['type'] in ('XOR', 'AND', 'OR'):
                predecessors.append(f"gateway_{src_node['type'].lower()}_{list(nodes.keys()).index(src)}")
            elif src_node['type'] == 'StartNode':
                #StartNode without text = just a structural start marker
                predecessors.append("start")

        #we need to find what comes after so basiclaly everything that points FROM this action node
        #and we append what comes after depending on its node type
        #now inistead of start we need to check the end node
        successors = []
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue
            #if the target is an actionable node with text, reference its action id
            if tgt_node['type'] in actionable_types and tgt_node['NodeText'].strip():
                successors.append(rid_to_id[tgt])
            elif tgt_node['type'] in ('XOR', 'AND', 'OR'):
                successors.append(f"gateway_{tgt_node['type'].lower()}_{list(nodes.keys()).index(tgt)}")
            #unnamed EndNodes are implicit termination, not listed in successors

        #now we finally build the final action object with all information
        #like cleaned readbale id, the actor, outgoing and incoming edges
        #conditions are NOT stored on actions - they belong on gateway branches only
        #to avoid duplication and confusion during training/validation
        #postconditions represent the state achieved after completing this action
        #this is useful for process supervision to track what has been done
        #deduplicate while preserving order (e.g. two edges from same gateway)
        predecessors = list(dict.fromkeys(predecessors))
        successors = list(dict.fromkeys(successors))

        action = {
            "id": action_id,
            "name": node['NodeText'].strip(),
            "actor": node['agent'].strip() if node['agent'].strip() else None,
            "predecessors": predecessors,
            "successors": successors,
            #added postconditions in order to define states as "postcondition" and future possible action
            "postconditions": [f"{action_id}_done"],
        }

        actions.append(action)


    #we have in the dataset 3 different types of gateway
    # xor, and or OR and we build again in the same way for the nodes which have as type gatyeway
    #an unique numbered ID
    gateways = []
    for rid, node in nodes.items():
        if node['type'] not in ('XOR', 'AND', 'OR'):
            continue

        gateway_id = f"gateway_{node['type'].lower()}_{list(nodes.keys()).index(rid)}"

        gateway_type_map = {
            'XOR': 'exclusive',
            'AND': 'parallel',
            'OR': 'inclusive'
        }

        #for each outgoing edge from gateway we create a branch
        #each branch has next (where it goes) and optional condition
        # Gateway with two branches:
        #   use_own_money (condition="not found")
        #   fund_procedure (condition="found")

        branches = []
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue
            #actionable nodes (Activity, StartNode/EndNode with text) -> use action id
            if tgt in rid_to_id:
                next_id = rid_to_id[tgt]
            elif tgt_node['type'] in ('XOR', 'AND', 'OR'):
                next_id = f"gateway_{tgt_node['type'].lower()}_{list(nodes.keys()).index(tgt)}"
            elif tgt_node['type'] == 'EndNode':
                #unnamed EndNode = implicit termination, skip this branch
                #the condition info is still visible in the other branches' conditions
                #and termination is implicit via available_next: [] in execution_states
                continue
            else:
                next_id = tgt

            branch = {"next": next_id}
            if cond.strip():
                branch["condition"] = cond.strip()
            branches.append(branch)

        #same logic here for for incoming
        incoming_from = []
        for src, cond in incoming.get(rid, []):
            src_node = nodes.get(src)
            if not src_node:
                continue
            #actionable nodes -> use action id
            if src in rid_to_id:
                incoming_from.append(rid_to_id[src])
            elif src_node['type'] in ('XOR', 'AND', 'OR'):
                incoming_from.append(f"gateway_{src_node['type'].lower()}_{list(nodes.keys()).index(src)}")

        #deduplicate while preserving order (e.g. two edges from same gateway)
        incoming_from = list(dict.fromkeys(incoming_from))

        #as we did for nodes now we build the gateaway object with id, type, incooming agdes and branches
        gateway = {
            "id": gateway_id,
            "type": gateway_type_map[node['type']],
            "incoming_from": incoming_from,
            "branches": branches,
        }
        if node['agent'].strip():
            gateway["actor"] = node['agent'].strip()

        gateways.append(gateway)

    #we enumerate all valid execution paths through the workflow using DFS
    #this properly handles different gateway types:
    # XOR (exclusive): only ONE branch is taken -> separate paths per branch
    # AND (parallel): ALL branches must complete -> interleave, then continue past join
    # OR (inclusive): ONE or MORE branches -> combinations
    #each path is a list of action ids in valid execution order
    from itertools import permutations, combinations
    from collections import deque, OrderedDict

    MAX_PATHS = 50  #to avoid way too many paths in case there are such cases
    MAX_LOOP_ITERATIONS = 2

    def is_split(rid):
        """A split gateway has multiple outgoing edges (fork)."""
        return len(outgoing.get(rid, [])) > 1

    def is_join(rid):
        """A join gateway has multiple incoming edges (synchronization point)."""
        return len(incoming.get(rid, [])) > 1

    def find_matching_join(split_rid, gateway_type):
        """Find the join gateway that corresponds to a split gateway.
        BFS from each branch to find the first common node that has
        multiple incoming edges and is the same gateway type.
        For AND split -> find AND join where all branches reconverge.
        For OR split -> find OR join."""
        branches_out = outgoing.get(split_rid, [])
        if len(branches_out) <= 1:
            return None

        #BFS from each branch to find all reachable nodes per branch
        branch_reachable = []
        for tgt, _ in branches_out:
            reachable = set()
            queue = deque([tgt])
            while queue:
                nid = queue.popleft()
                if nid in reachable or nid == split_rid:
                    continue
                reachable.add(nid)
                for next_tgt, _ in outgoing.get(nid, []):
                    queue.append(next_tgt)
            branch_reachable.append(reachable)

        #find nodes reachable from ALL branches (convergence points)
        if not branch_reachable:
            return None
        common = set.intersection(*branch_reachable)
        if not common:
            return None

        #among common nodes find the closest one with multiple incoming edges
        #BFS from split to get closest first
        queue = deque()
        for tgt, _ in branches_out:
            queue.append(tgt)
        visited = set()
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            if nid in common and is_join(nid):
                node = nodes.get(nid)
                #prefer same gateway type but accept any join
                if node and node['type'] in ('AND', 'OR', 'XOR'):
                    return nid
            for next_tgt, _ in outgoing.get(nid, []):
                queue.append(next_tgt)

        return None

    def enumerate_paths_from(current_rid, path, visit_counts=None, stop_at=None):
        """DFS to enumerate all valid execution paths from a given node.
        Returns a list of complete paths where each path is a list of action ids.
        visit_counts tracks how many times each node has been visited on the current path
        to detect and limit cycles like loops in the workflow like revise-rework-revise.
        stop_at is a set of node rids where the DFS should stop (used for AND/OR joins
        so that branches don't continue past the join point)."""
        if visit_counts is None:
            visit_counts = defaultdict(int)

        #stop at join gateway when we're inside a parallel/inclusive branch
        #this prevents branches from continuing past the join and duplicating post-join actions
        if stop_at and current_rid in stop_at:
            return [path]

        if visit_counts[current_rid] >= MAX_LOOP_ITERATIONS:
            return [path]

        visit_counts = defaultdict(int, visit_counts)
        visit_counts[current_rid] += 1

        node = nodes.get(current_rid)
        if not node:
            return [path]

        #EndNode: mark the path as complete
        #named EndNodes (with text) get added as actions (e.g. "loan_application_rejected")
        #unnamed EndNodes just terminate the path (no "end" token in the path)
        if node['type'] == 'EndNode':
            if current_rid in rid_to_id:
                return [path + [rid_to_id[current_rid]]]
            return [path]

        #if activity node add it to the current path and follow outgoing edges
        #StartNodes with text are also treated as actions (e.g. "Loan application received")
        if node['type'] == 'Activity' or (node['type'] == 'StartNode' and current_rid in rid_to_id):
            new_path = path + [rid_to_id[current_rid]]
            next_edges = outgoing.get(current_rid, [])
            if not next_edges:
                return [new_path]
            all_paths = []
            for tgt, _ in next_edges:
                all_paths.extend(enumerate_paths_from(tgt, new_path, visit_counts, stop_at))
            return all_paths

        #XOR gateway: exclusive choice, only one branch is taken
        #creates separate paths for each branch
        if node['type'] == 'XOR':
            #XOR join (multiple incoming, 1 outgoing): just pass through
            #XOR split (1 incoming, multiple outgoing): explore each branch
            #both cases: follow each outgoing edge independently
            all_paths = []
            for tgt, _ in outgoing.get(current_rid, []):
                all_paths.extend(enumerate_paths_from(tgt, path, visit_counts, stop_at))
            return all_paths

        #AND gateway: parallel execution, ALL branches must be taken
        if node['type'] == 'AND':
            branches_out = outgoing.get(current_rid, [])
            if not branches_out:
                return [path]

            #AND JOIN (multiple incoming, typically 1 outgoing)
            #all incoming branches must have completed before we pass through
            #in our DFS this is handled by the fact that we only reach here
            #after the AND split has already collected and interleaved all branches
            #so we just pass through to the outgoing edge
            if is_join(current_rid) and not is_split(current_rid):
                all_paths = []
                for tgt, _ in branches_out:
                    all_paths.extend(enumerate_paths_from(tgt, path, visit_counts, stop_at))
                return all_paths

            #AND SPLIT (multiple outgoing): all branches execute in parallel
            #1. find the matching join where branches reconverge
            #2. collect actions from each branch, stopping at the join
            #3. generate all valid orderings (permutations) of parallel branches
            #4. continue from after the join
            join_rid = find_matching_join(current_rid, 'AND')

            #collect paths from each parallel branch, stopping at the join
            branch_stop = {join_rid} if join_rid else set()
            branch_paths = []
            for tgt, _ in branches_out:
                bp = enumerate_paths_from(tgt, [], visit_counts, stop_at=branch_stop)
                branch_paths.append(bp)

            #generate all valid interleavings: pick one sub-path per branch,
            #then permute the order (since parallel = any order is valid)
            from itertools import product
            branch_combos = list(product(*branch_paths))
            all_paths = []
            for combo in branch_combos[:MAX_PATHS]:
                branch_seqs = [seq for seq in combo if seq]
                if not branch_seqs:
                    #all branches were empty, continue from join
                    if join_rid:
                        all_paths.extend(enumerate_paths_from(join_rid, path, visit_counts, stop_at))
                    else:
                        all_paths.append(path)
                    continue
                for perm in permutations(range(len(branch_seqs))):
                    interleaved = path[:]
                    for idx in perm:
                        interleaved.extend(branch_seqs[idx])
                    #after interleaving all branches, continue from the join
                    if join_rid:
                        continuations = enumerate_paths_from(join_rid, interleaved, visit_counts, stop_at)
                        all_paths.extend(continuations)
                    else:
                        all_paths.append(interleaved)
                    if len(all_paths) >= MAX_PATHS:
                        break
                if len(all_paths) >= MAX_PATHS:
                    break
            return all_paths[:MAX_PATHS]

        #OR gateway: inclusive choice, one or more branches taken
        if node['type'] == 'OR':
            branches_out = outgoing.get(current_rid, [])
            if not branches_out:
                return [path]

            #OR JOIN: pass through
            if is_join(current_rid) and not is_split(current_rid):
                all_paths = []
                for tgt, _ in branches_out:
                    all_paths.extend(enumerate_paths_from(tgt, path, visit_counts, stop_at))
                return all_paths

            #OR SPLIT: pick one or more branches
            join_rid = find_matching_join(current_rid, 'OR')
            branch_stop = {join_rid} if join_rid else set()

            branch_paths = []
            for tgt, _ in branches_out:
                bp = enumerate_paths_from(tgt, [], visit_counts, stop_at=branch_stop)
                branch_paths.append(bp)

            all_paths = []
            #try all non-empty subsets of branches (OR = pick 1 or more)
            for r in range(1, len(branch_paths) + 1):
                for subset in combinations(range(len(branch_paths)), r):
                    chosen = [branch_paths[i] for i in subset]
                    from itertools import product as iprod
                    for combo in iprod(*chosen):
                        merged = path[:]
                        for seq in combo:
                            merged.extend(seq)
                        #continue from join after merging branches
                        if join_rid:
                            continuations = enumerate_paths_from(join_rid, merged, visit_counts, stop_at)
                            all_paths.extend(continuations)
                        else:
                            all_paths.append(merged)
                        if len(all_paths) >= MAX_PATHS:
                            break
                    if len(all_paths) >= MAX_PATHS:
                        break
                if len(all_paths) >= MAX_PATHS:
                    break
            return all_paths[:MAX_PATHS]

        #StartNode: just follow outgoing edges
        if node['type'] == 'StartNode':
            all_paths = []
            for tgt, _ in outgoing.get(current_rid, []):
                all_paths.extend(enumerate_paths_from(tgt, path, visit_counts, stop_at))
            return all_paths

        return [path]

    start_rids = [rid for rid, n in nodes.items() if n['type'] == 'StartNode']

    #enumerate all valid paths from all start nodes
    all_execution_paths = []
    for start in start_rids:
        all_execution_paths.extend(enumerate_paths_from(start, []))

    #remove duplicate paths
    unique_paths = []
    seen_paths = set()
    for p in all_execution_paths:
        key = tuple(p)
        if key not in seen_paths:
            seen_paths.add(key)
            unique_paths.append(p)

    #build execution states by merging ALL paths into a single state map
    #at each prefix (completed actions so far), we collect ALL valid next actions
    #across all paths that share that prefix
    #this way at an XOR gateway we see both branches as available
    #and at an AND gateway we see all parallel branches as available
    #e.g. after ["collect_receipts"], available_next = ["use_own_money", "fund_procedure"]
    state_map = OrderedDict()  #key: tuple of completed actions -> set of valid next actions

    for path in unique_paths[:MAX_PATHS]:
        for step_idx in range(len(path) + 1):
            completed = tuple(path[:step_idx])
            next_action = path[step_idx] if step_idx < len(path) else None
            if completed not in state_map:
                state_map[completed] = set()
            if next_action:
                state_map[completed].add(next_action)
            #paths that terminate here naturally have available_next: [] (implicit end)

    #convert to list format
    execution_states = []
    for completed, available in state_map.items():
        execution_states.append({
            "completed_actions": list(completed),
            "available_next": sorted(available),
        })

    #final workflow schema with the index file number for each procedure, the text and our created worflow
    #with the excution paths as well  
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

samples = [extract_workflow(data[i]) for i in range(2)]

output_path = output_dir / 'workflow_samples.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

for i, s in enumerate(samples):
    w = s['workflow']
    print(f"Record {i}: {len(w['actions'])} actions, {len(w['gateways'])} gateways, actors: {w['actors']}")

print(f"\nSaved to {output_path}")
