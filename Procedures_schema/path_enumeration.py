from collections import defaultdict, deque, OrderedDict
from itertools import permutations, combinations, product



MAX_PATHS = 60  #to avoid way too many paths in case there are such cases when we have loops
MAX_LOOP_ITERATIONS = 2


def is_split(rid, outgoing):
    return len(outgoing.get(rid, [])) > 1


def is_join(rid, incoming):
    return len(incoming.get(rid, [])) > 1


def find_matching_join(split_rid, nodes, outgoing, incoming):
    """Find the join gateway that corresponds to a split gateway.
    BFS from each branch to find the first common convergence node
    that has multiple incoming edges."""
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
        if nid in common and is_join(nid, incoming):
            node = nodes.get(nid)
            if node and node['type'] in ('AND', 'OR', 'XOR'):
                return nid
        for next_tgt, _ in outgoing.get(nid, []):
            queue.append(next_tgt)

    return None

#
def enumerate_paths(nodes, outgoing, incoming, rid_to_id, start_rids):

    def _dfs(current_rid, path, conditions, visit_counts=None, stop_at=None):
        """DFS to enumerate all valid execution paths from a given node.
        Returns list of (path, conditions) tuples.
        conditions: list of (condition_str, path_index) tuples."""
        if visit_counts is None:
            visit_counts = defaultdict(int)

        #stop at join gateway when we're inside a parallel/inclusive branch
        #this prevents branches from continuing past the join
        if stop_at and current_rid in stop_at:
            return [(path, conditions)]

        if visit_counts[current_rid] >= MAX_LOOP_ITERATIONS:
            return [(path, conditions)]

        visit_counts = defaultdict(int, visit_counts)
        visit_counts[current_rid] += 1

        node = nodes.get(current_rid)
        if not node:
            return [(path, conditions)]

        #named EndNodes (with text) get added as actions
        #unnamed EndNodes just terminate the path
        if node['type'] == 'EndNode':
            if current_rid in rid_to_id:
                return [(path + [rid_to_id[current_rid]], conditions)]
            return [(path, conditions)]

        #Activity node or StartNode with text we add to path
        if node['type'] == 'Activity' or (node['type'] == 'StartNode' and current_rid in rid_to_id):
            new_path = path + [rid_to_id[current_rid]]
            next_edges = outgoing.get(current_rid, [])
            if not next_edges:
                return [(new_path, conditions)]
            all_paths = []
            for tgt, cond in next_edges:
                new_conds = conditions + ([(cond.strip(), len(new_path))] if cond.strip() else [])
                all_paths.extend(_dfs(tgt, new_path, new_conds, visit_counts, stop_at))
            return all_paths

        #XOR creates separate paths for each branch, recording the chosen condition
        if node['type'] == 'XOR':
            all_paths = []
            for tgt, cond in outgoing.get(current_rid, []):
                new_conds = conditions + ([(cond.strip(), len(path))] if cond.strip() else [])
                all_paths.extend(_dfs(tgt, path, new_conds, visit_counts, stop_at))
            return all_paths

        #AND we take all branches
        if node['type'] == 'AND':
            return _handle_and(current_rid, path, conditions, visit_counts, stop_at)

        #OR one or more is taken
        if node['type'] == 'OR':
            return _handle_or(current_rid, path, conditions, visit_counts, stop_at)

        #StartNode without text: just follow outgoing edges
        if node['type'] == 'StartNode':
            all_paths = []
            for tgt, cond in outgoing.get(current_rid, []):
                new_conds = conditions + ([(cond.strip(), len(path))] if cond.strip() else [])
                all_paths.extend(_dfs(tgt, path, new_conds, visit_counts, stop_at))
            return all_paths

        return [(path, conditions)]

    def _handle_and(current_rid, path, conditions, visit_counts, stop_at):
        branches_out = outgoing.get(current_rid, [])
        if not branches_out:
            return [(path, conditions)]

        #AND JOIN
        if is_join(current_rid, incoming) and not is_split(current_rid, outgoing):
            all_paths = []
            for tgt, cond in branches_out:
                new_conds = conditions + ([(cond.strip(), len(path))] if cond.strip() else [])
                all_paths.extend(_dfs(tgt, path, new_conds, visit_counts, stop_at))
            return all_paths

        #AND SPLIT: each branch runs independently, collecting its own sub-path and conditions
        join_rid = find_matching_join(current_rid, nodes, outgoing, incoming)
        branch_stop = {join_rid} if join_rid else set()
        branch_results = []
        for tgt, cond in branches_out:
            edge_conds = [(cond.strip(), 0)] if cond.strip() else []
            bp = _dfs(tgt, [], edge_conds, visit_counts, stop_at=branch_stop)
            branch_results.append(bp)  # list of (sub_path, sub_conditions)

        #generate all valid interleavings: pick one result per branch,
        #then permute the order because any order works for AND
        branch_combos = list(product(*branch_results))
        all_paths = []
        for combo in branch_combos[:MAX_PATHS]:
            branch_seqs = [(bp, bc) for bp, bc in combo if bp]
            if not branch_seqs:
                if join_rid:
                    all_paths.extend(_dfs(join_rid, path, conditions, visit_counts, stop_at))
                else:
                    all_paths.append((path, conditions))
                continue
            for perm in permutations(range(len(branch_seqs))):
                interleaved = path[:]
                merged_conds = conditions[:]
                for idx in perm:
                    offset = len(interleaved)
                    interleaved.extend(branch_seqs[idx][0])
                    merged_conds.extend([(c, ci + offset) for c, ci in branch_seqs[idx][1]])
                if join_rid:
                    all_paths.extend(_dfs(join_rid, interleaved, merged_conds, visit_counts, stop_at))
                else:
                    all_paths.append((interleaved, merged_conds))
                if len(all_paths) >= MAX_PATHS:
                    break
            if len(all_paths) >= MAX_PATHS:
                break
        return all_paths[:MAX_PATHS]

    def _handle_or(current_rid, path, conditions, visit_counts, stop_at):
        branches_out = outgoing.get(current_rid, [])
        if not branches_out:
            return [(path, conditions)]

        #OR JOIN: pass through
        if is_join(current_rid, incoming) and not is_split(current_rid, outgoing):
            all_paths = []
            for tgt, cond in branches_out:
                new_conds = conditions + ([(cond.strip(), len(path))] if cond.strip() else [])
                all_paths.extend(_dfs(tgt, path, new_conds, visit_counts, stop_at))
            return all_paths

        #OR SPLIT: pick one or more branches, collecting sub-paths and conditions
        join_rid = find_matching_join(current_rid, nodes, outgoing, incoming)
        branch_stop = {join_rid} if join_rid else set()

        branch_results = []
        for tgt, cond in branches_out:
            edge_conds = [(cond.strip(), 0)] if cond.strip() else []
            bp = _dfs(tgt, [], edge_conds, visit_counts, stop_at=branch_stop)
            branch_results.append(bp)

        all_paths = []
        for r in range(1, len(branch_results) + 1):
            for subset in combinations(range(len(branch_results)), r):
                chosen = [branch_results[i] for i in subset]
                for combo in product(*chosen):
                    merged = path[:]
                    merged_conds = conditions[:]
                    for bp, bc in combo:
                        offset = len(merged)
                        merged.extend(bp)
                        merged_conds.extend([(c, ci + offset) for c, ci in bc])
                    if join_rid:
                        all_paths.extend(_dfs(join_rid, merged, merged_conds, visit_counts, stop_at))
                    else:
                        all_paths.append((merged, merged_conds))
                    if len(all_paths) >= MAX_PATHS:
                        break
                if len(all_paths) >= MAX_PATHS:
                    break
            if len(all_paths) >= MAX_PATHS:
                break
        return all_paths[:MAX_PATHS]

    #enumerate all valid paths from all start nodes
    all_execution_paths = []
    for start in start_rids:
        all_execution_paths.extend(_dfs(start, [], []))

    #remove duplicate (path, conditions) pairs
    unique_paths = []
    seen_paths = set()
    for p, c in all_execution_paths:
        key = (tuple(p), tuple(c))
        if key not in seen_paths:
            seen_paths.add(key)
            unique_paths.append((p, c))

    return unique_paths

def build_execution_states(unique_paths):

    seen = set()
    execution_states = []

    for path, conditions in unique_paths[:MAX_PATHS]:
        for step_idx in range(len(path) + 1):
            completed = tuple(path[:step_idx])
            next_action = path[step_idx] if step_idx < len(path) else None

            #only include conditions whose gateway was traversed by this step
            #initial state (step_idx=0) never has conditions — no gateway traversed yet
            active_conds = [c for c, idx in conditions if step_idx > 0 and idx <= step_idx]

            key = (completed, next_action, tuple(active_conds))
            if key in seen:
                continue
            seen.add(key)

            state = {
                "completed_actions": list(completed),
                "conditions_met": active_conds,
            }
            if next_action:
                state["available_next"] = [next_action]
                
            else:
                state["available_next"] = []
                state["can_terminate"] = True

            execution_states.append(state)

    return execution_states
