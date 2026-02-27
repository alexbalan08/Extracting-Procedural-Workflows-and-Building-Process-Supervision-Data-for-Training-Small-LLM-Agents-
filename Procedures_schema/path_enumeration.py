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


def enumerate_paths(nodes, outgoing, incoming, rid_to_id, start_rids):
    """Enumerate all valid execution paths through the workflow"""

    def _dfs(current_rid, path, visit_counts=None, stop_at=None):
        """DFS to enumerate all valid execution paths from a given node"""
        if visit_counts is None:
            visit_counts = defaultdict(int)

        #stop at join gateway when we're inside a parallel/inclusive branch
        #this prevents branches from continuing past the join
        if stop_at and current_rid in stop_at:
            return [path]

        if visit_counts[current_rid] >= MAX_LOOP_ITERATIONS:
            return [path]

        visit_counts = defaultdict(int, visit_counts)
        visit_counts[current_rid] += 1

        node = nodes.get(current_rid)
        if not node:
            return [path]

        #EndNode: named EndNodes (with text) get added as actions
        #unnamed EndNodes just terminate the path
        if node['type'] == 'EndNode':
            if current_rid in rid_to_id:
                return [path + [rid_to_id[current_rid]]]
            return [path]

        #Activity node or StartNode with text we add to path
        if node['type'] == 'Activity' or (node['type'] == 'StartNode' and current_rid in rid_to_id):
            new_path = path + [rid_to_id[current_rid]]
            next_edges = outgoing.get(current_rid, [])
            if not next_edges:
                return [new_path]
            all_paths = []
            for tgt, _ in next_edges:
                all_paths.extend(_dfs(tgt, new_path, visit_counts, stop_at))
            return all_paths

        #XOR creates separate paths for each branch
        if node['type'] == 'XOR':
            all_paths = []
            for tgt, _ in outgoing.get(current_rid, []):
                all_paths.extend(_dfs(tgt, path, visit_counts, stop_at))
            return all_paths

        #AND we take all branches
        if node['type'] == 'AND':
            return _handle_and(current_rid, path, visit_counts, stop_at)

        #OR one or more is takes
        if node['type'] == 'OR':
            return _handle_or(current_rid, path, visit_counts, stop_at)

        #StartNode without text: just follow outgoing edges
        if node['type'] == 'StartNode':
            all_paths = []
            for tgt, _ in outgoing.get(current_rid, []):
                all_paths.extend(_dfs(tgt, path, visit_counts, stop_at))
            return all_paths

        return [path]

    def _handle_and(current_rid, path, visit_counts, stop_at):
        branches_out = outgoing.get(current_rid, [])
        if not branches_out:
            return [path]

        #AND JOIN
        if is_join(current_rid, incoming) and not is_split(current_rid, outgoing):
            all_paths = []
            for tgt, _ in branches_out:
                all_paths.extend(_dfs(tgt, path, visit_counts, stop_at))
            return all_paths

        #AND SPLIT
        join_rid = find_matching_join(current_rid, nodes, outgoing, incoming)
        branch_stop = {join_rid} if join_rid else set()
        branch_paths = []
        for tgt, _ in branches_out:
            bp = _dfs(tgt, [], visit_counts, stop_at=branch_stop)
            branch_paths.append(bp)

        #generate all valid interleavings: pick one sub-path per branch,
        #then permute the order becasue any order works
        branch_combos = list(product(*branch_paths))
        all_paths = []
        for combo in branch_combos[:MAX_PATHS]:
            branch_seqs = [seq for seq in combo if seq]
            if not branch_seqs:
                if join_rid:
                    all_paths.extend(_dfs(join_rid, path, visit_counts, stop_at))
                else:
                    all_paths.append(path)
                continue
            for perm in permutations(range(len(branch_seqs))):
                interleaved = path[:]
                for idx in perm:
                    interleaved.extend(branch_seqs[idx])
                if join_rid:
                    all_paths.extend(_dfs(join_rid, interleaved, visit_counts, stop_at))
                else:
                    all_paths.append(interleaved)
                if len(all_paths) >= MAX_PATHS:
                    break
            if len(all_paths) >= MAX_PATHS:
                break
        return all_paths[:MAX_PATHS]

    def _handle_or(current_rid, path, visit_counts, stop_at):
        branches_out = outgoing.get(current_rid, [])
        if not branches_out:
            return [path]

        #OR JOIN: pass through
        if is_join(current_rid, incoming) and not is_split(current_rid, outgoing):
            all_paths = []
            for tgt, _ in branches_out:
                all_paths.extend(_dfs(tgt, path, visit_counts, stop_at))
            return all_paths

        #OR SPLIT: pick one or more branches
        join_rid = find_matching_join(current_rid, nodes, outgoing, incoming)
        branch_stop = {join_rid} if join_rid else set()

        branch_paths = []
        for tgt, _ in branches_out:
            bp = _dfs(tgt, [], visit_counts, stop_at=branch_stop)
            branch_paths.append(bp)

        all_paths = []
        for r in range(1, len(branch_paths) + 1):
            for subset in combinations(range(len(branch_paths)), r):
                chosen = [branch_paths[i] for i in subset]
                for combo in product(*chosen):
                    merged = path[:]
                    for seq in combo:
                        merged.extend(seq)
                    if join_rid:
                        all_paths.extend(_dfs(join_rid, merged, visit_counts, stop_at))
                    else:
                        all_paths.append(merged)
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
        all_execution_paths.extend(_dfs(start, []))

    #remove duplicate paths
    unique_paths = []
    seen_paths = set()
    for p in all_execution_paths:
        key = tuple(p)
        if key not in seen_paths:
            seen_paths.add(key)
            unique_paths.append(p)

    return unique_paths

#path 1: A, B, C
#path 2: A, B, D
#Then after A, B, available next  C, D
def build_execution_states(unique_paths):
    """Build execution states by merging ALL paths into a single state map.
    At each prefix (completed actions so far), we collect ALL valid next actions"""
    state_map = OrderedDict()
    #track which states have at least one path that terminates there
    terminal_states = set()

    for path in unique_paths[:MAX_PATHS]:
        for step_idx in range(len(path) + 1):
            completed = tuple(path[:step_idx])
            next_action = path[step_idx] if step_idx < len(path) else None
            if completed not in state_map:
                state_map[completed] = set()
            if next_action:
                state_map[completed].add(next_action)
            else:
                #this path terminates at this prefix
                terminal_states.add(completed)

    execution_states = []
    for completed, available in state_map.items():
        state = {
            "completed_actions": list(completed),
            "available_next": sorted(available),
        }
        #mark states where the process can terminate (even if other actions are also available)
        if completed in terminal_states:
            state["can_terminate"] = True
        execution_states.append(state)

    return execution_states
