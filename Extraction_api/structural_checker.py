from dataclasses import dataclass, field


@dataclass
class CheckResult:
    is_valid: bool
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "Extraction is valid."
        return (
            f"Found {len(self.issues)} issue(s):\n"
            + "\n".join(f"  {i+1}. {iss}" for i, iss in enumerate(self.issues))
        )


class StructuralChecker:
    #we will check the structure of extracted components for 3 main things
    #first, we should have only UNIQUE id actions and gateways
    #next, using BFS we verify all action states are reachable from start. if an action never gets reached it s an issue
    #if you have start, A, B, gateway, {C, D}, both must be reachable.
    #and then we make sure at least one action has empty successors (terminal node)
    #note: execution_states are now derived algorithmically — we only check graph structure here

    #first unique checker
    @staticmethod
    def _check_unique_ids(items, label):
        seen, dupes = set(), set()
        for item in items:
            val = item.get("id", "")
            (dupes if val in seen else seen).add(val)
        return [f"Duplicate {label} ID: '{d}'." for d in sorted(dupes)]

    @staticmethod
    def _check_start_reachability(w):
        #bfs search
        actions = w.get("actions", [])
        gateways = w.get("gateways", [])
        if not actions:
            return []

        action_ids = {a["id"] for a in actions}
        gateway_ids = {g["id"] for g in gateways}
        all_ids = action_ids | gateway_ids

        #build adjacency: node -> successors
        adj = {nid: set() for nid in all_ids}
        for a in actions:
            for s in a.get("successors", []):
                if s in all_ids:
                    adj[a["id"]].add(s)
        for g in gateways:
            for b in g.get("branches", []):
                nxt = b.get("next")
                if nxt and nxt in all_ids:
                    adj[g["id"]].add(nxt)

        #find start nodes: explicit "start" predecessor, or fallback to nodes with no incoming edges
        has_incoming = set()
        for a in actions:
            for s in a.get("successors", []):
                if s in all_ids:
                    has_incoming.add(s)
        for g in gateways:
            for b in g.get("branches", []):
                nxt = b.get("next")
                if nxt and nxt in all_ids:
                    has_incoming.add(nxt)

        start_nodes = {a["id"] for a in actions if "start" in a.get("predecessors", [])}
        if not start_nodes:
            start_nodes = all_ids - has_incoming
        if not start_nodes:
            return ["No start node found — no action has 'start' as predecessor."]

        visited = set()
        queue = list(start_nodes)
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj.get(node, []))

        return [
            f"Node '{uid}' is unreachable from start."
            for uid in sorted(all_ids - visited)
        ]

    @staticmethod
    def _check_terminal_state(w):
        actions = w.get("actions", [])
        if not actions:
            return []
        if not any(not a.get("successors") for a in actions):
            return ["No terminal action found — at least one action must have empty successors."]
        return []

    #!!! by the way i will re use this method inside the api extractor!!!!

    def check(self, workflow) -> CheckResult:
        if workflow is None:
            return CheckResult(False, ["Output could not be parsed as valid JSON."])
            #it happens for procedures that require more than 6k tokens for output

        issues = self._check_unique_ids(workflow.get("actions", []), "action")
        issues += self._check_unique_ids(workflow.get("gateways", []), "gateway")
        issues += self._check_start_reachability(workflow)
        issues += self._check_terminal_state(workflow)
        return CheckResult(len(issues) == 0, issues)
