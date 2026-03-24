from dataclasses import dataclass, field



@dataclass
class CheckResult:
    is_valid: bool
    issues: list = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "Extraction is valid."
        return (
            f"Found {len(self.issues)} issue(s):\n"
            + "\n".join(f"  {i+1}. {iss}" for i, iss in enumerate(self.issues))
        )




class StructuralChecker:
    #we will check the structure of extracted components for 3 main thigs
    #first, we should have only UNIQUE id actions and gateways
    #next, using BFS we verify all action states are reachable from start. if an action never gets reached it s an issue
    #if you have start, A, B, gateway, {C, D}, both must be reachable.
    #and then we make sure at least on execution trace has can_terminate=true or no avaiolable next
    
    #first unique chcker
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
        #gateway and list of next IDs it can reach
        gw_targets = {
            g["id"]: [b["next"] for b in g.get("branches", []) if b.get("next")]
            for g in gateways
        }

        reachable = set()
        visited_gw = set()
        #
        frontier = set()
        for a in actions:
            if "start" in a.get("predecessors", []):
                reachable.add(a["id"])
                frontier.update(a.get("successors", []))
        for g in gateways:
            if "start" in g.get("incoming_from", []):
                frontier.add(g["id"])

        changed = True
        while changed:
            changed = False
            new_frontier = set()
            for nid in frontier:
                if nid in gw_targets and nid not in visited_gw:
                    visited_gw.add(nid)
                    new_frontier.update(gw_targets[nid])
                    changed = True
                elif nid in action_ids and nid not in reachable:
                    reachable.add(nid)
                    a = next(a for a in actions if a["id"] == nid)
                    new_frontier.update(a.get("successors", []))
                    changed = True
            # also check actions with predecessors now reachable
            for a in actions:
                if a["id"] not in reachable:
                    for p in a.get("predecessors", []):
                        if p in reachable or p in visited_gw:
                            reachable.add(a["id"])
                            new_frontier.update(a.get("successors", []))
                            changed = True
                            break
            frontier = new_frontier

        return [
            f"Action '{a['id']}' is not reachable from 'start'."
            for a in actions if a["id"] not in reachable
        ]
    
    #last checker
    @staticmethod
    def _check_terminal_state(w):
        states = w.get("execution_states", [])
        if not states:
            return []
        if not any(s.get("can_terminate") or not s.get("available_next") for s in states):
            return ["No terminal execution state found."]
        return []

    #!!! by the way i will re use this method inside the api extractor!!!!
    
    def check(self, workflow) -> CheckResult:
        if workflow is None:
            return CheckResult(False, ["Output could not be parsed as valid JSON."])#it happens to to too long
        #procedures that require more than 6k tokens for output. 

        issues = self._check_unique_ids(workflow.get("actions", []), "action")
        issues += self._check_unique_ids(workflow.get("gateways", []), "gateway")
        issues += self._check_start_reachability(workflow)
        issues += self._check_terminal_state(workflow)
        return CheckResult(len(issues) == 0, issues)
