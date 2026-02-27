from collections import defaultdict


def build_graph(record):
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
        #we will need this to label each XOR node type by OR, or AND for example or just merge XOR (where we have 2 ingoing edges and one outgoing)

    return nodes, outgoing, incoming


def make_action_id(node_text, seen_ids):
    """
    The idea is to lower case everything, remove spaces and replace with _
    Remove 's 
    Make actions readable for YAML"""



    action_id = node_text.strip().lower()
    action_id = action_id.replace(' ', '_').replace("'", "")
    action_id = ''.join(c for c in action_id if c.isalnum() or c == '_')


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


def make_gateway_id(nodes, rid, node):
    return f"gateway_{node['type'].lower()}_{list(nodes.keys()).index(rid)}"


ACTIONABLE_TYPES = {'Activity', 'StartNode', 'EndNode'}

GATEWAY_TYPE_MAP = {
    'XOR': 'exclusive',
    'AND': 'parallel',
    'OR': 'inclusive'
}

#"sid-A151...": "check_inventory"
def build_rid_to_id(nodes):
    seen_ids = set()
    rid_to_id = {}

    for rid, node in nodes.items():
        #we dont do it for the gateaways
        if node['type'] in ACTIONABLE_TYPES and node['NodeText'].strip():
            rid_to_id[rid] = make_action_id(node['NodeText'], seen_ids)
    return rid_to_id


def extract_actions(nodes, outgoing, incoming, rid_to_id):
    actions = []
    for rid, node in nodes.items():
        if node['type'] not in ACTIONABLE_TYPES or not node['NodeText'].strip():
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
            if src_node['type'] in ACTIONABLE_TYPES and src_node['NodeText'].strip():
                predecessors.append(rid_to_id[src])
            elif src_node['type'] in ('XOR', 'AND', 'OR'):
                predecessors.append(make_gateway_id(nodes, src, src_node))
            elif src_node['type'] == 'StartNode':
                #StartNode without text 
                predecessors.append("start")

        #we need to find what comes after so basiclaly everything that points FROM this action node
        #and we append what comes after depending on its node type
        successors = []
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue
            #if the target is an actionable node with text reference its action id
            if tgt_node['type'] in ACTIONABLE_TYPES and tgt_node['NodeText'].strip():
                successors.append(rid_to_id[tgt])
            elif tgt_node['type'] in ('XOR', 'AND', 'OR'):
                successors.append(make_gateway_id(nodes, tgt, tgt_node))
            #unnamed EndNodes are termination


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

    return actions


def extract_gateways(nodes, outgoing, incoming, rid_to_id):
    gateways = []
    for rid, node in nodes.items():
        if node['type'] not in ('XOR', 'AND', 'OR'):
            continue

        gateway_id = make_gateway_id(nodes, rid, node)

        #for each outgoing edge from gateway we create a branch
        #each branch has next (where it goes) and optional condition
        branches = []
        for tgt, cond in outgoing.get(rid, []):
            tgt_node = nodes.get(tgt)
            if not tgt_node:
                continue

            if tgt in rid_to_id:
                next_id = rid_to_id[tgt]
            elif tgt_node['type'] in ('XOR', 'AND', 'OR'):
                next_id = make_gateway_id(nodes, tgt, tgt_node)
            elif tgt_node['type'] == 'EndNode':
                #unnamed EndNode = termination. Use null to indicate process ends here
                #this preserves the branch condition (e.g. "Inventory Level Above Minimum")
                next_id = None
            else:
                next_id = tgt

            branch = {"next": next_id}
            if cond.strip():
                branch["condition"] = cond.strip()
            branches.append(branch)

        #same logic here for incoming
        incoming_from = []
        for src, cond in incoming.get(rid, []):
            src_node = nodes.get(src)
            if not src_node:
                continue
            if src in rid_to_id:
                incoming_from.append(rid_to_id[src])
            elif src_node['type'] in ('XOR', 'AND', 'OR'):
                incoming_from.append(make_gateway_id(nodes, src, src_node))

        #deduplicate while preserving order (e.g. two edges from same gateway)
        incoming_from = list(dict.fromkeys(incoming_from))


        in_degree = len(incoming.get(rid, []))
        out_degree = len(outgoing.get(rid, []))
        if in_degree <= 1 and out_degree > 1:
            role = "split"
        elif in_degree > 1 and out_degree <= 1:
            role = "merge"
            #maybe some gateaways are both merge and split? to avoid any issues anyways
        elif in_degree > 1 and out_degree > 1:
            role = "join_split"
            #just to avoid any issue
        else:
            role = "pass_through"

        gateway = {
            "id": gateway_id,
            "type": GATEWAY_TYPE_MAP[node['type']],
            "role": role,
            "incoming_from": incoming_from,
            "branches": branches,
        }
        if node['agent'].strip():
            gateway["actor"] = node['agent'].strip()

        gateways.append(gateway)

    return gateways
