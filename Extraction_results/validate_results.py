#METRICS we use 
#witch action F1 we see how well the predicted actions match the ground truth actions
#two actions match if they share at least one nonstopword (ROUGE-1 > 0) and then we sort all cadidates by Jaccardi score and always pick the highest. This works quite well
#we also consider cases such as gt: "processing", extraction: "process" and we simply catch those if the previous method fails and this is edit distcnace in characters bascially
#and we need it over 0.6

#Edge F1 measures how well the predicted flow matches ground truth flow graph 
#Edges are action  pairs A then B, but without gateways so that A then gateway then B
#this avoids penalising a missing gateway twice as we ve seen that missing gateways is one of the biggest issues

#Gateway counts F1 measures if the right number of gateways were extracted. for the type and order i defined another metric

#Gateway type accuracy measures for the gateways that were matched what fraction have the correct type
#exclusive  parallel  inclusive, BUT ONLY if they re stored in the same order as GT so this also basically will check for order extraction of the gqatways that were 
#extratced. 
#The ones missing are not included, neither ones invented, but that s why we have gateway counts to see those numbers exaclty

#Branch tuple F1 where each branch is a tuple (gateway_id, condition, next_action) that we have in reaoning traces so with this we basicallly make nsure the reasoning traces are correct
#Gateway IDs must match exactly, conditions are matched with jaccard bigger than 0.3 or edit distance and next-action IDs
#are matched the same with rouge 1 and Jacardi. This metric will capture any routing errors and it will be lower since we see that we quite miss some gateways. 

#then if all those metrics are good, then the reasoning traces will be correct as they re built deterministic from the graph extraction

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


#i included common verbs here as well which i ve seen they appear quite a lot in actions ids
#things like "process request" and "processing the request" should obviously match
STOPWORDS = {"the", "a", "an", "it", "its", "is", "are", "was", "were",
             "be", "been", "to", "of", "in", "on", "at", "for", "and", "or"}

def normalize(text):
    return " ".join(text.strip().lower().split()).rstrip(";").rstrip(".")

def word_set(text):
    words = set(normalize(text).split())
    return words - STOPWORDS


#character level similarity using python s built in SequenceMatcher which is basically edit distance ratio
#"recieve" vs "receive" this is useful becasue many action ids have spelling mistakes in the dataset...
def char_similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


#first tries word overlap then falls back to character similarity
#example char match where gt="initialise" pred="initialize" no word overlap but char_similarity around 0.90
def fuzzy_name_match(name_a, name_b, char_threshold=0.6):
    wa, wb = word_set(name_a), word_set(name_b)
    if wa and wb and len(wa & wb) >= 1:
        return True
    return char_similarity(name_a, name_b) >= char_threshold



def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



def match_actions(gt_actions, pred_actions):
    matched_pred = set()
    mapping = {}

    for i, gt_a in enumerate(gt_actions):
        best_j, best_score = -1, 0.0
        for j, pred_a in enumerate(pred_actions):
            if j in matched_pred:
                continue
            gt_words = word_set(gt_a["name"])
            pred_words = word_set(pred_a["name"])
            # word overlap (Jaccard)
            if gt_words and pred_words:
                shared = len(gt_words & pred_words)
                if shared > 0:
                    score = shared / len(gt_words | pred_words)
                    if score > best_score:
                        best_score = score
                        best_j = j
                    continue
            #character fallback 
            score = char_similarity(gt_a["name"], pred_a["name"])
            if score >= 0.6 and score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            mapping[i] = best_j
            matched_pred.add(best_j)
            #always pick the best action with highest jacard

    tp = len(mapping)
    fn = len(gt_actions) - tp
    fp = len(pred_actions) - tp
    return tp, fn, fp, mapping


#main validation function which takes one gt workflow dict and one predicted workflow dict with the matching file index and returns all metrics
def validate_record(gt_workflow, pred_workflow):
    gt_w = gt_workflow
    pred_w = pred_workflow
    metrics = {}

    gt_actions = gt_w.get("actions", [])
    pred_actions = pred_w.get("actions", [])
    gt_gateways = gt_w.get("gateways", [])
    pred_gateways = pred_w.get("gateways", [])

    #we match the gt actions names to predicitons actions nnames so we can compare them 
    tp, fn, fp, action_map = match_actions(gt_actions, pred_actions)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["action_f1"] = f1(p, r)
    metrics["action_precision"] = p
    metrics["action_recall"] = r
    metrics["action_fn"] = fn
    metrics["action_fp"] = fp


    #this is necessary because the extractor invents its own IDs like "action_1", "action_2" etc
    id_map = {}
    for gi, pi in action_map.items():
        id_map[gt_actions[gi]["id"]] = pred_actions[pi]["id"]




    #so if the successor of action A is a gateway, we keep walking through that gateway s branches until we hit actions
    def get_action_edges(actions, gateways):
        gw_branches = {g["id"]: g.get("branches", []) for g in gateways}
        action_ids = {a["id"] for a in actions}

        def reach_actions(node_id, visited=frozenset()):
            #visited set prevents infinite loops if there are cycles in the graph which can happen with badly extracted loops
            if node_id in visited:
                return set()
            visited = visited | {node_id}
            if node_id in action_ids:
                return {node_id}
            result = set()
            for b in gw_branches.get(node_id, []):
                nxt = b.get("next")
                if nxt:
                    result = reach_actions(nxt, visited)
            return result

        edges = set()
        for a in actions:
            for s in a.get("successors", []):
                for t in reach_actions(s):
                    edges.add((a["id"], t))
        return edges

    gt_edges_raw = get_action_edges(gt_actions, gt_gateways)
    pred_edges_raw = get_action_edges(pred_actions, pred_gateways)


    gt_edges = {(id_map.get(u, u), id_map.get(v, v)) for u, v in gt_edges_raw}
    pred_edges = pred_edges_raw

    
    if not gt_edges and not pred_edges:
        metrics["edge_f1"] = 1.0
    else:
        e_tp = len(gt_edges & pred_edges)
        e_p = e_tp / len(pred_edges) if pred_edges else 0.0
        e_r = e_tp / len(gt_edges) if gt_edges else 0.0
        metrics["edge_f1"] = f1(e_p, e_r)


    #we match gateways by position in the list so gateway 0 in gt is compared to gateway 0 in pred
    #the order of gateways in the workflow reflects execution order and we want to penalise wrong order
    #if pred has fewer gateways than gt the extra gt ones are just false negatives and vice versa
    gt_gw_count = len(gt_gateways)
    pred_gw_count = len(pred_gateways)


    compared = min(gt_gw_count, pred_gw_count)
    type_correct = sum(
        1 for i in range(compared)
        if gt_gateways[i]["type"] == pred_gateways[i]["type"]
    )
    metrics["gateway_count_gt"] = gt_gw_count
    metrics["gateway_count_pred"] = pred_gw_count
    metrics["gateway_fn"] = max(0, gt_gw_count - pred_gw_count)
    metrics["gateway_fp"] = max(0, pred_gw_count - gt_gw_count)
    metrics["gateway_type_acc"] = type_correct / compared if compared else 1.0


    #gateway "gw_1" at index 0, pred gateway "gateway_0" also at index 0 so they get mapped
    gw_id_map = {}
    for i in range(compared):
        gw_id_map[gt_gateways[i]["id"]] = pred_gateways[i]["id"]


    all_id_map = {**id_map, **gw_id_map}

   
    #we translate all gt IDs to pred ID space before matching so comparisons are fair
    #condition matching uses jaccard with threshold 0.3 because conditions are often paraphrased
    def build_branch_list(gateways, translate=None):

        tuples = []
        for g in gateways:
            gid = translate.get(g["id"], g["id"]) if translate else g["id"]
            for b in g.get("branches", []):
                cond = normalize(b.get("condition", "") or "")
                nxt = b.get("next")
                if translate and nxt:
                    nxt = translate.get(nxt, nxt)
                tuples.append((gid, cond, nxt))
        return tuples

    def match_branch_tuples(gt_list, pred_list, gt_names, pred_names,
                            cond_threshold=0.3):
        #gateway id must match exactly, then we check next node and condition
        #we use 0.3 jaccard for conditions because theyre often short phrases and paraphrasing is common
        matched_pred = set()
        tp = 0
        for gt_gid, gt_cond, gt_nxt in gt_list:
            for j, (pred_gid, pred_cond, pred_nxt) in enumerate(pred_list):
                if j in matched_pred:
                    continue
                if gt_gid != pred_gid:
                    continue


                if gt_nxt == pred_nxt:
                    next_ok = True
                elif gt_nxt is None or pred_nxt is None:
                    next_ok = False
                else:
                    #gt_nxt may have been translated to pred ID space if the action was matched
                    #but if the action was unmatched it s still in gt ID space so we look up the name from both sides
                    gt_name = pred_names.get(gt_nxt) or gt_names.get(gt_nxt, gt_nxt)
                    pred_name = pred_names.get(pred_nxt, pred_nxt)
                    # word overlap first, character-level fallback for typos
                    next_ok = fuzzy_name_match(gt_name, pred_name)

                if not next_ok:
                    continue


                gt_w = word_set(gt_cond)
                pred_w = word_set(pred_cond)
                #if both conditions are empty (unconditional branch) thats a match
                if not gt_w and not pred_w:
                    cond_ok = True
                #if only one side is empty they re describing the branch differently -> no match
                elif not gt_w or not pred_w:
                    cond_ok = False
                else:
                    jaccard = len(gt_w & pred_w) / len(gt_w | pred_w)
                    cond_ok = jaccard >= cond_threshold or char_similarity(gt_cond, pred_cond) >= 0.5 #0.5 because they can get longer than actions and 0.6 would be too strict
                    #for conditions we fallback also to edit distance because we can have "approval" ad "approved"

                if cond_ok:
                    tp += 1
                    matched_pred.add(j)
                    break
        return tp


    gt_action_names = {a["id"]: a.get("name", "") for a in gt_actions}
    pred_action_names = {a["id"]: a.get("name", "") for a in pred_actions}

    gt_tuples = build_branch_list(gt_gateways, translate=all_id_map)
    pred_tuples = build_branch_list(pred_gateways)


    if not gt_tuples and not pred_tuples:
        metrics["branch_tuple_f1"] = 1.0
    else:
        bt_tp = match_branch_tuples(gt_tuples, pred_tuples, gt_action_names, pred_action_names)
        bt_p = bt_tp / len(pred_tuples) if pred_tuples else 0.0
        bt_r = bt_tp / len(gt_tuples) if gt_tuples else 0.0
        metrics["branch_tuple_f1"] = f1(bt_p, bt_r)


    gw_tp = min(gt_gw_count, pred_gw_count)
    gw_p = gw_tp / pred_gw_count if pred_gw_count else (1.0 if gt_gw_count == 0 else 0.0)
    gw_r = gw_tp / gt_gw_count if gt_gw_count else (1.0 if pred_gw_count == 0 else 0.0)
    metrics["gateway_f1"] = f1(gw_p, gw_r)

    return metrics



def main():
    parser = argparse.ArgumentParser(description="Validate extraction predictions against ground truth")
    _here = Path(__file__).parent
    parser.add_argument("--predictions", type=Path,
                        default=_here / "extraction_predictions.json",
                        help="Path to predictions JSON")
    parser.add_argument("--gt", type=Path,
                        default=_here.parent / "Data" / "Processed" / "extracted_test.json",
                        help="Path to ground truth JSON (default: extracted_test.json)")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(args.gt, encoding="utf-8") as f:
        ground_truth = json.load(f)


    gt_by_idx = {r["file_index"]: r for r in ground_truth}

    metric_keys = [
        "action_f1", "action_precision", "action_recall", "action_fn", "action_fp",
        "edge_f1",
        "gateway_f1", "gateway_type_acc", "gateway_fn", "gateway_fp",
        "branch_tuple_f1",
    ]
    totals = {k: 0.0 for k in metric_keys}
    n = 0

    print(f"Predictions: {args.predictions}")
    print(f"Ground truth: {args.gt}")
    print()

    for pred in predictions:
        fidx = pred["file_index"]
        gt_rec = gt_by_idx.get(fidx)
        if gt_rec is None:
            print(f"  WARNING: file_index={fidx} not found in GT — skipping")
            continue
        if pred.get("workflow") is None:
            #null workflow means the extractor gave up or returned unparseable json, we skip but dont crash
            print(f"  WARNING: file_index={fidx} has null workflow — skipping")
            continue

        m = validate_record(gt_rec["workflow"], pred["workflow"])
        n += 1

        for k in metric_keys:
            totals[k] += m[k]


        #print(f"file_index={fidx}  "
              #f"act_f1={m['action_f1']:.2f} (fn={m['action_fn']} fp={m['action_fp']})  "
              #f"edge_f1={m['edge_f1']:.2f}  "
              #f"gw_f1={m['gateway_f1']:.2f} (fn={m['gateway_fn']} fp={m['gateway_fp']})  "
              #f"gw_type={m['gateway_type_acc']:.2f}  "
              #f"branch_f1={m['branch_tuple_f1']:.2f}")


    print()
    print(f"--- AVERAGES ({n} records) ---")
    print(f"Action Precision:   {totals['action_precision']/n:.3f}")
    print(f"Action Recall:      {totals['action_recall']/n:.3f}")
    print(f"Action F1:          {totals['action_f1']/n:.3f}  (avg FN={totals['action_fn']/n:.1f}, avg FP={totals['action_fp']/n:.1f})")
    print(f"Edge F1:            {totals['edge_f1']/n:.3f}")

    print(f"Gateway F1:         {totals['gateway_f1']/n:.3f}  (avg FN={totals['gateway_fn']/n:.1f}, avg FP={totals['gateway_fp']/n:.1f})")
    print(f"Gateway type acc:   {totals['gateway_type_acc']/n:.3f}")

    print(f"Branch tuple F1:    {totals['branch_tuple_f1']/n:.3f}")


if __name__ == "__main__":
    main()