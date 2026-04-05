"""
METRICS we use 

Action F1 measures how well the predicted actions match the ground truth actions.
    Two actions match if they share at least one nonstopword (ROUGE-1 > 0) and then
    Jaccard score is used as a tiebreaker when multiple candidates overlap which will be the case in practice
    in most of the cases.

Edge F1
    Measures how well the predicted flow matches ground truth flow graph 
    Edges are action  pairs (A then B), but without gateways so that A then gateway then B
    becomes edge (A, B).
    This avoids penalising a missing gateway twice (once in gateway F1,
    once in edge F1), as we ve seen that missing gateways is one of the biggest issues.

Gateway counts F1
    Measures whether the right number of gateways were extracted.

Gateway Type Accuracy
    In the gateways that were matched, what fraction have the correct type
    (exclusive / parallel / inclusive), BUT ONLY if they re stored in the same order
    as GT so this also basically will check for order extraction of the gqatways that were 
    extratced. The ones missing are not included, neither ones invented, but that s why we have 
    Gateway counts to see those numbers exaclty.

Branch Tuple F1
    Each branch is a tuple (gateway_id, condition, next_action).
    Gateway IDs must match exactly, conditions are matched with Jaccard >= 0.3, and next-action IDs
    are matched the same with rouge 1 and Jacardi. This will capture errors in branch conditions
    and wrong routing.
"""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STOPWORDS = {"the", "a", "an", "it", "its", "is", "are", "was", "were",
             "be", "been", "to", "of", "in", "on", "at", "for", "and", "or"}


def normalize(text):
    return " ".join(text.strip().lower().split()).rstrip(";").rstrip(".")


def word_set(text):
    words = set(normalize(text).split())
    return words - STOPWORDS


def char_similarity(a, b):
    """Character-level similarity (fallback for typos / morphological variants)."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def fuzzy_name_match(name_a, name_b, char_threshold=0.6):
    """Match two action names: word overlap first, then character-level fallback."""
    wa, wb = word_set(name_a), word_set(name_b)
    if wa and wb and len(wa & wb) >= 1:
        return True
    return char_similarity(name_a, name_b) >= char_threshold


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# action matching (fuzzy by word overlap)
# ---------------------------------------------------------------------------

def match_actions(gt_actions, pred_actions):
    """Match GT actions to predicted actions.
    Uses word overlap (Jaccard) first, falls back to character-level similarity
    for typos and morphological variants.
    Returns (tp, fn, fp, mapping) where mapping is {gt_idx: pred_idx}."""
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
            # character-level fallback (typos, morphological variants)
            score = char_similarity(gt_a["name"], pred_a["name"])
            if score >= 0.6 and score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            mapping[i] = best_j
            matched_pred.add(best_j)

    tp = len(mapping)
    fn = len(gt_actions) - tp
    fp = len(pred_actions) - tp
    return tp, fn, fp, mapping


# ---------------------------------------------------------------------------
# per-record validation
# ---------------------------------------------------------------------------

def validate_record(gt_workflow, pred_workflow):
    gt_w = gt_workflow
    pred_w = pred_workflow
    metrics = {}

    gt_actions = gt_w.get("actions", [])
    pred_actions = pred_w.get("actions", [])
    gt_gateways = gt_w.get("gateways", [])
    pred_gateways = pred_w.get("gateways", [])

    # ---- actions ----
    tp, fn, fp, action_map = match_actions(gt_actions, pred_actions)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["action_f1"] = f1(p, r)
    metrics["action_precision"] = p
    metrics["action_recall"] = r
    metrics["action_fn"] = fn
    metrics["action_fp"] = fp

    # build id mapping from matched actions: gt_id -> pred_id
    id_map = {}
    for gi, pi in action_map.items():
        id_map[gt_actions[gi]["id"]] = pred_actions[pi]["id"]

    # ---- successor edges (flow) ----
    # flatten through gateway intermediaries: compare action-to-action reachability
    # so A → gateway → B becomes edge (A, B), avoiding penalising missing/extra gateways twice

    def get_action_edges(actions, gateways):
        gw_branches = {g["id"]: g.get("branches", []) for g in gateways}
        action_ids = {a["id"] for a in actions}

        def reach_actions(node_id, visited=frozenset()):
            if node_id in visited:
                return set()
            visited = visited | {node_id}
            if node_id in action_ids:
                return {node_id}
            result = set()
            for b in gw_branches.get(node_id, []):
                nxt = b.get("next")
                if nxt:
                    result |= reach_actions(nxt, visited)
            return result

        edges = set()
        for a in actions:
            for s in a.get("successors", []):
                for t in reach_actions(s):
                    edges.add((a["id"], t))
        return edges

    gt_edges_raw = get_action_edges(gt_actions, gt_gateways)
    pred_edges_raw = get_action_edges(pred_actions, pred_gateways)

    # translate GT action IDs to pred IDs for comparison
    gt_edges = {(id_map.get(u, u), id_map.get(v, v)) for u, v in gt_edges_raw}
    pred_edges = pred_edges_raw

    if not gt_edges and not pred_edges:
        metrics["edge_f1"] = 1.0
    else:
        e_tp = len(gt_edges & pred_edges)
        e_p = e_tp / len(pred_edges) if pred_edges else 0.0
        e_r = e_tp / len(gt_edges) if gt_edges else 0.0
        metrics["edge_f1"] = f1(e_p, e_r)

    # ---- gateways ----
    gt_gw_count = len(gt_gateways)
    pred_gw_count = len(pred_gateways)

    # match gateways by index order
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

    # build gateway id mapping from index-order match
    gw_id_map = {}
    for i in range(compared):
        gw_id_map[gt_gateways[i]["id"]] = pred_gateways[i]["id"]

    # combined id map for translating any reference (action or gateway)
    all_id_map = {**id_map, **gw_id_map}

    # ---- branch tuples: (gateway_id, condition, next) ----
    # condition matching is fuzzy (Jaccard 0.3) to handle paraphrasing

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
        matched_pred = set()
        tp = 0
        for gt_gid, gt_cond, gt_nxt in gt_list:
            for j, (pred_gid, pred_cond, pred_nxt) in enumerate(pred_list):
                if j in matched_pred:
                    continue
                if gt_gid != pred_gid:
                    continue

                # next: exact ID match first, then fuzzy name match
                if gt_nxt == pred_nxt:
                    next_ok = True
                elif gt_nxt is None or pred_nxt is None:
                    next_ok = False
                else:
                    # gt_nxt may already be translated to pred ID space (matched actions)
                    # or still in GT ID space (unmatched) — look up name from either side
                    gt_name = pred_names.get(gt_nxt) or gt_names.get(gt_nxt, gt_nxt)
                    pred_name = pred_names.get(pred_nxt, pred_nxt)
                    # word overlap first, character-level fallback for typos
                    next_ok = fuzzy_name_match(gt_name, pred_name)

                if not next_ok:
                    continue

                # condition: fuzzy match
                gt_w = word_set(gt_cond)
                pred_w = word_set(pred_cond)
                if not gt_w and not pred_w:
                    cond_ok = True
                elif not gt_w or not pred_w:
                    cond_ok = False
                else:
                    cond_ok = len(gt_w & pred_w) / len(gt_w | pred_w) >= cond_threshold

                if cond_ok:
                    tp += 1
                    matched_pred.add(j)
                    break
        return tp

    # name lookups needed for fuzzy next-action matching
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

    # gateway F1 based on count
    gw_tp = min(gt_gw_count, pred_gw_count)
    gw_p = gw_tp / pred_gw_count if pred_gw_count else (1.0 if gt_gw_count == 0 else 0.0)
    gw_r = gw_tp / gt_gw_count if gt_gw_count else (1.0 if pred_gw_count == 0 else 0.0)
    metrics["gateway_f1"] = f1(gw_p, gw_r)

    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
    # compares action-to-action edges flattened through gateways (A→gw→B becomes edge A→B)
    # GT IDs translated to pred IDs via fuzzy action match before comparison
    print(f"Gateway F1:         {totals['gateway_f1']/n:.3f}  (avg FN={totals['gateway_fn']/n:.1f}, avg FP={totals['gateway_fp']/n:.1f})")
    print(f"Gateway type acc:   {totals['gateway_type_acc']/n:.3f}")

    print(f"Branch tuple F1:    {totals['branch_tuple_f1']/n:.3f}")


if __name__ == "__main__":
    main()
