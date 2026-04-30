#YAML version of validate_results.py  same metrics reads yaml files instead of json

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
import yaml
from difflib import SequenceMatcher
from pathlib import Path

#reuse all the matching logic from the json validator
from validate_results import (
    match_actions, validate_record,
)


#PyYAML defaults to YAML 1.1 and parses unquoted Yes/No/On/Off/True/False as booleans.
#Branch conditions like "Yes"/"No"/"Approved" must stay as strings, so we strip the
#bool resolver from a SafeLoader subclass for any read of the predictions file.
class _SafeLoaderNoBool(yaml.SafeLoader):
    pass

_SafeLoaderNoBool.yaml_implicit_resolvers = {
    k: [(tag, regex) for tag, regex in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def main():
    parser = argparse.ArgumentParser(description="Validate YAML extraction predictions against YAML ground truth")
    _here = Path(__file__).parent
    parser.add_argument("--predictions", type=Path,
                        default=_here / "extraction_predictions_yaml.yaml",
                        help="Path to predictions YAML")
    parser.add_argument("--gt", type=Path,
                        default=_here.parent / "Data" / "Processed" / "extracted_test.json",
                        help="Path to ground truth JSON (same as json validator)")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = yaml.load(f, Loader=_SafeLoaderNoBool) or []

    #keep only the same file_indices as the json extraction so the comparison is fair
    #the json predictions already had the holdout removed and the filter applied
    import json
    json_preds_path = _here / "extraction_predictions.json"
    if json_preds_path.exists():
        with open(json_preds_path, encoding="utf-8") as f:
            json_indices = {r["file_index"] for r in json.load(f)}
        before_align = len(predictions)
        predictions = [r for r in predictions if r["file_index"] in json_indices]
        print(f"Aligned to JSON predictions — {before_align} → {len(predictions)} (matched {len(json_indices)} json file_indices)")

    #same filter as json version
    total_before = len(predictions)
    removed = []
    kept = []
    for r in predictions:
        n = len(r.get("execution_states", []))
        if r.get("workflow") is None:
            removed.append((r["file_index"], "null workflow"))
        elif n == 0:
            removed.append((r["file_index"], "0 execution states"))
        elif n > 100:
            removed.append((r["file_index"], f"too complex ({n} execution states)"))
        else:
            kept.append(r)
    predictions = kept
    null_wf = sum(1 for _, r in removed if r == "null workflow")
    zero_states = sum(1 for _, r in removed if r == "0 execution states")
    too_complex = len(removed) - null_wf - zero_states
    print(f"Filter step — Total: {total_before} | Removed: {len(removed)} | Kept: {len(kept)}")
    print(f"  Failed extractions (null workflow): {null_wf}")
    print(f"  Failed traces (0 execution states): {zero_states}")
    print(f"  Too complex (>100 execution states): {too_complex}")
    if removed:
        for file_index, reason in removed:
            print(f"  file_index={file_index}  ({reason})")
    with open(args.predictions, "w", encoding="utf-8") as f:
        yaml.dump(kept, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print()

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

    #per-complexity buckets: linear (0 gateways), simple (1 gateway), complex (2+ gateways)
    #we group by ground truth gateway count so the categories are stable across runs
    #3 complexity buckets based on ground truth gateway count
    complexity_buckets = {
        "linear (0 gw)": [],
        "simple (1 gw)": [],
        "complex (2+ gw)": [],
    }

    def _bucket_name(gt_gw_count):
        if gt_gw_count == 0:
            return "linear (0 gw)"
        if gt_gw_count == 1:
            return "simple (1 gw)"
        return "complex (2+ gw)"

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

        gt_gw_count = len(gt_rec["workflow"].get("gateways", []))
        complexity_buckets[_bucket_name(gt_gw_count)].append(m)


    print()
    print(f"--- AVERAGES ({n} records) ---")
    print(f"Action Precision:   {totals['action_precision']/n:.3f}")
    print(f"Action Recall:      {totals['action_recall']/n:.3f}")
    print(f"Action F1:          {totals['action_f1']/n:.3f}  (avg FN={totals['action_fn']/n:.1f}, avg FP={totals['action_fp']/n:.1f})")
    print(f"Edge F1:            {totals['edge_f1']/n:.3f}")

    print(f"Gateway F1:         {totals['gateway_f1']/n:.3f}  (avg FN={totals['gateway_fn']/n:.1f}, avg FP={totals['gateway_fp']/n:.1f})")
    print(f"Gateway type acc:   {totals['gateway_type_acc']/n:.3f}")

    print(f"Branch tuple F1:    {totals['branch_tuple_f1']/n:.3f}")

    total_completion = sum(p.get("completion_tokens", 0) for p in predictions)
    print(f"Avg completion tokens: {total_completion // n if n else 0:,}")

    #per-complexity breakdown so we can see where the extractor struggles
    print()
    print(f"--- BY COMPLEXITY ---")
    for bucket_name, bucket_metrics in complexity_buckets.items():
        bn = len(bucket_metrics)
        if bn == 0:
            print(f"  {bucket_name:20s}  (no records)")
            continue
        avg = {k: sum(m[k] for m in bucket_metrics) / bn for k in metric_keys}
        print(f"  {bucket_name:20s}  n={bn:3d}  "
              f"act_f1={avg['action_f1']:.3f}  edge_f1={avg['edge_f1']:.3f}  "
              f"gw_f1={avg['gateway_f1']:.3f}  gw_type={avg['gateway_type_acc']:.3f}  "
              f"branch_f1={avg['branch_tuple_f1']:.3f}")


if __name__ == "__main__":
    main()
