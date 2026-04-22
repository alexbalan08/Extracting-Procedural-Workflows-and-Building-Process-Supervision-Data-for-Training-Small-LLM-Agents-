#runs the validator on all 5 experiment configs and prints a side by side comparison
#each config points to its own output file saved by extract_procedure with --output flag

import json
from pathlib import Path

from validate_results import validate_record


_here = Path(__file__).parent
_gt_path = _here.parent / "Data" / "Processed" / "extracted_test.json"

#map each config label to its output file
EXPERIMENTS = {
    "exp1 gpt alone":        _here / "exp1_gpt_alone.json",
    "exp2 critic":           _here / "exp2_critic.json",
    "exp3 critic + rag1":    _here / "exp3_critic_rag1.json",
    "exp4 critic2 + rag1":   _here / "exp4_critic2_rag1.json",
    "exp5 critic2 + rag2":   _here / "exp5_critic2_rag2.json",
    "exp6 critic2 + rag3":   _here / "exp6_critic2_rag3.json",
    "exp7 critic3 + rag1":   _here / "exp7_critic3_rag1.json",
}


def filter_predictions(preds):
    #same filter as the main validator — drop null workflows 0 exec states and over 100
    kept = []
    failed_null = failed_zero = failed_big = 0
    for r in preds:
        n = len(r.get("execution_states", []))
        if r.get("workflow") is None:
            failed_null += 1
        elif n == 0:
            failed_zero += 1
        elif n > 100:
            failed_big += 1
        else:
            kept.append(r)
    return kept, failed_null, failed_zero, failed_big


def run_experiment(name, path, gt_by_idx):
    if not path.exists():
        return {"name": name, "missing": True}

    with open(path, encoding="utf-8") as f:
        preds = json.load(f)

    kept, f_null, f_zero, f_big = filter_predictions(preds)

    metric_keys = ["action_f1", "edge_f1", "gateway_f1", "gateway_type_acc", "branch_tuple_f1"]
    totals = {k: 0.0 for k in metric_keys}
    n = 0
    total_completion = 0

    for pred in kept:
        fidx = pred["file_index"]
        gt_rec = gt_by_idx.get(fidx)
        if gt_rec is None or pred.get("workflow") is None:
            continue
        m = validate_record(gt_rec["workflow"], pred["workflow"])
        n += 1
        for k in metric_keys:
            totals[k] += m[k]
        total_completion += pred.get("completion_tokens", 0)

    result = {
        "name": name,
        "total": len(preds),
        "failed_null": f_null,
        "failed_zero": f_zero,
        "failed_big": f_big,
        "n_validated": n,
        "avg_completion_tokens": total_completion // max(n, 1),
    }
    for k in metric_keys:
        result[k] = totals[k] / n if n else 0.0
    return result


def main():
    with open(_gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    gt_by_idx = {r["file_index"]: r for r in ground_truth}

    results = []
    for name, path in EXPERIMENTS.items():
        print(f"Validating {name} ...")
        results.append(run_experiment(name, path, gt_by_idx))

    #print side by side table
    print()
    print("=" * 80)
    print(f"{'Config':<22s} {'act_f1':>8s} {'edge_f1':>8s} {'gw_f1':>7s} {'gw_type':>8s} {'branch_f1':>10s}")
    print("-" * 80)
    for r in results:
        if r.get("missing"):
            print(f"{r['name']:<22s} (file missing)")
            continue
        print(f"{r['name']:<22s} "
              f"{r['action_f1']:>8.3f} "
              f"{r['edge_f1']:>8.3f} "
              f"{r['gateway_f1']:>7.3f} "
              f"{r['gateway_type_acc']:>8.3f} "
              f"{r['branch_tuple_f1']:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
