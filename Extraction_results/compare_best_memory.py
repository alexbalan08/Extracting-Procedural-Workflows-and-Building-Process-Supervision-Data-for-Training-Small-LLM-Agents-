#comparison of the winning config (exp4) with and without memory
#we see the effect of memeory

import json
from pathlib import Path

from compare_experiments import run_experiment


_here = Path(__file__).parent
_gt_path = _here.parent / "Data" / "Processed" / "extracted_test.json"

PAIR = [
    ("exp4 critic2+rag1 mem",    "exp4_critic2_rag1.json"),
    ("exp4 critic2+rag1 no_mem", "exp4_nomem.json"),
]


def main():
    with open(_gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    gt_by_idx = {r["file_index"]: r for r in ground_truth}

    print()
    print("=" * 90)
    print(f"{'Config':<28s} {'act_f1':>8s} {'edge_f1':>8s} {'gw_f1':>7s} {'gw_type':>8s} {'branch_f1':>10s}")
    print("-" * 90)
    for label, fname in PAIR:
        r = run_experiment(label, _here / fname, gt_by_idx)
        if r.get("missing"):
            print(f"{label:<28s} (file missing)")
            continue
        print(f"{label:<28s} "
              f"{r['action_f1']:>8.3f} "
              f"{r['edge_f1']:>8.3f} "
              f"{r['gateway_f1']:>7.3f} "
              f"{r['gateway_type_acc']:>8.3f} "
              f"{r['branch_tuple_f1']:>10.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
