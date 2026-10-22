"""Freeze the previously checked serial commands into a dependency graph."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    reference = ROOT / "reports/serial_commands_reference.jsonl"
    calls = [json.loads(line) for line in reference.read_text().splitlines()]
    calls = [[a for a in call if a != "-u"] for call in calls]
    old_source = calls[1][0].split("/experiments/historical_scripts/")[0]
    old_run = calls[1][calls[1].index("--data-dir") + 1].removesuffix("/data")
    definitions = {
        1: ("train_model6", [], "model6_unetpp_resnet34/best.pt"),
        2: ("train_model62", [1], "model62_ft_weighted_safeaug/best.pt"),
        3: ("train_model97", [2], "model97_native_patch/best.pt"),
        4: ("train_model98", [3], "model98_hard_negative/best.pt"),
        5: ("train_model109", [4], "model109_aspp_native/best.pt"),
        6: ("train_restart51", [5], "restart51_aspp_768_finetune/best.pt"),
        7: ("train_restart59", [5], "restart59_aspp768_seed456/best.pt"),
        8: ("train_restart61", [6], "restart61_pairwise_margin/best.pt"),
        9: ("train_model123", [5], "model123_ema/best.pt"),
        10: ("train_model181", [5], "model181_raw_zcdf_aspp/best.pt"),
        11: ("infer_route47", [3, 4, 5], "restart47_router/cache"),
        12: ("build_model183_cache", [11, 10, 8, 6, 5, 7, 9], "model183_metric_calibration/entries.pkl"),
        13: ("train_model194", [5], "model194_multistage_seed2028/best.pt"),
        15: ("train_model195", [13], "model195_native_resume/valid_probs.npz"),
        16: ("infer_model220_tent", [15], "model220_native_tent_extended/valid_probs_06.npz"),
        17: ("train_model239", [15], "model239_native_stem/best.pt"),
        18: ("train_model242", [17], "model242_native_stem_ema/best_ema.pt"),
        19: ("train_model257", [18], "model257_full_image_stats_1x/best.pt"),
        20: ("train_model269", [19], "model269_fullstats_1x_extended/valid_probs.npz"),
        51: ("train_model336", [], "model336_hrnet_best/last.pt"),
        52: ("evaluate_model336", [51], "model336_hrnet_best/valid_eval/valid_probabilities.npz"),
        53: ("train_model347", [51], "model347_hrnet_inverse_ce_finetune/last.pt"),
        54: ("evaluate_model347", [53], "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz"),
        55: ("route_model335", [12, 15, 16, 20, 30, 40], "model335_exhaustive_b4_routes/results.json"),
        56: ("route_model338", [55], "model338_fine_gamma_route_search/results.json"),
        57: ("route_model339", [56], "model339_joint_gamma_route_search/results.json"),
        58: ("route_model341", [57, 50], "model341_model329_joint_route_search/results.json"),
        59: ("route_model346", [58, 52], "model346_model336_into_341_route/results.json"),
        60: ("route_model349", [59, 54], "model349_model347_into_346_route/results.json"),
    }
    families = [(21, 322, "model322_team_b4_group_oof"),
                (31, 326, "model326_hybrid_fullstats_group_oof"),
                (41, 329, "model329_fullstats_width4000_uniform_group_oof")]
    for start, model, out in families:
        for fold in range(1, 5):
            index = start + (fold - 1) * 2
            definitions[index] = (f"train_model{model}_fold{fold}", [], f"{out}/fold{fold}/last.pt")
            definitions[index + 1] = (f"oof_model{model}_fold{fold}", [index], f"{out}/fold{fold}_oof/probabilities.npz")
        definitions[start + 8] = (f"aggregate_model{model}", list(range(start + 1, start + 8, 2)), f"{out}/train_oof_probs.npz")
        definitions[start + 9] = (f"valid_model{model}", [start + 8], f"{out}/valid_ensemble/probabilities.npz")
    tasks = []
    for index, (name, deps, produces) in sorted(definitions.items()):
        argv = [a.replace(old_source, "{source}").replace(old_run, "{run}") for a in calls[index]]
        if index >= 55:
            argv[argv.index("--workers") + 1] = "{workers}"
        kind = "cpu" if name.startswith(("aggregate_", "route_model")) else "gpu"
        priority = 0 if index == 1 else 1 if index == 51 else 100 if "_fold" in name and name.startswith("train_") else 2
        # Input checkpoint/cache paths are checked immediately before launching.
        output_options = {"--out", "--output-dir", "--output-json", "--output"}
        requires = []
        for j, value in enumerate(argv):
            if value.startswith("{run}/outputs/") and (j == 0 or argv[j - 1] not in output_options):
                if "--runs" in argv and j == argv.index("--runs") + 1 and name.startswith("aggregate_"):
                    continue
                requires.append(value)
        tasks.append({"id": name, "serial_index": index, "kind": kind, "priority": priority,
                      "deps": [definitions[d][0] for d in deps], "argv": argv,
                      "requires": requires, "produces": ["{run}/outputs/" + produces]})
    assert len(tasks) == 59
    assert {t["serial_index"] for t in tasks} == set(range(1, 61)) - {14}
    owners = []
    by_id = {t["id"]: t for t in tasks}
    for task in tasks:
        argv = task["argv"]
        for option in ("--out", "--output-dir", "--output-json", "--output"):
            if option in argv:
                destination = argv[argv.index(option) + 1]
                if destination.endswith(".json"):
                    destination = str(Path(destination).parent)
                owners.append((destination, task["id"]))
    def ancestors(name):
        deps = set(by_id[name]["deps"])
        return deps | set().union(*(ancestors(dep) for dep in deps))
    for task in tasks:
        for required in task["requires"]:
            matching = [(prefix, owner) for prefix, owner in owners
                        if required == prefix or required.startswith(prefix + "/")]
            if not matching:
                raise ValueError(f"No producer for {task['id']}: {required}")
            prefix, owner = max(matching, key=lambda pair: len(pair[0]))
            if owner not in ancestors(task["id"]):
                raise ValueError(f"Missing dependency: {task['id']} requires {owner}")
            if required not in by_id[owner]["produces"]:
                by_id[owner]["produces"].append(required)
    recipe = {"description": "Model349 unchanged per-model recipe; dependency-aware multi-GPU scheduling",
              "serial_reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "prepare_cv_argv": [a.replace(old_source, "{source}").replace(old_run, "{run}") for a in calls[14]],
              "tasks": tasks}
    (ROOT / "pipeline_tasks.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"Wrote {len(tasks)} tasks: {sum(t['kind'] == 'gpu' for t in tasks)} GPU / {sum(t['kind'] == 'cpu' for t in tasks)} CPU")


if __name__ == "__main__":
    main()
