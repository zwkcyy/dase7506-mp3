#!/usr/bin/env python3
"""CPU-only, explicitly posthoc robustness analysis of all 2,040 formal calls.

    python tools/analyze_robustness.py --self-check
    python tools/analyze_robustness.py

Reads enhanced_analysis_plan.json first. Writes a NEW enhanced_analysis/ only;
never changes the original freezes, prepared inputs, raw logs or analyses.
Statistics use stdlib; the compact overview requires numpy and matplotlib.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNS = ("qwen3_main", "qwen3_time16", "qwen25_main")
PERSONAS = ("P01", "P02", "P03", "P04")
ACTIONS = ("study", "go_class", "exercise", "chat", "rest", "eat", "wander")
TARGETS = {"peer_signal": "study", "energy_control": "rest", "hunger_control": "eat"}
RUN_LABELS = {"qwen3_main": "Q3 11:00", "qwen3_time16": "Q3 16:00", "qwen25_main": "Q2.5 11:00"}


class RobustnessError(Exception):
    pass


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_original_analyzer():
    path = Path(__file__).with_name("analyze_experiment.py")
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("ak7506_original_analysis", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    finally:
        sys.dont_write_bytecode = previous


def binomial_sum(n, p, lower, upper):
    return math.fsum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(lower, upper + 1))


@lru_cache(maxsize=None)
def clopper_pearson(x, n, alpha=0.05):
    """Invert finite binomial tails; no beta/scipy approximation or dependency."""
    if type(x) is not int or type(n) is not int or n <= 0 or not 0 <= x <= n or not 0 < alpha < 1:
        raise RobustnessError("Invalid CP input")
    if x == 0:
        lower = 0.0
    else:
        left, right = 0.0, 1.0
        for _ in range(64):
            mid = (left + right) / 2
            if binomial_sum(n, mid, x, n) < alpha / 2:
                left = mid
            else:
                right = mid
        lower = (left + right) / 2
    if x == n:
        upper = 1.0
    else:
        left, right = 0.0, 1.0
        for _ in range(64):
            mid = (left + right) / 2
            if binomial_sum(n, mid, 0, x) > alpha / 2:
                left = mid
            else:
                right = mid
        upper = (left + right) / 2
    return [lower, upper]


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def bootstrap_summary(values, planned):
    defined = [v for v in values if v is not None]
    return {"replicates": planned, "defined_replicates": len(defined),
            "undefined_replicates": planned - len(defined),
            "percentile95": [percentile(defined, 0.025), percentile(defined, 0.975)] if defined else None,
            "empirical_support_degenerate": bool(defined) and max(defined) == min(defined),
            "warning": "Percentile resampling cannot generate unobserved successes. Zero-width boundary intervals do not establish a zero population probability or zero effect."}


def self_check():
    for x in range(31):
        lo, hi = clopper_pearson(x, 30)
        rlo, rhi = clopper_pearson(30 - x, 30)
        assert 0 <= lo <= x / 30 <= hi <= 1
        assert math.isclose(lo, 1 - rhi, abs_tol=1e-12)
        assert math.isclose(hi, 1 - rlo, abs_tol=1e-12)
        adjusted = clopper_pearson(x, 30, 0.05 / 24)
        assert adjusted[0] <= lo + 1e-12 and adjusted[1] >= hi - 1e-12
    assert math.isclose(clopper_pearson(0, 30)[1], 1 - 0.025 ** (1 / 30), abs_tol=1e-12)
    assert math.isclose(clopper_pearson(30, 30)[0], 0.025 ** (1 / 30), abs_tol=1e-12)
    # Enumerate the complete binomial sample space over a fixed probability grid.
    for j in range(101):
        p = j / 100
        covered = math.fsum(math.comb(30, x) * p ** x * (1 - p) ** (30 - x)
                            for x in range(31) if clopper_pearson(x, 30)[0] - 1e-14 <= p <= clopper_pearson(x, 30)[1] + 1e-14)
        assert covered >= 0.95 - 1e-12
    assert percentile([0, 1, 2, 3], 0.5) == 1.5
    assert bootstrap_summary([0.0, 0.0, None], 3)["undefined_replicates"] == 1
    print("CP symmetry/boundaries/monotonicity/enumerated coverage and bootstrap missingness checks: PASS")


def validate_plan(plan):
    if tuple(plan.get("formal_runs", [])) != RUNS or plan.get("formal_calls") != 2040 or plan.get("pilots_included") is not False:
        raise RobustnessError("Enhancement plan does not match all three frozen formal runs")
    if plan["batch_bootstrap"]["resamples"] != 10000 or plan["batch_bootstrap"]["master_seed"] != 20261123:
        raise RobustnessError("Unexpected bootstrap plan; no adaptive resampling permitted")
    for name, contrasts, endpoints in (("peer_family", 12, 24), ("controls_family", 16, 32)):
        family = plan["multiplicity"][name]
        if family["contrasts"] != contrasts or family["unique_endpoint_proportions"] != endpoints or family["family_alpha"] != 0.05:
            raise RobustnessError("Multiplicity family differs from the recorded plan")
        if not math.isclose(family["two_sided_CP_alpha_per_endpoint"], family["family_alpha"] / endpoints, rel_tol=1e-12):
            raise RobustnessError("CP endpoint alpha does not match the Bonferroni allocation")


def load_all(project, original):
    returned = project / "remote_return"
    freeze = read_json(project / "protocol_freeze.json")
    source_hashes, runs = {}, {}
    for entry in freeze["formal_plans"]:
        name = entry["name"]
        if name not in RUNS:
            raise RobustnessError("Unknown formal run in original freeze")
        prepared, run_path = returned / entry["prepared"], returned / "runs" / name
        cells, by_id, schedule, by_call, hashes = original.load_prepared(prepared)
        manifest, finals, fixture, run_hashes = original.load_run(run_path, by_id, by_call, hashes, False)
        if fixture or manifest is None or len(finals) != entry["calls"] or len(finals) != len(schedule):
            raise RobustnessError("Each formal run must be complete and non-fixture")
        identity = manifest["identity"]
        if identity.get("model_requested") != entry["model"] or identity.get("model_revision") != entry["revision"]:
            raise RobustnessError("Run checkpoint differs from original freeze")
        analysis_path = returned / "analysis" / name / "analysis.json"
        analysis = read_json(analysis_path)
        if analysis["status"] != "complete" or analysis["fixture"] or analysis["run_hashes"] != run_hashes or analysis["prepared_hashes"] != hashes:
            raise RobustnessError("Original analysis hashes/status disagree with raw inputs")
        records = defaultdict(list)
        batches = defaultdict(list)
        for row in finals:
            records[row["cell_id"]].append(row)
            if type(row.get("batch_index")) is not int or row["batch_index"] < 1:
                raise RobustnessError("Formal final lacks a computational batch index")
            batches[row["batch_index"]].append(row)
        summaries = [original.summarize_cell(cell, records[cell["cell_id"]]) for cell in cells]
        if original.canonical(summaries) != original.canonical(analysis["cells"]):
            raise RobustnessError("Raw recomputation disagrees with original per-cell analysis")
        recomputed_effects, _ = original.make_effects(summaries)
        if original.canonical(recomputed_effects) != original.canonical(analysis["effects"]):
            raise RobustnessError("Recomputed effects disagree with the original analysis")
        for rows in batches.values():
            indices = [r.get("row_index") for r in rows]
            if len(set(indices)) != len(indices) or len({r.get("seed_batch") for r in rows}) != 1:
                raise RobustnessError("Batch row indices or recorded seed are inconsistent")
        run = {"name": name, "model": entry["model"], "revision": entry["revision"],
               "prepared_cells": cells, "cells": {c["cell_id"]: c for c in summaries},
               "by_key": {(c["persona_id"], c["arm"], c["level"]): c for c in summaries},
               "analysis": analysis, "finals": finals, "batches": dict(batches), "identity": identity}
        runs[name] = run
        for path in [analysis_path, run_path / "manifest.json", run_path / "final.jsonl", prepared / "cells.jsonl", prepared / "schedule.jsonl", prepared / "preflight.json"]:
            source_hashes[str(path.relative_to(project)).replace("\\", "/")] = digest(path.read_bytes())
    if set(runs) != set(RUNS) or sum(len(run["finals"]) for run in runs.values()) != 2040:
        raise RobustnessError("Incomplete 2040-call census")
    return runs, source_hashes


def effect_spec(run, effect, family, alpha):
    action = TARGETS[effect["arm"]]
    a, b = run["cells"][effect["a_cell_id"]], run["cells"][effect["b_cell_id"]]
    x_a, n_a = a["actions"][action]["valid_count"], a["completed"]
    x_b, n_b = b["actions"][action]["valid_count"], b["completed"]
    ca, cb = clopper_pearson(x_a, n_a, alpha), clopper_pearson(x_b, n_b, alpha)
    cp = [ca[0] - cb[1], ca[1] - cb[0]]
    sensitivity = [ca[0] - clopper_pearson(x_b + b["invalid"], n_b, alpha)[1],
                   clopper_pearson(x_a + a["invalid"], n_a, alpha)[1] - cb[0]]
    old = effect["primary_study"] if action == "study" else effect["arm_target"]
    return {"run": run["name"], "persona_id": effect["persona_id"], "arm": effect["arm"],
            "action": action, "contrast": effect["contrast"], "a_cell_id": a["cell_id"], "b_cell_id": b["cell_id"],
            "contrast_status": "frozen_scientific_contrast", "enhancement_status": "posthoc_robustness_analysis",
            "estimate": x_a / n_a - x_b / n_b,
            "original_newcombe95": old["valid_and_action_per_completed"]["newcombe_wilson95"],
            "family": family, "cp_alpha_per_endpoint": alpha, "simultaneous_cp95": cp,
            "cp_endpoint_a": {"count": x_a, "n": n_a, "interval": ca}, "cp_endpoint_b": {"count": x_b, "n": n_b, "interval": cb},
            "valid_only_estimate": old["valid_only"]["estimate"], "valid_only_newcombe95": old["valid_only"]["newcombe_wilson95"],
            "invalid_assignment_identification_range": old["invalid_outcome_identification_range"],
            "invalid_assignment_simultaneous_cp_envelope": sensitivity,
            "invalid_a": a["invalid"], "invalid_b": b["invalid"]}


def bootstrap_run(run, specs, original, count, seed):
    ids = list(run["cells"])
    index = {cid: i for i, cid in enumerate(ids)}
    matrices = []
    for batch_id in sorted(run["batches"]):
        grouped = defaultdict(lambda: [0, 0, 0, 0, 0])
        for row in run["batches"][batch_id]:
            val = grouped[index[row["cell_id"]]]
            val[0] += 1
            valid = original.validity(row)[0]
            val[1] += valid
            if valid and row["action"] in ("study", "rest", "eat"):
                val[("study", "rest", "eat").index(row["action"]) + 2] += 1
        matrices.append(list(grouped.items()))
    choices = [(index[s["a_cell_id"]], index[s["b_cell_id"]], ("study", "rest", "eat").index(s["action"]) + 2) for s in specs]
    values = [[] for _ in specs]
    conditional = [[] for _ in specs]
    rng = random.Random(seed)
    for _ in range(count):
        result = [[0, 0, 0, 0, 0] for _ in ids]
        selected = Counter(rng.choices(range(len(matrices)), k=len(matrices)))
        for batch, weight in selected.items():
            for idx, stats in matrices[batch]:
                target = result[idx]
                for k in range(5):
                    target[k] += weight * stats[k]
        for i, (ia, ib, outcome) in enumerate(choices):
            a, b = result[ia], result[ib]
            values[i].append(a[outcome] / a[0] - b[outcome] / b[0] if a[0] and b[0] else None)
            conditional[i].append(a[outcome] / a[1] - b[outcome] / b[1] if a[1] and b[1] else None)
    for i, spec in enumerate(specs):
        spec["batch_bootstrap"] = bootstrap_summary(values[i], count)
        spec["batch_bootstrap_valid_only"] = bootstrap_summary(conditional[i], count)
    return {spec["persona_id"] + "/" + spec["arm"]: values[i] for i, spec in enumerate(specs)}


def context_comparisons(runs, boot, original, count):
    dose, interactions = [], []
    for kind, a_name, b_name in (("time", "qwen3_time16", "qwen3_main"), ("checkpoint", "qwen25_main", "qwen3_main")):
        a_run, b_run = runs[a_name], runs[b_name]
        for persona in PERSONAS:
            for level in ("q10", "q50", "q90"):
                a = a_run["by_key"][(persona, "peer_signal", level)]
                b = b_run["by_key"][(persona, "peer_signal", level)]
                xa, xb = a["actions"]["study"]["valid_count"], b["actions"]["study"]["valid_count"]
                result = original.difference(xa, a["completed"], xb, b["completed"])
                dose.append({"comparison": kind, "a_run": a_name, "b_run": b_name, "persona_id": persona, "level": level,
                             "status": "posthoc_exploratory_fixed_conditions", "estimate": result["estimate"],
                             "pointwise_newcombe95": result["newcombe_wilson95"],
                             "valid_only": original.difference(xa, a["valid"], xb, b["valid"])})
            key = persona + "/peer_signal"
            av, bv = boot[a_name][key], boot[b_name][key]
            pairs = [a - b if a is not None and b is not None else None for a, b in zip(av, bv)]
            def peer_effect(run):
                high, low = [run["by_key"][(persona, "peer_signal", level)] for level in ("q90", "q10")]
                return high["actions"]["study"]["valid_count"] / high["completed"] - low["actions"]["study"]["valid_count"] / low["completed"]
            interactions.append({"comparison": kind, "a_run": a_name, "b_run": b_name, "persona_id": persona,
                                 "status": "posthoc_exploratory_fixed_conditions", "estimand": "(high-low peer response in a_run) minus (high-low peer response in b_run)",
                                 "estimate": peer_effect(a_run) - peer_effect(b_run), "batch_bootstrap": bootstrap_summary(pairs, count),
                                 "interval_status": "pointwise exploratory sensitivity, not multiplicity-adjusted confirmatory inference"})
    return dose, interactions


def action_rows(runs):
    output, heatmap = [], []
    for name in RUNS:
        run = runs[name]
        for cell in run["cells"].values():
            for action in (*ACTIONS, "invalid"):
                x = cell["invalid"] if action == "invalid" else cell["actions"][action]["valid_count"]
                output.append({"run": name, "persona_id": cell["persona_id"], "arm": cell["arm"], "level": cell["level"],
                               "action": action, "count": x, "completed": cell["completed"], "valid": cell["valid"],
                               "per_completed": x / cell["completed"], "per_valid": x / cell["valid"] if action != "invalid" and cell["valid"] else None})
        for persona in PERSONAS:
            cells = [run["by_key"][(persona, "peer_signal", level)] for level in ("q10", "q50", "q90")]
            n = sum(c["completed"] for c in cells)
            row = {"run": name, "persona_id": persona, "completed": n, "pooling": "peer arm only, equal 30 calls at each of three signal levels"}
            for action in ACTIONS:
                row[action] = sum(c["actions"][action]["valid_count"] for c in cells) / n
            row["invalid"] = sum(c["invalid"] for c in cells) / n
            if not math.isclose(sum(row[action] for action in (*ACTIONS, "invalid")), 1):
                raise RobustnessError("Action composition does not sum to all completed calls")
            heatmap.append(row)
    return output, heatmap


def flatten_effect(row):
    output = {key: row[key] for key in ("run", "persona_id", "arm", "action", "contrast", "estimate", "family", "cp_alpha_per_endpoint", "valid_only_estimate", "invalid_a", "invalid_b")}
    for name in ("original_newcombe95", "simultaneous_cp95", "valid_only_newcombe95", "invalid_assignment_identification_range", "invalid_assignment_simultaneous_cp_envelope"):
        value = row[name]
        output[name + "_lower"], output[name + "_upper"] = value if value is not None else (None, None)
    boot = row["batch_bootstrap"]
    output["batch_percentile_lower"], output["batch_percentile_upper"] = boot["percentile95"] if boot["percentile95"] is not None else (None, None)
    output["bootstrap_defined"] = boot["defined_replicates"]
    output["bootstrap_boundary_degenerate"] = boot["empirical_support_degenerate"]
    return output


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overview(output, peer, heatmap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    ordered = [next(row for row in peer if row["run"] == run and row["persona_id"] == persona) for run in RUNS for persona in PERSONAS]
    fig, (forest, heat) = plt.subplots(1, 2, figsize=(9.0, 5.0), gridspec_kw={"width_ratios": [1.08, 1.25]})
    for y, row in enumerate(ordered):
        color = colors[y % 4]
        cp, nominal, point = row["simultaneous_cp95"], row["original_newcombe95"], 100 * row["estimate"]
        forest.plot([100 * cp[0], 100 * cp[1]], [y, y], color="#B7BCC5", linewidth=1.5, zorder=1)
        forest.plot([100 * nominal[0], 100 * nominal[1]], [y, y], color=color, linewidth=2.8, zorder=2)
        forest.scatter([point], [y], color=color, s=16, zorder=3)
    forest.set_yticks(range(12), [RUN_LABELS[row["run"]] + "  " + row["persona_id"] for row in ordered], fontsize=7.7)
    forest.set_ylim(11.6, -0.6)
    forest.set_xlim(-100, 100)
    forest.set_xticks([-100, -50, 0, 50, 100])
    forest.axvline(0, linestyle="--", linewidth=0.8, color="#555555")
    forest.set_xlabel("High minus low peer effect (percentage points)", fontsize=8)
    forest.set_title("A  Valid-study response", loc="left", fontweight="bold", fontsize=9)
    forest.spines[["top", "right"]].set_visible(False)
    matrix = np.array([[row[action] for action in (*ACTIONS, "invalid")] for row in heatmap])
    image = heat.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    heat.set_xticks(range(8), ["Study", "Class", "Exerc.", "Chat", "Rest", "Eat", "Wander", "Invalid"], rotation=45, ha="right", fontsize=7.5)
    heat.set_yticks([])
    for y in range(12):
        for x in range(8):
            heat.text(x, y, f"{matrix[y,x] * 100:.0f}", ha="center", va="center", color="white" if matrix[y,x] > 0.5 else "#222222", fontsize=7.3)
    heat.set_title("B  Peer-arm action composition (%)", loc="left", fontweight="bold", fontsize=9)
    for ax in (forest, heat):
        for y in (3.5, 7.5):
            ax.axhline(y, color="#89909A", linewidth=0.8)
    bar = fig.colorbar(image, ax=heat, fraction=0.042, pad=0.03)
    bar.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=7)
    fig.text(0.01, 0.065, "Colored bars: original pointwise Newcombe 95%. Gray bars: conservative CP/Bonferroni simultaneous 95% family (12 peer contrasts).", fontsize=6.8)
    fig.text(0.01, 0.035, "Heatmap: all 90 peer-arm calls per fixed profile/run, pooling three equally sampled signal levels; invalid is separate. Study excludes go_class.", fontsize=6.8)
    fig.text(0.01, 0.005, "Q3 = Qwen3-4B-Instruct-2507; Q2.5 = Qwen2.5-7B-Instruct. Posthoc robustness layer; fixed checkpoints/profiles only; no population inference.", fontsize=6.8)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.23, wspace=0.18)
    for extension in ("pdf", "png"):
        fig.savefig(output / ("robustness_overview." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze(project, plan_path, output):
    if output.exists():
        raise RobustnessError("Output already exists; choose a new directory")
    if any(part.lower() in ("reference", "remote_return", "prepared", "prepared_time16") for part in output.parts):
        raise RobustnessError("Output cannot be inside immutable inputs")
    plan = read_json(plan_path)
    validate_plan(plan)
    original, original_path = load_original_analyzer()
    runs, source_hashes = load_all(project, original)
    peer, controls, boot = [], [], {}
    for name in RUNS:
        run = runs[name]
        specs = []
        for effect in run["analysis"]["effects"]:
            family = "peer_family" if effect["arm"] == "peer_signal" else "controls_family"
            alpha = plan["multiplicity"][family]["two_sided_CP_alpha_per_endpoint"]
            spec = effect_spec(run, effect, family, alpha)
            specs.append(spec)
            (peer if effect["arm"] == "peer_signal" else controls).append(spec)
        print(f"Bootstrapping {name}: {len(run['batches'])} computational blocks, {plan['batch_bootstrap']['resamples']} resamples", flush=True)
        boot[name] = bootstrap_run(run, specs, original, plan["batch_bootstrap"]["resamples"], plan["batch_bootstrap"]["run_seeds"][name])
    if len(peer) != 12 or len(controls) != 16:
        raise RobustnessError("Incomplete unselected contrast families")
    expected_peer = {(run, persona, "peer_signal") for run in RUNS for persona in PERSONAS}
    expected_controls = {(run, persona, arm) for run in ("qwen3_main", "qwen25_main") for persona in PERSONAS for arm in ("energy_control", "hunger_control")}
    for rows, expected in ((peer, expected_peer), (controls, expected_controls)):
        if {(row["run"], row["persona_id"], row["arm"]) for row in rows} != expected:
            raise RobustnessError("Contrast family contains repeated or omitted fixed profiles/arms")
    dose, interactions = context_comparisons(runs, boot, original, plan["batch_bootstrap"]["resamples"])
    actions, heatmap = action_rows(runs)
    averages = []
    for name in RUNS:
        specs = [row for row in peer if row["run"] == name]
        samples = [sum(values) / 4 if all(value is not None for value in values) else None
                   for values in zip(*(boot[name][persona + "/peer_signal"] for persona in PERSONAS))]
        averages.append({"run": name, "fixed_four_profiles_mean": sum(s["estimate"] for s in specs) / 4,
                         "simultaneous_cp_family_implied_mean_bounds": [sum(s["simultaneous_cp95"][k] for s in specs) / 4 for k in (0, 1)],
                         "batch_bootstrap": bootstrap_summary(samples, plan["batch_bootstrap"]["resamples"]),
                         "interpretation": "Equal-weight description of these same four fixed text profiles, not a sample of a student population."})
    result = {"schema_version": 1, "analysis_stage": "posthoc_cpu_robustness_enhancement", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": digest(plan_path.read_bytes()), "tool_sha256": digest(Path(__file__).read_bytes()),
              "original_analyzer_sha256": digest(original_path.read_bytes()), "source_hashes": source_hashes,
              "method_plan": plan, "formal_calls": 2040, "pilots_included": False,
              "runs": {name: {"totals": runs[name]["analysis"]["totals"], "model": runs[name]["model"], "revision": runs[name]["revision"],
                               "computational_batches": len(runs[name]["batches"]), "batch_sizes": dict(Counter(len(v) for v in runs[name]["batches"].values()))} for name in RUNS},
              "peer_effects": peer, "control_effects": controls, "fixed_profile_averages": averages,
              "exploratory_per_dose_time_checkpoint_comparisons": dose, "exploratory_difference_of_peer_responses": interactions,
              "peer_arm_action_composition": heatmap,
              "limitations": ["All enhancements were specified after inspecting the original results; original freezes remain unchanged.",
                              "CP/Bonferroni intervals are conservative and depend on binomial marginal sampling assumptions; two separately defined families are not a joint 95% family.",
                              "Computational-batch resampling is a sensitivity analysis, not proof of within-batch dependence or a more correct randomized-cluster sampling model.",
                              "Percentile bootstrap is degenerate at observed boundaries and cannot replace nonzero binomial uncertainty limits.",
                              "Time and checkpoint differences are posthoc exploratory contrasts for fixed conditions; different checkpoints confound size, training and family.",
                              "Invalid sensitivity permits hypothetical binary assignments; invalid outputs need not possess a well-defined hidden action.",
                              "Four bundled text profiles do not isolate individual personality traits or represent real students."]}
    for relative, expected in source_hashes.items():
        if digest((project / relative).read_bytes()) != expected:
            raise RobustnessError("An immutable source changed during computation")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "robustness.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    write_csv(output / "peer_effects.csv", [flatten_effect(row) for row in peer])
    write_csv(output / "control_effects.csv", [flatten_effect(row) for row in controls])
    write_csv(output / "action_distribution.csv", actions)
    write_csv(output / "exploratory_dose_comparisons.csv", [{key: value for key, value in row.items() if key != "valid_only"} for row in dose])
    write_csv(output / "exploratory_peer_response_differences.csv", [{key: value for key, value in row.items() if key != "batch_bootstrap"} |
              {"batch_lower": row["batch_bootstrap"]["percentile95"][0] if row["batch_bootstrap"]["percentile95"] is not None else None,
               "batch_upper": row["batch_bootstrap"]["percentile95"][1] if row["batch_bootstrap"]["percentile95"] is not None else None,
               "bootstrap_defined": row["batch_bootstrap"]["defined_replicates"], "bootstrap_undefined": row["batch_bootstrap"]["undefined_replicates"]} for row in interactions])
    overview(output, peer, heatmap)
    versions = {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy"), "matplotlib": importlib.metadata.version("matplotlib")}
    (output / "DEPENDENCIES.md").write_text("# Reproduction dependencies\n\n" + "\n".join(f"- {key}: {value}" for key, value in versions.items()) +
        "\n\nStatistics use Python stdlib only. CP endpoints invert finite binomial tails (64 bisection steps); no scipy/statsmodels runtime dependency. NumPy/Matplotlib are used only for the figure.\n\nMethod reference: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html\n", encoding="utf-8")
    receipt = {"analysis_stage": result["analysis_stage"], "formal_calls": 2040, "versions": versions,
               "plan_sha256": result["plan_sha256"], "tool_sha256": result["tool_sha256"],
               "files_sha256": {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote posthoc enhancement of all 2040 formal calls to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--plan", type=Path, default=PROJECT / "enhanced_analysis_plan.json")
    parser.add_argument("--out", type=Path, default=PROJECT / "enhanced_analysis")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_check:
            self_check()
        else:
            analyze(args.project.resolve(), args.plan.resolve(), args.out.resolve())
    except (RobustnessError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Robustness analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
