#!/usr/bin/env python3
"""Independent raw-text measurement sensitivity; no model calls or data edits.

    python tools/analyze_measurement_audit.py --self-check
    python tools/analyze_measurement_audit.py

Writes a NEW measurement_audit/ JSON/CSV/PDF/PNG directory. The frozen
strict primary analysis is preserved. Cached parsed/action/parse_status fields
are used only for comparison after independent classification, never as input
to the classification itself.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNS = ("qwen3_main", "qwen3_time16", "qwen25_main")
PERSONAS = ("P01", "P02", "P03", "P04")
ACTIONS = ("study", "go_class", "exercise", "chat", "rest", "eat", "wander")
TARGETS = {"peer_signal": "study", "energy_control": "rest", "hunger_control": "eat"}


class MeasurementError(Exception):
    pass


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    """Safe diagnostic text; never change the object used by classification.

    The frozen Python json.loads accepts NaN/Infinity extensions. Preserve that
    decoder behavior, while tagging such values explicitly in diagnostic JSON
    strings instead of crashing or writing nonstandard numbers in audit files.
    """
    def tagged(item):
        if isinstance(item, float) and not math.isfinite(item):
            label = "NaN" if math.isnan(item) else ("Infinity" if item > 0 else "-Infinity")
            return {"__python_json_nonfinite__": label}
        if isinstance(item, dict):
            return {key: tagged(child) for key, child in item.items()}
        if isinstance(item, list):
            return [tagged(child) for child in item]
        return item
    return json.dumps(tagged(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_original():
    path = Path(__file__).with_name("analyze_experiment.py")
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("ak7506_measurement_hash_validation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    finally:
        sys.dont_write_bytecode = old


def independent_parse(content):
    """Frozen extraction, independently implemented; no malformed repair."""
    result = {"object_parsed": False, "parse_category": "missing_text", "object": None,
              "action": None, "action_legal": False, "optional_failures": [],
              "duration_present": False, "duration_type": "missing", "duration_json": "",
              "effects_present": False, "effects_type": "missing", "raw_action_json": ""}
    if not isinstance(content, str):
        return result
    start, end = content.find("{"), content.rfind("}") + 1
    try:
        obj = json.loads(content[start:end]) if start >= 0 and end > start else None
    except (ValueError, TypeError):
        obj = None
    if not isinstance(obj, dict):
        result["parse_category"] = "invalid_json"
        return result
    result.update(object_parsed=True, object=obj, parse_category="object")
    action = obj.get("action")
    result["raw_action_json"] = canonical(action)
    if isinstance(action, str) and action.strip().lower() in ACTIONS:
        result["action"] = action.strip().lower()
        result["action_legal"] = True
    else:
        result["parse_category"] = "illegal_action"
    if "duration_ticks" in obj:
        value = obj["duration_ticks"]
        result.update(duration_present=True, duration_type=type(value).__name__, duration_json=canonical(value))
        if type(value) is not int:
            result["optional_failures"].append("duration_wrong_type")
        elif not 1 <= value <= 6:
            result["optional_failures"].append("duration_out_of_range")
    if "contextual_effects" in obj:
        value = obj["contextual_effects"]
        result.update(effects_present=True, effects_type=type(value).__name__)
        if not isinstance(value, dict):
            result["optional_failures"].append("contextual_effects_not_object")
    return result


def classify(content, eos, termination_log_consistent=True):
    result = independent_parse(content)
    base = result["object_parsed"] and result["action_legal"] and eos and termination_log_consistent
    strict = base and not result["optional_failures"]
    result.update(eos=bool(eos), termination_log_consistent=bool(termination_log_consistent),
                  strict_valid=bool(strict), action_only_valid=bool(base), recovered_auxiliary=bool(base and not strict))
    if not result["object_parsed"]:
        category = result["parse_category"]
    elif not result["action_legal"]:
        category = "illegal_action"
    elif not eos or not termination_log_consistent:
        category = "non_eos_or_termination_mismatch"
    elif result["optional_failures"]:
        category = "recoverable_action_auxiliary_schema_failure"
    else:
        category = "strict_valid"
    result["category"] = category
    return result


def self_check():
    for action in ACTIONS:
        result = classify(json.dumps({"action": action}), True)
        assert result["strict_valid"] and result["action_only_valid"]
    valid = classify('prefix```json\n{"action":" Study ","duration_ticks":6,"contextual_effects":{}}\n```', True)
    assert valid["strict_valid"] and valid["action"] == "study"
    for value, reason in ((300, "duration_out_of_range"), (True, "duration_wrong_type"), (3.0, "duration_wrong_type"), ("3", "duration_wrong_type"), (0, "duration_out_of_range")):
        result = classify(json.dumps({"action": "exercise", "duration_ticks": value}), True)
        assert result["action_only_valid"] and not result["strict_valid"] and reason in result["optional_failures"]
    result = classify('{"action":"study","contextual_effects":[]}', True)
    assert result["action_only_valid"] and not result["strict_valid"]
    for content in ('{"action":"studying"}', '{"action":study}', '{"action":"study"} {"action":"rest"}', None):
        result = classify(content, True)
        assert not result["strict_valid"] and not result["action_only_valid"]
    assert not classify('{"action":"study"}', False)["action_only_valid"]
    assert not classify('{"action":"study"}', True, False)["action_only_valid"]
    for extension in ("NaN", "Infinity", "-Infinity"):
        # Match the frozen decoder and only relax optional-field gates. No
        # nonfinite replacement ever reaches the classification object.
        result = classify('{"action":' + extension + '}', True)
        assert not result["strict_valid"] and not result["action_only_valid"]
        assert "__python_json_nonfinite__" in result["raw_action_json"]
        result = classify('{"action":"study","duration_ticks":' + extension + '}', True)
        assert not result["strict_valid"] and result["action_only_valid"]
        assert result["optional_failures"] == ["duration_wrong_type"]
        assert "__python_json_nonfinite__" in result["duration_json"]
        result = classify('{"action":"study","contextual_effects":' + extension + '}', True)
        assert not result["strict_valid"] and result["action_only_valid"]
        # The frozen rule checks object type, not nested effects/unknown keys.
        assert classify('{"action":"study","contextual_effects":{"x":' + extension + '}}', True)["strict_valid"]
        assert classify('{"action":"study","unknown":' + extension + '}', True)["strict_valid"]
    print("Independent JSON/action/auxiliary-field/EOS fixtures: PASS")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_termination(row):
    metadata = row.get("generation_metadata")
    if not isinstance(metadata, dict):
        raise MeasurementError("Missing raw generation metadata")
    tokens, eos_ids = metadata.get("generated_token_ids"), metadata.get("eos_token_ids")
    if (not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens) or
            not isinstance(eos_ids, list) or not eos_ids or any(type(token) is not int for token in eos_ids)):
        raise MeasurementError("Missing/invalid raw generated token IDs or EOS IDs")
    eos_positions = [i for i, token in enumerate(tokens) if token in eos_ids]
    eos = eos_positions == [len(tokens) - 1]
    logged = row.get("finish_reason") in ("stop", "eos") and row.get("completion_truncated") is not True
    return eos, eos == logged


def audit(project, plan_path, output):
    if output.exists():
        raise MeasurementError("Output already exists; choose a NEW directory")
    if any(part.lower() in ("reference", "remote_return", "prepared", "prepared_time16") for part in output.parts):
        raise MeasurementError("Audit output must be outside immutable sources")
    plan = read_json(plan_path)
    if tuple(plan.get("formal_runs", [])) != RUNS or plan.get("formal_calls") != 2040 or plan.get("analysis_stage") != "post_result_measurement_sensitivity":
        raise MeasurementError("Unexpected measurement-audit plan")
    original, original_path = load_original()
    source_hashes, call_rows, all_cells, run_summaries = {}, [], [], {}
    duration_counts, reason_counts = Counter(), Counter()
    strict_disagreements, termination_disagreements = [], []
    analyses = {}
    freeze = read_json(project / "protocol_freeze.json")
    frozen_runs = {entry["name"]: entry for entry in freeze["formal_plans"]}
    for name in RUNS:
        entry = frozen_runs[name]
        prepared = project / "remote_return" / entry["prepared"]
        run = project / "remote_return/runs" / name
        cells, by_id, schedule, by_call, hashes = original.load_prepared(prepared)
        manifest, finals, fixture, run_hashes = original.load_run(run, by_id, by_call, hashes, False)
        if fixture or manifest is None or len(finals) != len(schedule) or len(finals) != entry["calls"]:
            raise MeasurementError("Every formal run must be complete and non-fixture")
        analysis_path = project / "remote_return/analysis" / name / "analysis.json"
        archived = read_json(analysis_path)
        if archived["run_hashes"] != run_hashes or archived["prepared_hashes"] != hashes:
            raise MeasurementError("Original strict analysis source hashes disagree")
        analyses[name] = archived
        classified = defaultdict(list)
        categories, recovered_actions = Counter(), Counter()
        for row in finals:
            cell = by_id[row["cell_id"]]
            eos, consistent = token_termination(row)
            independent = classify(row.get("raw_content"), eos, consistent)
            if not consistent:
                termination_disagreements.append({"run": name, "planned_call_id": row["planned_call_id"]})
            # Cached flags are consulted only after independent classification.
            recorded_strict = original.validity(row)[0]
            if independent["strict_valid"] != recorded_strict:
                strict_disagreements.append({"run": name, "planned_call_id": row["planned_call_id"], "independent": independent["strict_valid"], "recorded": recorded_strict})
            base = {"run": name, "planned_call_id": row["planned_call_id"], "cell_id": row["cell_id"],
                    "persona_id": cell["persona_id"], "arm": cell["arm"], "level": cell["level"]}
            record = {**base, "category": independent["category"], "strict_valid": independent["strict_valid"],
                      "action_only_valid": independent["action_only_valid"], "recovered_auxiliary": independent["recovered_auxiliary"],
                      "action": independent["action"], "raw_action_json": independent["raw_action_json"],
                      "parse_category": independent["parse_category"], "optional_failures": ";".join(independent["optional_failures"]),
                      "duration_present": independent["duration_present"], "duration_type": independent["duration_type"],
                      "duration_json": independent["duration_json"], "effects_present": independent["effects_present"], "effects_type": independent["effects_type"],
                      "token_eos": eos, "termination_log_consistent": consistent}
            call_rows.append(record)
            classified[row["cell_id"]].append(record)
            categories[independent["category"]] += 1
            if independent["recovered_auxiliary"]:
                recovered_actions[independent["action"]] += 1
            for reason in independent["optional_failures"]:
                reason_counts[(name, cell["persona_id"], cell["arm"], cell["level"], reason, independent["action"] or "illegal_or_missing")] += 1
            duration_counts[(name, cell["persona_id"], cell["arm"], cell["level"], independent["duration_type"], independent["duration_json"], independent["action"] or "illegal_or_missing")] += 1
        for cell in cells:
            records = classified[cell["cell_id"]]
            summary = {"run": name, "cell_id": cell["cell_id"], "persona_id": cell["persona_id"], "arm": cell["arm"],
                       "level": cell["level"], "level_rendered": cell["level_rendered"], "completed": len(records), "planned": cell["n_replicates"]}
            for mode, flag in (("strict", "strict_valid"), ("action_only", "action_only_valid")):
                counts = Counter(record["action"] for record in records if record[flag])
                summary[mode] = {"valid": sum(record[flag] for record in records), "invalid": sum(not record[flag] for record in records),
                                 "actions": {action: counts[action] for action in ACTIONS}}
            summary["recovered_auxiliary"] = sum(record["recovered_auxiliary"] for record in records)
            summary["study_count_changed"] = summary["strict"]["actions"]["study"] != summary["action_only"]["actions"]["study"]
            summary["categories"] = dict(Counter(record["category"] for record in records))
            old = next(old for old in archived["cells"] if old["cell_id"] == cell["cell_id"])
            if summary["strict"]["valid"] != old["valid"] or any(summary["strict"]["actions"][action] != old["actions"][action]["valid_count"] for action in ACTIONS):
                raise MeasurementError("Independent strict cell counts disagree with the archived primary analysis")
            all_cells.append(summary)
        run_cells = [cell for cell in all_cells if cell["run"] == name]
        run_summaries[name] = {"completed": len(finals), "strict_valid": sum(cell["strict"]["valid"] for cell in run_cells),
                               "action_only_valid": sum(cell["action_only"]["valid"] for cell in run_cells),
                               "categories": dict(categories), "recovered_actions": dict(recovered_actions),
                               "study_strict": sum(cell["strict"]["actions"]["study"] for cell in run_cells),
                               "study_action_only": sum(cell["action_only"]["actions"]["study"] for cell in run_cells)}
        for path in (run / "final.jsonl", run / "manifest.json", analysis_path, prepared / "cells.jsonl", prepared / "schedule.jsonl", prepared / "preflight.json"):
            source_hashes[str(path.relative_to(project)).replace("\\", "/")] = sha(path.read_bytes())
    if len(call_rows) != 2040 or len(all_cells) != 68:
        raise MeasurementError("The audit does not cover all 2040 calls and 68 cells")
    peer, controls = [], []
    for name in RUNS:
        by_cell = {cell["cell_id"]: cell for cell in all_cells if cell["run"] == name}
        for old in analyses[name]["effects"]:
            action = TARGETS[old["arm"]]
            a, b = by_cell[old["a_cell_id"]], by_cell[old["b_cell_id"]]
            result = {"run": name, "persona_id": old["persona_id"], "arm": old["arm"], "action": action,
                      "contrast": old["contrast"], "a_cell_id": a["cell_id"], "b_cell_id": b["cell_id"],
                      "analysis_stage": "post_result_measurement_sensitivity"}
            for mode in ("strict", "action_only"):
                ca, cb = a[mode]["actions"][action], b[mode]["actions"][action]
                result[mode] = original.difference(ca, a["completed"], cb, b["completed"])
            result["estimate_changed"] = result["strict"]["estimate"] != result["action_only"]["estimate"]
            result["change_in_estimate"] = result["action_only"]["estimate"] - result["strict"]["estimate"]
            (peer if old["arm"] == "peer_signal" else controls).append(result)
    if len(peer) != 12 or len(controls) != 16:
        raise MeasurementError("Missing frozen peer/control contrasts")
    source_hashes[str(plan_path.relative_to(project)).replace("\\", "/")] = sha(plan_path.read_bytes())
    result = {"schema_version": 1, "analysis_stage": "post_result_measurement_sensitivity", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan": plan, "tool_sha256": sha(Path(__file__).read_bytes()), "hash_validation_and_interval_helper_sha256": sha(original_path.read_bytes()),
              "source_hashes": source_hashes, "formal_calls": len(call_rows), "pilots_included": False,
              "original_primary_preserved": True, "cached_parsed_fields_used_for_classification": False,
              "termination_disagreements": termination_disagreements, "strict_validity_disagreements": strict_disagreements,
              "runs": run_summaries, "cells": all_cells, "peer_effects": peer, "control_effects": controls,
              "study_changed_cells": [cell["run"] + "/" + cell["cell_id"] for cell in all_cells if cell["study_count_changed"]],
              "limitations": ["This is a post-result alternative measurement rule, not a replacement for the frozen strict primary outcome.",
                              "Recovering a legal action does not repair an executable decision with invalid duration/effects or validate its explanation.",
                              "Malformed JSON and illegal action labels remain rejected; only optional-field checks differ.",
                              "Both estimates retain completed-call denominators. Validity rates change their meaning under the two rules.",
                              "Intervals are pointwise Wilson/Newcombe summaries, not a new confirmatory multiple-comparison family.",
                              "Duration values are recorded as generated; no seconds-to-ticks conversion or inferred cause is imposed."]}
    for relative, expected in source_hashes.items():
        if sha((project / relative).read_bytes()) != expected:
            raise MeasurementError("An immutable source changed during the audit")
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "call_classification.csv", call_rows)
    flat_cells = []
    action_rows = []
    for cell in all_cells:
        flat = {key: cell[key] for key in ("run", "cell_id", "persona_id", "arm", "level", "level_rendered", "completed", "planned", "recovered_auxiliary", "study_count_changed")}
        for mode in ("strict", "action_only"):
            flat[mode + "_valid"], flat[mode + "_invalid"] = cell[mode]["valid"], cell[mode]["invalid"]
            for action in ACTIONS:
                count = cell[mode]["actions"][action]
                flat[mode + "_" + action] = count
                action_rows.append({"run": cell["run"], "cell_id": cell["cell_id"], "persona_id": cell["persona_id"], "arm": cell["arm"], "level": cell["level"],
                                    "mode": mode, "action": action, "count": count, "completed": cell["completed"], "per_completed": count / cell["completed"]})
        flat_cells.append(flat)
    write_csv(output / "all_cells.csv", flat_cells)
    write_csv(output / "action_distributions.csv", action_rows)
    write_csv(output / "duration_values.csv", [dict(zip(("run", "persona_id", "arm", "level", "duration_type", "duration_json", "action"), key)) | {"count": value} for key, value in sorted(duration_counts.items())])
    write_csv(output / "optional_failure_reasons.csv", [dict(zip(("run", "persona_id", "arm", "level", "reason", "action"), key)) | {"count": value} for key, value in sorted(reason_counts.items())])
    def flat_contrast(row):
        flat = {key: row[key] for key in ("run", "persona_id", "arm", "action", "contrast", "analysis_stage", "estimate_changed", "change_in_estimate")}
        for mode in ("strict", "action_only"):
            flat[mode + "_estimate"] = row[mode]["estimate"]
            flat[mode + "_lower"], flat[mode + "_upper"] = row[mode]["newcombe_wilson95"]
            flat[mode + "_a_count"], flat[mode + "_b_count"] = row[mode]["a"]["count"], row[mode]["b"]["count"]
        return flat
    write_csv(output / "peer_contrasts.csv", [flat_contrast(row) for row in peer])
    write_csv(output / "control_contrasts.csv", [flat_contrast(row) for row in controls])
    figure(output, all_cells, original)
    receipt = {"analysis_stage": result["analysis_stage"], "formal_calls": 2040, "tool_sha256": result["tool_sha256"],
               "source_hashes": source_hashes, "files_sha256": {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"formal_calls": 2040, "runs": run_summaries, "strict_disagreements": len(strict_disagreements),
                      "termination_disagreements": len(termination_disagreements), "study_changed_cells": len(result["study_changed_cells"]),
                      "peer_contrasts_changed": sum(row["estimate_changed"] for row in peer), "control_contrasts_changed": sum(row["estimate_changed"] for row in controls)}, ensure_ascii=False))
    return result


def figure(output, cells, original):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 4, figsize=(9.4, 6.1), sharex=True, sharey=True)
    by_key = {(cell["run"], cell["persona_id"], cell["arm"], cell["level"]): cell for cell in cells}
    labels = {"qwen3_main": "Qwen3, 11:00", "qwen3_time16": "Qwen3, 16:00", "qwen25_main": "Qwen2.5, 11:00"}
    for i, run in enumerate(RUNS):
        for j, persona in enumerate(PERSONAS):
            ax = axes[i][j]
            rows = [by_key[(run, persona, "peer_signal", level)] for level in ("q10", "q50", "q90")]
            xs = [float(row["level_rendered"]) for row in rows]
            study_changed = any(row["study_count_changed"] for row in rows)
            for mode, color, linestyle, marker in (("strict", "#0072B2", "-", "o"), ("action_only", "#D55E00", "--", "s")):
                if mode == "action_only" and not study_changed:
                    continue
                rates = [original.proportion(row[mode]["actions"]["study"], row["completed"]) for row in rows]
                ys = [rate["estimate"] for rate in rates]
                if mode == "strict":
                    errors = [[rate["estimate"] - rate["wilson95"][0] for rate in rates], [rate["wilson95"][1] - rate["estimate"] for rate in rates]]
                    ax.errorbar(xs, ys, yerr=errors, color=color, linestyle=linestyle, marker=marker, linewidth=1.3, markersize=3.2, capsize=2, elinewidth=0.7)
                else:
                    ax.plot(xs, ys, color=color, linestyle=linestyle, marker=marker, markerfacecolor="none", linewidth=1.1, markersize=4.2)
            recovered = sum(row["recovered_auxiliary"] for row in rows)
            changed = sum(row["study_count_changed"] for row in rows)
            # Keep the annotation clear of Qwen2.5 P02's high study rates.
            annotation_y, annotation_va = (0.07, "bottom") if max(row["strict"]["actions"]["study"] / row["completed"] for row in rows) > 0.60 else (0.95, "top")
            ax.text(0.97, annotation_y, f"aux recovered: {recovered}/90\nstudy cells changed: {changed}/3", transform=ax.transAxes, ha="right", va=annotation_va, fontsize=6.9)
            ax.set_ylim(-0.025, 1.025)
            ax.set_xlim(0.04, 0.26)
            ax.set_xticks([0.06, 0.13, 0.24], [".06", ".13", ".24"])
            ax.set_yticks([0, 0.5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            if i == 0:
                ax.set_title(persona, fontweight="bold", fontsize=9)
            if j == 0:
                ax.set_ylabel(labels[run] + "\nStudy / completed", fontsize=7.6)
    handles = [Line2D([0], [0], color="#0072B2", marker="o", label="Frozen parser; pointwise Wilson 95%")]
    if any(cell["study_count_changed"] for cell in cells if cell["arm"] == "peer_signal"):
        handles.append(Line2D([0], [0], color="#D55E00", marker="s", markerfacecolor="none", linestyle="--", label="Action-only sensitivity where study counts differ"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.54, 0.995), frameon=False, fontsize=8)
    fig.text(0.52, 0.092, "Recent peer study signal shown in prompt", ha="center", fontsize=8)
    fig.text(0.01, 0.045, "Post-result sensitivity; N = 30 completed calls per cell. Action-only overlays appear only where study counts differ; zero observations do not imply zero probability.", fontsize=6.7)
    fig.text(0.01, 0.024, "Aux = legal actions recovered by ignoring optional-field schema checks. Malformed JSON and illegal actions remain rejected; frozen primary is unchanged.", fontsize=6.7)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.15, hspace=0.26, wspace=0.20)
    for extension in ("pdf", "png"):
        fig.savefig(output / ("peer_dose_measurement_audit." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--plan", type=Path, default=PROJECT / "measurement_audit_plan.json")
    parser.add_argument("--out", type=Path, default=PROJECT / "measurement_audit")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_check:
            self_check()
        else:
            audit(args.project.resolve(), args.plan.resolve(), args.out.resolve())
    except (MeasurementError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Measurement audit failed: {exc}\n")


if __name__ == "__main__":
    main()
