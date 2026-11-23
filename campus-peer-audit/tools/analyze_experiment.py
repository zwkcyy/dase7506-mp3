#!/usr/bin/env python3
"""Analyze finalized experiment records using only Python's standard library.

    python tools/analyze_experiment.py --prepared prepared --run runs/model01 --out analysis/model01
    python tools/analyze_experiment.py --prepared prepared --run runs/not_started --out analysis/pending

Reads prepared inputs and a runner manifest/final.jsonl; never sends requests or
changes inputs. Existing output directories and output inside reference/ are
refused. Fixture runs require --allow-fixture and remain marked TEST FIXTURE.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


PROJECT = Path(__file__).resolve().parents[1]
ACTIONS = ("study", "go_class", "exercise", "chat", "rest", "eat", "wander")
ARM_TARGET = {"peer_signal": "study", "energy_control": "rest", "hunger_control": "eat"}
METHOD_SOURCE = "https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/proportion.py"
Z95 = NormalDist().inv_cdf(0.975)


class AnalysisError(Exception):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_json(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("not_object")
        return result
    except (OSError, ValueError, UnicodeError):
        raise AnalysisError(f"Cannot read valid JSON object: {path.name}") from None


def load_jsonl(path):
    rows = []
    try:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not_object")
                rows.append(row)
    except (OSError, ValueError, UnicodeError):
        raise AnalysisError(f"Cannot read valid JSONL objects: {path.name}") from None
    return rows


def proportion(count, denominator):
    """Wilson score 95% interval without continuity correction; zero n -> null."""
    if denominator == 0:
        return {"count": count, "denominator": 0, "estimate": None, "wilson95": None}
    if not 0 <= count <= denominator:
        raise AnalysisError("Invalid binomial count")
    p = count / denominator
    z2 = Z95 * Z95
    scale = 1 + z2 / denominator
    center = (p + z2 / (2 * denominator)) / scale
    half = Z95 * math.sqrt(p * (1 - p) / denominator + z2 / (4 * denominator * denominator)) / scale
    return {"count": count, "denominator": denominator, "estimate": p,
            "wilson95": [max(0.0, center - half), min(1.0, center + half)]}


def difference(count_a, n_a, count_b, n_b):
    """Newcombe independent-proportions Wilson hybrid interval (A minus B)."""
    a, b = proportion(count_a, n_a), proportion(count_b, n_b)
    if n_a == 0 or n_b == 0:
        return {"estimate": None, "newcombe_wilson95": None, "a": a, "b": b}
    pa, pb = a["estimate"], b["estimate"]
    la, ua = a["wilson95"]
    lb, ub = b["wilson95"]
    delta = pa - pb
    lower = delta - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2)
    upper = delta + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)
    return {"estimate": delta, "newcombe_wilson95": [max(-1.0, lower), min(1.0, upper)], "a": a, "b": b}


def load_prepared(folder):
    cells = load_jsonl(folder / "cells.jsonl")
    schedule = load_jsonl(folder / "schedule.jsonl")
    if not cells or not schedule:
        raise AnalysisError("Prepared cells and schedule must not be empty")
    hashes = {"cells_sha256": digest((folder / "cells.jsonl").read_bytes()),
              "schedule_sha256": digest((folder / "schedule.jsonl").read_bytes())}
    by_id = {}
    for cell in cells:
        cid = cell.get("cell_id")
        if not isinstance(cid, str) or not cid or cid in by_id:
            raise AnalysisError("Invalid or duplicate prepared cell_id")
        if cell.get("arm") not in ARM_TARGET or not isinstance(cell.get("persona_id"), str):
            raise AnalysisError("Unknown arm or missing persona_id")
        if type(cell.get("n_replicates")) is not int or cell["n_replicates"] <= 0:
            raise AnalysisError("Invalid planned cell replicate count")
        for role in ("system", "user"):
            value = cell.get(role + "_prompt")
            if not isinstance(value, str) or cell.get(role + "_prompt_sha256") != digest(value.encode("utf-8")):
                raise AnalysisError("Prepared prompt hash mismatch")
        messages = [{"role": role, "content": cell[role + "_prompt"]} for role in ("system", "user")]
        if cell.get("input_sha256") != digest(canonical(messages)):
            raise AnalysisError("Prepared input hash mismatch")
        if cell.get("prompt_sha256", cell["user_prompt_sha256"]) != cell["user_prompt_sha256"]:
            raise AnalysisError("Prepared prompt alias hash mismatch")
        by_id[cid] = cell
    by_call, repetitions = {}, defaultdict(set)
    for item in schedule:
        pid, cid, rep = item.get("planned_call_id"), item.get("cell_id"), item.get("replicate")
        if not isinstance(pid, str) or not pid or pid in by_call or cid not in by_id:
            raise AnalysisError("Invalid or duplicate planned_call_id")
        if type(rep) is not int or not 1 <= rep <= by_id[cid]["n_replicates"] or rep in repetitions[cid]:
            raise AnalysisError("Invalid or repeated schedule replicate")
        if item.get("repetition") != rep - 1 or item.get("input_sha256") != by_id[cid]["input_sha256"]:
            raise AnalysisError("Schedule metadata/hash mismatch")
        repetitions[cid].add(rep)
        by_call[pid] = item
    if set(repetitions) != set(by_id) or any(repetitions[cid] != set(range(1, cell["n_replicates"] + 1)) for cid, cell in by_id.items()):
        raise AnalysisError("Schedule does not match all planned cell replicates")
    preflight_path = folder / "preflight.json"
    if preflight_path.exists():
        preflight = load_json(preflight_path)
        hashes["preflight_sha256"] = digest(preflight_path.read_bytes())
        for name, key in (("cells.jsonl", "cells_sha256"), ("schedule.jsonl", "schedule_sha256")):
            if preflight.get("output_sha256", {}).get(name) != hashes[key]:
                raise AnalysisError("Prepared files disagree with preflight hashes")
        if preflight.get("n_cells") != len(cells) or preflight.get("planned_calls_per_model") != len(schedule):
            raise AnalysisError("Prepared files disagree with preflight counts")
        if any(cell["n_replicates"] != preflight.get("n_replicates") for cell in cells):
            raise AnalysisError("Prepared files disagree with preflight n_replicates")
    return cells, by_id, schedule, by_call, hashes


def validity(row):
    reasons = []
    if row.get("error_code") not in (None, ""):
        reasons.append("nonempty_error_code")
    if row.get("parse_status") != "ok":
        reasons.append("parse_status_not_ok")
    if row.get("finish_reason") not in ("stop", "eos"):
        reasons.append("finish_reason_not_stop_or_eos")
    if row.get("completion_truncated") is True:
        reasons.append("completion_truncated")
    if row.get("action") not in ACTIONS:
        reasons.append("missing_or_illegal_action")
    # Recompute validity from the recorded evidence, never from is_study or
    # analysis_eligible flags (which may come from a different runner version).
    return not reasons, reasons


def load_run(folder, by_id, by_call, hashes, allow_fixture):
    manifest_path, final_path = folder / "manifest.json", folder / "final.jsonl"
    initial_hashes = {"manifest_sha256": digest(manifest_path.read_bytes()) if manifest_path.exists() else None,
                      "final_sha256": digest(final_path.read_bytes()) if final_path.exists() else None}
    manifest = load_json(manifest_path) if manifest_path.exists() else None
    finals = load_jsonl(final_path) if final_path.exists() else []
    if manifest is None:
        if finals:
            raise AnalysisError("Final records require a runner manifest")
        return None, [], False, initial_hashes
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or type(identity.get("fixture")) is not bool:
        raise AnalysisError("Manifest identity/fixture flag is missing")
    engine = identity.get("execution_engine", "http")
    if engine == "transformers_batch":
        if ("endpoint" not in identity or identity["endpoint"] is not None or
                identity.get("invocation_schema") != "canonical_generation_spec_v1"):
            raise AnalysisError("Unsupported transformers batch manifest schema")
    elif engine != "http":
        raise AnalysisError("Unsupported execution engine")
    if identity.get("prepared_hashes") != hashes:
        raise AnalysisError("Run manifest prepared hashes do not match these inputs")
    if manifest.get("planned_calls") != len(by_call) or manifest.get("cells") != len(by_id):
        raise AnalysisError("Manifest planned counts disagree with prepared inputs")
    fixture = identity["fixture"]
    if fixture and not allow_fixture:
        raise AnalysisError("TEST FIXTURE is rejected by default; explicitly use --allow-fixture")
    seen = set()
    for row in finals:
        pid = row.get("planned_call_id")
        if pid not in by_call or pid in seen or row.get("event") != "final":
            raise AnalysisError("Duplicate, unplanned, or non-final final record")
        seen.add(pid)
        item = by_call[pid]
        cell = by_id[item["cell_id"]]
        if row.get("cell_id") != item["cell_id"] or row.get("replicate") != item["replicate"]:
            raise AnalysisError("Final record cell/replicate disagrees with schedule")
        if type(row.get("fixture")) is not bool or row["fixture"] != fixture:
            raise AnalysisError("Final and manifest fixture flags disagree")
        if row.get("prepared_input_sha256") != cell["input_sha256"]:
            raise AnalysisError("Final prepared input hash mismatch")
        for row_key, cell_key in (("system_prompt_sha256", "system_prompt_sha256"), ("prepared_user_prompt_sha256", "user_prompt_sha256")):
            if row.get(row_key) != cell[cell_key]:
                raise AnalysisError("Final prepared prompt hash mismatch")
        suffix = identity.get("user_prompt_suffix")
        if not isinstance(suffix, str) or row.get("user_prompt_suffix") != suffix:
            raise AnalysisError("Final prompt suffix disagrees with manifest")
        messages = [{"role": "system", "content": cell["system_prompt"]}, {"role": "user", "content": cell["user_prompt"] + suffix}]
        payload = {"model": identity.get("model_requested"), "messages": messages, **identity.get("sampling_params", {})}
        expected = {"actual_user_prompt_sha256": digest((cell["user_prompt"] + suffix).encode("utf-8")),
                    "actual_messages_sha256": digest(canonical(messages))}
        if engine == "transformers_batch":
            # An in-process generation spec is a logical invocation, not an
            # HTTP request. Never accept a fabricated HTTP body/status here.
            if (row.get("transport") != "in_process" or
                    any(key not in row or row[key] is not None for key in ("endpoint", "request_body_sha256", "http_status"))):
                raise AnalysisError("Batch final must explicitly record in-process transport and null HTTP fields")
            if (row.get("execution_engine", engine) != engine or
                    row.get("invocation_schema", identity["invocation_schema"]) != identity["invocation_schema"]):
                raise AnalysisError("Batch final engine/schema disagrees with manifest")
            expected["invocation_spec_sha256"] = digest(canonical(payload))
        else:
            # Legacy HTTP validation remains unchanged and requires the actual
            # serialized request body fingerprint, never a logical-spec alias.
            expected["request_body_sha256"] = digest(canonical(payload))
        if any(row.get(key) != value for key, value in expected.items()):
            raise AnalysisError("Final submitted-input hash mismatch")
    return manifest, finals, fixture, initial_hashes


def summarize_cell(cell, records):
    valid_records, invalid_reasons = [], Counter()
    reported_actions = Counter()
    for row in records:
        reported_actions[row["action"] if row.get("action") in ACTIONS else "missing_or_illegal"] += 1
        valid, reasons = validity(row)
        if valid:
            valid_records.append(row)
        else:
            invalid_reasons.update(reasons)
    completed, valid = len(records), len(valid_records)
    invalid = completed - valid
    counts = Counter(row["action"] for row in valid_records)
    target = ARM_TARGET[cell["arm"]]
    actions = {}
    for action in ACTIONS:
        actions[action] = {"valid_count": counts[action],
                           "valid_and_action_per_completed": proportion(counts[action], completed),
                           "action_per_valid": proportion(counts[action], valid),
                           "invalid_outcome_identification_range": [counts[action] / completed, (counts[action] + invalid) / completed] if completed else None}
    return {"cell_id": cell["cell_id"], "arm": cell["arm"], "persona_id": cell["persona_id"],
            "level": cell.get("level"), "level_rendered": cell.get("level_rendered"),
            "planned": cell["n_replicates"], "completed": completed,
            "not_completed": cell["n_replicates"] - completed, "valid": valid, "invalid": invalid,
            "status": "pending" if not completed else ("complete" if completed == cell["n_replicates"] else "partial"),
            "valid_per_completed": proportion(valid, completed), "invalid_per_completed": proportion(invalid, completed),
            "invalid_reason_counts_nonexclusive": dict(invalid_reasons),
            "reported_action_counts_all_completed": {action: reported_actions[action] for action in (*ACTIONS, "missing_or_illegal")},
            "actions": actions, "primary_action": "study", "arm_target_action": target,
            "primary": actions["study"], "arm_target": actions[target]}


def make_effect(a, b, action):
    ca, cb = a["actions"][action]["valid_count"], b["actions"][action]["valid_count"]
    primary = difference(ca, a["completed"], cb, b["completed"])
    valid_only = difference(ca, a["valid"], cb, b["valid"])
    bounds_a, bounds_b = a["actions"][action]["invalid_outcome_identification_range"], b["actions"][action]["invalid_outcome_identification_range"]
    identified = [bounds_a[0] - bounds_b[1], bounds_a[1] - bounds_b[0]] if bounds_a is not None and bounds_b is not None else None
    return {"action": action, "valid_and_action_per_completed": primary,
            "valid_only": valid_only, "invalid_outcome_identification_range": identified}


def make_effects(summaries):
    grouped = defaultdict(dict)
    for cell in summaries:
        key = (cell["persona_id"], cell["arm"])
        if cell["level"] in grouped[key]:
            raise AnalysisError("Repeated persona/arm/level cell")
        grouped[key][cell["level"]] = cell
    effects = []
    for (persona, arm), levels in sorted(grouped.items()):
        if "q10" not in levels or "q90" not in levels:
            continue
        alevel, blevel = ("q90", "q10") if arm == "peer_signal" else ("q10", "q90")
        a, b = levels[alevel], levels[blevel]
        target = ARM_TARGET[arm]
        effects.append({"persona_id": persona, "arm": arm, "contrast": f"{alevel}_minus_{blevel}",
                        "a_cell_id": a["cell_id"], "b_cell_id": b["cell_id"],
                        "status": "pending" if not a["completed"] or not b["completed"] else "observed",
                        "primary_study": make_effect(a, b, "study"), "arm_target": make_effect(a, b, target)})
    averages = []
    for arm in ARM_TARGET:
        group = [effect for effect in effects if effect["arm"] == arm]
        values = [effect["primary_study"]["valid_and_action_per_completed"]["estimate"] for effect in group]
        target_values = [effect["arm_target"]["valid_and_action_per_completed"]["estimate"] for effect in group]
        ready = len(group) == 4 and all(value is not None for value in values)
        averages.append({"arm": arm, "profiles_with_observed_contrast": sum(value is not None for value in values),
                         "fixed_four_profiles_mean_study_effect": sum(values) / 4 if ready else None,
                         "fixed_four_profiles_mean_target_effect": sum(target_values) / 4 if ready else None,
                         "interpretation": "Equal-weight description of these four fixed text profiles only; no population inference and no aggregate confidence interval."})
    return effects, averages


def fmt(value):
    return "pending" if value is None else f"{value:.3f}"


def interval(value):
    return "pending" if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def render_report(result):
    title = "TEST FIXTURE — 实验分析" if result["fixture"] else "固定上下文实验分析"
    totals = result["totals"]
    lines = [f"# {title}", "", f"状态：**{result['status']}**。计划 {totals['planned']}；已完成 {totals['completed']}；未完成 {totals['not_completed']}；有效 {totals['valid']}；无效 {totals['invalid']}。", "",
             "未完成调用保留为 not_completed，不计为失败或零动作。主指标严格为 `action == 'study'`；`go_class` 单列。比例分母分别为全部已完成调用及其中有效调用。", ""]
    if result["fixture"]:
        lines += ["**TEST FIXTURE：仅用于验证分析流程，不能作为课程项目实验结果。**", ""]
    if not totals["completed"]:
        lines += ["尚无最终调用记录。所有行为概率及效应均为 pending/null；没有填入任何模型结果。", ""]
    lines += ["| 格子 | 档案 | 实验臂/档位 | 完成/计划 | 有效/无效 | 有效且目标/完成，Wilson 95% | 目标/有效，Wilson 95% |", "|---|---|---|---:|---:|---|---|"]
    for cell in result["cells"]:
        a = cell["arm_target"]["valid_and_action_per_completed"]
        b = cell["arm_target"]["action_per_valid"]
        lines.append(f"| {cell['cell_id']} | {cell['persona_id']} | {cell['arm']}/{cell['level']} | {cell['completed']}/{cell['planned']} | {cell['valid']}/{cell['invalid']} | {fmt(a['estimate'])} {interval(a['wilson95'])} | {fmt(b['estimate'])} {interval(b['wilson95'])} |")
    lines += ["", "| 档案 | 实验臂/方向 | study差，Newcombe 95% | 对照目标差，Newcombe 95% | study无效结果最坏范围 |", "|---|---|---|---|---|"]
    for effect in result["effects"]:
        a = effect["primary_study"]["valid_and_action_per_completed"]
        b = effect["arm_target"]["valid_and_action_per_completed"]
        lines.append(f"| {effect['persona_id']} | {effect['arm']}/{effect['contrast']} | {fmt(a['estimate'])} {interval(a['newcombe_wilson95'])} | {fmt(b['estimate'])} {interval(b['newcombe_wilson95'])} | {interval(effect['primary_study']['invalid_outcome_identification_range'])} |")
    lines += ["", "peer_signal 方向为高减低，目标 study；energy_control 和 hunger_control 方向为低减高，目标分别为 rest、eat。完整七类动作分布、valid-only效应和各分母见 analysis.json。", "",
              "最坏范围允许每个已完成但无效输出的潜在动作任意取值，它是识别范围，非置信区间；不把未完成调用纳入该范围。未完成的随机计划子集不能自动代表最终完整实验。", "",
              "Wilson区间用于各二项比例；差值使用两个独立比例的 Newcombe Wilson hybrid 区间，无连续性校正。区间按单个比较计算，未做多重比较调整。仅在单次调用可视为独立随机输出时有相应解释。没有计算p值，也不据区间宣称显著或无影响。", "",
              f"公式核对来源：[statsmodels 官方源码]({METHOD_SOURCE})。", "",
              "四份档案的平均值只描述这四份固定文本，不能推广到真实学生群体；若任一档案端点无完成记录，四档案平均保持 null。模型版本、服务状态及抽样参数限制仍需在论文中交代。", ""]
    return "\n".join(lines)


def analyze(prepared, run, output, allow_fixture=False):
    if output.exists():
        raise AnalysisError("Output already exists; use --out with a new directory")
    if any(part.lower() == "reference" for part in output.parts):
        raise AnalysisError("Output must be outside reference/")
    if output == prepared or prepared in output.parents or output == run or run in output.parents:
        raise AnalysisError("Analysis output must be outside prepared and run inputs")
    cells, by_id, schedule, by_call, hashes = load_prepared(prepared)
    manifest, finals, fixture, run_hashes = load_run(run, by_id, by_call, hashes, allow_fixture)
    records = defaultdict(list)
    for row in finals:
        records[row["cell_id"]].append(row)
    summaries = [summarize_cell(cell, records[cell["cell_id"]]) for cell in cells]
    effects, averages = make_effects(summaries)
    totals = {key: sum(cell[key] for cell in summaries) for key in ("planned", "completed", "not_completed", "valid", "invalid")}
    status = "pending" if not finals else ("complete" if len(finals) == len(schedule) else "partial")
    # Deliberately omit endpoints and raw responses from the analysis artifact.
    identity = manifest["identity"] if manifest else {}
    done_ids = {row["planned_call_id"] for row in finals}
    result = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "fixture": fixture, "data_kind": "TEST FIXTURE" if fixture else "EXPERIMENT",
        "prepared_hashes": hashes, "run_hashes": run_hashes, "analysis_script_sha256": digest(Path(__file__).read_bytes()),
        "model_requested": identity.get("model_requested"),
        "execution_engine": identity.get("execution_engine", "http") if manifest else None,
        "invocation_schema": identity.get("invocation_schema"),
        "models_served_counts": dict(Counter(str(row.get("model_served")) for row in finals)),
        "sampling_params": identity.get("sampling_params"), "totals": totals,
        "validity_rule": "error_code empty AND parse_status ok AND finish_reason stop/eos AND not truncated AND action in legal seven-action set; runner eligibility/action-indicator flags are not used",
        "primary_definition": "Count(valid AND action == study) / completed; invalid completions are not interpreted as observed non-study behavior",
        "missingness_rule": "Calls lacking a final record are not_completed; excluded from all outcome denominators, never imputed as failures or zero actions",
        "interval_method": {"proportion": "Wilson score 95%, no continuity correction", "difference": "Newcombe independent two-proportions Wilson hybrid 95%, no continuity correction", "formula_source": METHOD_SOURCE, "multiple_comparison_adjustment": False},
        "cells": summaries, "effects": effects, "fixed_profile_averages": averages,
        "not_completed_planned_call_ids": [item["planned_call_id"] for item in schedule if item["planned_call_id"] not in done_ids],
        "limitations": ["No p-values or significance/no-effect claims are generated.", "Intervals are individual, unadjusted comparisons and presume independent calls.", "Invalid-output identification ranges are not confidence intervals.", "Only four fixed text profiles are described; no student-population inference.", "No inference is made for effects with an endpoint having zero completed calls.", "A partial run may have unequal cell completion and does not substitute for the full planned experiment."],
    }
    # Fail closed if a run changed while it was being read.
    for name, key in (("manifest.json", "manifest_sha256"), ("final.jsonl", "final_sha256")):
        path = run / name
        current = digest(path.read_bytes()) if path.exists() else None
        if current != run_hashes[key]:
            raise AnalysisError("Run inputs changed during analysis; retry on a stable snapshot")
    for name, key in (("cells.jsonl", "cells_sha256"), ("schedule.jsonl", "schedule_sha256"), ("preflight.json", "preflight_sha256")):
        path = prepared / name
        current = digest(path.read_bytes()) if path.exists() else None
        if current != hashes.get(key):
            raise AnalysisError("Prepared inputs changed during analysis; retry on a stable snapshot")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "analysis.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "report.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(render_report(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=PROJECT / "prepared")
    parser.add_argument("--run", type=Path, required=True, help="Runner output directory (a not-yet-created directory produces a pending report)")
    parser.add_argument("--out", type=Path, required=True, help="New analysis output directory")
    parser.add_argument("--allow-fixture", action="store_true", help="Explicitly permit TEST FIXTURE data; results remain labelled as fixtures")
    args = parser.parse_args()
    try:
        result = analyze(args.prepared.resolve(), args.run.resolve(), args.out.resolve(), args.allow_fixture)
    except (AnalysisError, OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Analysis failed: {exc}\n")
    print(f"{result['data_kind']}: {result['status']}; completed {result['totals']['completed']}/{result['totals']['planned']}; wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
