#!/usr/bin/env python3
"""Independent standard-library audit of the post-result measurement sensitivity.

Reads all raw formal calls; imports the reviewed implementation only for isolated
fixture comparisons. No GPU, model, network, raw-data or primary-analysis writes.
Optional --out writes a NEW receipt, never replacing an existing file.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNS = ("qwen3_main", "qwen3_time16", "qwen25_main")
ACTIONS = ("study", "go_class", "exercise", "chat", "rest", "eat", "wander")
PERSONAS = ("P01", "P02", "P03", "P04")
TARGETS = {"peer_signal": "study", "energy_control": "rest", "hunger_control": "eat"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def oracle(content, eos=True, consistent=True):
    """Independent rule, with no imported parser or cached response fields."""
    obj = None
    category = "missing_text" if not isinstance(content, str) else "invalid_json"
    if isinstance(content, str):
        try:
            obj = json.loads(content[content.index("{"):content.rindex("}") + 1])
        except (ValueError, TypeError):
            pass
    if not isinstance(obj, dict):
        return {"strict_valid": False, "action_only_valid": False, "recovered_auxiliary": False,
                "action": None, "category": category, "optional_failures": []}
    value = obj.get("action")
    action = value.strip().lower() if isinstance(value, str) else None
    legal = action in ACTIONS
    failures = []
    if "duration_ticks" in obj:
        value = obj["duration_ticks"]
        if value.__class__ != int:
            failures.append("duration_wrong_type")
        elif value < 1 or value > 6:
            failures.append("duration_out_of_range")
    if "contextual_effects" in obj and not isinstance(obj["contextual_effects"], dict):
        failures.append("contextual_effects_not_object")
    base = legal and eos and consistent
    strict = base and not failures
    category = ("illegal_action" if not legal else "non_eos_or_termination_mismatch" if not eos or not consistent
                else "recoverable_action_auxiliary_schema_failure" if failures else "strict_valid")
    return {"strict_valid": bool(strict), "action_only_valid": bool(base), "recovered_auxiliary": bool(base and not strict),
            "action": action if legal else None, "category": category, "optional_failures": failures}


def termination(row):
    meta = row["generation_metadata"]
    tokens, ends = meta["generated_token_ids"], meta["eos_token_ids"]
    require(isinstance(tokens, list) and len(tokens) > 0 and all(type(x) is int for x in tokens), "Invalid token evidence")
    require(isinstance(ends, list) and len(ends) > 0 and all(type(x) is int for x in ends), "Invalid EOS vocabulary")
    eos = tokens[-1] in set(ends) and all(x not in set(ends) for x in tokens[:-1])
    log_ok = row.get("finish_reason") in {"stop", "eos"} and row.get("completion_truncated") is not True
    return eos, eos == log_ok


def import_reviewed():
    path = Path(__file__).with_name("analyze_measurement_audit.py")
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("measurement_implementation_under_review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    finally:
        sys.dont_write_bytecode = old


def fixtures(reviewed):
    # Literal expected outcomes deliberately do not come from either parser.
    cases = [("missing_optional", '{"action":"study"}', True, True),
             ("duration_min", '{"action":"study","duration_ticks":1}', True, True),
             ("duration_max", '{"action":"study","duration_ticks":6}', True, True),
             ("normalized", 'prefix```json\n{"action":" Study ","contextual_effects":{}}\n```', True, True),
             ("unknown_extra_ignored", '{"action":"study","unknown":[]}', True, True),
             ("nested_effects_unvalidated", '{"action":"study","contextual_effects":{"arbitrary":null}}', True, True),
             ("registered_brace_extraction", '[{"action":"study"}]', True, True),
             ("duplicate_last_key", '{"action":"rest","action":"study"}', True, True)]
    for name, literal in (("duration_false", "false"), ("duration_true", "true"), ("duration_float", "3.0"),
                          ("duration_null", "null"), ("duration_string", '"3"'), ("duration_low", "0"),
                          ("duration_high", "7"), ("duration_negative", "-1"), ("duration_300", "300"),
                          ("duration_nonfinite_nan", "NaN"), ("duration_nonfinite_inf", "Infinity")):
        cases.append((name, '{"action":"exercise","duration_ticks":' + literal + '}', False, True))
    for name, literal in (("null", "null"), ("list", "[]"), ("string", '"x"'), ("integer", "2")):
        cases.append(("effects_" + name, '{"action":"study","contextual_effects":' + literal + '}', False, True))
    for name, content in (("malformed_json", '{"action":study}'), ("two_objects", '{"action":"study"} {"action":"rest"}'),
                          ("missing_text", None), ("empty_object", '{}'), ("missing_action", '{"duration_ticks":3}'),
                          ("illegal_label", '{"action":"studying"}'), ("illegal_label2", '{"action":"wap inflater"}'),
                          ("action_null", '{"action":null}'), ("action_bool", '{"action":true}'),
                          ("action_number", '{"action":1}'), ("action_object", '{"action":{}}'),
                          ("action_nan", '{"action":NaN}'), ("no_braces", 'study')):
        cases.append((name, content, False, False))
    for action in ACTIONS:
        cases.append(("action_" + action, json.dumps({"action": action}), True, True))
    checked = []
    for name, content, strict, loose in cases:
        expected = oracle(content)
        require((expected["strict_valid"], expected["action_only_valid"]) == (strict, loose), "Fixture oracle mismatch: " + name)
        got = reviewed.classify(content, True)
        require(all(got[k] == expected[k] for k in expected), "Reviewed fixture mismatch: " + name)
        require(not strict or loose, "Strict must imply action-only")
        checked.append(name)
    # EOS/truncation gates must exclude even otherwise legal, normalized actions.
    terminal_cases = [("terminal_stop", [8, 2], [2], "stop", False, True, True),
                      ("terminal_eos", [8, 9], [2, 9], "eos", False, True, True),
                      ("no_eos_length", [8, 9], [2], "length", True, False, True),
                      ("no_eos_stop", [8, 9], [2], "stop", False, False, False),
                      ("early_eos", [2, 8], [2], "stop", False, False, False),
                      ("repeated_eos", [2, 2], [2], "stop", False, False, False),
                      ("eos_truncated", [8, 2], [2], "stop", True, True, False),
                      ("eos_length", [8, 2], [2], "length", False, True, False)]
    for name, tokens, ends, finish, truncated, eos, consistent in terminal_cases:
        row = {"generation_metadata": {"generated_token_ids": tokens, "eos_token_ids": ends},
               "finish_reason": finish, "completion_truncated": truncated}
        require(termination(row) == (eos, consistent), "Termination oracle fixture: " + name)
        require(reviewed.token_termination(row) == (eos, consistent), "Reviewed termination fixture: " + name)
        got = reviewed.classify('{"action":"study","duration_ticks":300}', eos, consistent)
        require(got["strict_valid"] is False and got["action_only_valid"] == (eos and consistent), "Termination gate fixture: " + name)
    return {"parser_fixture_count": len(checked), "parser_fixture_names": checked,
            "termination_fixture_count": len(terminal_cases),
            "nonfinite_policy": "Same Python json.loads semantics as the registered parser; nonfinite auxiliary duration is wrong type, never strict-valid.",
            "semantic_scope": "Only optional-field validation is removed. Illegal labels and malformed JSON are not repaired; termination remains required."}


def interval(count, n):
    # Independent expression of the Wilson score solution to the quadratic.
    z = NormalDist().inv_cdf(0.975)
    z2 = z*z
    center = (2*count + z2) / (2*(n+z2))
    radius = z * math.sqrt(z2 + 4*count*(1-count/n)) / (2*(n+z2))
    return max(0.0, center-radius), min(1.0, center+radius)


def difference(xa, na, xb, nb):
    pa, pb = xa/na, xb/nb
    la, ua = interval(xa, na)
    lb, ub = interval(xb, nb)
    delta = pa-pb
    return delta, [max(-1.0, delta-math.hypot(pa-la, ub-pb)), min(1.0, delta+math.hypot(ua-pa, pb-lb))]


def near(a, b, label):
    require(math.isclose(a, b, rel_tol=0.0, abs_tol=2e-14), "Numeric mismatch: " + label)


def census(project, audit_dir, reviewed_path):
    result = read_json(audit_dir / "audit.json")
    receipt = read_json(audit_dir / "receipt.json")
    require(result["analysis_stage"] == "post_result_measurement_sensitivity" and result["formal_calls"] == 2040, "Audit stage/census mismatch")
    require(result["original_primary_preserved"] is True and result["pilots_included"] is False and result["cached_parsed_fields_used_for_classification"] is False, "Audit scope declarations mismatch")
    require(result["tool_sha256"] == sha(reviewed_path) == receipt["tool_sha256"], "Reviewed source hash mismatch")
    require(result["source_hashes"] == receipt["source_hashes"], "Source maps disagree")
    for relative, expected in result["source_hashes"].items():
        require(sha(project / relative) == expected, "Input source hash mismatch: " + relative)
    for name, expected in receipt["files_sha256"].items():
        require(sha(audit_dir / name) == expected, "Audit artifact hash mismatch: " + name)
    require(not result["termination_disagreements"] and not result["strict_validity_disagreements"], "Audit records primary/termination disagreement")
    require(not result["study_changed_cells"], "Unexpected study-count change")
    freeze = read_json(project / "protocol_freeze.json")
    plans = {entry["name"]: entry for entry in freeze["formal_plans"]}
    call_csv = read_csv(audit_dir / "call_classification.csv")
    calls_by_key = {(row["run"], row["planned_call_id"]): row for row in call_csv}
    require(len(call_csv) == len(calls_by_key) == 2040, "Duplicate/missing call CSV rows")
    oracle_cells, summaries, raw_sources = {}, {}, {}
    recovered_durations, reasons, durations = Counter(), Counter(), Counter()
    recovered_personas = Counter()
    actual_keys = set()
    for name in RUNS:
        plan = plans[name]
        folder = project / "remote_return" / plan["prepared"]
        raw = project / "remote_return/runs" / name
        cell_rows = read_jsonl(folder / "cells.jsonl")
        cells = {cell["cell_id"]: cell for cell in cell_rows}
        schedule = read_jsonl(folder / "schedule.jsonl")
        scheduled = {row["planned_call_id"]: row for row in schedule}
        finals = read_jsonl(raw / "final.jsonl")
        manifest = read_json(raw / "manifest.json")
        identity = manifest["identity"]
        require(identity["model_requested"] == plan["model"] and identity["model_revision"] == plan["revision"] and identity["fixture"] is False, "Frozen model identity mismatch")
        require(len(finals) == len(schedule) == len(scheduled) == plan["calls"], "Incomplete formal run")
        require({row["planned_call_id"] for row in finals} == set(scheduled), "Final call IDs do not cover the schedule")
        groups = defaultdict(list)
        categories, recovered_actions = Counter(), Counter()
        for row in finals:
            key = (name, row["planned_call_id"])
            require(key not in actual_keys, "Duplicate final call ID")
            actual_keys.add(key)
            item = scheduled[row["planned_call_id"]]
            require(row["cell_id"] == item["cell_id"] and row["replicate"] == item["replicate"], "Schedule/cell binding mismatch")
            require(row["model_requested"] == plan["model"] and row["model_revision"] == plan["revision"] and row["fixture"] is False, "Row frozen model mismatch")
            meta = row["generation_metadata"]
            require(hashlib.sha256(row["raw_content"].encode("utf-8")).hexdigest() == meta["content_sha256"], "Raw content hash mismatch")
            require(hashlib.sha256(encoded(meta["generated_token_ids"])).hexdigest() == meta["output_ids_sha256"], "Generated token hash mismatch")
            require(len(meta["generated_token_ids"]) == meta["completion_tokens"] == row["usage"]["completion_tokens"], "Actual token count mismatch")
            eos, consistent = termination(row)
            require(eos and consistent, "Formal call has missing/inconsistent EOS evidence")
            own = oracle(row.get("raw_content"), eos, consistent)
            # Cached flags are checked only AFTER the independent raw rule. The
            # runner uses decision_* error codes for parser rejection, too.
            expected_parse = {"strict_valid": "ok", "recoverable_action_auxiliary_schema_failure": "invalid_schema",
                              "illegal_action": "invalid_action", "invalid_json": "invalid_json", "missing_text": "missing_text"}[own["category"]]
            expected_error = None if expected_parse == "ok" else "decision_" + expected_parse
            require(row["parse_status"] == expected_parse and row.get("error_code") == expected_error, "Raw rule versus original parser/error evidence mismatch")
            recorded = calls_by_key[key]
            cell = cells[row["cell_id"]]
            for field in ("persona_id", "arm", "level"):
                require(recorded[field] == cell[field], "Call CSV metadata mismatch")
            require(recorded["cell_id"] == row["cell_id"], "Call CSV cell mismatch")
            for field in ("strict_valid", "action_only_valid", "recovered_auxiliary"):
                require(recorded[field] == str(own[field]), "Call CSV rule mismatch: " + field)
            require(recorded["action"] == (own["action"] or "") and recorded["category"] == own["category"], "Call CSV action/category mismatch")
            require(recorded["optional_failures"] == ";".join(own["optional_failures"]), "Call CSV auxiliary reason mismatch")
            require(recorded["token_eos"] == "True" and recorded["termination_log_consistent"] == "True", "Call CSV termination mismatch")
            groups[row["cell_id"]].append(own)
            categories[own["category"]] += 1
            # Parse again only for independent auxiliary-value summary checks.
            content = row.get("raw_content")
            try:
                obj = json.loads(content[content.index("{"):content.rindex("}")+1])
            except (ValueError, TypeError, AttributeError):
                obj = None
            has_duration = isinstance(obj, dict) and "duration_ticks" in obj
            value = obj["duration_ticks"] if has_duration else None
            dtype = type(value).__name__ if has_duration else "missing"
            djson = encoded(value).decode("utf-8") if has_duration else ""
            require(recorded["duration_present"] == str(has_duration) and recorded["duration_type"] == dtype and recorded["duration_json"] == djson, "Call CSV duration mismatch")
            effects_present = isinstance(obj, dict) and "contextual_effects" in obj
            etype = type(obj["contextual_effects"]).__name__ if effects_present else "missing"
            require(recorded["effects_present"] == str(effects_present) and recorded["effects_type"] == etype, "Call CSV effects type mismatch")
            common = (name, cell["persona_id"], cell["arm"], cell["level"])
            durations[common + (dtype, djson, own["action"] or "illegal_or_missing")] += 1
            for failure in own["optional_failures"]:
                reasons[common + (failure, own["action"] or "illegal_or_missing")] += 1
            if own["recovered_auxiliary"]:
                recovered_actions[own["action"]] += 1
                recovered_durations[(dtype, djson)] += 1
                recovered_personas[(name, cell["persona_id"])] += 1
        for cid, cell in cells.items():
            rows = groups[cid]
            require(len(rows) == cell["n_replicates"] == 30, "Cell denominator mismatch")
            summary = {"run": name, "cell_id": cid, "persona_id": cell["persona_id"], "arm": cell["arm"],
                       "level": cell["level"], "completed": len(rows), "planned": cell["n_replicates"],
                       "level_rendered": cell["level_rendered"], "categories": dict(Counter(r["category"] for r in rows))}
            for mode, flag in (("strict", "strict_valid"), ("action_only", "action_only_valid")):
                counts = Counter(r["action"] for r in rows if r[flag])
                valid = sum(counts.values())
                summary[mode] = {"valid": valid, "invalid": len(rows)-valid, "actions": {a: counts[a] for a in ACTIONS}}
            summary["recovered_auxiliary"] = sum(r["recovered_auxiliary"] for r in rows)
            summary["study_count_changed"] = summary["strict"]["actions"]["study"] != summary["action_only"]["actions"]["study"]
            oracle_cells[(name, cid)] = summary
        rcells = [v for (run, _), v in oracle_cells.items() if run == name]
        summaries[name] = {"completed": len(finals), "strict_valid": sum(c["strict"]["valid"] for c in rcells),
                           "action_only_valid": sum(c["action_only"]["valid"] for c in rcells), "categories": dict(categories),
                           "recovered_actions": dict(recovered_actions), "study_strict": sum(c["strict"]["actions"]["study"] for c in rcells),
                           "study_action_only": sum(c["action_only"]["actions"]["study"] for c in rcells)}
        for path in (raw / "final.jsonl", raw / "manifest.json", folder / "cells.jsonl", folder / "schedule.jsonl", folder / "preflight.json"):
            raw_sources[str(path.relative_to(project)).replace("\\", "/")] = sha(path)
    require(actual_keys == set(calls_by_key) and len(actual_keys) == 2040, "Call coverage differs")
    require(summaries == result["runs"], "Independent run summaries differ")
    audited_cells = {(c["run"], c["cell_id"]): c for c in result["cells"]}
    require(len(result["cells"]) == len(audited_cells) == 68 and oracle_cells == audited_cells, "Independent 68-cell census differs")
    for name in RUNS:
        original = read_json(project / "remote_return/analysis" / name / "analysis.json")
        for cell in original["cells"]:
            own = oracle_cells[(name, cell["cell_id"])]["strict"]
            require(own["valid"] == cell["valid"] and all(own["actions"][a] == cell["actions"][a]["valid_count"] for a in ACTIONS), "Original strict primary differs")
    # Derive the entire target family from factorial design, not archived effects.
    by_design = {(c["run"], c["persona_id"], c["arm"], c["level"]): c for c in oracle_cells.values()}
    expected = {}
    for name in RUNS:
        for persona in PERSONAS:
            for arm, target in TARGETS.items():
                if name == "qwen3_time16" and arm != "peer_signal":
                    continue
                alevel, blevel = ("q90", "q10") if arm == "peer_signal" else ("q10", "q90")
                a, b = by_design[(name, persona, arm, alevel)], by_design[(name, persona, arm, blevel)]
                expected[(name, persona, arm)] = (target, a, b, alevel + "_minus_" + blevel)
    effects = result["peer_effects"] + result["control_effects"]
    keyed = {(e["run"], e["persona_id"], e["arm"]): e for e in effects}
    require(len(result["peer_effects"]) == 12 and len(result["control_effects"]) == 16 and len(effects) == len(keyed) == len(expected) == 28 and set(keyed) == set(expected), "Missing/duplicate target contrasts")
    max_residual = 0.0
    for key, (target, a, b, contrast) in expected.items():
        row = keyed[key]
        require(row["action"] == target and row["a_cell_id"] == a["cell_id"] and row["b_cell_id"] == b["cell_id"] and row["contrast"] == contrast, "Target contrast orientation mismatch")
        for mode in ("strict", "action_only"):
            ca, cb = a[mode]["actions"][target], b[mode]["actions"][target]
            delta, bounds = difference(ca, a["completed"], cb, b["completed"])
            got = row[mode]
            near(delta, got["estimate"], "target estimate")
            for side in (0, 1):
                near(bounds[side], got["newcombe_wilson95"][side], "Newcombe interval")
                max_residual = max(max_residual, abs(bounds[side]-got["newcombe_wilson95"][side]))
            require(got["a"]["count"] == ca and got["b"]["count"] == cb and got["a"]["denominator"] == a["completed"] and got["b"]["denominator"] == b["completed"], "Target completed denominator mismatch")
        require(row["strict"] == row["action_only"] and row["estimate_changed"] is False and row["change_in_estimate"] == 0, "Sensitivity changes a target contrast")
    # Cross-check the seven CSVs independently of the artifact hash receipt.
    flat_cells = read_csv(audit_dir / "all_cells.csv")
    require(len(flat_cells) == 68, "Cell CSV coverage mismatch")
    for row in flat_cells:
        c = oracle_cells[(row["run"], row["cell_id"])]
        require(int(row["completed"]) == 30 and int(row["planned"]) == 30, "Cell CSV denominator mismatch")
        for mode in ("strict", "action_only"):
            require(int(row[mode+"_valid"]) == c[mode]["valid"] and int(row[mode+"_invalid"]) == c[mode]["invalid"], "Cell CSV validity mismatch")
            for action in ACTIONS:
                require(int(row[mode+"_"+action]) == c[mode]["actions"][action], "Cell CSV action mismatch")
    action_csv = read_csv(audit_dir / "action_distributions.csv")
    require(len(action_csv) == 68*2*7 and len({(r["run"], r["cell_id"], r["mode"], r["action"]) for r in action_csv}) == len(action_csv), "Action CSV coverage/uniqueness mismatch")
    for row in action_csv:
        c = oracle_cells[(row["run"], row["cell_id"])]
        require(int(row["count"]) == c[row["mode"]]["actions"][row["action"]] and int(row["completed"]) == 30, "Action CSV count/denominator mismatch")
        near(float(row["per_completed"]), int(row["count"])/30, "Action CSV probability")
    for filename, fields, counter in (("duration_values.csv", ("run", "persona_id", "arm", "level", "duration_type", "duration_json", "action"), durations),
                                      ("optional_failure_reasons.csv", ("run", "persona_id", "arm", "level", "reason", "action"), reasons)):
        rows = read_csv(audit_dir / filename)
        observed = {tuple(r[k] for k in fields): int(r["count"]) for r in rows}
        require(len(rows) == len(observed) and observed == dict(counter), "Auxiliary CSV mismatch: " + filename)
    for filename, family in (("peer_contrasts.csv", result["peer_effects"]), ("control_contrasts.csv", result["control_effects"])):
        rows = read_csv(audit_dir / filename)
        require(len(rows) == len(family), "Contrast CSV coverage mismatch")
        family_map = {(r["run"], r["persona_id"], r["arm"]): r for r in family}
        require(len({(r["run"], r["persona_id"], r["arm"]) for r in rows}) == len(rows), "Contrast CSV duplicates")
        for r in rows:
            e = family_map[(r["run"], r["persona_id"], r["arm"])]
            for mode in ("strict", "action_only"):
                near(float(r[mode+"_estimate"]), e[mode]["estimate"], "Contrast CSV estimate")
                near(float(r[mode+"_lower"]), e[mode]["newcombe_wilson95"][0], "Contrast CSV lower")
                near(float(r[mode+"_upper"]), e[mode]["newcombe_wilson95"][1], "Contrast CSV upper")
                require(int(r[mode+"_a_count"]) == e[mode]["a"]["count"] and int(r[mode+"_b_count"]) == e[mode]["b"]["count"], "Contrast CSV counts")
    require(recovered_durations == {("int", "300"): 36, ("int", "360"): 1}, "Unexpected recovered duration distribution")
    require(recovered_personas == {("qwen3_main", "P03"): 19, ("qwen3_time16", "P03"): 18}, "Unexpected recovered profile distribution")
    require(sum(c["strict"]["valid"] for c in oracle_cells.values()) == 1999 and sum(c["action_only"]["valid"] for c in oracle_cells.values()) == 2036, "Overall validity totals differ")
    # Sources and public artifacts must stay unchanged throughout this read-only review.
    for relative, value in raw_sources.items():
        require(sha(project / relative) == value, "Raw source changed during review")
    for name, expected_hash in receipt["files_sha256"].items():
        require(sha(audit_dir / name) == expected_hash, "Audit artifact changed during review")
    return {"formal_calls": 2040, "unique_run_call_ids": 2040, "cells": 68, "calls_per_cell": 30,
            "peer_contrasts": 12, "control_contrasts": 16, "seven_csvs_independently_checked": True,
            "all_eos_and_termination_consistent": True, "runs": summaries,
            "strict_valid_total": 1999, "action_only_valid_total": 2036, "recovered_action_total": 37,
            "recovered_action": "exercise", "recovered_duration_values": {"300": 36, "360": 1},
            "recovered_persona": "P03", "study_cells_changed": 0, "target_contrasts_changed": 0,
            "max_independent_newcombe_endpoint_residual": max_residual,
            "source_hashes": result["source_hashes"], "audit_artifact_hashes": receipt["files_sha256"],
            "audit_receipt_sha256": sha(audit_dir / "receipt.json"), "protocol_freeze_sha256": sha(project / "protocol_freeze.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.out is not None:
            require(not args.out.exists(), "Receipt output already exists")
        project = args.project.resolve()
        audit_dir = args.audit.resolve() if args.audit else project / "measurement_audit"
        reviewed, reviewed_path = import_reviewed()
        result = {"schema_version": 1, "status": "passed", "scope": "independent CPU fixture matrix and raw formal-call census; no new model calls",
                  "fixtures": fixtures(reviewed), "census": census(project, audit_dir, reviewed_path),
                  "reviewed_script_sha256": sha(reviewed_path), "reviewer_script_sha256": sha(Path(__file__)),
                  "interpretation": "This reclassifies the same observed outputs under an auxiliary-field measurement rule. It is not a causal intervention on model behavior, does not repair executable decisions, and leaves the registered primary analysis unchanged."}
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"status": "passed", "parser_fixtures": result["fixtures"]["parser_fixture_count"], "termination_fixtures": result["fixtures"]["termination_fixture_count"],
                          "formal_calls": 2040, "cells": 68, "target_contrasts": 28, "strict_valid": 1999, "action_only_valid": 2036, "recovered_exercise": 37}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Independent measurement review failed: {exc}\n")


if __name__ == "__main__":
    main()
