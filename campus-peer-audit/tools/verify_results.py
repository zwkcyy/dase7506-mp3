"""Independently verify a completed native-batch run, without models or GPU imports.

Example (paths are resolved under --root):
  python tools/verify_results.py --root . --run runs/qwen3_main \
      --prepared prepared --analysis analysis/qwen3_main --out checks/qwen3_main.json

Only the checks JSON is written, exclusively. Input artifacts are never changed.
No response text, profile text, or token arrays are printed in the checks output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ACTIONS = ("study", "go_class", "exercise", "chat", "rest", "eat", "wander")
TARGETS = {"peer_signal": "study", "energy_control": "rest", "hunger_control": "eat"}
SUFFIX = " /no_think"
FIXED = {"do_sample": True, "top_k": 0, "typical_p": 1.0,
         "repetition_penalty": 1.0, "num_beams": 1, "use_model_defaults": False}


class VerificationError(Exception):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Checks:
    def __init__(self):
        self.groups = defaultdict(lambda: {"checked": 0, "failed": 0})

    def check(self, code, condition):
        group = self.groups[code]
        group["checked"] += 1
        if not condition:
            group["failed"] += 1
        return bool(condition)

    def result(self):
        return [{"check": key, "passed": group["failed"] == 0, **group}
                for key, group in sorted(self.groups.items())]


def read_artifact(path, label, snapshots, lines=False):
    try:
        raw = path.read_bytes()
        snapshots[label] = (path, sha(raw))
        values = ([json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                  if lines else json.loads(raw.decode("utf-8")))
        if lines and any(not isinstance(v, dict) for v in values):
            raise ValueError()
        if not lines and not isinstance(values, dict):
            raise ValueError()
        return values
    except (OSError, ValueError, UnicodeError):
        raise VerificationError("missing_or_invalid_" + label.replace(".", "_")) from None


def required_object(value, code):
    if not isinstance(value, dict):
        raise VerificationError(code)
    return value


def index_unique(rows, key, audit, code):
    indexed = {}
    for row in rows:
        value = row.get(key)
        if not audit.check(code, isinstance(value, str) and bool(value) and value not in indexed):
            continue
        indexed[value] = row
    return indexed


def decision_from_text(text):
    """Independent specification of the registered parser; no runner import."""
    if not isinstance(text, str):
        return {"parse_status": "missing_text", "parsed": None}
    obj = None
    begin, end = text.find("{"), text.rfind("}") + 1
    if begin >= 0 and end > begin:
        try:
            obj = json.loads(text[begin:end])
        except ValueError:
            pass
    if not isinstance(obj, dict):
        return {"parse_status": "invalid_json", "parsed": None}
    action = obj.get("action")
    if not isinstance(action, str) or action.strip().lower() not in ACTIONS:
        return {"parse_status": "invalid_action", "parsed": obj}
    if ("duration_ticks" in obj and
            (type(obj["duration_ticks"]) is not int or not 1 <= obj["duration_ticks"] <= 6)):
        return {"parse_status": "invalid_schema", "parsed": obj}
    if "contextual_effects" in obj and not isinstance(obj["contextual_effects"], dict):
        return {"parse_status": "invalid_schema", "parsed": obj}
    action = action.strip().lower()
    return {"parse_status": "ok", "parsed": obj, "action": action,
            "is_study": action == "study", "is_learning": action in ("study", "go_class")}


def valid_tokens(value):
    return isinstance(value, list) and all(type(v) is int and v >= 0 for v in value)


def same_number(actual, expected):
    if expected is None:
        return actual is None
    return (isinstance(actual, (float, int)) and not isinstance(actual, bool) and
            math.isfinite(actual) and abs(actual - expected) <= 1e-12)


def check_proportion(value, count, denominator, audit):
    value = required_object(value, "analysis_missing_proportion")
    audit.check("analysis.proportion_counts", value.get("count") == count and value.get("denominator") == denominator)
    audit.check("analysis.proportion_estimates", same_number(value.get("estimate"), count / denominator if denominator else None))


def verify(args, audit, snapshots):
    root = args.root.resolve()
    resolve = lambda p: (p if p.is_absolute() else root / p).resolve()
    prepared, run = resolve(args.prepared), resolve(args.run)
    cells = read_artifact(prepared / "cells.jsonl", "prepared.cells", snapshots, True)
    schedule = read_artifact(prepared / "schedule.jsonl", "prepared.schedule", snapshots, True)
    preflight = read_artifact(prepared / "preflight.json", "prepared.preflight", snapshots)
    manifest = read_artifact(run / "manifest.json", "run.manifest", snapshots)
    engine = read_artifact(run / "engine_identity.json", "run.engine_identity", snapshots)
    finals = read_artifact(run / "final.jsonl", "run.final", snapshots, True)
    generations = read_artifact(run / "generation.jsonl", "run.generation", snapshots, True)
    attempts = read_artifact(run / "attempts.jsonl", "run.attempts", snapshots, True)
    freeze = read_artifact(root / "protocol_freeze.json", "protocol.freeze", snapshots)
    execution_freeze = read_artifact(root / "execution_freeze.json", "execution.freeze", snapshots)
    identity = required_object(manifest.get("identity"), "missing_manifest_identity")
    by_cell = index_unique(cells, "cell_id", audit, "prepared.unique_cells")
    by_call = index_unique(schedule, "planned_call_id", audit, "prepared.unique_plan_ids")
    if not cells or not schedule:
        raise VerificationError("empty_prepared_plan")
    hashes = {key: snapshots[label][1] for key, label in
              (("cells_sha256", "prepared.cells"), ("schedule_sha256", "prepared.schedule"),
               ("preflight_sha256", "prepared.preflight"))}
    audit.check("prepared.preflight_hashes", preflight.get("output_sha256") ==
                {"cells.jsonl": hashes["cells_sha256"], "schedule.jsonl": hashes["schedule_sha256"]})
    audit.check("prepared.preflight_counts", preflight.get("n_cells") == len(cells) and
                preflight.get("planned_calls_per_model") == len(schedule))
    rep_sets = defaultdict(set)
    for cell in cells:
        n = cell.get("n_replicates")
        audit.check("prepared.replicates", type(n) is int and n > 0 and n == preflight.get("n_replicates"))
        for role in ("system", "user"):
            text = cell.get(role + "_prompt")
            audit.check("prepared.prompt_hashes", isinstance(text, str) and
                        sha(text.encode("utf-8")) == cell.get(role + "_prompt_sha256"))
        messages = [{"role": role, "content": cell[role + "_prompt"]} for role in ("system", "user")]
        audit.check("prepared.message_hashes", sha(canonical(messages)) == cell.get("input_sha256"))
        audit.check("prepared.prompt_alias", cell.get("prompt_sha256", cell.get("user_prompt_sha256")) == cell.get("user_prompt_sha256"))
    for item in schedule:
        cid, rep = item.get("cell_id"), item.get("replicate")
        cell = by_cell.get(cid)
        if cell is None:
            raise VerificationError("schedule_references_unknown_cell")
        audit.check("prepared.schedule_replicates", type(rep) is int and 1 <= rep <= cell["n_replicates"] and rep not in rep_sets[cid]
                    and type(item.get("repetition")) is int and item["repetition"] == rep - 1)
        rep_sets[cid].add(rep)
        audit.check("prepared.schedule_hashes", item.get("input_sha256") == cell["input_sha256"])
    audit.check("prepared.complete_cell_quotas", all(rep_sets[cid] == set(range(1, c["n_replicates"] + 1)) for cid, c in by_cell.items()))

    fixture = identity.get("fixture")
    audit.check("run.fixture_policy", type(fixture) is bool and (not fixture or args.allow_fixture))
    audit.check("run.native_engine", identity.get("execution_engine") == "transformers_batch" and
                identity.get("transport") == "in_process" and identity.get("invocation_schema") == "canonical_generation_spec_v1")
    audit.check("run.no_http_manifest", "endpoint" in identity and identity["endpoint"] is None)
    audit.check("run.prepared_fingerprint", identity.get("prepared_hashes") == hashes)
    audit.check("run.planned_quotas", manifest.get("planned_calls") == len(schedule) and manifest.get("cells") == len(cells)
                and manifest.get("max_calls_this_run") == len(schedule))
    plan_name = args.plan or run.name
    plans = [p for p in freeze.get("formal_plans", []) if p.get("name") == plan_name]
    if len(plans) != 1:
        raise VerificationError("formal_plan_name_not_unique_in_protocol_freeze")
    plan = plans[0]
    model, revision = plan.get("model"), plan.get("revision")
    audit.check("run.frozen_model_revision", identity.get("model_requested") == model and identity.get("model_revision") == revision
                and engine.get("model_id") == model and engine.get("revision") == revision)
    audit.check("run.frozen_plan", plan.get("calls") == len(schedule) and resolve(Path(plan["prepared"])) == prepared)
    frozen_sampling = required_object(freeze.get("sampling"), "missing_frozen_sampling")
    sampling = identity.get("sampling_params")
    audit.check("run.frozen_sampling", sampling == {"temperature": frozen_sampling.get("temperature"),
                "top_p": frozen_sampling.get("top_p"), "max_tokens": frozen_sampling.get("max_new_tokens")})
    audit.check("execution.frozen_sampling", all(execution_freeze.get("sampling", {}).get(k) == v
                for k, v in frozen_sampling.items() if k != "do_sample"))
    audit.check("run.neutral_fixed_sampling", identity.get("sampling_fixed") == FIXED and all(
                frozen_sampling.get(k) == v for k, v in FIXED.items() if k not in ("typical_p", "num_beams")))
    audit.check("run.suffix", identity.get("user_prompt_suffix") == SUFFIX)
    for name, key in (("cells.jsonl", "cells_sha256"), ("schedule.jsonl", "schedule_sha256"), ("preflight.json", "preflight_sha256")):
        relative = prepared.relative_to(root).as_posix() + "/" + name
        audit.check("run.frozen_prepared_hashes", freeze.get("source_hashes", {}).get(relative) == hashes[key])
        audit.check("execution.frozen_prepared_hashes", execution_freeze.get("remote_source_sha256", {}).get(relative) == hashes[key])
    for filename, field in (("run_experiment_batch.py", "runner_sha256"), ("run_experiment.py", "shared_parser_validator_sha256")):
        path = root / "tools" / filename
        if path.exists():
            digest = sha(path.read_bytes())
            snapshots["source." + filename] = (path, digest)
            audit.check("run.source_fingerprints", identity.get(field) == digest)
            audit.check("execution.source_fingerprints", execution_freeze.get("remote_source_sha256", {}).get("tools/" + filename) == digest)
        else:
            audit.check("run.source_fingerprints", False)
    batch_size = identity.get("batch_size")
    if type(batch_size) is not int or batch_size not in (4, 8, 16):
        raise VerificationError("unsupported_batch_size")
    seed_base = identity.get("seed_base")
    if type(seed_base) is not int:
        raise VerificationError("invalid_seed_base")
    audit.check("run.seed_base", seed_base == preflight.get("seed"))
    audit.check("execution.frozen_engine_batch_seed", execution_freeze.get("execution_engine") == "transformers_batch"
                and execution_freeze.get("transport") == "in_process" and execution_freeze.get("batch_size") == batch_size
                and execution_freeze.get("seed_base") == seed_base
                and execution_freeze.get("formal_calls") == sum(p["calls"] for p in freeze["formal_plans"]))
    try:
        audit.check("execution.frozen_before_run", datetime.fromisoformat(execution_freeze["frozen_at_utc"]) <
                    datetime.fromisoformat(manifest["created_at_utc"]))
    except (KeyError, ValueError, TypeError):
        audit.check("execution.frozen_before_run", False)
    audit.check("run.shared_rng_documented", isinstance(identity.get("seed_rule"), str) and
                "seed_base + batch_index" in identity["seed_rule"] and "shared RNG" in identity["seed_rule"])
    audit.check("engine.padding", engine.get("padding_side") == "left")
    audit.check("engine.frozen_dtype_attention", str(engine.get("dtype", "")).removeprefix("torch.") == execution_freeze.get("dtype")
                and engine.get("attention_implementation") == execution_freeze.get("attention_implementation"))
    audit.check("engine.configuration", engine.get("max_total_tokens") == 4096 and
                "use_model_defaults=False" in str(engine.get("generation_config_source")))
    eos_ids, pad = engine.get("eos_token_ids"), engine.get("pad_token_id")
    if not valid_tokens(eos_ids) or not eos_ids or type(pad) is not int:
        raise VerificationError("invalid_engine_eos_or_pad")
    eos_set = set(eos_ids)
    final_by_id = index_unique(finals, "planned_call_id", audit, "run.unique_final_ids")
    gen_by_id = index_unique(generations, "planned_call_id", audit, "run.unique_generation_ids")
    expected_order = [r["planned_call_id"] for r in schedule]
    audit.check("run.complete_final_schedule", [r.get("planned_call_id") for r in finals] == expected_order)
    audit.check("run.complete_generation_schedule", [r.get("planned_call_id") for r in generations] == expected_order)
    expected_batches = math.ceil(len(schedule) / batch_size)
    audit.check("run.batch_event_count", len(attempts) == 2 * expected_batches)
    by_batch = defaultdict(list)
    independent = defaultdict(lambda: {"completed": 0, "valid": 0, "actions": Counter(), "reported": Counter()})
    neutral = {"num_return_sequences": 1, "min_length": 0, "min_new_tokens": None,
               "min_p": None, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
               "no_repeat_ngram_size": 0, "bad_words_ids": None, "force_words_ids": None,
               "forced_bos_token_id": None, "forced_eos_token_id": None, "suppress_tokens": None,
               "begin_suppress_tokens": None, "sequence_bias": None, "stop_strings": None,
               "watermarking_config": None, "guidance_scale": None}
    for offset, item in enumerate(schedule):
        pid, cid = item["planned_call_id"], item["cell_id"]
        row, gen = final_by_id.get(pid), gen_by_id.get(pid)
        if row is None or gen is None:
            audit.check("run.row_artifacts_present", False)
            continue
        audit.check("run.row_artifacts_present", True)
        cell = by_cell[cid]
        bindex, rindex = offset // batch_size + 1, offset % batch_size
        expected_seed = seed_base + bindex
        messages = [{"role": "system", "content": cell["system_prompt"]},
                    {"role": "user", "content": cell["user_prompt"] + SUFFIX}]
        invocation = {"model": model, "messages": messages, **sampling}
        invocation_hash = sha(canonical(invocation))
        audit.check("row.native_no_http", row.get("execution_engine") == "transformers_batch" and row.get("transport") == "in_process"
                    and row.get("invocation_schema") == "canonical_generation_spec_v1" and all(
                        k in row and row[k] is None for k in ("endpoint", "http_status", "request_body_sha256", "response_id")))
        audit.check("row.identity", row.get("fixture") == fixture and row.get("event") == "final" and row.get("cell_id") == cid
                    and row.get("replicate") == item["replicate"] and row.get("model_requested") == model
                    and row.get("model_served") == model and row.get("model_revision") == revision
                    and gen.get("model") == model and gen.get("revision") == revision)
        audit.check("row.batch_and_seed", all(record.get("batch_index") == bindex and record.get("row_index") == rindex
                    and record.get("seed_batch") == expected_seed for record in (row, gen)))
        audit.check("row.prepared_hashes", row.get("prepared_input_sha256") == cell["input_sha256"]
                    and row.get("system_prompt_sha256") == cell["system_prompt_sha256"]
                    and row.get("prepared_user_prompt_sha256") == cell["user_prompt_sha256"])
        audit.check("row.actual_input_hashes", row.get("user_prompt_suffix") == SUFFIX
                    and row.get("actual_user_prompt_sha256") == sha((cell["user_prompt"] + SUFFIX).encode("utf-8"))
                    and row.get("actual_messages_sha256") == sha(canonical(messages)))
        audit.check("row.invocation_hashes", row.get("sampling_params") == sampling and
                    row.get("invocation_spec_sha256") == invocation_hash and gen.get("invocation_spec_sha256") == invocation_hash)
        audit.check("row.generation_copy", row.get("generation_metadata") == gen)
        prompt_ids, new_ids = gen.get("prompt_token_ids"), gen.get("generated_token_ids")
        if not valid_tokens(prompt_ids) or not valid_tokens(new_ids) or not prompt_ids or not new_ids:
            audit.check("row.token_arrays", False)
            continue
        audit.check("row.token_arrays", True)
        width = gen.get("padded_input_width")
        if type(width) is not int or width < len(prompt_ids):
            raise VerificationError("invalid_padded_width")
        padding = width - len(prompt_ids)
        audit.check("row.token_hashes", gen.get("input_ids_sha256") == sha(canonical(prompt_ids)) and
                    gen.get("output_ids_sha256") == sha(canonical(new_ids)))
        audit.check("row.padding_hashes", gen.get("padded_input_ids_sha256") == sha(canonical([pad] * padding + prompt_ids)) and
                    gen.get("attention_mask_sha256") == sha(canonical([0] * padding + [1] * len(prompt_ids))))
        audit.check("row.token_counts", gen.get("prompt_tokens") == len(prompt_ids) and gen.get("completion_tokens") == len(new_ids)
                    and row.get("usage") == {"prompt_tokens": len(prompt_ids), "completion_tokens": len(new_ids),
                                            "total_tokens": len(prompt_ids) + len(new_ids)})
        content = row.get("raw_content")
        if not isinstance(content, str):
            raise VerificationError("nontext_raw_content")
        audit.check("row.content_hash", gen.get("content_sha256") == sha(content.encode("utf-8")))
        config = required_object(gen.get("generation_config"), "missing_generation_config")
        expected_config = {**{k: v for k, v in FIXED.items() if k != "use_model_defaults"},
                           "temperature": sampling["temperature"], "top_p": sampling["top_p"],
                           "max_new_tokens": sampling["max_tokens"], "use_cache": True,
                           "pad_token_id": pad}
        audit.check("row.generation_sampling", gen.get("use_model_defaults") is False and
                    all(config.get(k) == v for k, v in expected_config.items()))
        audit.check("row.neutral_generation_defaults", all(k in config and config[k] == v for k, v in neutral.items()))
        config_eos = config.get("eos_token_id")
        config_eos = config_eos if isinstance(config_eos, list) else [config_eos]
        audit.check("row.checkpoint_eos", set(config_eos) == eos_set and gen.get("eos_token_ids") == eos_ids and gen.get("pad_token_id") == pad)
        finish = "stop" if new_ids[-1] in eos_set else "length"
        audit.check("row.eos_trim_and_length", not any(t in eos_set for t in new_ids[:-1]) and
                    len(new_ids) <= sampling["max_tokens"] and (finish == "stop" or len(new_ids) == sampling["max_tokens"])
                    and row.get("finish_reason") == finish and gen.get("finish_reason") == finish
                    and width + sampling["max_tokens"] <= engine["max_total_tokens"])
        parsed = decision_from_text(content)
        audit.check("row.independent_parser", all(row.get(k) == v for k, v in parsed.items()) and all(
                    k in parsed or k not in row for k in ("action", "is_study", "is_learning")))
        error = None if parsed["parse_status"] == "ok" else "decision_" + parsed["parse_status"]
        if finish == "length":
            error = "completion_truncated"
        valid = error is None and finish == "stop"
        audit.check("row.validity_and_error", row.get("error_code") == error and row.get("analysis_eligible") == valid
                    and row.get("completion_truncated") == (finish == "length") and row.get("retryable") is False)
        audit.check("row.timing_scope", row.get("elapsed_scope") == "entire_shared_batch_not_per_row"
                    and row.get("elapsed_seconds") == gen.get("batch_elapsed_seconds"))
        by_batch[bindex].append((len(prompt_ids), width))
        count = independent[cid]
        count["completed"] += 1
        count["reported"][row.get("action") if row.get("action") in ACTIONS else "missing_or_illegal"] += 1
        if valid:
            count["valid"] += 1
            count["actions"][parsed["action"]] += 1
    for bindex in range(1, expected_batches + 1):
        segment = schedule[(bindex - 1) * batch_size:bindex * batch_size]
        records = by_batch.get(bindex, [])
        audit.check("batch.padded_width", len(records) == len(segment) and bool(records) and
                    all(width == max(length for length, _ in records) for _, width in records))
        events = attempts[2 * (bindex - 1):2 * bindex]
        if len(events) != 2:
            audit.check("batch.event_pair", False)
            continue
        start, end = events
        invocations = [{"model": model, "messages": [
            {"role": "system", "content": by_cell[i["cell_id"]]["system_prompt"]},
            {"role": "user", "content": by_cell[i["cell_id"]]["user_prompt"] + SUFFIX}], **sampling} for i in segment]
        audit.check("batch.event_pair", start.get("event") == "batch_started" and end.get("event") == "batch_finished"
                    and all(e.get("batch_index") == bindex and e.get("seed_batch") == seed_base + bindex for e in events))
        audit.check("batch.invocation_schedule", start.get("planned_call_ids") == [i["planned_call_id"] for i in segment]
                    and start.get("invocation_specs") == invocations and start.get("invocation_specs_sha256") == sha(canonical(invocations))
                    and end.get("completed_rows") == len(segment))
    audit.check("run.per_cell_completion", all(independent[cid]["completed"] == c["n_replicates"] for cid, c in by_cell.items()))

    if args.analysis is not None:
        analysis_path = resolve(args.analysis)
        if analysis_path.is_dir():
            analysis_path /= "analysis.json"
        analysis = read_artifact(analysis_path, "analysis.result", snapshots)
        audit.check("analysis.identity", analysis.get("status") == "complete" and analysis.get("fixture") == fixture
                    and analysis.get("prepared_hashes") == hashes and analysis.get("model_requested") == model
                    and analysis.get("execution_engine") == "transformers_batch" and analysis.get("sampling_params") == sampling)
        audit.check("analysis.run_fingerprints", analysis.get("run_hashes") == {
                    "manifest_sha256": snapshots["run.manifest"][1], "final_sha256": snapshots["run.final"][1]})
        valid_total = sum(c["valid"] for c in independent.values())
        completed_total = sum(c["completed"] for c in independent.values())
        audit.check("analysis.total_counts", analysis.get("totals") == {"planned": len(schedule), "completed": completed_total,
                    "not_completed": len(schedule) - completed_total, "valid": valid_total, "invalid": completed_total - valid_total}
                    and analysis.get("not_completed_planned_call_ids") == [])
        audit.check("analysis.served_model_counts", analysis.get("models_served_counts") == {model: len(finals)})
        summaries = index_unique(analysis.get("cells", []), "cell_id", audit, "analysis.unique_cells")
        audit.check("analysis.cell_coverage", set(summaries) == set(by_cell))
        for cid, cell in by_cell.items():
            summary = required_object(summaries.get(cid), "missing_analysis_cell")
            count, n = independent[cid], cell["n_replicates"]
            completed, valid = count["completed"], count["valid"]
            audit.check("analysis.cell_counts", summary.get("planned") == n and summary.get("completed") == completed
                        and summary.get("not_completed") == n - completed and summary.get("valid") == valid
                        and summary.get("invalid") == completed - valid and summary.get("status") == "complete")
            audit.check("analysis.cell_labels", all(summary.get(k) == cell.get(k) for k in ("persona_id", "arm", "level")))
            audit.check("analysis.reported_counts", summary.get("reported_action_counts_all_completed") ==
                        {a: count["reported"][a] for a in (*ACTIONS, "missing_or_illegal")})
            check_proportion(summary.get("valid_per_completed"), valid, completed, audit)
            check_proportion(summary.get("invalid_per_completed"), completed - valid, completed, audit)
            for action in ACTIONS:
                value = required_object(summary.get("actions", {}).get(action), "missing_analysis_action")
                k = count["actions"][action]
                audit.check("analysis.action_counts", value.get("valid_count") == k)
                check_proportion(value.get("valid_and_action_per_completed"), k, completed, audit)
                check_proportion(value.get("action_per_valid"), k, valid, audit)
            audit.check("analysis.primary_target", summary.get("primary_action") == "study"
                        and summary.get("arm_target_action") == TARGETS[cell["arm"]]
                        and summary.get("primary") == summary["actions"]["study"]
                        and summary.get("arm_target") == summary["actions"][TARGETS[cell["arm"]]])
    return {"fixture": fixture, "planned_samples": len(schedule), "final_samples": len(finals),
            "generation_records": len(generations), "batches": expected_batches,
            "batch_size": batch_size, "http_requests": 0, "analysis_checked": args.analysis is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("prepared"))
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--plan", help="Protocol formal-plan name; default: run directory basename")
    parser.add_argument("--out", type=Path, required=True, help="New checks JSON file")
    parser.add_argument("--allow-fixture", action="store_true", help="Explicitly verify labeled implementation fixtures")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    resolve = lambda p: (p if p.is_absolute() else root / p).resolve()
    out = resolve(args.out)
    protected = [resolve(args.run), resolve(args.prepared), root / "reference"]
    if args.analysis is not None:
        protected.append(resolve(args.analysis))
    if out.exists() or any(out == p or p in out.parents for p in protected):
        print("ERROR: output_exists_or_overlaps_read_only_inputs", file=sys.stderr)
        return 2
    audit, snapshots, summary, error = Checks(), {}, {}, None
    try:
        summary = verify(args, audit, snapshots)
    except VerificationError as exc:
        error = str(exc)
        audit.check("verification.structural_error", False)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        error = "invalid_artifact_structure_or_io"
        audit.check("verification.structural_error", False)
    for label, (path, original) in snapshots.items():
        try:
            stable = sha(path.read_bytes()) == original
        except OSError:
            stable = False
        audit.check("inputs.stable_during_verification", stable)
    checks = audit.result()
    passed = bool(checks) and all(c["passed"] for c in checks)
    output = {"schema_version": 1, "verification": "passed" if passed else "failed",
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "error_code": error,
              "verifier_sha256": sha(Path(__file__).read_bytes()), "summary": summary,
              "input_sha256": {label: digest for label, (_path, digest) in sorted(snapshots.items())},
              "checks": checks,
              "limits": ["No tokenizer decode or model generation is repeated.",
                         "Token hashes establish internal artifact consistency, not independent model-output reproduction.",
                         "Completion acceptance requires the full frozen plan; partial runs fail quota checks."]}
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError):
        print("ERROR: cannot_write_new_checks_file", file=sys.stderr)
        return 2
    print(json.dumps({"verification": output["verification"], "failed_check_groups": sum(not c["passed"] for c in checks),
                      **summary}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
