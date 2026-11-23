"""Validate a frozen single-step schedule, or explicitly execute it using stdlib HTTP.

Validation never creates result files. Execution reads only AK7506_* credentials.
attempts.jsonl is a write-ahead log: one started and one finished event per request.
The hard request budget counts started events, including interrupted requests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

SUFFIX = " /no_think"
ACTIONS = {"study", "go_class", "exercise", "chat", "rest", "eat", "wander"}
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


class RunError(Exception):
    """Messages are fixed diagnostic codes, never provider exception strings."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_jsonl(path):
    try:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RunError("jsonl_record_must_be_object")
                rows.append(row)
        return rows
    except (OSError, ValueError, UnicodeError):
        raise RunError("cannot_read_valid_jsonl") from None


def validate_prepared(prepared):
    cells_path, schedule_path = prepared / "cells.jsonl", prepared / "schedule.jsonl"
    cells, schedule = read_jsonl(cells_path), read_jsonl(schedule_path)
    if not cells or not schedule:
        raise RunError("prepared_files_must_not_be_empty")
    by_id = {}
    for cell in cells:
        cid = cell.get("cell_id")
        if not isinstance(cid, str) or not cid or cid in by_id:
            raise RunError("invalid_or_duplicate_cell_id")
        if type(cell.get("n_replicates")) is not int or cell["n_replicates"] < 1:
            raise RunError("each_cell_requires_positive_n_replicates")
        for role in ("system", "user"):
            prompt = cell.get(role + "_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise RunError("missing_prompt")
            if cell.get(role + "_prompt_sha256") != digest(prompt.encode("utf-8")):
                raise RunError("prompt_hash_mismatch")
        if cell.get("prompt_sha256", cell["user_prompt_sha256"]) != cell["user_prompt_sha256"]:
            raise RunError("prompt_alias_hash_mismatch")
        messages = [{"role": role, "content": cell[role + "_prompt"]}
                    for role in ("system", "user")]
        if cell.get("input_sha256") != digest(canonical(messages)):
            raise RunError("input_hash_mismatch")
        by_id[cid] = cell
    ids, reps = set(), defaultdict(set)
    for item in schedule:
        pid, cid, rep = item.get("planned_call_id"), item.get("cell_id"), item.get("replicate")
        if not isinstance(pid, str) or not pid or pid in ids:
            raise RunError("invalid_or_duplicate_planned_call_id")
        if cid not in by_id or type(rep) is not int or not 1 <= rep <= by_id[cid]["n_replicates"]:
            raise RunError("invalid_schedule_cell_or_replicate")
        if rep in reps[cid] or type(item.get("repetition")) is not int or item["repetition"] != rep - 1:
            raise RunError("invalid_or_duplicate_repetition")
        if item.get("input_sha256") != by_id[cid]["input_sha256"]:
            raise RunError("schedule_input_hash_mismatch")
        ids.add(pid)
        reps[cid].add(rep)
    if set(reps) != set(by_id) or any(v != set(range(1, by_id[cid]["n_replicates"] + 1))
                                    for cid, v in reps.items()):
        raise RunError("schedule_replicates_disagree_with_cell_plan")
    hashes = {"cells_sha256": digest(cells_path.read_bytes()),
              "schedule_sha256": digest(schedule_path.read_bytes())}
    try:
        preflight_path = prepared / "preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RunError("requires_valid_preflight_json") from None
    if (preflight.get("n_cells") != len(cells) or preflight.get("planned_calls_per_model") != len(schedule)
            or any(c["n_replicates"] != preflight.get("n_replicates") for c in cells)):
        raise RunError("preflight_counts_disagree_with_prepared_plan")
    output_hashes = preflight.get("output_sha256", {})
    if (output_hashes.get("cells.jsonl") != hashes["cells_sha256"] or
            output_hashes.get("schedule.jsonl") != hashes["schedule_sha256"]):
        raise RunError("preflight_output_hash_mismatch")
    hashes["preflight_sha256"] = digest(preflight_path.read_bytes())
    return by_id, schedule, hashes


def endpoint_from_env():
    base = os.environ.get("AK7506_BASE_URL", "").rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise RunError("AK7506_BASE_URL_requires_plain_http_url_without_credentials_or_query")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, key) for k, v in value.items()}
    return value


def parse_decision(content):
    if not isinstance(content, str):
        return {"parse_status": "missing_text", "parsed": None}
    try:
        start, end = content.find("{"), content.rfind("}") + 1
        obj = json.loads(content[start:end]) if start >= 0 and end > start else None
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        return {"parse_status": "invalid_json", "parsed": None}
    action = obj.get("action")
    if not isinstance(action, str) or action.strip().lower() not in ACTIONS:
        return {"parse_status": "invalid_action", "parsed": obj}
    action = action.strip().lower()
    if ("duration_ticks" in obj and
            (type(obj["duration_ticks"]) is not int or not 1 <= obj["duration_ticks"] <= 6)):
        return {"parse_status": "invalid_schema", "parsed": obj}
    if "contextual_effects" in obj and not isinstance(obj["contextual_effects"], dict):
        return {"parse_status": "invalid_schema", "parsed": obj}
    return {"parse_status": "ok", "parsed": obj, "action": action,
            "is_study": action == "study", "is_learning": action in {"study", "go_class"}}


def request_once(endpoint, key, body, timeout):
    started, start_clock = now(), time.monotonic()
    result = {"started_at_utc": started, "raw_content": None, "model_served": None,
              "usage": None, "finish_reason": None, "parse_status": "not_attempted",
              "response_id": None, "generation_metadata": None,
              "error_code": None, "retryable": False, "http_status": None,
              "completion_truncated": False, "analysis_eligible": False}
    try:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = Request(endpoint, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=timeout) as response:
            result["http_status"] = response.status
            raw = response.read()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            result["error_code"] = "response_not_json"
        else:
            if not isinstance(data, dict):
                result["error_code"] = "response_not_object"
            else:
                result["model_served"] = data.get("model")
                result["usage"] = data.get("usage")
                result["response_id"] = data.get("id")
                result["generation_metadata"] = data.get("ak_metadata")
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    result["error_code"] = "response_missing_choice"
                else:
                    choice = choices[0]
                    result["finish_reason"] = choice.get("finish_reason")
                    message = choice.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    result["raw_content"] = content
                    result.update(parse_decision(content))
                    if result["parse_status"] != "ok":
                        result["error_code"] = "decision_" + result["parse_status"]
                    result["completion_truncated"] = result["finish_reason"] == "length"
                    result["analysis_eligible"] = (result["parse_status"] == "ok" and
                                                   isinstance(result["finish_reason"], str) and
                                                   result["finish_reason"] in {"stop", "eos"})
                    if result["completion_truncated"]:
                        result["error_code"] = "completion_truncated"
    except HTTPError as exc:
        result.update(http_status=exc.code, error_code="http_" + str(exc.code),
                      retryable=exc.code in TRANSIENT)
        exc.close()  # Do not read or record error bodies or response headers.
    except (socket.timeout, TimeoutError):
        result.update(error_code="request_timeout", retryable=True)
    except URLError as exc:
        result.update(error_code="request_timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout))
                      else "network_error", retryable=True)
    except (OSError, ValueError):
        result.update(error_code="request_error", retryable=False)
    result["ended_at_utc"] = now()
    result["elapsed_seconds"] = round(time.monotonic() - start_clock, 6)
    return redact(result, key)


def append_record(handle, row):
    handle.write(canonical(row).decode("utf-8") + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def execute(args, cells, schedule, hashes):
    endpoint = endpoint_from_env()
    key = os.environ.get("AK7506_API_KEY", "")
    model = args.model or os.environ.get("AK7506_MODEL")
    if not model:
        raise RunError("execute_requires_model")
    if not key and urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RunError("remote_execute_requires_AK7506_API_KEY")
    if args.fixture and urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RunError("fixture_requires_loopback_endpoint")
    if args.out is None:
        raise RunError("execute_requires_out_directory")
    sampling = {"temperature": 1.0, "top_p": 1.0, "max_tokens": args.max_tokens}
    identity = {"schema_version": 1, "model_requested": model, "endpoint": endpoint,
                "prepared_hashes": hashes, "sampling_params": sampling,
                "user_prompt_suffix": SUFFIX, "fixture": args.fixture}
    manifest_path = args.out / "manifest.json"
    if args.resume:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RunError("resume_requires_valid_existing_manifest") from None
        if manifest.get("identity") != identity:
            raise RunError("resume_identity_mismatch_model_endpoint_prepared_or_sampling")
    else:
        try:
            args.out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise RunError("out_exists_use_new_directory_or_resume") from None
        manifest = {"created_at_utc": now(), "identity": identity,
                    "planned_calls": len(schedule), "cells": len(cells)}
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    lock_path = args.out / ".run.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RunError("output_locked_check_existing_runner_before_removing_stale_lock") from None
    try:
        lock.write(str(os.getpid()))
        lock.close()
        return execute_locked(args, cells, schedule, identity, key)
    finally:
        lock.close()
        lock_path.unlink(missing_ok=True)


def execute_locked(args, cells, schedule, identity, key):
    apath, fpath = args.out / "attempts.jsonl", args.out / "final.jsonl"
    attempts = read_jsonl(apath) if apath.exists() else []
    finals = read_jsonl(fpath) if fpath.exists() else []
    planned_ids = {row["planned_call_id"] for row in schedule}
    done = set()
    starts, finishes, by_call = {}, {}, defaultdict(list)
    for row in finals:
        pid = row.get("planned_call_id")
        if pid not in planned_ids or pid in done:
            raise RunError("invalid_or_duplicate_final_record")
        done.add(pid)
    for row in attempts:
        aid, pid, event = row.get("attempt_id"), row.get("planned_call_id"), row.get("event")
        if pid not in planned_ids or not isinstance(aid, str):
            raise RunError("invalid_attempt_log")
        if event == "attempt_started":
            if aid in starts:
                raise RunError("duplicate_attempt_start")
            starts[aid] = row
            by_call[pid].append(row)
        elif event == "attempt_finished" and aid in starts and aid not in finishes:
            finishes[aid] = row
        else:
            raise RunError("invalid_attempt_event_order")
    used = len(starts)
    with apath.open("a", encoding="utf-8", newline="\n") as alog, fpath.open("a", encoding="utf-8", newline="\n") as flog:
        # An interrupted request may already have been charged. Never resend it automatically.
        for pid, prior in by_call.items():
            if pid in done:
                continue
            last = prior[-1]
            recovered = dict(finishes.get(last["attempt_id"], last))
            recovered.update(event="final", recovered_on_resume=True,
                             recovery_note="no_automatic_resend_after_interruption")
            if last["attempt_id"] not in finishes:
                recovered.update(error_code="interrupted_attempt_unknown_outcome",
                                 parse_status="not_attempted", retryable=False)
            append_record(flog, recovered)
            done.add(pid)
        for item in schedule:
            pid, cid = item["planned_call_id"], item["cell_id"]
            if pid in done:
                continue
            if used >= args.max_requests:
                break
            cell = cells[cid]
            messages = [{"role": "system", "content": cell["system_prompt"]},
                        {"role": "user", "content": cell["user_prompt"] + SUFFIX}]
            payload = {"model": identity["model_requested"], "messages": messages,
                       **identity["sampling_params"]}
            body = canonical(payload)
            common = {"planned_call_id": pid, "cell_id": cid,
                      "replicate": item["replicate"], "fixture": args.fixture,
                      "model_requested": identity["model_requested"],
                      "endpoint": identity["endpoint"], "sampling_params": identity["sampling_params"],
                      "prepared_input_sha256": cell["input_sha256"],
                      "system_prompt_sha256": cell["system_prompt_sha256"],
                      "prepared_user_prompt_sha256": cell["user_prompt_sha256"],
                      "actual_user_prompt_sha256": digest((cell["user_prompt"] + SUFFIX).encode("utf-8")),
                      "actual_messages_sha256": digest(canonical(messages)),
                      "request_body_sha256": digest(body), "user_prompt_suffix": SUFFIX}
            for index in range(1, 4):  # Initial request + at most two transient retries.
                used += 1
                attempt = {**common, "attempt_id": pid + "-A" + str(index),
                           "attempt_number": index, "request_ordinal": used,
                           "reserved_at_utc": now(), "event": "attempt_started"}
                append_record(alog, attempt)
                result = request_once(identity["endpoint"], key, body, args.timeout)
                completed = {**attempt, **result, "event": "attempt_finished"}
                append_record(alog, completed)
                retry = result["retryable"] and index < 3 and used < args.max_requests
                if retry:
                    time.sleep(args.retry_delay * index)
                    continue
                terminal = {**completed, "event": "final", "recovered_on_resume": False}
                if result["retryable"]:
                    terminal["terminal_reason"] = ("request_budget_exhausted" if used >= args.max_requests
                                                   else "retry_limit_reached")
                else:
                    terminal["terminal_reason"] = "observed_response" if not result["error_code"] else "nonretryable_error"
                append_record(flog, terminal)
                done.add(pid)
                break
    summary = {"fixture": args.fixture, "planned_calls": len(schedule), "final_records": len(done),
               "pending_calls": len(schedule) - len(done), "requests_started_total": used,
               "request_budget": args.max_requests, "budget_exhausted": used >= args.max_requests}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if len(done) == len(schedule) else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true", help="offline validation (default)")
    mode.add_argument("--execute", action="store_true", help="explicitly permit HTTP requests")
    parser.add_argument("--prepared", type=Path, default=Path(__file__).resolve().parents[1] / "prepared")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-requests", type=int, default=840, help="lifetime HTTP request ceiling, including retries")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--fixture", action="store_true", help="mark local HTTP test output; loopback only")
    args = parser.parse_args(argv)
    try:
        if args.max_requests < 1 or args.max_tokens < 1 or args.timeout <= 0 or args.retry_delay < 0:
            raise RunError("budgets_and_timeout_must_be_positive")
        cells, schedule, hashes = validate_prepared(args.prepared)
        if not args.execute:
            print(json.dumps({"validation": "passed", "network_requests": 0,
                              "cells": len(cells), "planned_calls": len(schedule),
                              "replicates_per_cell": sorted({c["n_replicates"] for c in cells.values()}),
                              "prepared_hashes": hashes}, ensure_ascii=False))
            return 0
        return execute(args, cells, schedule, hashes)
    except RunError as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, KeyError):
        print("ERROR: invalid_local_state_or_io_failure", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
