"""Run the prepared schedule by direct, explicitly recorded Transformers batches.

There is no HTTP service or request. The canonical invocation specification hashes
the model/messages/sampling supplied to this in-process engine. A started batch is
never retried automatically, and existing output directories are always rejected.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from run_experiment import (RunError, SUFFIX, append_record, canonical, digest,
                            now, parse_decision, validate_prepared)

ENGINE = "transformers_batch"
INVOCATION_SCHEMA = "canonical_generation_spec_v1"


def trim_generated(ids, eos_ids):
    """Remove only batch padding after the first generated EOS, retaining EOS."""
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[:index + 1], "stop"
    return ids, "length"


def prepare_rows(cells, schedule, model_id, sampling):
    rows = []
    for item in schedule:
        cell = cells[item["cell_id"]]
        messages = [{"role": "system", "content": cell["system_prompt"]},
                    {"role": "user", "content": cell["user_prompt"] + SUFFIX}]
        invocation = {"model": model_id, "messages": messages, **sampling}
        common = {
            "planned_call_id": item["planned_call_id"], "cell_id": item["cell_id"],
            "replicate": item["replicate"], "fixture": False,
            "model_requested": model_id, "endpoint": None,
            "execution_engine": ENGINE, "transport": "in_process",
            "invocation_schema": INVOCATION_SCHEMA,
            "sampling_params": sampling,
            "prepared_input_sha256": cell["input_sha256"],
            "system_prompt_sha256": cell["system_prompt_sha256"],
            "prepared_user_prompt_sha256": cell["user_prompt_sha256"],
            "actual_user_prompt_sha256": digest((cell["user_prompt"] + SUFFIX).encode()),
            "actual_messages_sha256": digest(canonical(messages)),
            "invocation_spec_sha256": digest(canonical(invocation)),
            "request_body_sha256": None, "http_status": None,
            "user_prompt_suffix": SUFFIX,
        }
        rows.append((common, messages, invocation))
    return rows


class BatchEngine:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch, self.GenerationConfig = torch, GenerationConfig
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), local_files_only=True, trust_remote_code=False,
            padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa").eval()
        self.eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.eos_ids = set(self.eos if isinstance(self.eos, list) else [self.eos])
        if None in self.eos_ids:
            raise RunError("checkpoint_requires_eos_token")
        self.pad = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id)
        self.tokenizer.pad_token_id = self.pad
        self.tokenizer.padding_side = "left"
        self.max_new = args.max_tokens
        self.identity = {
            "loaded_at_utc": now(), "model_id": args.model,
            "revision": args.revision, "model_dir": str(args.model_dir),
            "dtype": str(self.model.dtype), "attention_implementation": "sdpa",
            "gpu": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "versions": {k: importlib.metadata.version(k) for k in
                         ("torch", "transformers", "accelerate", "tokenizers")},
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "tokenizer_chat_template_sha256": digest((self.tokenizer.chat_template or "").encode()),
            "tokenizer_file_sha256": {p.name: digest(p.read_bytes()) for p in
                                      args.model_dir.iterdir() if p.is_file() and p.name in
                                      {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}},
            "config_sha256": digest((args.model_dir / "config.json").read_bytes()),
            "generation_config_source": "fresh GenerationConfig; use_model_defaults=False",
            "padding_side": "left", "eos_token_ids": sorted(self.eos_ids),
            "pad_token_id": self.pad, "max_total_tokens": 4096,
        }

    def generate(self, rows, seed):
        torch, tokenizer = self.torch, self.tokenizer
        rendered = [tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=False)
                    for _common, messages, _invocation in rows]
        inputs = tokenizer(rendered, add_special_tokens=False, padding=True,
                           return_attention_mask=True, return_tensors="pt")
        padded_width = inputs["input_ids"].shape[1]
        if padded_width + self.max_new > 4096:
            raise RunError("context_capacity_exceeded")
        padded_ids = inputs["input_ids"].tolist()
        masks = inputs["attention_mask"].tolist()
        prompt_ids = [[token for token, keep in zip(ids, mask) if keep]
                      for ids, mask in zip(padded_ids, masks)]
        # Verify that rendering then batching produced exactly the unpadded chat
        # input of the original per-call tokenizer path.
        for ids, (_common, messages, _invocation) in zip(prompt_ids, rows):
            if ids != tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True):
                raise RunError("batched_tokenization_disagrees_with_single_chat_template")
        config = self.GenerationConfig(
            max_new_tokens=self.max_new, do_sample=True, temperature=1.0, top_p=1.0,
            top_k=0, typical_p=1.0, repetition_penalty=1.0, num_beams=1,
            eos_token_id=self.eos, pad_token_id=self.pad, use_cache=True)
        inputs = inputs.to("cuda:0")
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        started, clock = now(), time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=config,
                                          use_model_defaults=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - clock
        generated = outputs[:, padded_width:].tolist()
        records = []
        for row_index, (ids, unpadded, padded, mask) in enumerate(
                zip(generated, prompt_ids, padded_ids, masks)):
            new_ids, finish = trim_generated(ids, self.eos_ids)
            content = tokenizer.decode(new_ids, skip_special_tokens=True)
            metadata = {
                "seed_batch": seed, "row_index": row_index,
                "prompt_tokens": len(unpadded), "padded_input_width": padded_width,
                "input_ids_sha256": digest(canonical(unpadded)),
                "padded_input_ids_sha256": digest(canonical(padded)),
                "attention_mask_sha256": digest(canonical(mask)),
                "output_ids_sha256": digest(canonical(new_ids)),
                "content_sha256": digest(content.encode()),
                "completion_tokens": len(new_ids), "finish_reason": finish,
                "generation_config": config.to_dict(), "use_model_defaults": False,
                "eos_token_ids": sorted(self.eos_ids), "pad_token_id": self.pad,
                "prompt_token_ids": unpadded, "generated_token_ids": new_ids,
                "started_at_utc": started, "ended_at_utc": now(),
                "batch_elapsed_seconds": elapsed,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            records.append((content, finish, metadata))
        return records


def execute(args, cells, schedule, hashes):
    if args.out is None or not args.model or not args.model_dir or not args.revision:
        raise RunError("execute_requires_out_model_model_dir_revision")
    if args.out.exists():
        raise RunError("output_exists_resume_and_automatic_batch_replay_are_disabled")
    sampling = {"temperature": 1.0, "top_p": 1.0, "max_tokens": args.max_tokens}
    identity = {
        "schema_version": 2, "model_requested": args.model,
        "model_revision": args.revision, "endpoint": None,
        "execution_engine": ENGINE, "transport": "in_process",
        "invocation_schema": INVOCATION_SCHEMA,
        "prepared_hashes": hashes, "sampling_params": sampling,
        "user_prompt_suffix": SUFFIX, "fixture": False, "batch_size": args.batch_size,
        "seed_base": args.seed, "seed_rule": "seed_base + batch_index; first batch_index=1; one shared RNG stream per batch; fixed schedule order",
        "sampling_fixed": {"do_sample": True, "top_k": 0, "typical_p": 1.0,
                           "repetition_penalty": 1.0, "num_beams": 1,
                           "use_model_defaults": False},
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "shared_parser_validator_sha256": digest(Path(__file__).with_name("run_experiment.py").read_bytes()),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at_utc": now(), "identity": identity,
                "planned_calls": len(schedule), "cells": len(cells),
                "max_calls_this_run": min(args.max_calls or len(schedule), len(schedule)),
                "interruption_rule": "A started but unfinished batch has unknown outcomes; never automatically replayed. Resume is disabled."}
    with (args.out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    engine = BatchEngine(args)
    with (args.out / "engine_identity.json").open("x", encoding="utf-8") as stream:
        json.dump(engine.identity, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    all_rows = prepare_rows(cells, schedule, args.model, sampling)
    rows = all_rows[:manifest["max_calls_this_run"]]
    completed = 0
    with ((args.out / "attempts.jsonl").open("x", encoding="utf-8", newline="\n") as alog,
          (args.out / "generation.jsonl").open("x", encoding="utf-8", newline="\n") as glog,
          (args.out / "final.jsonl").open("x", encoding="utf-8", newline="\n") as flog):
        for offset in range(0, len(rows), args.batch_size):
            batch_index = offset // args.batch_size + 1
            seed = args.seed + batch_index
            batch = rows[offset:offset + args.batch_size]
            append_record(alog, {"event": "batch_started", "batch_index": batch_index,
                                "seed_batch": seed, "started_at_utc": now(),
                                "planned_call_ids": [r[0]["planned_call_id"] for r in batch],
                                "invocation_specs": [r[2] for r in batch],
                                "invocation_specs_sha256": digest(canonical([r[2] for r in batch]))})
            # Exceptions intentionally leave batch_started without a completion.
            results = engine.generate(batch, seed)
            for row_index, ((common, _messages, _invocation), (content, finish, metadata)) in enumerate(zip(batch, results)):
                parsed = parse_decision(content)
                error = None if parsed["parse_status"] == "ok" else "decision_" + parsed["parse_status"]
                truncated = finish == "length"
                if truncated:
                    error = "completion_truncated"
                eligible = error is None and not truncated and finish == "stop"
                generation = {**metadata, "batch_index": batch_index,
                              "planned_call_id": common["planned_call_id"],
                              "model": args.model, "revision": args.revision,
                              "invocation_spec_sha256": common["invocation_spec_sha256"]}
                append_record(glog, generation)
                final = {
                    **common, **parsed, "event": "final", "batch_index": batch_index,
                    "row_index": row_index, "seed_batch": seed,
                    "model_served": args.model, "model_revision": args.revision,
                    "raw_content": content, "finish_reason": finish,
                    "usage": {"prompt_tokens": metadata["prompt_tokens"],
                              "completion_tokens": metadata["completion_tokens"],
                              "total_tokens": metadata["prompt_tokens"] + metadata["completion_tokens"]},
                    "response_id": None, "generation_metadata": generation,
                    "started_at_utc": metadata["started_at_utc"],
                    "ended_at_utc": metadata["ended_at_utc"],
                    "elapsed_seconds": metadata["batch_elapsed_seconds"],
                    "elapsed_scope": "entire_shared_batch_not_per_row",
                    "error_code": error, "retryable": False,
                    "completion_truncated": truncated, "analysis_eligible": eligible,
                    "recovered_on_resume": False,
                    "terminal_reason": "observed_response" if error is None else "nonretryable_error",
                }
                append_record(flog, final)
                completed += 1
            append_record(alog, {"event": "batch_finished", "batch_index": batch_index,
                                "seed_batch": seed, "ended_at_utc": now(),
                                "completed_rows": len(batch)})
            print(json.dumps({"completed": completed, "planned": len(schedule),
                              "batch_index": batch_index}), flush=True)
    return 0 if completed == len(schedule) else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepared", type=Path, default=Path(__file__).resolve().parents[1] / "prepared")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    try:
        if args.batch_size not in {4, 8, 16} or args.max_tokens != 1024 or (args.max_calls is not None and args.max_calls < 1):
            raise RunError("batch_size4_8_16_max_tokens1024_and_positive_call_budget_required")
        cells, schedule, hashes = validate_prepared(args.prepared)
        if not args.execute:
            print(json.dumps({"validation": "passed", "execution_engine": ENGINE,
                              "model_loads": 0, "gpu_calls": 0, "http_requests": 0,
                              "cells": len(cells), "planned_calls": len(schedule),
                              "prepared_hashes": hashes}))
            return 0
        return execute(args, cells, schedule, hashes)
    except RunError as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
