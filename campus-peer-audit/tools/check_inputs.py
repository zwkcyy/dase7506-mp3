"""Verify released anonymous prompts and schedules without private source data."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root.resolve()
    vendor = root / "vendor/agent_kernel_campus"
    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("released_prompt_builder", vendor / "prompt_builder.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    calibration_bytes = (root / "data/calibration_factors.jsonl").read_bytes()
    ledger = json.loads((root / "data/calibration_provenance.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(calibration_bytes).hexdigest() == ledger["data_sha256"]
    calibration = [json.loads(x) for x in calibration_bytes.decode("utf-8").splitlines() if x]
    assert len(calibration) == ledger["rows"] == 458
    assert all(set(row) == set(ledger["fields"]) and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
                   for x in row.values()) for row in calibration)
    total = 0
    for folder in ("prepared", "prepared_time16"):
        meta = json.loads((root / folder / "preflight.json").read_text(encoding="utf-8"))
        for factor in ledger["fields"]:
            values = sorted(row[factor] for row in calibration)
            for label, probability in (("q10", .1), ("q50", .5), ("q90", .9)):
                position = (len(values) - 1) * probability
                lo, hi = math.floor(position), math.ceil(position)
                q = values[lo] + (values[hi] - values[lo]) * (position - lo)
                assert q == meta["quantiles"][factor]["quantiles"][label]
        cells = [json.loads(x) for x in (root / folder / "cells.jsonl").read_text(encoding="utf-8").splitlines() if x]
        schedule = [json.loads(x) for x in (root / folder / "schedule.jsonl").read_text(encoding="utf-8").splitlines() if x]
        personas = {p["persona_id"]: p["text"] for p in meta["personas"]}
        for cell in cells:
            c, f = cell["context"], cell["factors_raw"]
            profile = SimpleNamespace(**c)
            payload = {"name": c["name"], "role": c["role"], **personas[cell["persona_id"]],
                       "current_state": {k: round(f[k], 2) for k in ("energy", "hunger", "social", "mood")}}
            duty = builder.check_has_duty(c["role"], c["hour_in_day"], c["is_weekday"], class_group=c["class_group"])
            user = builder.build_llm_prompt(payload, c["hour"], c["hour_in_day"], c["day_type"], c["weather_zh"],
                duty, c["recent_str"], profile, c["memory_context"], builder.build_peer_info_block(c["peer_rows"], f["peer_cum_at_prompt"]))
            assert duty == c["has_duty"] and user == cell["user_prompt"]
            assert builder.get_system_prompt(c["role"]) == cell["system_prompt"]
            assert hashlib.sha256(user.encode()).hexdigest() == cell["user_prompt_sha256"]
        expected = [{"schema_version": 1, "planned_call_id": f"{c['cell_id']}-R{r:02d}", "cell_id": c["cell_id"],
                     "replicate": r, "repetition": r-1, "input_sha256": c["input_sha256"]}
                    for c in cells for r in range(1, c["n_replicates"]+1)]
        random.Random(meta["seed"]).shuffle(expected)
        assert expected == schedule
        total += len(cells)
    print(json.dumps({"input_check": "passed", "cells": total, "model_calls": 0}))

if __name__ == "__main__":
    main()
