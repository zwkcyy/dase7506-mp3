"""Document exact numerical extraction; the original private trace is optional."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads((root / "data/calibration_provenance.json").read_text(encoding="utf-8"))
    raw = args.trace.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ledger["source_trace_sha256"]:
        raise SystemExit("Source trace SHA mismatch")
    output = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") != "student" or row.get("decision_source") != "llm" or row.get("llm_parse_ok") is not True:
            continue
        assert row.get("event") == "decision"
        factors = {k: float(row[k]) for k in ledger["fields"]}
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in factors.values())
        output.append(json.dumps(factors, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    data = b"".join(output)
    assert len(output) == ledger["rows"] and hashlib.sha256(data).hexdigest() == ledger["data_sha256"]
    with args.out.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"numeric_rows": len(output), "fields": len(ledger["fields"]), "private_text_written": False}))

if __name__ == "__main__":
    main()
