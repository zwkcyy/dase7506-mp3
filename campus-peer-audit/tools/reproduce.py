"""CPU-only verification and recomputation of the released formal experiment."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

MEASUREMENT_CSVS = ("call_classification.csv", "all_cells.csv", "action_distributions.csv", "duration_values.csv",
                    "optional_failure_reasons.csv", "peer_contrasts.csv", "control_contrasts.csv")
MEASUREMENT_FIGURES = ("peer_dose_measurement_audit.pdf", "peer_dose_measurement_audit.png")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def prepare_scratch(root, out, extracted, plan_names):
    """Copy verified raw records and original analysis bytes into a new layout."""
    scratch = out / "analysis_scratch"
    returned = scratch / "remote_return"
    shutil.copytree(extracted / "runs", returned / "runs")
    shutil.copytree(root / "analysis", returned / "analysis")
    for folder in ("prepared", "prepared_time16"):
        shutil.copytree(root / folder, returned / folder)
    for name in ("protocol_freeze.json", "execution_freeze.json", *plan_names):
        shutil.copyfile(root / name, scratch / name)
    return scratch

def verify_measurement_receipt(folder, extra_files=()):
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    expected = {"audit.json", *MEASUREMENT_CSVS, *MEASUREMENT_FIGURES}
    if set(receipt["files_sha256"]) != expected or {p.name for p in folder.iterdir() if p.is_file()} != expected | {"receipt.json", *extra_files}:
        raise SystemExit("Measurement output inventory differs")
    for name in expected:
        if digest((folder / name).read_bytes()) != receipt["files_sha256"][name]:
            raise SystemExit("Measurement output receipt hash mismatch")
    if not (folder / MEASUREMENT_FIGURES[0]).read_bytes().startswith(b"%PDF-"):
        raise SystemExit("Measurement PDF missing or invalid")
    if not (folder / MEASUREMENT_FIGURES[1]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("Measurement PNG missing or invalid")
    return receipt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=Path("reproduced"))
    p.add_argument("--figures", action="store_true")
    p.add_argument("--enhanced", action="store_true", help="Recompute the posthoc CPU robustness analysis and its figure")
    p.add_argument("--measurement", action="store_true", help="Recompute the separate post-result raw-text measurement sensitivity")
    args = p.parse_args()
    root = args.root.resolve()
    out = (root / args.out).resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        raise SystemExit("Use a new output directory inside the release")
    release = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    for name, spec in release["files"].items():
        data = (root / name).read_bytes()
        if len(data) != spec["bytes"] or digest(data) != spec["sha256"]:
            raise SystemExit("Release hash mismatch")
    subprocess.run([sys.executable, "-B", str(root / "tools/check_inputs.py"), "--root", str(root)], check=True)
    out.mkdir(parents=True, exist_ok=False)
    extracted = out / "extracted"
    inventory = json.loads((root / "data/formal_runs_manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(root / "data/formal_runs.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(inventory["files"]):
            raise SystemExit("Archive inventory mismatch")
        for name, spec in inventory["files"].items():
            pure = PurePosixPath(name)
            target = extracted.joinpath(*pure.parts).resolve()
            if pure.is_absolute() or "\\" in name or not target.is_relative_to(extracted):
                raise SystemExit("Unsafe archive entry")
            data = archive.read(name)
            if len(data) != spec["bytes"] or digest(data) != spec["sha256"]:
                raise SystemExit("Raw entry hash mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    plans = json.loads((root / "protocol_freeze.json").read_text(encoding="utf-8"))["formal_plans"]
    summaries = []
    for plan in plans:
        name, prepared = plan["name"], root / plan["prepared"]
        run, analysis = extracted / "runs" / name, out / "analysis" / name
        subprocess.run([sys.executable, "-B", str(root / "tools/analyze_experiment.py"), "--prepared", str(prepared),
                        "--run", str(run), "--out", str(analysis)], check=True)
        subprocess.run([sys.executable, "-B", str(root / "tools/verify_results.py"), "--root", str(root), "--prepared", str(prepared),
                        "--run", str(run), "--analysis", str(analysis), "--out", str(out / "checks" / (name + ".json"))], check=True)
        actual = json.loads((analysis / "analysis.json").read_text(encoding="utf-8"))
        included = json.loads((root / "analysis" / name / "analysis.json").read_text(encoding="utf-8"))
        for key in ("totals", "cells", "effects", "fixed_profile_averages"):
            if actual[key] != included[key]:
                raise SystemExit("Recomputed scientific content differs")
        if args.figures:
            subprocess.run([sys.executable, "-B", str(root / "tools/plot_results.py"), "--analysis", str(analysis / "analysis.json"),
                            "--out", str(out / "figures" / name)], check=True)
        summaries.append({"plan": name, "calls": actual["totals"]["completed"], "numerical_content_matches": True})
    plan_names = (["enhanced_analysis_plan.json"] if args.enhanced else []) + (["measurement_audit_plan.json"] if args.measurement else [])
    scratch = prepare_scratch(root, out, extracted, plan_names) if plan_names else None
    enhanced = None
    if args.enhanced:
        enhanced_out = out / "enhanced_analysis"
        subprocess.run([sys.executable, "-B", str(root / "tools/analyze_robustness.py"), "--project", str(scratch),
                        "--plan", str(scratch / "enhanced_analysis_plan.json"), "--out", str(enhanced_out)], check=True)
        subprocess.run([sys.executable, "-B", str(root / "tools/review_robustness_independent.py"), "--project", str(scratch),
                        "--out", str(out / "checks/robustness_independent.json")], check=True)
        actual = json.loads((enhanced_out / "robustness.json").read_text(encoding="utf-8"))
        included = json.loads((root / "enhanced_analysis/robustness.json").read_text(encoding="utf-8"))
        metadata = {"generated_at_utc", "plan_sha256", "tool_sha256", "original_analyzer_sha256", "source_hashes", "versions"}
        scientific_actual = {k: v for k, v in actual.items() if k not in metadata}
        scientific_included = {k: v for k, v in included.items() if k not in metadata}
        if scientific_actual != scientific_included:
            raise SystemExit("Recomputed enhancement scientific content differs")
        # The original bytes and same relative layout should also preserve these hashes.
        for key in ("plan_sha256", "tool_sha256", "original_analyzer_sha256", "source_hashes"):
            if actual[key] != included[key]:
                raise SystemExit("Enhancement source provenance differs")
        csv_names = sorted(p.name for p in (root / "enhanced_analysis").glob("*.csv"))
        if len(csv_names) != 5 or any((enhanced_out / name).read_bytes() != (root / "enhanced_analysis" / name).read_bytes()
                                      for name in csv_names):
            raise SystemExit("Recomputed enhancement CSV differs")
        review = json.loads((out / "checks/robustness_independent.json").read_text(encoding="utf-8"))
        if review["status"] != "passed" or review["census"]["formal_call_census"] != 2040:
            raise SystemExit("Independent enhancement review failed")
        enhanced = {"stage": actual["analysis_stage"], "scientific_content_matches": True, "source_provenance_matches": True,
                    "csv_files_identical": csv_names, "independent_review": "passed", "formal_calls": actual["formal_calls"]}
    measurement = None
    if args.measurement:
        measurement_out = out / "measurement_audit"
        subprocess.run([sys.executable, "-B", str(root / "tools/analyze_measurement_audit.py"), "--self-check"], check=True)
        subprocess.run([sys.executable, "-B", str(root / "tools/analyze_measurement_audit.py"), "--project", str(scratch),
                        "--plan", str(scratch / "measurement_audit_plan.json"), "--out", str(measurement_out)], check=True)
        actual = json.loads((measurement_out / "audit.json").read_text(encoding="utf-8"))
        included_folder = root / "measurement_audit"
        included = json.loads((included_folder / "audit.json").read_text(encoding="utf-8"))
        metadata = {"created_at_utc", "tool_sha256", "hash_validation_and_interval_helper_sha256", "source_hashes"}
        if {k: v for k, v in actual.items() if k not in metadata} != {k: v for k, v in included.items() if k not in metadata}:
            raise SystemExit("Recomputed measurement scientific content differs")
        for key in ("tool_sha256", "hash_validation_and_interval_helper_sha256", "source_hashes"):
            if actual[key] != included[key]:
                raise SystemExit("Measurement source provenance differs")
        for name in MEASUREMENT_CSVS:
            if (measurement_out / name).read_bytes() != (included_folder / name).read_bytes():
                raise SystemExit("Recomputed measurement CSV differs")
        actual_receipt = verify_measurement_receipt(measurement_out)
        included_receipt = verify_measurement_receipt(included_folder, ("independent_review.json",))
        for key in ("analysis_stage", "formal_calls", "tool_sha256", "source_hashes"):
            if actual_receipt[key] != included_receipt[key] or actual_receipt[key] != actual[key]:
                raise SystemExit("Measurement receipt provenance differs")
        review_path = out / "checks/measurement_independent.json"
        subprocess.run([sys.executable, "-B", str(root / "tools/review_measurement_independent.py"), "--project", str(scratch),
                        "--audit", str(measurement_out), "--out", str(review_path)], check=True)
        review = json.loads(review_path.read_text(encoding="utf-8"))
        included_review = json.loads((included_folder / "independent_review.json").read_text(encoding="utf-8"))
        if review["status"] != "passed" or review["census"]["formal_calls"] != 2040:
            raise SystemExit("Independent measurement review failed")
        for key in ("fixtures", "reviewed_script_sha256", "reviewer_script_sha256"):
            if review[key] != included_review[key]:
                raise SystemExit("Independent measurement review differs")
        if review["reviewed_script_sha256"] != actual["tool_sha256"]:
            raise SystemExit("Independent reviewer checked another measurement tool")
        review_metadata = {"audit_artifact_hashes", "audit_receipt_sha256"}
        if {k: v for k, v in review["census"].items() if k not in review_metadata} != {k: v for k, v in included_review["census"].items() if k not in review_metadata}:
            raise SystemExit("Independent measurement census differs")
        measurement = {"stage": actual["analysis_stage"], "formal_calls": actual["formal_calls"],
                       "scientific_content_matches": True, "source_provenance_matches": True,
                       "csv_files_identical": list(MEASUREMENT_CSVS), "tool_self_check": "passed",
                       "independent_review": "passed", "independent_census_matches": True,
                       "figure_receipt_hashes_verified": True,
                       "figure_hashes": {name: actual_receipt["files_sha256"][name] for name in MEASUREMENT_FIGURES},
                       "png_bytes_identical": actual_receipt["files_sha256"][MEASUREMENT_FIGURES[1]] == included_receipt["files_sha256"][MEASUREMENT_FIGURES[1]]}
    result = {"reproduction": "passed", "model_calls": 0, "network_calls": 0, "plans": summaries,
              "enhanced": enhanced, "measurement": measurement}
    (out / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
