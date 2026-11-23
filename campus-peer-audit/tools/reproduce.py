"""CPU-only verification and recomputation of the released formal experiment."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=Path("reproduced"))
    p.add_argument("--figures", action="store_true")
    p.add_argument("--enhanced", action="store_true", help="Recompute the posthoc CPU robustness analysis and its figure")
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
    enhanced = None
    if args.enhanced:
        scratch = out / "enhanced_scratch"
        returned = scratch / "remote_return"
        shutil.copytree(extracted / "runs", returned / "runs")
        shutil.copytree(root / "analysis", returned / "analysis")
        for folder in ("prepared", "prepared_time16"):
            shutil.copytree(root / folder, returned / folder)
        for name in ("protocol_freeze.json", "execution_freeze.json", "enhanced_analysis_plan.json"):
            shutil.copyfile(root / name, scratch / name)
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
    result = {"reproduction": "passed", "model_calls": 0, "network_calls": 0, "plans": summaries,
              "enhanced": enhanced}
    (out / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
