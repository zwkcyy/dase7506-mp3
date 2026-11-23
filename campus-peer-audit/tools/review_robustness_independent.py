"""Independent CPU audit fixtures and a read-only census of frozen formal inputs.

No model, Pod or network calls. The optional --out receipt is a new file only.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import analyze_robustness as reviewed


def decimal_tail(n, p, low, high):
    with localcontext() as context:
        context.prec = 60
        prob = Decimal(str(p))
        term = (1 - prob) ** n
        total = Decimal(0)
        for k in range(n + 1):
            if low <= k <= high:
                total += term
            if k < n:
                term *= Decimal(n - k) / Decimal(k + 1) * prob / (1 - prob)
        return float(total)


def check_cp():
    worst_residual = 0.0
    minimum_coverage = {}
    for alpha in (0.05, 0.05 / 24, 0.05 / 32):
        intervals = [reviewed.clopper_pearson(x, 30, alpha) for x in range(31)]
        assert math.isclose(intervals[0][1], 1 - (alpha / 2) ** (1 / 30), abs_tol=1e-14)
        assert math.isclose(intervals[30][0], (alpha / 2) ** (1 / 30), abs_tol=1e-14)
        for x, (lo, hi) in enumerate(intervals):
            assert 0 <= lo <= x / 30 <= hi <= 1
            if x:
                residual = abs(decimal_tail(30, lo, x, 30) - alpha / 2)
                worst_residual = max(worst_residual, residual)
                assert residual < 2e-13
            if x < 30:
                residual = abs(decimal_tail(30, hi, 0, x) - alpha / 2)
                worst_residual = max(worst_residual, residual)
                assert residual < 2e-13
        covered = []
        for j in range(10001):
            p = j / 10000
            value = math.fsum(math.comb(30, x) * p**x * (1 - p)**(30 - x)
                              for x, (lo, hi) in enumerate(intervals) if lo - 1e-14 <= p <= hi + 1e-14)
            covered.append(value)
        minimum_coverage[str(alpha)] = min(covered)
        assert min(covered) >= 1 - alpha - 1e-12
    return {"independent_decimal_recurrence_max_tail_residual": worst_residual,
            "coverage_grid_points": 10001, "minimum_grid_coverage": minimum_coverage}


def check_bootstrap():
    run = {"cells": {"A": {}, "B": {}}, "batches": {
        10: [{"cell_id": "A", "action": "study", "fixture_valid": True},
             {"cell_id": "A", "action": "study", "fixture_valid": False}],
        50: [{"cell_id": "B", "action": "rest", "fixture_valid": True}]}}
    specs = [{"persona_id": "TEST_ONLY", "arm": "peer_signal", "a_cell_id": "A", "b_cell_id": "B", "action": "study"}]
    selections = iter([[0, 0], [0, 1], [1, 0], [1, 1]])

    class FixedSelections:
        def choices(self, population, k):
            assert list(population) == [0, 1] and k == 2
            return next(selections)

    oracle = SimpleNamespace(validity=lambda row: (row["fixture_valid"], []))
    with patch.object(reviewed.random, "Random", return_value=FixedSelections()):
        values = reviewed.bootstrap_run(run, specs, oracle, 4, 123)
    assert values["TEST_ONLY/peer_signal"] == [None, 0.5, 0.5, None]
    assert specs[0]["batch_bootstrap"]["defined_replicates"] == 2
    assert specs[0]["batch_bootstrap"]["undefined_replicates"] == 2
    assert specs[0]["batch_bootstrap"]["percentile95"] == [0.5, 0.5]
    assert specs[0]["batch_bootstrap_valid_only"]["percentile95"] == [1.0, 1.0]
    assert reviewed.bootstrap_summary([None, None], 2)["percentile95"] is None
    # Positive completed denominators but zero valid denominators must remain
    # defined for the joint outcome and undefined for the valid-only outcome.
    invalid_run = json.loads(json.dumps(run))
    for rows in invalid_run["batches"].values():
        for row in rows:
            row["fixture_valid"] = False
    selections = iter([[0, 0], [0, 1], [1, 0], [1, 1]])
    with patch.object(reviewed.random, "Random", return_value=FixedSelections()):
        invalid_values = reviewed.bootstrap_run(invalid_run, specs, oracle, 4, 123)
    assert invalid_values["TEST_ONLY/peer_signal"] == [None, 0.0, 0.0, None]
    assert specs[0]["batch_bootstrap_valid_only"]["undefined_replicates"] == 4
    assert specs[0]["batch_bootstrap_valid_only"]["percentile95"] is None
    return {"unequal_batch_sizes": [2, 1], "nonconsecutive_batch_ids": [10, 50],
            "enumerated_batch_draws": 4, "undefined_primary_draws": 2,
            "observed_primary_effect": 0.5, "observed_valid_only_effect": 1.0,
            "zero_valid_positive_completed_denominator_handling": "passed"}


def check_did():
    runs, boot = {}, {}
    for name, low, high, values in (("qwen3_main", 30, 0, [-1, -1, None, 0]),
                                    ("qwen3_time16", 0, 30, [1, 1, 1, 0]),
                                    ("qwen25_main", 15, 15, [0, 0, 0, 0])):
        by_key = {}
        for persona in reviewed.PERSONAS:
            for level, count in (("q10", low), ("q50", 15), ("q90", high)):
                by_key[(persona, "peer_signal", level)] = {
                    "actions": {"study": {"valid_count": count}}, "completed": 30, "valid": 30}
        runs[name] = {"by_key": by_key}
        boot[name] = {persona + "/peer_signal": list(values) for persona in reviewed.PERSONAS}
    oracle = SimpleNamespace(difference=lambda a, na, b, nb: {"estimate": a / na - b / nb,
                                                              "newcombe_wilson95": None})
    dose, differences = reviewed.context_comparisons(runs, boot, oracle, 4)
    assert len(dose) == 24 and len(differences) == 8
    for row in differences:
        assert row["estimate"] == (2 if row["comparison"] == "time" else 1)
        assert row["batch_bootstrap"]["undefined_replicates"] == 1
    return {"time_extreme_difference_of_differences": 2,
            "equivalent_percentage_points": 200,
            "parameter_domain_probability_units": [-2, 2],
            "cross_run_difference_undefined_preserved": True}


def read_only_census(project):
    plan = reviewed.read_json(project / "enhanced_analysis_plan.json")
    reviewed.validate_plan(plan)
    for family in ("peer_family", "controls_family"):
        altered = json.loads(json.dumps(plan))
        altered["multiplicity"][family]["two_sided_CP_alpha_per_endpoint"] = 0.05
        try:
            reviewed.validate_plan(altered)
        except reviewed.RobustnessError:
            pass
        else:
            raise AssertionError("Misallocated endpoint alpha was accepted")
    original, _ = reviewed.load_original_analyzer()
    runs, _ = reviewed.load_all(project, original)
    endpoints = {"peer_family": set(), "controls_family": set()}
    rows_by_role = {"peer_endpoints": 0, "peer_center": 0, "state_control_endpoints": 0}
    per_run = {}
    for name, run in runs.items():
        per_run[name] = {"rows": len(run["finals"]), "batches": len(run["batches"])}
        assert sum(len(rows) for rows in run["batches"].values()) == len(run["finals"])
        for effect in run["analysis"]["effects"]:
            family = "peer_family" if effect["arm"] == "peer_signal" else "controls_family"
            target = reviewed.TARGETS[effect["arm"]]
            for key in ("a_cell_id", "b_cell_id"):
                endpoints[family].add((name, effect[key], target))
        for cell in run["cells"].values():
            role = "state_control_endpoints" if cell["arm"] != "peer_signal" else (
                "peer_center" if cell["level"] == "q50" else "peer_endpoints")
            rows_by_role[role] += cell["completed"]
    assert len(endpoints["peer_family"]) == 24 and len(endpoints["controls_family"]) == 32
    assert rows_by_role == {"peer_endpoints": 720, "peer_center": 360, "state_control_endpoints": 960}
    assert sum(rows_by_role.values()) == 2040
    return {"runs": per_run, "unique_endpoint_proportions": {k: len(v) for k, v in endpoints.items()},
            "rows_by_role": rows_by_role, "formal_call_census": 2040,
            "coverage_argument": "Each family allocates 0.05/M to M two-sided endpoint intervals. If all endpoints cover, every endpoint-difference interval covers. The union bound gives family coverage at least 0.95 under the stated marginal binomial assumptions; it needs no independence among endpoints."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=reviewed.PROJECT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {"status": "passed", "scope": "independent CPU mathematical fixtures and read-only frozen-data census",
              "cp": check_cp(), "bootstrap": check_bootstrap(), "exploratory_did": check_did(),
              "census": read_only_census(args.project.resolve()),
              "reviewed_script_sha256": hashlib.sha256(Path(reviewed.__file__).read_bytes()).hexdigest(),
              "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
