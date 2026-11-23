#!/usr/bin/env python3
"""Make paper figures from a real analyze_experiment.py analysis.json.

    python tools/plot_results.py --analysis analysis/model01/analysis.json --out figures/model01

Requires matplotlib and numpy. Writes PDF/300-dpi PNG figures, captions.md,
and figures.json with source hashes. Pending and fixture data are always refused.
Incomplete real runs require --allow-partial and are visibly marked PARTIAL.
Accepts the 28-cell main design or the 12-cell peer-only time extension.
This script performs no model calls and never derives outcomes from old traces.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


PERSONAS = ("P01", "P02", "P03", "P04")
ACTIONS = ("study", "go_class", "exercise", "chat", "rest", "eat", "wander")
COLORS = {"P01": "#0072B2", "P02": "#D55E00", "P03": "#009E73", "P04": "#CC79A7"}
MARKERS = {"P01": "o", "P02": "s", "P03": "^", "P04": "D"}


class PlotError(Exception):
    pass


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load_analysis(path, allow_partial):
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise PlotError("Cannot read a valid analysis.json") from None
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise PlotError("Unsupported analysis schema")
    if data.get("fixture") is not False or data.get("data_kind") != "EXPERIMENT":
        raise PlotError("Fixture or unlabelled data cannot be plotted as experimental results")
    if data.get("status") == "pending" or data.get("totals", {}).get("completed", 0) == 0:
        raise PlotError("Pending data contain no observed effects; no figure was created")
    if data.get("status") not in ("partial", "complete"):
        raise PlotError("Unknown analysis completion status")
    if data.get("status") != "complete" and not allow_partial:
        raise PlotError("Incomplete run: finish the planned run or explicitly use --allow-partial")
    cells = data.get("cells")
    effects = data.get("effects")
    if not isinstance(cells, list) or len(cells) not in (12, 28) or not isinstance(effects, list):
        raise PlotError("Expected the 28-cell main or 12-cell peer-only four-profile schema")
    by_cell = {}
    for cell in cells:
        key = (cell.get("persona_id"), cell.get("arm"), cell.get("level"))
        if key in by_cell or key[0] not in PERSONAS:
            raise PlotError("Unexpected or repeated persona/arm/level")
        if cell.get("completed", 0) <= 0:
            raise PlotError("A plotted cell has zero completed calls; no outcome is imputed")
        count_keys = ("planned", "completed", "not_completed", "valid", "invalid")
        if any(type(cell.get(key)) is not int or cell[key] < 0 for key in count_keys):
            raise PlotError("A plotted cell has invalid completion counts")
        if (cell["planned"] != cell["completed"] + cell["not_completed"] or
                cell["completed"] != cell["valid"] + cell["invalid"]):
            raise PlotError("A plotted cell has inconsistent completed/valid/missing counts")
        try:
            action_counts = [cell["actions"][action]["valid_count"] for action in ACTIONS]
        except (KeyError, TypeError):
            raise PlotError("The complete seven-action valid distribution is missing") from None
        if any(type(value) is not int or value < 0 for value in action_counts) or sum(action_counts) != cell["valid"]:
            raise PlotError("The seven-action valid distribution does not match the valid total")
        try:
            level = float(cell["level_rendered"])
        except (KeyError, ValueError, TypeError):
            raise PlotError("A plotted cell lacks a numeric rendered manipulation level") from None
        if not math.isfinite(level) or not 0 <= level <= 1:
            raise PlotError("A plotted manipulation level is outside [0,1]")
        by_cell[key] = cell
    arms = [("peer_signal", ("q10", "q50", "q90"))]
    if len(cells) == 28:
        arms += [("energy_control", ("q10", "q90")), ("hunger_control", ("q10", "q90"))]
    expected = {(persona, arm, level) for persona in PERSONAS for arm, levels in arms for level in levels}
    if set(by_cell) != expected:
        raise PlotError("The plotted cells do not match the complete planned design")
    if any(sum(c[key] for c in cells) != data["totals"].get(key) for key in count_keys):
        raise PlotError("Analysis completion totals are inconsistent")
    if data["status"] == "complete" and data["totals"]["not_completed"] != 0:
        raise PlotError("A complete analysis still contains uncompleted calls")
    by_effect = {(row.get("persona_id"), row.get("arm")): row for row in effects}
    if len(by_effect) != len(effects):
        raise PlotError("Repeated persona/arm effect")
    for persona in PERSONAS:
        effect = by_effect.get((persona, "peer_signal"))
        if not effect or effect.get("contrast") != "q90_minus_q10":
            raise PlotError("Peer contrast is missing or has an unexpected direction")
        high, low = by_cell[(persona, "peer_signal", "q90")], by_cell[(persona, "peer_signal", "q10")]
        if effect.get("a_cell_id") != high["cell_id"] or effect.get("b_cell_id") != low["cell_id"]:
            raise PlotError("Peer effect endpoints disagree with the high/low cells")
        try:
            primary = effect["primary_study"]
            result = primary["valid_and_action_per_completed"]
            if primary["action"] != "study":
                raise PlotError("Peer primary effect must use study only")
            for endpoint, cell in (("a", high), ("b", low)):
                if (result[endpoint]["denominator"] != cell["completed"] or
                        result[endpoint]["count"] != cell["actions"]["study"]["valid_count"]):
                    raise PlotError("Peer effect does not use valid-study counts per completed call")
            expected_difference = (result["a"]["count"] / result["a"]["denominator"] -
                                   result["b"]["count"] / result["b"]["denominator"])
            if not math.isclose(result["estimate"], expected_difference, rel_tol=1e-12, abs_tol=1e-12):
                raise PlotError("Peer effect estimate disagrees with the displayed endpoint probabilities")
        except (KeyError, TypeError):
            raise PlotError("Peer primary effect count/denominator metadata are missing") from None
    return data, by_cell, by_effect, sha(raw)


def get_rate(cell, action):
    try:
        rate = cell["actions"][action]["valid_and_action_per_completed"]
        point = float(rate["estimate"])
        lower, upper = map(float, rate["wilson95"])
    except (KeyError, TypeError, ValueError):
        raise PlotError("A plotted action probability or Wilson interval is missing") from None
    if not all(math.isfinite(x) for x in (point, lower, upper)) or not 0 <= lower <= point <= upper <= 1:
        raise PlotError("A plotted probability interval is inconsistent")
    if (rate.get("denominator") != cell["completed"] or
            rate.get("count") != cell["actions"][action]["valid_count"] or
            not math.isclose(point, rate["count"] / cell["completed"], rel_tol=1e-12, abs_tol=1e-12)):
        raise PlotError("Plotted rates must use valid-action counts divided by completed calls")
    return point, max(0.0, point - lower), max(0.0, upper - point)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def response_panel(ax, by_cell, arm, action, levels, title, xlabel, percent_formatter):
    ticks = None
    for persona in PERSONAS:
        rows = [by_cell[(persona, arm, level)] for level in levels]
        xs = [float(row["level_rendered"]) for row in rows]
        values = [get_rate(row, action) for row in rows]
        points = [value[0] for value in values]
        errors = [[value[index] for value in values] for index in (1, 2)]
        ax.errorbar(xs, points, yerr=errors, color=COLORS[persona], marker=MARKERS[persona],
                    linewidth=1.4, markersize=4.5, capsize=3, elinewidth=1.0, label=persona)
        if ticks is None:
            ticks = xs
        elif xs != ticks:
            raise PlotError("Profiles do not share identical numeric manipulation levels")
    ax.set_xticks(ticks, [f"{x:.2f}" for x in ticks])
    width = max(ticks) - min(ticks)
    ax.set_xlim(max(0, min(ticks) - width * 0.13), min(1, max(ticks) + width * 0.13))
    ax.set_ylim(-0.025, 1.025)
    ax.yaxis.set_major_formatter(percent_formatter(xmax=1))
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Valid {action} / completed calls")
    style_axes(ax)


def forest_panel(ax, by_effect):
    extents = [0.0]
    invalid_shown = False
    for y, persona in enumerate(PERSONAS):
        row = by_effect[(persona, "peer_signal")]["primary_study"]
        result = row["valid_and_action_per_completed"]
        try:
            point = float(result["estimate"]) * 100
            lower, upper = [float(x) * 100 for x in result["newcombe_wilson95"]]
        except (KeyError, ValueError, TypeError):
            raise PlotError("A peer effect or Newcombe interval is missing") from None
        if not all(math.isfinite(x) for x in (point, lower, upper)) or not -100 <= lower <= point <= upper <= 100:
            raise PlotError("A peer effect interval is inconsistent")
        extents.extend((lower, upper))
        bounds = row.get("invalid_outcome_identification_range")
        if bounds is not None and (len(bounds) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bounds) or not -1 <= bounds[0] <= bounds[1] <= 1):
            raise PlotError("Invalid-output identification bounds are malformed")
        if bounds is not None and abs(bounds[1] - bounds[0]) > 1e-12:
            ax.plot([100 * bounds[0], 100 * bounds[1]], [y, y], color="#B3B9C3", linewidth=4,
                    solid_capstyle="butt", alpha=0.65, label="Invalid-output bounds" if not invalid_shown else None)
            invalid_shown = True
            extents.extend([100 * bounds[0], 100 * bounds[1]])
        ax.errorbar(point, y, xerr=[[max(0, point - lower)], [max(0, upper - point)]],
                    color=COLORS[persona], marker=MARKERS[persona], markersize=5,
                    linewidth=1.3, capsize=4, linestyle="none")
    ax.axvline(0, color="#555C65", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_yticks(range(len(PERSONAS)), PERSONAS)
    ax.set_ylim(len(PERSONAS) - 0.5, -0.5)
    span = max(10.0, max(extents) - min(extents))
    ax.set_xlim(max(-100, min(extents) - span * 0.12), min(100, max(extents) + span * 0.12))
    ax.set_title("B  High minus low peer signal", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Study effect (percentage points)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D9DDE3", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if invalid_shown:
        ax.legend(loc="lower right", fontsize=7, frameon=False)


def create_figures(path, output, allow_partial=False, model_label=None, condition_label=None):
    if output.exists():
        raise PlotError("Output already exists; use a new --out directory")
    if any(part.lower() == "reference" for part in output.parts):
        raise PlotError("Output must be outside reference/")
    data, cells, effects, source_hash = load_analysis(path, allow_partial)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        import numpy
    except ImportError:
        raise PlotError("Install plotting dependencies in this environment: python -m pip install matplotlib numpy") from None
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    title = str(model_label or data.get("model_requested") or "Observed model")[:100]
    if condition_label:
        title += " | " + str(condition_label)[:60]
    partial = data["status"] != "complete"
    if partial:
        title = "PARTIAL RUN: " + title
    totals = data["totals"]
    footer = (f"Completed {totals['completed']}/{totals['planned']}; valid {totals['valid']}; invalid {totals['invalid']}. "
              "Pointwise 95% intervals; four fixed profiles.")
    figure1, (curve, forest) = plt.subplots(1, 2, figsize=(10.0, 4.1), gridspec_kw={"width_ratios": [1.3, 1]})
    response_panel(curve, cells, "peer_signal", "study", ("q10", "q50", "q90"),
                   "A  Study response to peer signal", "Peer study signal shown in prompt", PercentFormatter)
    handles, labels = curve.get_legend_handles_labels()
    figure1.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False)
    forest_panel(forest, effects)
    figure1.suptitle(title, fontsize=11, y=0.97)
    figure1.text(0.02, 0.035, footer, fontsize=7.5)
    figure1.text(0.02, 0.005, "Curves: Wilson intervals. Effects: Newcombe Wilson hybrid intervals. Gray ranges, if shown: invalid-output identification bounds (not CIs).", fontsize=7.0)
    figure1.subplots_adjust(left=0.075, right=0.98, top=0.84, bottom=0.20, wspace=0.40)
    figures = [(figure1, "fig1_peer_response")]
    has_controls = len(data["cells"]) == 28
    if has_controls:
        figure2, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
        response_panel(axes[0], cells, "energy_control", "rest", ("q10", "q90"),
                       "A  Own energy: rest control", "Energy shown in prompt\n(lower = more tired)", PercentFormatter)
        response_panel(axes[1], cells, "hunger_control", "eat", ("q10", "q90"),
                       "B  Own satiety: eat control", "Satiety shown in prompt\n(lower = hungrier)", PercentFormatter)
        # Both controls use the same probability scale. A single left-axis
        # label avoids squeezing a rotated second label against its tick text.
        axes[0].set_ylabel("Valid target action / completed calls")
        axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
        axes[0].yaxis.tick_left()
        axes[0].tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
        handles, labels = axes[0].get_legend_handles_labels()
        figure2.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False)
        figure2.suptitle(title, fontsize=11, y=0.97)
        figure2.text(0.02, 0.035, footer, fontsize=7.5)
        figure2.text(0.02, 0.005, "Outcome = valid target action / completed calls. Error bars: Wilson 95% intervals. Control contrasts in analysis.json are low minus high.", fontsize=7.0)
        figure2.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.24, wspace=0.24)
        figures.append((figure2, "fig2_state_controls"))
    if sha(path.read_bytes()) != source_hash:
        plt.close("all")
        raise PlotError("Analysis changed during plotting; retry on a stable file")
    output.mkdir(parents=True, exist_ok=False)
    names = []
    for figure, basename in figures:
        for extension in ("pdf", "png"):
            filename = f"{basename}.{extension}"
            figure.savefig(output / filename, dpi=300, bbox_inches="tight")
            names.append(filename)
        plt.close(figure)
    controls_caption = """**Figure 2.** State-response controls for the same four profiles. Panel A shows valid rest decisions under low and high own energy. Panel B shows valid eat decisions under low and high satiety; the source variable is named `hunger`, but larger values mean more satiated. The denominator is completed calls, and error bars are Wilson 95% intervals. Control contrasts in analysis.json use low minus high.""" if has_controls else "This peer-only time extension does not repeat the state controls."
    captions = f"""# Figure captions

{('**PARTIAL RUN: these figures are provisional and do not represent the complete planned experiment.**' if partial else '')}

**Figure 1.** Response to the peer study signal for four fixed anonymous persona texts. Panel A shows the valid-study indicator divided by completed calls at the three numeric signal levels actually shown in the prompt; error bars are Wilson 95% intervals. Connecting segments guide the eye. Panel B shows the high-minus-low contrast separately for each profile, with Newcombe Wilson hybrid 95% intervals. Any gray ranges show the worst-case identification bounds obtained by allowing invalid completed outputs to take either latent action value; those ranges are not confidence intervals.

{controls_caption}

All intervals are pointwise and unadjusted for multiple comparisons. Outcomes with no final record are not counted as failures. `study` is the primary action; `go_class` is separate. These plots describe four fixed profiles for {title}; they do not estimate a student-population effect. Complete seven-action distributions, valid-only sensitivity analyses, and missingness counts remain in analysis.json.
"""
    (output / "captions.md").write_text(captions, encoding="utf-8")
    metadata = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_analysis_sha256": source_hash, "analysis_status": data["status"],
        "model_label": title, "fixture": False, "partial_run": partial,
        "condition_label": condition_label,
        "execution_engine": data.get("execution_engine"), "invocation_schema": data.get("invocation_schema"),
        "prepared_hashes": data.get("prepared_hashes"), "run_hashes": data.get("run_hashes"),
        "plot_script_sha256": sha(Path(__file__).read_bytes()),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": numpy.__version__},
        "outcome_denominator": "completed calls", "primary_action": "study",
        "intervals": {"curve": "Wilson 95%", "effect": "Newcombe independent-proportions Wilson hybrid 95%"},
        "files_sha256": {name: sha((output / name).read_bytes()) for name in names + ["captions.md"]},
    }
    (output / "figures.json").write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True, help="Real analysis.json from analyze_experiment.py")
    parser.add_argument("--out", type=Path, required=True, help="New figure output directory")
    parser.add_argument("--model-label", help="Optional short display label; does not alter results")
    parser.add_argument("--condition-label", help="Optional explicit context label, e.g. 11:00 or 16:00; does not alter results")
    parser.add_argument("--allow-partial", action="store_true", help="Permit nonempty partial runs with all plotted cells observed; figures are marked PARTIAL")
    args = parser.parse_args()
    try:
        result = create_figures(args.analysis.resolve(), args.out.resolve(), args.allow_partial, args.model_label, args.condition_label)
    except (PlotError, OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Plotting refused: {exc}\n")
    print(f"Wrote PDF+PNG figures for {result['analysis_status']} real data to {args.out.resolve()}")


if __name__ == "__main__":
    main()
