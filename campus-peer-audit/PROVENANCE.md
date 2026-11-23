# Evidence boundary

The repository contains the 2,040 finalized calls of the first frozen experiment:
840 Qwen3-4B calls at 11:00, 360 Qwen3-4B peer-only calls at 16:00, and 840
Qwen2.5-7B calls at 11:00. Pilots and historical archive results are excluded.

The framework and existing probe idea are inherited. New work comprises input
and source auditing, an independent executable protocol, anonymous controlled
inputs, newly recorded checkpoint-specific calls, and uncertainty/QA analysis.
Four fixed texts are not four sampled students; repeated calls are not new human
participants. There is no human survey dataset in this release.

The preflight quantiles summarize 458 selected decisions from one supplied
historical simulation trace. Only its five numerical factors are redistributed
in data/calibration_factors.jsonl; there are no names, agent IDs, ticks, actions
or free-text logs in that table. The full trace and original named profiles are
excluded. The source SHA, selection and quantile rule are retained for provenance.
tools/extract_calibration.py documents exact extraction if a rights holder has
the original trace. CPU reproduction needs no private source archive: it recomputes
all calibration quantiles, re-renders each anonymous prompt with the preserved
prompt sources, and verifies the shuffled schedule.

Original prepared, plan, execution, analysis and QA files are byte-identical
copies. The 15 formal run files are losslessly packed in data/formal_runs.zip.
Their per-entry hashes are in data/formal_runs_manifest.json. Runtime directories,
cloud control tools, host addresses, SSH details and provider credentials are
excluded. Model hashes/revisions, numeric seeds, software versions and scientific
timestamps are retained. A successful artifact check is not a claim of independent
model-output regeneration or human behavioral validity.

The separate enhanced_analysis_plan.json records a posthoc CPU robustness layer
specified after the original outcomes had been inspected. Its exact plan, source,
JSON/CSV outputs and overview figures are included under enhanced_analysis/.
The layer adds conservative CP/Bonferroni family intervals, computational-batch
bootstrap sensitivity, and explicitly exploratory time/checkpoint comparisons.
It does not alter the original plans or observations and is not retrospective
preregistration. CPU reproduction can recompute it from the same formal records.
