# Campus LLM peer-signal experiment

Reproducible artifacts for DASE7506 by **Wenkang Zhu**.
Repository: https://github.com/zwkcyy/dase7506-mp3/tree/main/campus-peer-audit.
The independent SQL project already in the repository is preserved.

The study compares next-action distributions for four fixed anonymous profile
texts under controlled peer intensity and state changes. It tests a single-step
decision interface, rather than a complete campus simulation. The original
framework, campus prompts and probe idea are credited in NOTICE.md.

## Reproduce tables and checks on CPU

Python 3.12.14 was used for the published CPU calculations. No pip dependency,
GPU, model download, API key, cloud account or network call is needed:

```sh
python -B tools/reproduce.py --out reproduced
```

The command verifies release/file hashes, safely extracts the 15 formal files
into a new output directory, re-renders/validates the anonymous inputs, recomputes
all three analyses, checks the raw/token/seed/protocol evidence independently,
and compares numerical results with the included analyses. Existing output
directories are refused. A failure returns a nonzero exit code.

Optional figures use exact package pins:

```sh
python -m pip install -r requirements-figures.lock
python -B tools/reproduce.py --out reproduced_with_figures --figures
```

Recompute the full **posthoc** robustness layer, including its 10,000-resample
batch sensitivity analysis and overview figures, with the same figure packages:

```sh
python -m pip install -r requirements-figures.lock
python -B tools/reproduce.py --out reproduced_enhanced --enhanced
```

This also runs independent mathematical fixtures and a read-only call census,
then compares all scientific JSON fields and CSV bytes against the included
enhancement. Generation timestamps and figure metadata may differ. Original
run and analysis bytes are copied into a separate scratch layout; the released
data remain untouched. The original outcomes were already inspected before
enhanced_analysis_plan.json was written. Its family intervals and exploratory
contrasts are reported separately from the original unadjusted analysis.

## Data and scope

The complete 2,040 formal calls are included in `data/formal_runs.zip`, with a
per-entry SHA-256 manifest. `prepared/` has the 28-cell/840-call 11:00 design;
`prepared_time16/` has the 12-cell/360-call peer-only extension. `analysis/` holds
the original CPU summaries, and `checks/` the independent acceptance checks.
No pilot or historical effect estimate is pooled into this dataset. The original
private source archive, named profiles, human-survey claims, and infrastructure
logs are excluded. The five-factor/458-row calibration table and separate
anonymous profile-text bundles are also included. See PROVENANCE.md and
PUBLIC_DATA_AUDIT.json. Reproduction does not require the private original ZIP.

Main action: `study`; `go_class` is separate. Invalid and truncated responses
remain in the accounting. Wilson/Newcombe intervals are unadjusted individual
intervals. An interval containing zero is not proof of no effect. Four fixed
profile texts and two checkpoints do not support student-population or model-size
causal claims; size and model generation vary together across checkpoints.

## Optional generation from the frozen checkpoints

This is separate from CPU reproduction and requires a suitable CUDA GPU. The
recorded environment and exact checkpoint revisions are in MODEL_PROVENANCE.json.
Install `requirements-gpu.lock` into a separate Linux/Python environment. These
pins describe the observed CUDA 12.8 generation environment, rather than a claim
of cross-platform or bitwise determinism.

Obtain each model from its official model card at the pinned revision, preserving
the downloaded LICENSE. For example:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B-Instruct-2507",
    revision="cdbee75f17c01a7cc42f958dc650907174af0554",
    local_dir="models/Qwen3-4B-Instruct-2507")
snapshot_download("Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    local_dir="models/Qwen2.5-7B-Instruct")
```

Generate into a new directory; do not overwrite released observations:

```sh
python tools/run_experiment_batch.py --execute --prepared prepared   --out runs_new/qwen3_main --model Qwen/Qwen3-4B-Instruct-2507   --model-dir models/Qwen3-4B-Instruct-2507   --revision cdbee75f17c01a7cc42f958dc650907174af0554   --batch-size 16 --max-tokens 1024 --seed 20261122
```

For Qwen2.5 use its ID/revision/model directory; for the time extension use
`--prepared prepared_time16`. Sampling is explicit: temperature 1, top-p 1,
top-k 0, and `use_model_defaults=False`. One PRNG stream is shared per batch;
batch seeds and row positions are recorded. The inherited ` /no_think` suffix is
preserved and hashed. No claim of reproducing the archive's historical serving
model is made. Model weights, inference outputs from a new rerun, and local
environments are ignored by Git.

## Report source

The English conference-style report source is in `paper/` (5 pages of main text,
followed by references and appendices). The abstract contains this public
repository link. The submission-only student identifier is intentionally not
public; `\StudentID` defaults to empty.

Compile using Tectonic 0.17.0+20260731 (or a compatible LaTeX toolchain with
IEEEtran, amsmath, graphicx, booktabs, array, hyperref and balance):

```sh
cd paper
tectonic main.tex
```

The first Tectonic build may download standard TeX packages. This is separate
from the offline statistical reproduction command. To regenerate the data-bound
LaTeX results and full control table, run from the repository root:

```sh
python tools/build_paper_tables.py --project . --out paper
```

`paper/figures/robustness_overview.pdf` is the enhancement figure; its numerical
source and generation script are included. Original and strengthened analyses
are distinguished in the report, including the post-result chronology.
