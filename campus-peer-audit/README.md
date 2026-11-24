# Campus LLM peer-signal experiment

Reproducible artifacts for DASE7506 by **Wenkang Zhu**.
Repository: https://github.com/zwkcyy/dase7506-mp3/tree/main/campus-peer-audit.
The independent SQL project already in the repository is preserved.

The study compares next-action distributions for four fixed anonymous profile
texts under controlled peer intensity and state changes. It tests a single-step
decision interface, rather than a complete campus simulation. The original
framework, campus prompts and probe idea are credited in NOTICE.md.

## First use

```sh
git clone https://github.com/zwkcyy/dase7506-mp3.git
cd dase7506-mp3
git checkout campus-audit-v2
cd campus-peer-audit
python --version
```

Run all commands below from this `campus-peer-audit` project directory, inside
the Git repository. The repository's top-level SQL project has separate files.
The stable reproduction version is the `campus-audit-v2` tag.
Use Python 3.12; the tested version is 3.12.14. Cloning, installing optional
packages, and downloading models use the network. The CPU calculations below
use only the files already present in the clone.

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

Optional figures use exact package pins, including their runtime dependencies.
For an isolated environment, first run `python -m venv .venv`, then activate it
with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Install and reproduce:

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

The separate **post-result measurement sensitivity** reparses every raw response
and compares the registered strict outcome with an action-only rule that permits
legal actions with invalid optional fields. Malformed JSON, illegal actions, and
inconsistent termination remain rejected. It preserves the original primary
analysis and requires the same figure packages:

```sh
python -B tools/reproduce.py --out reproduced_measurement --measurement
```

Run both post-result analyses together with one verified scratch copy:

```sh
python -B tools/reproduce.py --out reproduced_all --enhanced --measurement
```

The measurement command compares every scientific JSON field, seven CSV files,
and source hashes with `measurement_audit/`; it also verifies each output receipt
and the figure files, then runs the independent raw-text, cell/contrast, CSV, and
boundary-fixture reviewer. Rendering metadata can differ across runs. This audit uses
the same 2,040 completions and makes no new model calls.

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
Install `requirements-gpu.lock` into a separate Linux/Python 3.12 environment:

```sh
python -m pip install -r requirements-gpu.lock
python -m pip check
python -B tools/run_experiment_batch.py --prepared prepared
```

The final command validates inputs without loading a model or using a GPU. The
recorded generation hardware was a 24 GB RTX 4090; smaller GPUs were not tested.
The pins describe the observed CUDA 12.8 generation environment, rather than a claim
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

Summarize a newly generated run separately from the released observations:

```sh
python -B tools/analyze_experiment.py --prepared prepared --run runs_new/qwen3_main --out reproduced_new/qwen3_main
```

Use the matching prepared directory and a new output path for each run. New
stochastic outputs need not equal the released results; `tools/reproduce.py`
recomputes the released observations, while this command analyzes a new run.

## Report source

The English conference-style report source is in `paper/` (6 pages of main text,
followed by references and appendices). The abstract contains this public
repository link. The submission-only student identifier is intentionally not
public; `\StudentID` defaults to empty.

Compile using Tectonic 0.17.0+20260731 (or a compatible LaTeX toolchain with
IEEEtran, amsmath, graphicx, booktabs, array, hyperref and balance):

```sh
cd paper
tectonic main.tex
cd ..
```

The first Tectonic build may download standard TeX packages. This is separate
from the offline statistical reproduction command. To regenerate the data-bound
LaTeX results and full control table, run from the `campus-peer-audit` project
directory:

```sh
python tools/build_paper_tables.py --project . --out paper
```

`paper/figures/robustness_overview.pdf` is the enhancement figure; its numerical
source and generation script are included. Original and strengthened analyses
are distinguished in the report, including the post-result chronology.
