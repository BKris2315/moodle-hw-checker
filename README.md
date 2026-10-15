# Moodle homework submission checker

Round 1: **who submitted what** (not correctness). Every problem of every
homework is marked `attempted` / `partial` / `missing` by a local LLM. The total
is then checked against `ceil(0.7 × total problems)`, computed exactly, so
6.1 becomes 7 and 7.0 stays 7.

## Pipeline (`run`)

1. **Sync.** `moodle-dl -p <moodle_dl.path>` updates the course folder, using the token in moodle-dl's own `config.json`.
   To skip it, use `--no-sync` or set `moodle_dl.sync_before_run: false`.
   When a file changes on Moodle, moodle-dl saves the new version as `X_01.ext`. The script always uses the newest version.
2. **Homeworks and problem lists.** The homeworks are the folders matching `layout.homework_glob`,
   plus the entries listed under `homeworks:` (see "Another course").
   A homework without `problems:` in the config gets its problem list drafted by the LLM from the assignment text.
   The text comes from `Assignment intro.md`, or from the practice notebook if the intro only points to it.
   Drafts are stored in `results/problems.yaml` and re-drafted automatically when the text changes on Moodle.
   The config file is never written to; to fix a draft, copy its entry into `homeworks:` and edit it there.
   Old folders go under `ignore_folders`.
3. **Extraction** (no LLM). Submitted `.ipynb` / `.py` files are read.
   Cells copied from the instructor's notebooks are removed, keeping only their headings as anchors:
   unchanged or lightly edited template cells, and cells found verbatim in most submissions.
4. **Check.** For each problem, the LLM decides `attempted` / `partial` / `missing`.
   Submissions with no student-written content skip the model.
   Replies are cached in `results/cache/`, so only new or changed submissions go to the GPU.
5. **Reports** in `results/`:
   - `summary.csv`: one row per student, with `k/n` per homework, solved, required, status and notes.
     Required = ceil(70% × the problems of homeworks **due by today**).
     Deadlines come from moodle-dl's `Events/`; override per homework with `due:` / `count:`.
     `solved` includes work handed in early.
   - `report.md`: the same table, plus per-submission evidence and ⚠ flags.
   - `details.csv`: one row per student × problem.

The roster is everyone who submitted anything in the course, practices included.

## Setup

1. Download the course with [moodle-dl](https://github.com/C0D3D3V/Moodle-DL) (`moodle-dl --init` in a folder of your choice).
   Its `config.json` holds your Moodle token: keep it out of this repo.
2. `python -m venv .venv && .venv/bin/pip install moodle-dl -r requirements.txt`
3. `cp homeworks.example.yaml homeworks.yaml` and set `course_dir`, `moodle_dl.path` and the homeworks.
   `homeworks.yaml` and `results/` are gitignored: they hold course material and student data.

## Usage

```sh
# On a GPU node, as a SLURM job (survives logout); the log is in results/slurm-<id>.log.
# Add your cluster's GPU options, e.g. `sbatch --gres=gpu:1 ...` or `sbatch -p <partition> -w <node> ...`.
# The job uses $HWCHECK_PYTHON, else the active venv, else ./.venv, else python3.
sbatch run_hwcheck.sbatch                         # sync + everything
sbatch run_hwcheck.sbatch --hw HW3                # check only HW3 (others come from the cache)
sbatch run_hwcheck.sbatch --grade --hw HW2 --model Qwen/Qwen3-14B --load-in-4bit   # scores + feedback

# Without a GPU (login node): who submitted, and exactly what the model will see
.venv/bin/python hwcheck.py scan [--sync] [--hw HW2]     # -> results/extracted/<HW>/<ID>.txt
.venv/bin/python hwcheck.py report               # rebuild the reports from cached verdicts, seconds

# Only draft problem lists
.venv/bin/python hwcheck.py propose              # append drafts for new homework folders
.venv/bin/python hwcheck.py propose --all        # print drafts for all folders, write nothing
```

You can also choose homeworks with `check_homeworks: [HW3, HW4]` in the YAML; `--hw` overrides it.
Unselected homeworks still count in the summary through their cached verdicts.
Submissions that were never checked show as `?`.

## Another course

One config file per course. Copy `homeworks.example.yaml` to `mycourse.yaml` and set:
- `course_dir`, `course_description` (goes into the model's instructions), and `results_dir` (a new folder).
- `moodle_dl`: add the course ID to `download_course_ids` in moodle-dl's `config.json`,
  or point `moodle_dl.path` to a separate moodle-dl folder. If the files don't come from Moodle, set `sync_before_run: false`.
- **The homeworks**, in either of two ways (or both):
  - **Repeating pattern**, in `layout:`:
    ```yaml
    layout:
      homework_glob: "week*/assignment"     # every matching folder is a homework
      submissions_dir: "handins"            # <homework>/handins/<student>/...
      assignment_text: ["task.md"]          # <homework>/task.md
      student_dir_regex: '^(?P<name>.+?)\s*\((?P<code>[^()]+)\)$'   # "Name (ID)"; otherwise the folder name is the ID
    ```
  - **Explicit paths**, in `homeworks:` (set `homework_glob: []` to use only these):
    ```yaml
    homeworks:
      - folder: "Final project"
        name: "Project"
        submissions: "/data/uploads/project"       # one subfolder per student
        assignment_text: "Final project/task.html"
        due: "2026-12-15 23:59"
    ```
- `problems:` is optional everywhere: leave it out and the first `run` drafts it into `<results_dir>/problems.yaml`. Review those drafts.

Then run `sbatch run_hwcheck.sbatch --config mycourse.yaml`.
`hwcheck.py scan --config mycourse.yaml` (no GPU) shows which homeworks and students it found.

Limits:
- Only `.ipynb`, `.py` (and `.zip` archives of those) are read. PDFs and other files are flagged "check manually".
- Template detection expects the instructor notebooks in the homework's parent folder; otherwise set `templates:`.
- Moodle deadlines are found from moodle-dl's `Events/` folder, whose event titles contain the homework folder name.
  For anything else, set `due:` or `count:`.

## Model

The model is set in the `model:` section of the config, which also lists the models usable on a 16 GB Tesla T4.
`--model`, `--load-in-4bit`, `--backend` and `--base-url` override it for one run.
Verdicts are cached per model, so switching models re-checks everything once.

## Grading (`--grade`)

With `--grade`, or `grading.enabled: true`, each submission gets a second pass.
The model sees the full assignment text and a longer version of the submission (more code, printed numbers),
then scores every problem from 0 to `points` (default 10), with feedback.
Problems the attempt check found missing always get 0.
The results appear in:
- `summary.csv`: `HWn score` columns, plus the total over the homeworks that are due.
- `details.csv`: `score`, `max_points`, `feedback`, `overall_feedback`.
- `report.md`: scores and feedback per problem.

Per homework, `grading_notes: "..."` can give the grader expected numbers or common mistakes.
This improves consistency a lot.

The scores are **suggestions**. In testing, the default 4B model made up written answers that did not exist,
so grade with `Qwen/Qwen3-14B` + `load_in_4bit: true`, and review before using the scores for marks.
The model cannot see figures; it judges plots from the code.

T4 notes: 16 GB VRAM, compute capability 7.5. There is **no native bf16**, so always
use fp16 (`--dtype half` in vLLM). There is no FlashAttention 2 either.
Models of 8B and up need 4-bit weights: AWQ/GPTQ for vLLM, `--load-in-4bit` for transformers, or GGUF for llama.cpp.
