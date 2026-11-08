# CE6190 Project 1: Deep Schema Grounding with open vision-language models

Code for the individual project report. It re-implements Deep Schema Grounding (DSG; Hsu et al., ICLR 2025),
the compared methods (Direct, Blind, zero-shot CoT, CCoT), the DSG component variants and controls, and two
extensions (DSG-X: role-exclusive grounding; DSG-Q: question-specific components). All models are open-weight
VLMs served by vLLM through its OpenAI-compatible API.

## Layout

| Path | Content |
| --- | --- |
| `dsg/client.py` | async chat client, persistent SQLite request cache, de-duplication of identical in-flight requests |
| `dsg/schema.py` | schema programs: parser (`p(...)` and `gen(...)` forms), dependency order, rendering, the notebook parser |
| `dsg/prompts.py` | all prompts (DSG prompts copied from the authors' notebook; schema prompt from the paper's appendix) |
| `dsg/methods.py` | Direct, Blind, CoT, CCoT, +Schema, DSG variants, controls, DSG-X and DSG-Q |
| `dsg/data.py`, `dsg/scoring.py`, `dsg/stats.py` | question items, answer matching, bootstrap CIs and McNemar tests |
| `scripts/prepare_data.py` | builds the manifests: VAD, SpatialEval Maze-Nav (with and without legend), AI2D concepts |
| `scripts/serve.py` | starts/stops one vLLM server (or replica) per model of `configs/models.json` |
| `scripts/gen_schemas.py`, `scripts/gen_qschemas.py` | schema extraction and DSG-Q components with the schema LLM |
| `scripts/run_suite.py` | runs a suite of (dataset, method, schema sample, settings) jobs; resumable |
| `scripts/analyze.py` | re-scores every saved output, adds DSG-SC (vote) and DSG-Gate, writes result tables |
| `scripts/make_tables.py`, `scripts/figures.py`, `scripts/cases.py` | LaTeX tables, figures and case selection |
| `scripts/package_submission.sh` | builds the code zip (no data, models, runs or external links) |
| `configs/` | model registry and the experiment suites (`suite_main`, `suite_hparam*`, `suite_improve*`) |
| `schemas/` | the schemas and DSG-Q components used in the report |
| `results/` | result tables (CSV) written by `analyze.py` |

## Data (not included)

* Visual Abstractions Dataset (VAD), Hsu et al. (2025): 180 images, 540 questions.
* SpatialEval (Wang et al., NeurIPS 2024), Maze-Nav task, vision-only format: first 200 mazes (600 questions).
* AI2D (Kembhavi et al., ECCV 2016), test split, 13 concept categories (category file of the original release),
  at most 60 questions per category (556 questions).

Place the raw files under `$DSG_DATA/raw` as described at the top of `scripts/prepare_data.py`, then run
`python scripts/prepare_data.py --raw $DSG_DATA/raw --out $DSG_DATA`.

## Running

```bash
export DSG_MODEL_ROOT=/path/to/models DSG_DATA=/path/to/data DSG_RUNS=/path/to/runs \
       DSG_CACHE=/path/to/cache DSG_LOG_DIR=/path/to/logs
python scripts/serve.py start qwen3-vl-8b qwen3.8-27b
python scripts/gen_schemas.py --model qwen3.8-27b --datasets vad maze ai2d --size 3 --samples 5 \
       --out schemas/qwen3.8-27b_c3.json
python scripts/gen_qschemas.py --model qwen3.8-27b --datasets vad maze maze_nolegend ai2d \
       --schemas schemas/qwen3.8-27b_c3.json --samples 3 --out schemas/qwen3.8-27b_c3_q.json
python scripts/run_suite.py --model qwen3-vl-8b --suite configs/suite_main.json
python scripts/run_suite.py --model qwen3-vl-8b --suite configs/suite_improve.json
python scripts/run_suite.py --model qwen3-vl-8b --suite configs/suite_improve_x.json
python scripts/analyze.py --runs $DSG_RUNS --out results
python scripts/make_tables.py --results results --out report/tables
python scripts/figures.py --results results --out report/figures
```

Decoding is greedy (temperature 0) for every VLM call; schemas are sampled at temperature 0.7 (five per concept),
and run r of a schema-based method uses schema sample r. Requests are cached, so a suite can stop and resume.
