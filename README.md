# CE6190 Project 1 — Deep Schema Grounding: reproduction, ablation and generalization study

This folder holds the code for the individual project report. It re-implements Deep Schema Grounding
(DSG; Hsu et al., ICLR 2025) and the compared methods, and evaluates them with open-weight
vision-language models (VLMs) served by vLLM.

## Layout

| Path | Content |
| --- | --- |
| `dsg/client.py` | async client for OpenAI-compatible servers, persistent request cache, in-flight de-duplication |
| `dsg/schema.py` | schema programs: parser (`p(...)` and `gen(...)` forms), dependency order, the released-notebook parser |
| `dsg/prompts.py` | every prompt (DSG prompts copied from the authors' notebook, CCoT, CoT, schema extraction) |
| `dsg/methods.py` | Direct, CoT, CCoT and the DSG variants and controls |
| `dsg/data.py`, `dsg/scoring.py`, `dsg/stats.py` | question items, answer matching, bootstrap CIs and McNemar tests |
| `scripts/prepare_data.py` | builds the question manifests (VAD, SpatialEval Maze-Nav with and without legend, AI2D concepts) |
| `scripts/serve.py` | starts one vLLM server per model of `configs/models.json` |
| `scripts/gen_schemas.py` | schema extraction with an LLM |
| `scripts/run_suite.py` | runs a suite of (dataset, method, schema sample, settings) jobs against one model |
| `scripts/analyze.py`, `scripts/figures.py` | tables, statistics and figures of the report |
| `configs/` | model registry and experiment suites |
| `schemas/` | the generated schemas used in the report |

## Datasets (not included)

* Visual Abstractions Dataset (VAD) of Hsu et al. (2025): 180 images, 540 questions.
* SpatialEval (Wang et al., 2024), Maze-Nav task, vision-only format: the first 200 mazes (600 questions).
* AI2D (Kembhavi et al., 2016), test split: the 13 concept categories, at most 60 questions each (556 questions).

Put the raw files under `$DSG_DATA/raw` as described in `scripts/prepare_data.py`, then run
`python scripts/prepare_data.py --raw $DSG_DATA/raw --out $DSG_DATA`.

## Running

```bash
export DSG_MODEL_ROOT=/path/to/models DSG_DATA=/path/to/data DSG_RUNS=/path/to/runs \
       DSG_CACHE=/path/to/cache DSG_LOG_DIR=/path/to/logs
python scripts/serve.py start qwen3-vl-8b qwen3.8-27b          # one server per model (GPUs in the registry)
python scripts/gen_schemas.py --model qwen3.8-27b --datasets vad maze ai2d --size 3 --samples 5 \
       --out schemas/qwen3.8-27b_c3.json
python scripts/run_suite.py --model qwen3-vl-8b --suite configs/suite_main.json
python scripts/analyze.py --runs $DSG_RUNS --out results
```

All decoding is greedy (temperature 0) except schema sampling (temperature 0.7, five samples per concept).
Requests are cached in SQLite, so a suite can be stopped and resumed.
