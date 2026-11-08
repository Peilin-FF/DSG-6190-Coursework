# CE6190 Project 1: Deep Schema Grounding with open-source vision-language models

Code for the individual project report "What Makes Deep Schema Grounding Work? A Study with Open-Source
Vision-Language Models". It re-implements Deep Schema Grounding (DSG; Hsu et al., ICLR 2025), the compared
methods, the ablation of the original paper, three controls on the schema, and two changes to DSG
(exclusive grounding and question components). All models are open-source VLMs served by vLLM through its
OpenAI-compatible API.

## Names in the code and in the report

| Code | Report |
| --- | --- |
| `vad` | Visual Abstractions Benchmark (VAB) of Hsu et al. (released as `VAD.zip`) |
| `direct`, `blind` | base VLM, language-only baseline |
| `cot`, `ccot` | CoT (zero-shot), Compositional CoT |
| `dsg_schema`, `dsg_seq`, `dsg_hier`, `dsg_full` | + schema, + grounding, + hierarchical, DSG |
| `dsg_release` | DSG, released code (behaviour of the authors' notebook) |
| `dsg_shuffle`, `dsg_wrong`, `dsg_generic` | groundings of another image, schema of another concept, generic schema |
| `dsg_x`, `dsg_q` | DSG + exclusive grounding, DSG + question components |
| `dsg_sc3`, `dsg_sc5` | DSG, vote over 3 or 5 schema samples |
| `maze`, `maze_nolegend`, `ai2d` | Maze-Nav, Maze-Nav without legend, AI2D (13 concept categories) |

## Layout

| Path | Content |
| --- | --- |
| `dsg/client.py` | async chat client, persistent SQLite request cache, de-duplication of identical requests in flight |
| `dsg/schema.py` | schema programs: parser (`p(...)` and `gen(...)` forms), dependency order, rendering, the notebook parser |
| `dsg/prompts.py` | all prompts (DSG prompts copied from the authors' notebook; schema prompt from the paper's appendix) |
| `dsg/methods.py` | every method and variant of the table above |
| `dsg/data.py`, `dsg/scoring.py`, `dsg/stats.py` | question items, answer matching, McNemar tests and a cluster bootstrap |
| `scripts/prepare_data.py` | builds the manifests: VAB, SpatialEval Maze-Nav (with and without legend), AI2D concepts |
| `scripts/serve.py` | starts and stops one vLLM server (or replica) per model of `configs/models.json` |
| `scripts/gen_schemas.py`, `scripts/gen_qschemas.py` | schema extraction and question components with the schema LLM |
| `scripts/run_suite.py` | runs a suite of (dataset, method, schema sample, settings) jobs; resumable |
| `scripts/analyze.py` | re-scores every saved output, adds the votes over schema samples, writes result tables |
| `scripts/grounding_stats.py` | shared groundings per resolved schema and accuracy per concept, from the run files |
| `scripts/answer_stats.py` | turn counts on Maze-Nav and yes/no answers on VAB |
| `scripts/make_tables.py`, `scripts/figures.py`, `scripts/cases.py` | LaTeX tables, figures and case selection |
| `scripts/package_submission.sh` | builds the code zip (no data, models, runs or external links) |
| `configs/` | model registry and the experiment suites (`suite_main`, `suite_hparam*`, `suite_improve*`) |
| `schemas/` | the schemas and question components used in the report |
| `results/` | result tables (CSV) written by the analysis scripts |

## Data (not included)

* Visual Abstractions Benchmark (VAB), Hsu et al. (2025): 180 images, 540 questions.
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
python scripts/grounding_stats.py --runs $DSG_RUNS --out results
python scripts/answer_stats.py --results results --data $DSG_DATA
python scripts/make_tables.py --results results --out report/tables
python scripts/figures.py --results results --out report/figures --data $DSG_DATA
```

Decoding is greedy (temperature 0) for every VLM call; schemas are sampled at temperature 0.7 (five per concept),
and run r of a schema-based method uses schema sample r. Requests are cached, so a suite can stop and resume.
