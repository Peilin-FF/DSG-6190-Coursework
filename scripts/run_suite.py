"""Run a suite of (dataset, method, schema sample, settings) jobs against one model server.

  python scripts/run_suite.py --model qwen3-vl-8b --suite configs/suite_main.json

Each job writes <runs>/<suite>/<model>/<dataset>/<method>[@tag]/run<r>.jsonl (resumable) and the
suite writes a summary.json beside it.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from dsg.client import ChatClient  # noqa: E402
from dsg.data import read_manifest  # noqa: E402
from dsg.methods import Runner, Settings  # noqa: E402
from dsg.schema import Schema  # noqa: E402
from dsg.scoring import match_choice  # noqa: E402


def load_schemas(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {c: [Schema.from_dict(d) for d in lst] for c, lst in raw["schemas"].items()}


def expand_jobs(suite: dict) -> list:
    jobs = []
    for j in suite["jobs"]:
        for m in j["methods"]:
            for r in j.get("runs", [0]):
                jobs.append({"dataset": j["dataset"], "method": m, "run": r, "tag": j.get("tag", ""),
                             "settings": j.get("settings", {}), "client": j.get("client", {}),
                             "schemas": j.get("schemas", suite.get("schemas")), "limit": j.get("limit"),
                             "qschemas": j.get("qschemas")})
    return jobs


def read_done(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except json.JSONDecodeError:
                    pass
    return done


async def main_async(a):
    suite = json.load(open(a.suite))
    reg = json.load(open(a.registry))
    cfg = reg[a.model]
    port = cfg["replicas"][a.replica - 1]["port"] if a.replica else cfg["port"]
    base_url = a.base_url or f"http://127.0.0.1:{port}/v1"
    jobs = expand_jobs(suite)
    if a.only:
        jobs = [j for j in jobs if any(s in f"{j['dataset']}/{j['method']}@{j['tag']}" for s in a.only.split(","))]
    datasets = {j["dataset"]: read_manifest(os.path.join(a.data, "manifests", f"{j['dataset']}.jsonl")) for j in jobs}
    schema_sets = {p: load_schemas(os.path.join(HERE, p) if not os.path.isabs(p) else p)
                   for p in {j["schemas"] for j in jobs if j["schemas"] and j["method"].startswith("dsg")}}
    clients, runners = {}, {}

    def client_for(over: dict):
        key = json.dumps(over, sort_keys=True)
        if key not in clients:
            ctk = dict(cfg.get("chat_template_kwargs", {}), **over.get("chat_template_kwargs", {}))
            clients[key] = ChatClient(base_url, a.model, os.path.join(a.cache, f"{a.model}.sqlite"),
                                      concurrency=a.concurrency, chat_template_kwargs=ctk, max_side=cfg.get("max_side", 1024),
                                      system_mode=over.get("system_mode", cfg.get("system_mode", "native")))
        return clients[key]

    def runner_for(j):
        key = (j["dataset"], json.dumps(j["settings"], sort_keys=True), json.dumps(j["client"], sort_keys=True),
               j["schemas"], j["qschemas"])
        if key not in runners:
            qextra = None
            if j["qschemas"]:
                qp = j["qschemas"] if os.path.isabs(j["qschemas"]) else os.path.join(HERE, j["qschemas"])
                qextra = json.load(open(qp))["extra"]
            runners[key] = Runner(client_for(j["client"]), a.data, schema_sets.get(j["schemas"], {}),
                                  Settings(**j["settings"]), datasets[j["dataset"]], qextra=qextra)
        return runners[key]

    out_root = os.path.join(a.runs, suite["name"], a.model)
    counts = defaultdict(int)
    t0 = time.time()

    async def run_job(j):
        items = datasets[j["dataset"]][: j["limit"]] if j["limit"] else datasets[j["dataset"]]
        name = j["method"] + (f"@{j['tag']}" if j["tag"] else "")
        path = os.path.join(out_root, j["dataset"], name, f"run{j['run']}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        done = read_done(path)
        todo = [it for it in items if it.id not in done]
        runner = runner_for(j)

        async def one(it):
            try:
                res = await runner.run(j["method"], it, j["run"])
            except Exception as e:  # recorded, scored as wrong
                res = {"output": "", "error": f"{type(e).__name__}: {e}"[:500]}
            idx, rule = match_choice(res.get("output", ""), it.choices, it.letters)
            return {"id": it.id, "dataset": it.dataset, "method": j["method"], "tag": j["tag"], "run": j["run"],
                    "concept": it.concept, "category": it.category, "qtype": it.qtype, "image_key": it.image_key,
                    "order": it.order, "answer_idx": it.answer_idx, "pred_idx": idx, "match_rule": rule,
                    "correct": idx == it.answer_idx, "n_choices": len(it.choices), **res}

        with open(path, "a") as f:
            for fut in asyncio.as_completed([one(it) for it in todo]):
                rec = await fut
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                counts["done"] += 1
        recs = list(read_done(path).values())
        acc = sum(r["correct"] for r in recs) / max(1, len(recs))
        print(f"[{time.time() - t0:7.0f}s] {j['dataset']}/{name}/run{j['run']}: n={len(recs)} acc={acc:.4f}", flush=True)
        return {"dataset": j["dataset"], "method": name, "run": j["run"], "n": len(recs), "acc": acc}

    async def progress():
        while True:
            await asyncio.sleep(60)
            st = defaultdict(int)
            for c in clients.values():
                for k, v in c.stats.items():
                    st[k] += v
            print(f"[{time.time() - t0:7.0f}s] items done {counts['done']} | client {dict(st)}", flush=True)

    prog = asyncio.create_task(progress())
    summaries = await asyncio.gather(*[run_job(j) for j in jobs])
    prog.cancel()
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=1)
    print(f"finished {len(jobs)} jobs in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--registry", default=os.path.join(HERE, "configs", "models.json"))
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    ap.add_argument("--runs", default=os.environ.get("DSG_RUNS", "runs"))
    ap.add_argument("--cache", default=os.environ.get("DSG_CACHE", "cache"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--replica", type=int, default=0, help="use the k-th replica server of the model")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--only", default="", help="comma-separated substrings of dataset/method@tag to run")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
