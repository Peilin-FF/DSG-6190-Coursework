"""DSG-Q: ask the schema LLM for question-specific components to append to each concept schema.

  python scripts/gen_qschemas.py --model qwen3.8-27b --datasets vad maze maze_nolegend ai2d \
      --schemas schemas/qwen3.8-27b_c3.json --samples 3 --out schemas/qwen3.8-27b_c3_q.json
"""

import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from dsg import prompts as P  # noqa: E402
from dsg.client import ChatClient, msg  # noqa: E402
from dsg.data import question_key, read_manifest  # noqa: E402
from dsg.schema import TERM_RE, Schema, clean_name, render_gen  # noqa: E402

MAX_EXTRA = 2


def parse_extra(text: str, schema: Schema) -> list:
    if not text or text.strip().upper().startswith("NONE"):
        return []
    known = set(schema.terms)
    out = []
    for head, cond in TERM_RE.findall(text):
        t = clean_name(head)
        if t.lower() == "image" or t in known or t in [x for x, _ in out]:
            continue
        deps = [clean_name(c) for c in cond.split(",") if c.strip() and not c.strip().lower().startswith("concept")]
        out.append((t, [d for d in deps if d in known or d in [x for x, _ in out]]))
        if len(out) == MAX_EXTRA:
            break
    return out


async def main_async(a):
    reg = json.load(open(a.registry))
    base = a.model.partition(":")[0]
    cfg = reg[base]
    port = cfg["replicas"][int(a.model.partition(":")[2]) - 1]["port"] if ":" in a.model else cfg["port"]
    client = ChatClient(f"http://127.0.0.1:{port}/v1", base, os.path.join(a.cache, f"{base}.sqlite"),
                        concurrency=64, chat_template_kwargs=cfg.get("chat_template_kwargs", {}))
    schemas = json.load(open(a.schemas if os.path.isabs(a.schemas) else os.path.join(HERE, a.schemas)))["schemas"]
    keys = {}
    for d in a.datasets:
        for it in read_manifest(os.path.join(a.data, "manifests", f"{d}.jsonl")):
            keys.setdefault(it.concept, set()).add(question_key(it))

    async def one(k, concept, q):
        sch = Schema.from_dict(schemas[concept][k])
        prompt = P.QSCHEMA_REQUEST.format(concept=concept, program=render_gen(sch), question=q,
                                          concept_tok=concept.replace(" ", "-"))
        out = await client.chat([msg("user", prompt)], temperature=0.0, max_tokens=300)
        return k, concept, q, parse_extra(out.get("text", ""), sch), out.get("text", "")

    tasks = [one(k, c, q) for k in range(a.samples) for c, qs in keys.items() for q in sorted(qs)]
    res = await asyncio.gather(*tasks)
    extra, raw = {}, {}
    for k, c, q, ex, txt in res:
        extra.setdefault(str(k), {}).setdefault(c, {})[q] = ex
        raw.setdefault(str(k), {}).setdefault(c, {})[q] = txt
    n_q = sum(len(v) for v in extra["0"].values())
    n_ext = sum(1 for v in extra["0"].values() for e in v.values() if e)
    print(f"{n_q} questions, {n_ext} got extra components (sample 0)")
    with open(a.out, "w") as f:
        json.dump({"meta": {"model": base, "schemas": a.schemas, "samples": a.samples, "max_extra": MAX_EXTRA},
                   "extra": extra, "raw": raw}, f, indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name, optionally name:k for the k-th replica")
    ap.add_argument("--datasets", nargs="+", default=["vad"])
    ap.add_argument("--schemas", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--registry", default=os.path.join(HERE, "configs", "models.json"))
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("DSG_CACHE", "cache"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
