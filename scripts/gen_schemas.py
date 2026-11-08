"""Extract schemas for every concept of the given datasets with an LLM served by vLLM.

  python scripts/gen_schemas.py --model qwen3.8-27b --datasets vad maze ai2d --size 3 --samples 5 \
      --out schemas/qwen3.8-27b_c3.json
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
from dsg.data import read_manifest  # noqa: E402
from dsg.schema import SchemaError, parse_program  # noqa: E402


async def main_async(a):
    reg = json.load(open(a.registry))
    cfg = reg[a.model]
    ctk = dict(cfg.get("chat_template_kwargs", {}))
    if "enable_thinking" in ctk:
        ctk["enable_thinking"] = a.thinking
    client = ChatClient(a.base_url or f"http://127.0.0.1:{cfg['port']}/v1", a.model,
                        os.path.join(a.cache, f"{a.model}.sqlite"), concurrency=64, chat_template_kwargs=ctk)
    concepts = sorted({it.concept for d in a.datasets
                       for it in read_manifest(os.path.join(a.data, "manifests", f"{d}.jsonl"))})

    async def sample(concept, k):
        prompt = P.schema_request(concept, a.size)
        last = ""
        for attempt in range(6):
            out = await client.chat([msg("user", prompt)], temperature=a.temperature, top_p=a.top_p,
                                    max_tokens=a.max_tokens, seed=1000 * k + attempt)
            last = out.get("text", "")
            try:
                s = parse_program(last, concept=concept, source=f"{a.model}/size{a.size}/k{k}/try{attempt}")
                if len(s.components) >= 1:
                    return s.to_dict()
            except SchemaError:
                continue
        raise RuntimeError(f"no valid schema for {concept} (last output: {last[:200]!r})")

    results = {}
    for concept in concepts:
        results[concept] = await asyncio.gather(*[sample(concept, k) for k in range(a.samples)])
        sizes = [len(s["components"]) for s in results[concept]]
        print(f"{concept:35s} sizes={sizes}  e.g. {[t for t, _ in results[concept][0]['components']]}", flush=True)
    meta = {"model": a.model, "size": a.size, "samples": a.samples, "temperature": a.temperature,
            "top_p": a.top_p, "thinking": a.thinking, "prompt_example": P.schema_request("<concept>", a.size)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "schemas": results}, f, indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["vad"])
    ap.add_argument("--size", type=int, default=3, choices=[3, 5, 7, 9])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--registry", default=os.path.join(HERE, "configs", "models.json"))
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("DSG_CACHE", "cache"))
    ap.add_argument("--base-url", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
