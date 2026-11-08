"""VQA methods: direct answering, zero-shot CoT, CCoT and the Deep Schema Grounding (DSG) family.

DSG variants (the component study of the paper plus controls):
  dsg_schema   schema given as a chain-of-thought plan inside one query (no grounding calls)
  dsg_seq      every component grounded independently (no dependency conditioning), results as text
  dsg_hier     hierarchical grounding along the schema DAG, results as text
  dsg_full     hierarchical grounding, results as text plus the full grounding dialogue (the paper's DSG)
  dsg_release  dsg_full as the released notebook runs it: the dialogue reaches only the first question of
               an image (later questions hit the cache) and the notebook parser drops dependencies whose
               name contains the concept word
  dsg_shuffle  dsg_hier, but grounded on another image of the same concept (is grounding image-specific?)
  dsg_wrong    dsg_full with the schema of a different concept (is the schema content relevant?)
  dsg_generic  dsg_full with a concept-agnostic objects/attributes/relations schema
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from . import prompts as P
from .client import image, msg, text
from .data import question_key
from .schema import Schema, extend_schema, parse_program


@dataclass
class Settings:
    ground_style: str = "phrase"
    ground_tokens: int = 300
    answer_tokens: int = 300
    reason_tokens: int = 2048
    logprobs: bool = True
    max_side: int | None = None


def context_text(concept: str, terms=(), answers=None) -> str:
    conds = "".join(P.FINAL_CONDITION.format(c=c, v=(answers or {}).get(c, "")) for c in terms)
    return P.FINAL_CONTEXT.format(concept=concept, conditions=conds)


def compose(context: str, it, suffix: str) -> str:
    """Context, question and instruction; the instruction gets its own line after lettered options."""
    return f"{context} {it.question}{chr(10) if it.letters else ' '}{suffix}"


def logprob_summary(out: dict) -> dict:
    lp = out.get("logprobs")
    if not lp:
        return {}
    return {"lp_sum": round(sum(lp), 4), "lp_mean": round(sum(lp) / len(lp), 5), "n_tok": len(lp)}


class Runner:
    def __init__(self, client, data_root: str, schemas: dict, settings: Settings, items: list, qextra: dict | None = None):
        self.client = client
        self.qextra = qextra or {}
        self.root = data_root
        self.schemas = schemas
        self.s = settings
        keys = defaultdict(OrderedDict)
        for it in items:
            keys[it.concept].setdefault(it.image_key, it)
        self.donor = {}
        for concept, od in keys.items():
            ks = list(od.keys())
            for i, k in enumerate(ks):
                self.donor[k] = od[ks[(i + 1) % len(ks)]]
        concepts = sorted(c for c in keys if c in schemas)
        shift = max(1, len(concepts) // 2)
        self.wrong = {c: concepts[(i + shift) % len(concepts)] for i, c in enumerate(concepts)}
        self.dispatch = {
            "direct": self.direct, "blind": self.blind, "cot": self.cot, "ccot": self.ccot, "dsg_schema": self.schema_cot,
            **{f"dsg_{v}": self._dsg(v) for v in ("seq", "hier", "full", "release", "shuffle", "wrong", "generic", "q", "x")},
        }

    # helpers -------------------------------------------------------------------------------------------
    def _img(self, it):
        return image(it.image_path(self.root), self.s.max_side)

    def schema_for(self, it, run: int, variant: str) -> Schema:
        if variant == "q":
            base = self.schema_for(it, run, "full")
            per_run = self.qextra[str(run % len(self.qextra))]
            extra = per_run.get(it.concept, {}).get(question_key(it), [])
            return extend_schema(base, [(t, d) for t, d in extra])
        if variant == "generic":
            return parse_program(P.GENERIC_PROGRAM.format(concept=it.concept), concept=it.concept, source="generic")
        concept = self.wrong[it.concept] if variant == "wrong" else it.concept
        samples = self.schemas[concept]
        return samples[run % len(samples)]

    async def _answer(self, it, prompt: str, history: list, max_tokens: int | None = None, logprobs=None):
        msgs = ([msg("system", P.GROUND_SYSTEM)] + history +
                [msg("system", P.FINAL_REMINDER), msg("user", [text(prompt), self._img(it)])])
        lp = self.s.logprobs if logprobs is None else logprobs
        return await self.client.chat(msgs, max_tokens=max_tokens or self.s.answer_tokens, logprobs=lp)

    @staticmethod
    def _result(out: dict, prompt: str, **extra) -> dict:
        r = {"output": out.get("text", ""), "final_prompt": prompt, "finish": out.get("finish"),
             "usage": out.get("usage"), **logprob_summary(out), **extra}
        if out.get("error"):
            r["error"] = out["error"]
        if out.get("reasoning"):
            r["reasoning_chars"] = len(out["reasoning"])
        return r

    # baselines -----------------------------------------------------------------------------------------
    async def direct(self, it, run):
        prompt = compose(context_text(it.concept), it, P.FINAL_SUFFIX)
        return self._result(await self._answer(it, prompt, []), prompt)

    async def blind(self, it, run):
        """Direct prompt without the image: the answer prior of the language model."""
        prompt = compose(context_text(it.concept), it, P.FINAL_SUFFIX)
        msgs = [msg("system", P.GROUND_SYSTEM), msg("system", P.FINAL_REMINDER), msg("user", [text(prompt)])]
        out = await self.client.chat(msgs, max_tokens=self.s.answer_tokens, logprobs=self.s.logprobs)
        return self._result(out, prompt)

    async def cot(self, it, run):
        prompt = compose(context_text(it.concept), it, P.COT_SUFFIX)
        out = await self._answer(it, prompt, [], max_tokens=self.s.reason_tokens, logprobs=False)
        return self._result(out, prompt)

    async def ccot(self, it, run):
        ctx = context_text(it.concept)
        sg_prompt = f"{ctx} " + P.CCOT_SCENE_GRAPH.format(question=it.question)
        sg = await self.client.chat([msg("system", P.GROUND_SYSTEM), msg("user", [text(sg_prompt), self._img(it)])],
                                    max_tokens=self.s.reason_tokens)
        prompt = P.CCOT_ANSWER.format(scene_graph=sg.get("text", "").strip(), question=f"{ctx} {it.question}")
        prompt = f"{prompt}{chr(10) if it.letters else ' '}{P.FINAL_SUFFIX}"
        return self._result(await self._answer(it, prompt, []), prompt, scene_graph=sg.get("text", ""),
                            aux_usage=[sg.get("usage")])

    async def schema_cot(self, it, run):
        schema = self.schema_for(it, run, "full")
        prompt = P.SCHEMA_COT.format(concept=it.concept, program=schema.render(),
                                     question=f"{context_text(it.concept)} {it.question}")
        out = await self._answer(it, prompt, [], max_tokens=self.s.reason_tokens, logprobs=False)
        return self._result(out, prompt)

    # DSG -----------------------------------------------------------------------------------------------
    async def ground(self, it, schema: Schema, hierarchical: bool, legacy: bool = False, source=None,
                     exclusive: bool = False):
        src = source or it
        img = self._img(src)
        concept_word = schema.concept.replace("-", " ").lower()
        answers, turns, records = {}, [], []
        for term, deps in schema.components:
            use = list(deps) if hierarchical else []
            if legacy:
                use = [d for d in use if concept_word not in d.lower()]
            conds = "".join(P.GROUND_CONDITION.format(c=d, v=answers.get(d, "")) for d in use) if use else ","
            q = P.GROUND_QUESTION.format(concept=it.concept, conditions=conds, term=term)
            if exclusive and records:
                items = " or ".join(f"the {r['term']} ({r['answer']})" for r in records)
                q += P.EXCLUSIVE_CLAUSE.format(items=items)
            q += P.GROUND_SUFFIX[self.s.ground_style]
            out = await self.client.chat([msg("system", P.GROUND_SYSTEM), msg("user", [text(q), img])],
                                         max_tokens=self.s.ground_tokens)
            a = out.get("text", "").strip().lower().replace(".", "")
            answers[term] = a
            turns += [msg("user", [text(q), img]), msg("assistant", a)]
            records.append({"term": term, "deps": use, "answer": a, "usage": out.get("usage")})
        return answers, turns, records

    def _dsg(self, variant: str):
        async def run_variant(it, run):
            schema = self.schema_for(it, run, variant)
            source = self.donor[it.image_key] if variant == "shuffle" else None
            answers, turns, records = await self.ground(it, schema, hierarchical=variant != "seq",
                                                        legacy=variant == "release", source=source,
                                                        exclusive=variant == "x")
            if variant in ("full", "wrong", "generic", "q", "x"):
                history = turns
            elif variant == "release":
                history = turns if it.order == 0 else []
            else:
                history = []
            prompt = compose(context_text(it.concept, schema.final, answers), it, P.FINAL_SUFFIX)
            out = await self._answer(it, prompt, history)
            return self._result(out, prompt, grounding=records, schema=schema.render(),
                                donor=source.image_key if source else None)
        return run_variant

    async def run(self, method: str, it, run: int) -> dict:
        return await self.dispatch[method](it, run)
