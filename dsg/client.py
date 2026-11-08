"""Async client for OpenAI-compatible chat servers (vLLM) with a persistent request cache.

Messages use an internal format so that cache keys never contain image bytes:
content parts are {"type": "text", "text": ...} or {"type": "image", "path": ..., "max_side": ...}.
"""

import asyncio
import hashlib
import json
import os
import random
import sqlite3
from collections import Counter

from openai import (APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError,
                    InternalServerError, RateLimitError)

from .images import encode_image, image_fingerprint

CACHE_VERSION = "v1"


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def image(path: str, max_side: int | None = None) -> dict:
    return {"type": "image", "path": path, "max_side": max_side}


def msg(role: str, content) -> dict:
    return {"role": role, "content": content}


class Cache:
    """SQLite key-value store shared by concurrent processes (WAL mode)."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=120, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))


class ChatClient:
    def __init__(self, base_url: str, model: str, cache_path: str, concurrency: int = 128,
                 chat_template_kwargs: dict | None = None, max_side: int = 1024,
                 system_mode: str = "native", timeout: float = 1800.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get("DSG_API_KEY", "EMPTY"),
                                  timeout=timeout, max_retries=0)
        self.model = model
        self.cache = Cache(cache_path)
        self.sem = asyncio.Semaphore(concurrency)
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.max_side = max_side
        self.system_mode = system_mode
        self.inflight: dict[str, asyncio.Future] = {}
        self.stats = Counter()

    def _normalize(self, messages: list[dict]) -> list[dict]:
        """Resolve image sizes and apply the system-message policy.

        native: keep a leading system message; a later system message is prepended to the next
        user turn as text (several chat templates reject mid-conversation system messages).
        merge_all: every system message is prepended to the next user turn.
        """
        out, pending = [], []
        for i, m in enumerate(messages):
            content = m["content"]
            if isinstance(content, str):
                content = [text(content)]
            content = [dict(p, max_side=p.get("max_side") or self.max_side) if p["type"] == "image" else p
                       for p in content]
            if m["role"] == "system":
                if i == 0 and self.system_mode == "native":
                    out.append({"role": "system", "content": content})
                else:
                    pending.extend(p["text"] for p in content if p["type"] == "text")
                continue
            if m["role"] == "user" and pending:
                content = [text(" ".join(pending))] + content
                pending = []
            out.append({"role": m["role"], "content": content})
        return out

    def _key(self, messages: list[dict], params: dict) -> str:
        def part_repr(p):
            if p["type"] == "image":
                return {"image": image_fingerprint(p["path"]), "max_side": p["max_side"]}
            return p
        rep = {"v": CACHE_VERSION, "model": self.model, "ctk": self.chat_template_kwargs, "params": params,
               "messages": [{"role": m["role"], "content": [part_repr(p) for p in m["content"]]} for m in messages]}
        return hashlib.sha256(json.dumps(rep, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _materialize(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            parts = []
            for p in m["content"]:
                if p["type"] == "image":
                    parts.append({"type": "image_url", "image_url": {"url": encode_image(p["path"], p["max_side"])}})
                else:
                    parts.append({"type": "text", "text": p["text"]})
            if m["role"] in ("system", "assistant") and all(p["type"] == "text" for p in parts):
                out.append({"role": m["role"], "content": "".join(p["text"] for p in parts)})
            else:
                out.append({"role": m["role"], "content": parts})
        return out

    async def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 300,
                   seed: int = 0, top_p: float | None = None, logprobs: bool = False) -> dict:
        norm = self._normalize(messages)
        params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed, "top_p": top_p,
                  "logprobs": logprobs}
        key = self._key(norm, params)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache_hit"] += 1
            return hit
        if key in self.inflight:
            self.stats["dedup"] += 1
            return await asyncio.shield(self.inflight[key])
        fut = asyncio.get_running_loop().create_future()
        fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self.inflight[key] = fut
        try:
            async with self.sem:
                out = await self._request(norm, params)
            if not out.get("error"):
                self.cache.put(key, out)
            fut.set_result(out)
            return out
        except BaseException as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            self.inflight.pop(key, None)

    async def _request(self, norm: list[dict], params: dict) -> dict:
        kwargs = {"model": self.model, "messages": self._materialize(norm),
                  "temperature": params["temperature"], "max_tokens": params["max_tokens"], "seed": params["seed"]}
        if params["top_p"] is not None:
            kwargs["top_p"] = params["top_p"]
        if params["logprobs"]:
            kwargs["logprobs"] = True
        if self.chat_template_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": self.chat_template_kwargs}
        delay = 2.0
        for attempt in range(10):
            try:
                r = await self.client.chat.completions.create(**kwargs)
                break
            except BadRequestError as e:
                self.stats["bad_request"] += 1
                return {"text": "", "finish": "error", "error": str(e)[:500]}
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError):
                self.stats["retry"] += 1
                if attempt == 9:
                    raise
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60.0)
        self.stats["request"] += 1
        ch = r.choices[0]
        out = {"text": ch.message.content or "", "finish": ch.finish_reason,
               "usage": {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}}
        reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None)
        if reasoning:
            out["reasoning"] = reasoning
        if params["logprobs"] and ch.logprobs is not None and ch.logprobs.content:
            out["logprobs"] = [round(t.logprob, 5) for t in ch.logprobs.content]
        return out
