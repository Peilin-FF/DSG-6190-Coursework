"""Start (or stop) one vLLM OpenAI-compatible server per model of the registry.

  python scripts/serve.py start qwen3-vl-8b qwen3.8-27b
  python scripts/serve.py stop all
  python scripts/serve.py status all
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def registry(path):
    with open(path) as f:
        return json.load(f)


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def resolve(reg: dict, name: str):
    """'model' or 'model:k' (k-th replica: same weights and served name, other GPUs and port)."""
    base, _, k = name.partition(":")
    cfg = dict(reg[base])
    if k:
        cfg.update(cfg["replicas"][int(k) - 1])
    cfg["served_name"] = base
    return cfg


def start(name, cfg, a):
    pid_file = os.path.join(a.log_dir, f"serve_{name}.pid")
    if healthy(cfg["port"]):
        print(f"{name}: already serving on {cfg['port']}")
        return
    gpus = cfg["gpus"]
    cmd = ["vllm", "serve", os.path.join(a.model_root, cfg["path"]),
           "--served-model-name", cfg["served_name"], "--host", "127.0.0.1", "--port", str(cfg["port"]),
           "--tensor-parallel-size", str(len(gpus.split(","))),
           "--max-model-len", str(cfg.get("max_model_len", a.max_model_len)),
           "--gpu-memory-utilization", str(cfg.get("gpu_mem", a.gpu_mem)),
           "--limit-mm-per-prompt", json.dumps({"image": a.max_images}),
           "--generation-config", "vllm", "--seed", "0"] + cfg.get("args", [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    log = open(os.path.join(a.log_dir, f"serve_{name}.log"), "a")
    log.write(f"\n=== {time.ctime()} {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    with open(pid_file, "w") as f:
        f.write(str(p.pid))
    print(f"{name}: started pid {p.pid} on GPUs {gpus}, port {cfg['port']}")


def stop(name, cfg, a):
    pid_file = os.path.join(a.log_dir, f"serve_{name}.pid")
    if not os.path.exists(pid_file):
        print(f"{name}: no pid file")
        return
    pid = int(open(pid_file).read().strip())
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"{name}: sent SIGTERM to process group {pid}")
    except ProcessLookupError:
        print(f"{name}: process {pid} not running")
    os.remove(pid_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "status", "wait"])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--registry", default=os.path.join(HERE, "configs", "models.json"))
    ap.add_argument("--model-root", default=os.environ.get("DSG_MODEL_ROOT", "models"))
    ap.add_argument("--log-dir", default=os.environ.get("DSG_LOG_DIR", "logs"))
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    raw = registry(a.registry)
    names = list(raw) if a.models == ["all"] else a.models
    for n in names:
        if n.partition(":")[0] not in raw:
            sys.exit(f"unknown model {n}")
    reg = {n: resolve(raw, n) for n in names}
    if a.action in ("start", "stop"):
        for n in names:
            (start if a.action == "start" else stop)(n, reg[n], a)
    if a.action in ("start", "wait"):
        t0 = time.time()
        pending = set(names)
        while pending and time.time() - t0 < a.timeout:
            for n in list(pending):
                if healthy(reg[n]["port"]):
                    print(f"{n}: healthy after {time.time() - t0:.0f}s", flush=True)
                    pending.discard(n)
            time.sleep(10)
        if pending:
            sys.exit(f"not healthy: {sorted(pending)}")
    if a.action == "status":
        for n in names:
            print(f"{n}: port {reg[n]['port']} {'UP' if healthy(reg[n]['port']) else 'down'}")


if __name__ == "__main__":
    main()
