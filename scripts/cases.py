"""Find and render qualitative cases (good: DSG fixes Direct in every run; bad: DSG breaks Direct in every run).

  python scripts/cases.py find   --model qwen3-vl-8b --out results/cases_qwen3-vl-8b.json
  python scripts/cases.py render --cases results/cases_selected.json --out report/figures/fig_cases
"""

import argparse
import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from dsg.data import read_manifest  # noqa: E402


def load(runs_root, suite, model, dataset, variant, run):
    p = os.path.join(runs_root, suite, model, dataset, variant, f"run{run}.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
                out[r["id"]] = r
            except json.JSONDecodeError:
                pass
    return out


def find(a):
    exclude = set()
    if a.exclude and os.path.exists(a.exclude):
        exclude = set(json.load(open(a.exclude)))  # VAD photos that already appear in the original paper
    res = []
    for ds in ("vad", "maze", "maze_nolegend", "ai2d"):
        items = {it.id: it for it in read_manifest(os.path.join(a.data, "manifests", f"{ds}.jsonl"))}
        direct = load(a.runs, "main", a.model, ds, "direct", 0)
        full = [load(a.runs, "main", a.model, ds, "dsg_full", r) for r in (0, 1, 2)]
        q = [load(a.runs, "improve", a.model, ds, "dsg_q", r) for r in (0, 1, 2)]
        for i, d in direct.items():
            if not all(i in f for f in full):
                continue
            fc = [f[i]["correct"] for f in full]
            kind = None
            if not d["correct"] and all(fc):
                kind = "good"
            elif d["correct"] and not any(fc):
                kind = "bad"
            elif not d["correct"] and not any(fc):
                kind = "both_wrong"
            if kind is None:
                continue
            it = items[i]
            if any(it.image.endswith(e) for e in exclude):
                continue
            qc = [g[i]["correct"] for g in q if i in g]
            res.append({"id": i, "dataset": ds, "kind": kind, "concept": it.concept, "qtype": it.qtype,
                        "image": it.image, "question": it.extra.get("question_open_ended") or it.question,
                        "choices": it.choices, "answer": it.answer, "direct": d["output"],
                        "dsg": full[0][i]["output"], "grounding": full[0][i].get("grounding"),
                        "schema": full[0][i].get("schema"), "dsg_q_correct": qc,
                        "dsg_q_grounding": (q[0].get(i) or {}).get("grounding")})
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    from collections import Counter
    print(Counter((r["dataset"], r["kind"]) for r in res))


def render(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    cases = json.load(open(a.cases))
    n = len(cases)
    cols = a.cols
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6.875, a.row_h * rows))
    for k, c in enumerate(cases):
        r, col = divmod(k, cols)
        x0, w = col / cols, 1 / cols
        y_top = 1 - r / rows
        ax_img = fig.add_axes([x0 + 0.01, y_top - 0.52 / rows, w - 0.02, 0.48 / rows])
        im = Image.open(os.path.join(a.data, c["image"])).convert("RGB")
        im.thumbnail((900, 900))
        ax_img.imshow(im)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        color = {"good": "#009E73", "bad": "#D55E00"}.get(c["kind"], "#555555")
        for s in ax_img.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2.0)
        ax_img.set_title(c.get("title", c["concept"]), fontsize=7.5, color=color, pad=2)
        lines = ["Q: " + c.get("short_q", c["question"])]
        g = c.get("show_grounding") or [f"{x['term']}: {x['answer']}" for x in (c.get("grounding") or [])][:4]
        if g:
            lines.append("Grounding: " + "; ".join(g))
        mark = lambda ok: r"$\checkmark$" if ok else r"$\times$"
        lines.append(f"Direct: {c['direct_short']} {mark(c['direct_ok'])}    DSG: {c['dsg_short']} {mark(c['dsg_ok'])}")
        if c.get("extra_line"):
            lines.append(c["extra_line"])
        lines.append(f"GT: {c['answer']}")
        txt = "\n".join(textwrap.fill(l, width=a.wrap) for l in lines)
        fig.text(x0 + 0.012, y_top - 0.545 / rows, txt, fontsize=6.4, va="top", ha="left", linespacing=1.15)
    fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["find", "render"])
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--runs", default=os.environ.get("DSG_RUNS", "runs"))
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases")
    ap.add_argument("--exclude", default=os.path.join(os.environ.get("DSG_DATA", "data"), "paper_figure_images.json"))
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--row-h", type=float, default=2.55)
    ap.add_argument("--wrap", type=int, default=44)
    a = ap.parse_args()
    find(a) if a.action == "find" else render(a)


if __name__ == "__main__":
    main()
