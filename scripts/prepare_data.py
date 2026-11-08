"""Build question manifests from the raw datasets.

Expected raw layout under --raw:
  VAD/<category>_concepts/<concept>/{vqa.json, *.jpg}            (Visual Abstractions Dataset)
  spatialeval/vqa/test-0000{0,1}-of-00002.parquet                 (SpatialEval, VQA format)
  ai2d/data/test-*.parquet  and  ai2d_orig/ai2d/categories.json   (AI2D test split + category file)

Outputs under --out: manifests/{vad,maze,maze_nolegend,ai2d}.jsonl and images/{maze,ai2d}/.
"""

import argparse
import glob
import io
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsg.data import Item, format_lettered, write_manifest  # noqa: E402

VAD_CONCEPTS = {
    "tic_tac_toe": "tic-tac-toe", "maze": "maze", "treasure_map": "treasure map",
    "solar_system": "solar system", "atom": "atom", "cell": "cell",
    "helping": "helping", "deceiving": "deceiving", "negotiating": "negotiating",
    "setting_up_table_for_two": "setting up a table for two",
    "tidying_up_guest_room": "tidying up a guest room",
    "putting_up_decorations_on_door": "putting up decorations on a door",
}
VAD_QTYPE = {"O": "open", "B": "binary", "C": "counting"}

AI2D_CONCEPTS = {
    "foodChainsWebs": "food web", "lifeCycles": "life cycle", "moonPhaseEquinox": "moon phases and seasons",
    "photosynthesisRespiration": "photosynthesis and respiration", "rockCycle": "rock cycle",
    "partsOfTheEarth": "layers of the Earth", "waterCNPCycle": "water and nutrient cycle",
    "rockStrata": "rock strata", "solarSystem": "solar system", "eclipses": "eclipse",
    "faultsEarthquakes": "faults and earthquakes", "volcano": "volcano", "atomStructure": "atom",
}
MAZE_QTYPE = {0: "right_turns", 1: "total_turns", 2: "relative_position"}
MAZE_MARKER = "Please answer the following question based on the provided information."


def parse_vad_choices(q: str) -> list:
    m = re.search(r"Answer with one of the following answer choices:\s*(.*)$", q, re.S)
    s = m.group(1).strip()
    s = s[:-1] if s.endswith(".") else s
    return [c.strip() for c in s.split(", ")]


def build_vad(raw: str, out: str) -> list:
    items = []
    for f in sorted(glob.glob(os.path.join(raw, "VAD", "*_concepts", "*", "vqa.json"))):
        category = f.split(os.sep)[-3].replace("_concepts", "")
        folder = f.split(os.sep)[-2]
        concept = VAD_CONCEPTS[folder]
        seen = defaultdict(int)
        for x in json.load(open(f))["questions"]:
            choices = parse_vad_choices(x["question_multiple_choice"])
            img = x["image_filename"]
            k = f"vad/{folder}/{os.path.splitext(img)[0]}"
            order = seen[k]
            seen[k] += 1
            items.append(Item(
                id=f"{k}/q{order}", dataset="vad", image=os.path.relpath(os.path.join(os.path.dirname(f), img), out),
                image_key=k, concept=concept, category=category, qtype=VAD_QTYPE[x["question_type"]],
                question=x["question_multiple_choice"].strip(), choices=choices,
                answer=x["answer_multiple_choice"].strip(), answer_idx=choices.index(x["answer_multiple_choice"].strip()),
                letters=False, order=order,
                extra={"folder": folder, "question_open_ended": x["question_open_ended"],
                       "answer_open_ended": x["answer_open_ended"]}))
    return items


def parse_lettered_options(block: str) -> list:
    opts = re.findall(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", block)
    return [o[1].rstrip(".").strip() for o in opts]


def maze_question_sentence(text: str) -> str:
    q = text.split(MAZE_MARKER)[-1]
    return q.split("Available options")[0].strip()


TURN_DEFINITIONS = (
    "A right turn is defined as a change in movement direction that is 90 degrees clockwise relative to the "
    "previous direction. A left turn is defined as a change in movement direction that is 90 degrees "
    "anticlockwise relative to the previous direction."
)


def nolegend_question(sentence: str) -> str:
    """Remove the colour legend and the S/E symbols; keep the definitions of the turn terms."""
    s = sentence.replace(" (marked by Blue)", "")
    s = s.replace("from S to E", "from the start to the exit")
    s = s.replace("the exit (E)", "the exit").replace("the starting point (S)", "the start")
    return f"The figure represents a maze. {TURN_DEFINITIONS} {s}"


def build_maze(raw: str, out: str, n_mazes: int) -> tuple:
    cols = ["id", "text", "image", "oracle_answer", "oracle_option"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in
                    sorted(glob.glob(os.path.join(raw, "spatialeval", "vqa", "test-*.parquet")))])
    df = df[df.id.str.startswith("mazenav.")].copy()
    df["i"] = df.id.str.split(".").str[2].astype(int)
    df["j"] = df.id.str.split(".").str[3].astype(int)
    df = df[df.i < n_mazes].sort_values(["i", "j"])
    img_dir = os.path.join(out, "images", "maze")
    os.makedirs(img_dir, exist_ok=True)
    legend, nolegend = [], []
    for r in df.itertuples():
        rel = os.path.join("images", "maze", f"{r.i}.png")
        if not os.path.exists(os.path.join(out, rel)):
            Image.open(io.BytesIO(r.image["bytes"])).save(os.path.join(out, rel))
        opt_block = r.text.split("Available options:")[-1]
        choices = parse_lettered_options(opt_block)
        idx = ord(r.oracle_option.strip()) - ord("A")
        assert choices[idx] == r.oracle_answer.strip(), (r.id, choices, r.oracle_answer)
        sentence = maze_question_sentence(r.text)
        base = dict(image=rel, image_key=f"maze/{r.i}", concept="maze", category="strategic",
                    qtype=MAZE_QTYPE[r.j], choices=choices, answer=choices[idx], answer_idx=idx,
                    letters=True, order=r.j)
        legend.append(Item(id=f"maze/{r.i}/q{r.j}", dataset="maze",
                           question=r.text.strip().rstrip("."), extra={"source_id": r.id}, **base))
        nolegend.append(Item(id=f"maze_nolegend/{r.i}/q{r.j}", dataset="maze_nolegend",
                             question=format_lettered(nolegend_question(sentence), choices),
                             extra={"source_id": r.id}, **base))
    return legend, nolegend


def build_ai2d(raw: str, out: str, per_cat: int, seed: int) -> list:
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(raw, "ai2d", "data", "test-*.parquet")))])
    df = df.reset_index(drop=True)
    df["row"] = range(len(df))
    df["fn"] = [x["path"] for x in df.image]
    cats = json.load(open(os.path.join(raw, "ai2d_orig", "ai2d", "categories.json")))
    df["cat"] = df.fn.map(cats)
    df = df[df.cat.isin(AI2D_CONCEPTS)]
    parts = []
    for cat, g in df.groupby("cat"):
        parts.append(g.sample(n=min(per_cat, len(g)), random_state=seed) if len(g) > per_cat else g)
    df = pd.concat(parts).sort_values("row")
    img_dir = os.path.join(out, "images", "ai2d")
    os.makedirs(img_dir, exist_ok=True)
    items, seen = [], defaultdict(int)
    for r in df.itertuples():
        rel = os.path.join("images", "ai2d", r.fn)
        if not os.path.exists(os.path.join(out, rel)):
            with open(os.path.join(out, rel), "wb") as f:
                f.write(r.image["bytes"])
        choices = [str(c).strip() for c in r.options]
        idx = int(r.answer)
        key = f"ai2d/{os.path.splitext(r.fn)[0]}"
        order = seen[key]
        seen[key] += 1
        label_q = all(len(c) <= 2 for c in choices)
        items.append(Item(
            id=f"{key}/r{r.row}", dataset="ai2d", image=rel, image_key=key, concept=AI2D_CONCEPTS[r.cat],
            category=r.cat, qtype="label" if label_q else "content",
            question=format_lettered(r.question.strip(), choices), choices=choices, answer=choices[idx],
            answer_idx=idx, letters=True, order=order, extra={"source_row": int(r.row), "file": r.fn}))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maze-n", type=int, default=200)
    ap.add_argument("--ai2d-per-cat", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    man = os.path.join(a.out, "manifests")
    vad = build_vad(a.raw, a.out)
    write_manifest(vad, os.path.join(man, "vad.jsonl"))
    legend, nolegend = build_maze(a.raw, a.out, a.maze_n)
    write_manifest(legend, os.path.join(man, "maze.jsonl"))
    write_manifest(nolegend, os.path.join(man, "maze_nolegend.jsonl"))
    ai2d = build_ai2d(a.raw, a.out, a.ai2d_per_cat, a.seed)
    write_manifest(ai2d, os.path.join(man, "ai2d.jsonl"))
    for name, its in [("vad", vad), ("maze", legend), ("maze_nolegend", nolegend), ("ai2d", ai2d)]:
        missing = [i.id for i in its if not os.path.exists(os.path.join(a.out, i.image))]
        print(f"{name}: {len(its)} items, {len({i.image_key for i in its})} images, missing images {len(missing)}")


if __name__ == "__main__":
    main()
