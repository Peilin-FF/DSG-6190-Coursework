"""Unified question items and manifest I/O for VAD, SpatialEval Maze-Nav and AI2D."""

import json
import os
import re
from dataclasses import asdict, dataclass, field


@dataclass
class Item:
    id: str
    dataset: str
    image: str            # path relative to the data root
    image_key: str        # questions with the same key share one image
    concept: str          # natural-language concept name given to every method
    category: str
    qtype: str
    question: str         # full question shown to the model, including the answer choices
    choices: list
    answer: str
    answer_idx: int
    letters: bool = False  # choices are presented with option letters (A, B, ...)
    order: int = 0         # index of this question among the questions of its image
    extra: dict = field(default_factory=dict)

    def image_path(self, root: str) -> str:
        return os.path.join(root, self.image)


def write_manifest(items: list, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def read_manifest(path: str) -> list:
    with open(path) as f:
        return [Item(**json.loads(line)) for line in f if line.strip()]


def letter(i: int) -> str:
    return chr(ord("A") + i)


def format_lettered(question: str, choices: list) -> str:
    opts = "\n".join(f"{letter(i)}. {c}" for i, c in enumerate(choices))
    return f"{question}\nOptions:\n{opts}"


MAZE_MARKER = "Please answer the following question based on the provided information."


def question_key(it: Item) -> str:
    """The question without answer choices or dataset legends (used to extend schemas per question)."""
    q = it.extra.get("question_open_ended") or it.question
    q = q.split(MAZE_MARKER)[-1]
    q = re.split(r"\n?Options:|Available options", q)[0]
    return q.strip()
