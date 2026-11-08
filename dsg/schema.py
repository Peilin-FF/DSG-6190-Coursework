"""Schema programs: parsing, dependency order and serialisation.

A schema is a short program that defines a directed acyclic graph (DAG) over the components of an
abstract concept, for example

    p(image | concept=tic-tac-toe) =
      p(board | concept=tic-tac-toe)
      p(symbols | concept=tic-tac-toe)
      p(strategy | concept=tic-tac-toe, symbols)
      p(image | board, symbols, strategy)
"""

import re
from dataclasses import asdict, dataclass, field

TERM_RE = re.compile(r"\b(?:p|gen)\(\s*([^|()]+?)\s*\|\s*([^()]*?)\s*\)")
FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")


class SchemaError(ValueError):
    pass


def clean_name(name: str) -> str:
    name = name.strip().strip("'\"`").replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


@dataclass
class Schema:
    concept: str
    components: list = field(default_factory=list)  # [(term, [deps])] in dependency order
    final: list = field(default_factory=list)       # terms that condition the final question
    program: str = ""
    source: str = ""

    @property
    def terms(self) -> list[str]:
        return [t for t, _ in self.components]

    def depth(self) -> dict:
        d = {}
        for t, deps in self.components:
            d[t] = 1 + max((d[x] for x in deps if x in d), default=0)
        return d

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(concept=d["concept"], components=[(t, list(deps)) for t, deps in d["components"]],
                   final=list(d["final"]), program=d.get("program", ""), source=d.get("source", ""))

    def render(self) -> str:
        c = self.concept
        lines = [f"p(image | concept={c}) ="]
        for t, deps in self.components:
            lines.append(f"  p({t} | " + ", ".join([f"concept={c}"] + deps) + ")")
        lines.append("  p(image | " + ", ".join(self.final) + ")")
        return "\n".join(lines)


def topo_order(components: list) -> list:
    """Stable Kahn ordering; dependencies on unknown terms or on later terms in a cycle are dropped."""
    names = [t for t, _ in components]
    known = set(names)
    deps = {t: [d for d in ds if d in known and d != t] for t, ds in components}
    placed, order = set(), []
    remaining = list(names)
    while remaining:
        for t in remaining:
            if all(d in placed for d in deps[t]):
                break
        else:
            t = remaining[0]
            deps[t] = [d for d in deps[t] if d in placed]
        remaining.remove(t)
        placed.add(t)
        order.append((t, deps[t]))
    return order


def parse_program(program: str, concept: str | None = None, source: str = "") -> Schema:
    src = FENCE_RE.sub("", program)
    terms = TERM_RE.findall(src)
    if not terms:
        raise SchemaError("no p(... | ...) terms found")
    parsed_concept, comps, final = None, [], None
    for head, cond in terms:
        head_c = clean_name(head)
        items = [c.strip() for c in cond.split(",") if c.strip()]
        concept_items = [c for c in items if re.match(r"^concept\s*=", c, re.I)]
        deps = [clean_name(c) for c in items if c not in concept_items]
        if head_c.lower() == "image":
            if concept_items and parsed_concept is None and not comps:
                parsed_concept = concept_items[0].split("=", 1)[1].strip()
            else:
                final = deps
            continue
        if head_c not in [t for t, _ in comps]:
            comps.append((head_c, deps))
    if not comps:
        raise SchemaError("schema has no components")
    comps = topo_order(comps)
    terms_set = [t for t, _ in comps]
    final = [t for t in (final or terms_set) if t in terms_set] or terms_set
    return Schema(concept=concept or parsed_concept or "", components=comps, final=final,
                  program=program.strip(), source=source)


def legacy_parse(program: str):
    """The parser of the released notebook, kept to document its behaviour.

    It drops every condition whose text contains the concept name, so a dependency such as
    "cell membrane" is lost for the concept "cell".
    """
    concept_pattern = r'[ \t]*image[ \t]*\|[ \t]*concept[ \t]*=[ \t]*(.*)[ \t]*\)[ \t]*='
    term_pattern = r'[ \t]*(.*)[ \t]*\|[ \t]*'
    final_pattern = r'[ \t]*.*[ \t]*\|[ \t]*(.*)[ \t]*\)'
    segments = program.replace('\n', '').split('p(')
    concept, final_conditioning, cond_programs = None, [], []
    for idx, segment in enumerate(segments):
        s = segment.strip()
        if not s:
            continue
        if s.endswith('='):
            m = re.search(concept_pattern, s)
            if m:
                concept = m.group(1)
        elif s.startswith('image') and (idx + 1 == len(segments)):
            m = re.search(final_pattern, s)
            if m:
                final_conditioning = [c.replace('-', ' ').strip() for c in m.group(1).split(',')]
        else:
            m = re.search(term_pattern, s)
            if m:
                term = m.group(1).replace('-', ' ').strip()
                raw = s.split('|')[-1].split(',')
                cs = [c.replace(')', '').replace('-', ' ').strip() for c in raw if concept not in c]
                cond_programs.append((term, cs))
    return concept, final_conditioning, cond_programs


def render_gen(schema: "Schema") -> str:
    """The schema in the gen(...) notation of the extraction prompt."""
    c = schema.concept.replace(" ", "-")
    lines = [f"gen(concept={c}) ="]
    for t, deps in schema.components:
        args = ", ".join([f"concept={c}"] + [d.replace(" ", "-") for d in deps])
        lines.append(f"gen({t.replace(' ', '-')} | {args})")
    return "\n".join(lines)


def extend_schema(schema: "Schema", extra: list) -> "Schema":
    """Append question-specific components (term, deps) after the concept components."""
    comps = [(t, list(d)) for t, d in schema.components]
    names = {t for t, _ in comps}
    for t, deps in extra:
        if t in names or not t:
            continue
        comps.append((t, [d for d in deps if d in names]))
        names.add(t)
    return Schema(concept=schema.concept, components=comps, final=[t for t, _ in comps],
                  program=schema.program, source=schema.source + "+q")
