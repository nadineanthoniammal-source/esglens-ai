"""Rule-based detector for POSSIBLE greenwashing: vague claims with no number, target year or evidence."""
import re

VAGUE = [
    r"committed to", r"strive|striving", r"eco[- ]friendly", r"environmentally (conscious|friendly)",
    r"sustainable future", r"green (future|solutions?|initiatives?)", r"planet[- ]friendly",
    r"carbon[- ]neutral", r"responsible(ly)? (corporate|organi[sz]ation|operations|business)",
    r"clean energy", r"nature[- ]positive", r"aspire|aspiration", r"working towards",
    r"best[- ]in[- ]class", r"leading (the way|sustainab)", r"we care",
]
ESG_HINT = re.compile(r"sustainab|emission|carbon|climate|green|renewable|environment|net[- ]zero|energy|waste|water", re.I)
EVIDENCE = re.compile(r"\d")  # any number: %, year, tonnes, etc.
# Questionnaire / template wording in BRSR forms is not a company claim
TEMPLATE = re.compile(r"^(please|provide|indicate|describe|state|specify|give details|details of|whether|"
                      r"does the entity|has the entity|is the entity|mention|list)\b|\?", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _dup(a: str, b: str) -> bool:
    """Chunks overlap, so the same sentence can appear twice (sometimes cut). Treat overlaps as duplicates."""
    return a == b or a[:40] in b or b[:40] in a


def detect(chunks: list[dict]) -> dict:
    vague_re = re.compile("|".join(VAGUE), re.I)
    flags, seen = [], []
    for c in chunks:
        for s in re.split(r"(?<=[.!?])\s+", c["text"]):
            s = s.strip()
            if len(s) < 40 or not s[0].isupper():   # skip fragments cut by chunk boundaries
                continue
            if TEMPLATE.search(s) or not ESG_HINT.search(s):
                continue
            n = _norm(s)
            if any(_dup(n, x) for x in seen):
                continue
            seen.append(n)
            if vague_re.search(s) and not EVIDENCE.search(s):
                flags.append({"page": c["page"], "claim": s[:300],
                              "reason": "Possible vague claim: no number, baseline or target year"})
    score = round(100 * len(flags) / len(seen), 1) if seen else 0.0
    return {"flags": flags, "esg_sentences": len(seen), "score": score}
