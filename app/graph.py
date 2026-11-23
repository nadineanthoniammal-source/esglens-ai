r"""LangGraph agent workflow:

 retrieve -> extract -> validate --(issues & retries left)--> extract
                                 \--(ok)--> analyze -> report
"""
import json
import re
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import greenwash
from .ingest import all_chunks
from .llm import call_llm, parse_json
from .ml_classifier import tag_chunks
from .schemas import METRIC_KEYS, ESGMetrics

QUERIES = [
    "Scope 1 greenhouse gas emissions tCO2e",
    "Scope 2 greenhouse gas emissions tCO2e",
    "Scope 3 emissions value chain",
    "total energy consumption GJ electricity",
    "total energy consumed from renewable sources GJ",
    "renewable energy share percentage",
    "total water consumption withdrawal kilolitres",
    "total waste generated tonnes",
    "net zero target year decarbonisation",
    "company name financial year reporting period",
]
MAX_RETRIES = 1


class State(TypedDict, total=False):
    col: Any
    revenue_inr_crore: Optional[float]
    context: str
    pages_text: dict
    metrics: dict
    issues: list
    retries: int
    analysis: dict
    report: str
    verification: dict
    api_key: Optional[str]


# ---------------- nodes ----------------
def retrieve(state: State) -> State:
    col = state["col"]
    seen, parts, pages_text = set(), [], {}
    for q in QUERIES:
        res = col.query(query_texts=[q], n_results=4)
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            if doc in seen:
                continue
            seen.add(doc)
            parts.append(f"[Page {meta['page']}] {doc}")
            pages_text.setdefault(meta["page"], "")
            pages_text[meta["page"]] += " " + doc
    return {"context": "\n\n".join(parts), "pages_text": pages_text, "issues": [], "retries": 0}


EXTRACT_PROMPT = """You are an ESG data extraction agent. From the report excerpts below, extract metrics.
Return ONLY a JSON object with exactly these keys:
company_name, reporting_year, net_zero_target_year (strings or null), and for each of
{keys} an object {{"value": number|null, "unit": string|null, "year": string|null, "page": integer|null}}.
renewable_energy_consumption = total energy consumed from renewable sources (A+B+C) in the same unit as total_energy_consumption.
Leave renewable_energy_share_percent null unless the report itself prints a percentage (it is computed separately).
Rules: use ONLY numbers that appear in the excerpts; "page" is the [Page N] label where the number was found;
renewable_energy_share_percent must be 0-100; if a metric is not present use null values. Do not guess.
{feedback}
EXCERPTS:
{context}
"""


def _unit_key(u):
    return (u or "").lower().replace(" ", "")


def _derive(m: dict) -> dict:
    """Compute renewable share in code (deterministic) from renewable and total energy."""
    r, t = m["renewable_energy_consumption"], m["total_energy_consumption"]
    if (r["value"] is not None and t["value"] and _unit_key(r["unit"]) and
            _unit_key(r["unit"]) == _unit_key(t["unit"])):
        m["renewable_energy_share_percent"] = {
            "value": round(r["value"] / t["value"] * 100, 2), "unit": "%",
            "year": t["year"] or r["year"], "page": r["page"], "derived": True}
    return m


def extract(state: State) -> State:
    feedback = ""
    if state.get("issues"):
        feedback = "A validator found problems with your previous answer. Fix them:\n- " + "\n- ".join(state["issues"])
    prompt = EXTRACT_PROMPT.format(keys=", ".join(METRIC_KEYS), feedback=feedback, context=state["context"])
    raw = call_llm(prompt, json_mode=True, api_key=state.get("api_key"))
    data = parse_json(raw)
    metrics = _derive(ESGMetrics.model_validate(data).model_dump())
    return {"metrics": metrics}


def _number_in_text(value: float, text: str) -> bool:
    """True if the number appears in the text. Ignores commas, so 36,459,294 and 3,64,59,294 both match."""
    plain = re.sub(r"(?<=\d),(?=\d)", "", text)
    for c in {f"{value:.0f}", f"{value:g}", f"{value:.2f}"}:
        if re.search(rf"(?<!\d){re.escape(c)}(?!\d)", plain):
            return True
    return False


def _page_text(col, page: int, cache: dict) -> str:
    if page not in cache:
        res = col.get(where={"page": page}, include=["documents"])
        cache[page] = " ".join(res["documents"])
    return cache[page]


def validate(state: State) -> State:
    """Check every extracted number against the FULL text of the page the model cited.
    Also records a per-metric verification status shown as a badge in the UI."""
    m, col, issues, cache, ver = state["metrics"], state["col"], [], {}, {}
    for key in METRIC_KEYS:
        met = m[key]
        page = met["page"]
        if met["value"] is None:
            ver[key] = {"status": "not disclosed", "page": None}
            continue
        before = len(issues)
        if met.get("derived"):
            if not (0 <= met["value"] <= 100):
                issues.append(f"{key}: derived value {met['value']} is outside 0-100.")
            ver[key] = {"status": "derived" if len(issues) == before else "unverified", "page": page}
            continue
        if met["value"] < 0:
            issues.append(f"{key}: negative value {met['value']}.")
        if key == "renewable_energy_share_percent" and not (0 <= met["value"] <= 100):
            issues.append(f"{key}: {met['value']} is outside 0-100.")
        if not met["unit"]:
            issues.append(f"{key}: value given without a unit.")
        if page is None:
            issues.append(f"{key}: no page was cited for the value.")
        elif not _number_in_text(met["value"], _page_text(col, page, cache)):
            issues.append(f"{key}: value {met['value']:.0f} not found on cited page {page}.")
        ver[key] = {"status": "verified" if len(issues) == before else "unverified", "page": page}
    return {"issues": issues, "verification": ver,
            "retries": state.get("retries", 0) + (1 if issues else 0)}


def route_after_validate(state: State) -> str:
    if state["issues"] and state["retries"] <= MAX_RETRIES:
        return "extract"
    return "analyze"


def analyze(state: State) -> State:
    chunks = all_chunks(state["col"])
    tags = tag_chunks(chunks)
    dist = tags["label"].value_counts().to_dict()
    gw = greenwash.detect(chunks)
    m = state["metrics"]
    s1, s2 = m["scope1_emissions"]["value"], m["scope2_emissions"]["value"]
    intensity = None
    rev = state.get("revenue_inr_crore")
    if rev and s1 is not None and s2 is not None:
        intensity = round((s1 + s2) / rev, 2)
    return {"analysis": {
        "esg_distribution": dist,
        "tags_df": tags,
        "greenwashing": gw,
        "scope12_total": (s1 or 0) + (s2 or 0) if (s1 is not None or s2 is not None) else None,
        "intensity_tco2e_per_crore": intensity,
        "unresolved_issues": state["issues"],
    }}


REPORT_PROMPT = """Write a concise one-page ESG summary in Markdown for an analyst.
Use ONLY the data below. Sections: Overview, Key Metrics (table), Risks & Red Flags, Questions for follow-up.
Mention page numbers for each metric. If a metric is null say "not extracted from the report" (do NOT claim the company failed to disclose it).
Metrics marked derived=true were computed in code (renewable energy / total energy x 100).
Questions for follow-up: only data gaps an analyst should check; do not give advice to the company.
Greenwashing items are automatic keyword flags, so call them "possible vague claims".
Unresolved validation issues (mention as data-quality caveats): {issues}

EXTRACTED METRICS:
{metrics}

ANALYSIS:
{analysis}
"""


def report(state: State) -> State:
    a = state["analysis"]
    slim = {
        "esg_distribution": a["esg_distribution"],
        "greenwashing_score_percent": a["greenwashing"]["score"],
        "top_vague_claims": a["greenwashing"]["flags"][:5],
        "scope1_plus_scope2": a["scope12_total"],
        "intensity_tco2e_per_inr_crore": (a["intensity_tco2e_per_crore"]
                                          if a["intensity_tco2e_per_crore"] is not None
                                          else "not computed (revenue not provided)"),
    }
    text = call_llm(REPORT_PROMPT.format(issues=a["unresolved_issues"] or "none",
                                         metrics=json.dumps(state["metrics"], indent=1),
                                         analysis=json.dumps(slim, indent=1)),
                    api_key=state.get("api_key"))
    return {"report": text}


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("analyze", analyze)
    g.add_node("report", report)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"extract": "extract", "analyze": "analyze"})
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return g.compile()


def run_pipeline(col, revenue_inr_crore: Optional[float] = None, api_key: Optional[str] = None) -> State:
    return build_graph().invoke({"col": col, "revenue_inr_crore": revenue_inr_crore, "api_key": api_key})


# ---------------- RAG Q&A ----------------
def answer_question(col, question: str, api_key: Optional[str] = None) -> str:
    res = col.query(query_texts=[question], n_results=5)
    ctx = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in zip(res["documents"][0], res["metadatas"][0]))
    prompt = (f"Answer the question using ONLY the report excerpts. Cite pages like (p. 12). "
              f"If the answer is not in the excerpts, say so.\n\nEXCERPTS:\n{ctx}\n\nQUESTION: {question}")
    return call_llm(prompt, api_key=api_key)
