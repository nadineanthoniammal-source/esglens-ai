"""Shared entry point used by the Streamlit UI and the REST API."""
import io
from typing import Optional

from .graph import run_pipeline
from .ingest import build_collection


def analyze_bytes(data: bytes, revenue_inr_crore: Optional[float] = None,
                  api_key: Optional[str] = None, embedding_function=None):
    """PDF bytes -> (collection, n_chunks, result)."""
    col, n = build_collection(io.BytesIO(data), data, embedding_function)
    result = run_pipeline(col, revenue_inr_crore, api_key)
    return col, n, result


def to_jsonable(result: dict) -> dict:
    """Strip non-serialisable objects (vector DB handle, DataFrame) for API output."""
    a = result["analysis"]
    return {
        "metrics": result["metrics"],
        "verification": result["verification"],
        "validator_issues": result["issues"],
        "scope1_plus_scope2": a["scope12_total"],
        "emission_intensity_tco2e_per_inr_crore": a["intensity_tco2e_per_crore"],
        "esg_distribution": a["esg_distribution"],
        "greenwashing_score_percent": a["greenwashing"]["score"],
        "greenwashing_flags": a["greenwashing"]["flags"][:10],
        "report_markdown": result["report"],
    }
