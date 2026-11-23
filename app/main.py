"""ESGLens AI - Streamlit UI. Run: streamlit run app/main.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.compare import build_comparison  # noqa: E402
from app.graph import answer_question  # noqa: E402
from app.schemas import METRIC_KEYS  # noqa: E402
from app.service import analyze_bytes  # noqa: E402

MAX_RUNS = int(os.getenv("DEMO_MAX_RUNS", "6"))          # analyses per visitor on the shared key
MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "15"))  # chat questions per visitor on the shared key
SERVER_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="ESGLens AI", page_icon="🌱", layout="wide")
st.title("🌱 ESGLens AI")
st.caption("Agentic ESG report analyzer · LangGraph + LangChain + ChromaDB + Groq · every number is verified against its source page")
st.session_state.setdefault("runs", 0)
st.session_state.setdefault("questions", 0)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Setup")
    own_key = ""
    with st.expander("Use your own Groq key (optional)", expanded=not SERVER_KEY):
        own_key = st.text_input("Groq API key", type="password",
                                help="Used only for your session. Not stored.")
    using_shared = not own_key
    has_key = bool(own_key or SERVER_KEY)
    if not has_key:
        st.warning("Add a Groq key above to start.")
    elif using_shared:
        st.info(f"Demo mode: {max(MAX_RUNS - st.session_state['runs'], 0)} analyses left this session. "
                "Add your own key for unlimited use.")
    st.caption("Tip: use text-based BRSR / sustainability PDFs (not scans).")


def can_run(n=1):
    return (not using_shared) or st.session_state["runs"] + n <= MAX_RUNS


def badge(v):
    s, p = v["status"], v["page"]
    if s == "verified":
        return f"✅ verified on p.{p}"
    if s == "derived":
        return f"🧮 computed from energy values (p.{p})"
    if s == "unverified":
        return f"⚠️ not confirmed on p.{p}" if p else "⚠️ no page cited"
    return "— not disclosed"


def auto_label(label, default, result):
    """If the user left the default label, name the column from the report itself."""
    if label.strip() != default:
        return label.strip()
    mm = result["metrics"]
    return f"{mm.get('company_name') or default} {mm.get('reporting_year') or ''}".strip()


def run_one(data, revenue, key):
    return analyze_bytes(data, revenue or None, api_key=key or None)


tab_single, tab_cmp = st.tabs(["📄 Analyze one report", "⚖️ Compare two reports"])

# ================= single report =================
with tab_single:
    up = st.file_uploader("Upload sustainability / BRSR report (PDF)", type=["pdf"], key="up_single")
    revenue = st.number_input("Revenue / turnover in ₹ crore (optional, gives emission intensity)",
                              min_value=0.0, value=0.0, key="rev_single")
    if st.button("Analyze report", type="primary", disabled=not (up and has_key), key="go_single"):
        if not can_run():
            st.error("Demo limit reached for this session. Add your own Groq key in the sidebar.")
        else:
            with st.status("Running agents...", expanded=True) as status:
                st.write("📄 Ingesting PDF → chunking → embedding into ChromaDB")
                st.write("🤖 Retrieve → Extract → Validate (retry) → Analyze → Report")
                try:
                    col, n, result = run_one(up.getvalue(), revenue, own_key)
                except Exception as e:  # show a readable error instead of a crash
                    status.update(label="Failed", state="error")
                    st.error(f"{type(e).__name__}: {e}")
                    st.stop()
                st.write(f"✅ {n} chunks indexed")
                status.update(label="Done", state="complete")
            st.session_state["runs"] += 1
            st.session_state["single_res"] = {"col": col, "result": result, "chat": []}

    res = st.session_state.get("single_res")
    if not res:
        st.info("Upload a report and click **Analyze report**.")
    else:
        result = res["result"]
        m, a, ver = result["metrics"], result["analysis"], result["verification"]
        st.subheader(f"{m.get('company_name') or 'Company'} — {m.get('reporting_year') or ''}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scope 1+2 (tCO₂e)", f"{a['scope12_total']:,.0f}" if a["scope12_total"] is not None else "n/a")
        c2.metric("Intensity (tCO₂e/₹ cr)",
                  a["intensity_tco2e_per_crore"] if a["intensity_tco2e_per_crore"] is not None else "n/a")
        c3.metric("Greenwashing score", f"{a['greenwashing']['score']}%")
        c4.metric("Net-zero target", m.get("net_zero_target_year") or "not disclosed")

        t1, t2, t3, t4, t5 = st.tabs(["📊 Metrics", "🏷️ E/S/G tags", "🚩 Greenwashing", "📝 Report", "💬 Ask the report"])
        with t1:
            rows = []
            for k in METRIC_KEYS:
                rows.append({"metric": k, **m[k], "verification": badge(ver[k])})
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")
            n_ver = sum(v["status"] in ("verified", "derived") for v in ver.values())
            n_tot = sum(v["status"] != "not disclosed" for v in ver.values())
            st.caption(f"{n_ver} of {n_tot} extracted values were confirmed by the validator agent (found on the cited page, or computed from verified values). "
                       "Always spot-check against the PDF.")
            st.download_button("Download metrics (.csv)", df.to_csv(index=False), file_name="esg_metrics.csv")
            if a["unresolved_issues"]:
                st.warning("Validator caveats:\n- " + "\n- ".join(a["unresolved_issues"]))
        with t2:
            st.bar_chart(pd.Series(a["esg_distribution"], name="chunks"))
            st.dataframe(a["tags_df"][["page", "label", "confidence", "text"]], width="stretch")
        with t3:
            gw = a["greenwashing"]
            st.write(f"{len(gw['flags'])} vague claims out of {gw['esg_sentences']} ESG-related sentences.")
            for f in gw["flags"]:
                st.markdown(f"**p.{f['page']}** — {f['claim']}  \n_{f['reason']}_")
        with t4:
            st.markdown(result["report"])
            st.download_button("Download report (.md)", result["report"], file_name="esg_report.md")
        with t5:
            for role, msg in res["chat"]:
                st.chat_message(role).write(msg)
            q = st.chat_input("Ask e.g. 'What is the net-zero target year?'")
            if q:
                if using_shared and st.session_state["questions"] >= MAX_QUESTIONS:
                    st.error("Question limit reached for this session. Add your own Groq key in the sidebar.")
                else:
                    st.chat_message("user").write(q)
                    try:
                        ans = answer_question(res["col"], q, api_key=own_key or None)
                    except Exception as e:
                        ans = f"Error: {type(e).__name__}: {e}"
                    st.chat_message("assistant").write(ans)
                    st.session_state["questions"] += 1
                    res["chat"] += [("user", q), ("assistant", ans)]

# ================= comparison =================
with tab_cmp:
    st.write("Compare **two years of one company** or **two companies**. Units must match for a % change.")
    ca, cb = st.columns(2)
    with ca:
        up_a = st.file_uploader("Report A (PDF)", type=["pdf"], key="up_a")
        label_a = st.text_input("Label A (leave as is to auto-name from the report)", "Report A")
        rev_a = st.number_input("Revenue A (₹ crore, optional)", min_value=0.0, value=0.0, key="rev_a")
    with cb:
        up_b = st.file_uploader("Report B (PDF)", type=["pdf"], key="up_b")
        label_b = st.text_input("Label B (leave as is to auto-name from the report)", "Report B")
        rev_b = st.number_input("Revenue B (₹ crore, optional)", min_value=0.0, value=0.0, key="rev_b")

    if st.button("Compare reports", type="primary", disabled=not (up_a and up_b and has_key), key="go_cmp"):
        if label_a.strip() == label_b.strip():
            st.error("Give the two reports different labels.")
        elif not can_run(2):
            st.error("Not enough demo analyses left (comparison needs 2). Add your own Groq key.")
        else:
            with st.status("Analyzing both reports...", expanded=True) as status:
                try:
                    st.write(f"▶ {label_a}")
                    _, _, ra = run_one(up_a.getvalue(), rev_a, own_key)
                    st.write(f"▶ {label_b}")
                    _, _, rb = run_one(up_b.getvalue(), rev_b, own_key)
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"{type(e).__name__}: {e}")
                    st.stop()
                status.update(label="Done", state="complete")
            st.session_state["runs"] += 2
            la, lb = auto_label(label_a, "Report A", ra), auto_label(label_b, "Report B", rb)
            if la == lb:
                lb += " (B)"
            st.session_state["cmp_res"] = (ra, rb, la, lb)

    cmp_res = st.session_state.get("cmp_res")
    if cmp_res:
        ra, rb, la, lb = cmp_res
        st.subheader(f"{la}  vs  {lb}")
        cdf = build_comparison(ra, rb, la, lb)
        st.dataframe(cdf, width="stretch")
        st.caption("Units are converted where possible (kg→tonnes, TJ/MWh→GJ, etc.). Percent metrics show the change in "
                   "percentage points. Absolute totals depend on company size, so for two companies compare "
                   "emission intensity (enter each company's revenue).")
        chart = cdf.set_index("metric")["change_%"].dropna()
        if not chart.empty:
            st.write(f"**Change from {la} to {lb} (%)**")
            st.bar_chart(chart)
        st.download_button("Download comparison (.csv)", cdf.to_csv(index=False), file_name="esg_comparison.csv")
