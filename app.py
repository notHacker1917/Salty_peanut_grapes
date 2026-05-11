import streamlit as st
import pandas as pd
from uuid import UUID

from cula.client import CulaClient
from cula.verification.fetch import fetch_sink_data
from cula.verification.normalize import normalize
from cula.verification.rules import run_rules
from cula.verification.scoring import score


st.set_page_config(page_title="Carbon Verification Dashboard", layout="wide")
st.title("Carbon Verification Dashboard")
st.caption("Verification dashboard: fetch → normalize → rules → scoring")


def group_results_by_severity(results):
    grouped = {"fail": [], "warn": [], "info": []}
    for r in results:
        grouped.setdefault(r.severity, []).append(r)
    return grouped


def categorize_result(result):
    code = result.code

    removal_vs_removal = {
        "TIMELINE_ORDER",
        "MASS_BALANCE",
        "SITE_CONTINUITY",
        "DELIVERY_DISTANCE",
    }
    removal_vs_machine = {
        "MACHINE_COVERAGE",
        "TEMP_PLAUSIBLE",
    }
    removal_vs_documents = {
        "PROOF_PRESENCE",
        "FILE_REUSE",
    }

    if code in removal_vs_removal:
        return "Removal vs. Removal"
    if code in removal_vs_machine:
        return "Removal vs. Machine Data"
    if code in removal_vs_documents:
        return "Removal vs. Documents"
    return "Other"


def build_summary_df(results):
    rows = []
    for r in results:
        rows.append(
            {
                "code": r.code,
                "severity": r.severity,
                "category": categorize_result(r),
                "message": r.message,
            }
        )
    return pd.DataFrame(rows)


def build_evidence_df(results):
    rows = []
    for r in results:
        if r.evidence:
            row = {
                "code": r.code,
                "severity": r.severity,
                "category": categorize_result(r),
            }
            for k, v in r.evidence.items():
                row[k] = str(v)
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render_result(result):
    label = f"[{result.code}] {result.message}"

    if result.severity == "fail":
        st.error(label)
    elif result.severity == "warn":
        st.warning(label)
    else:
        st.info(label)

    if result.evidence:
        with st.expander("Show evidence"):
            st.json(result.evidence)


st.sidebar.header("Input")
sink_id_input = st.sidebar.text_input("Sink ID")

menu = st.sidebar.selectbox(
    "Choose section",
    [
        "Overview",
        "Verification Summary",
        "Scoring",
        "Failures and Warnings",
        "Removal vs. Removal",
        "Removal vs. Machine Data",
        "Removal vs. Documents",
        "Evidence Table",
        "Raw Results",
    ],
)

run_button = st.sidebar.button("Run Verification")


if run_button:
    if not sink_id_input.strip():
        st.sidebar.warning("Please enter a sink ID.")
        st.stop()

    try:
        sink_id = UUID(sink_id_input.strip())
    except ValueError:
        st.sidebar.error("Sink ID must be a valid UUID.")
        st.stop()

    try:
        # 팀 코드에 따라 생성 방식이 다를 수 있음
        client = CulaClient()

        fetch_result = fetch_sink_data(client, sink_id, fetch_documents=False)
        ctx = normalize(fetch_result)
        rule_results = run_rules(ctx)

        st.session_state["fetch_result"] = fetch_result
        st.session_state["ctx"] = ctx
        st.session_state["rule_results"] = rule_results

    except Exception as e:
        st.error(f"Verification failed: {e}")
        st.stop()


fetch_result = st.session_state.get("fetch_result")
ctx = st.session_state.get("ctx")
rule_results = st.session_state.get("rule_results")

if fetch_result is None or ctx is None or rule_results is None:
    st.info("Enter a Sink ID in the sidebar and click 'Run Verification'.")
    st.stop()

verification_report = score(ctx.sink_id, rule_results)

grouped = group_results_by_severity(rule_results)
summary_df = build_summary_df(rule_results)
evidence_df = build_evidence_df(rule_results)

fail_count = len(grouped.get("fail", []))
warn_count = len(grouped.get("warn", []))
info_count = len(grouped.get("info", []))

removal_results = [r for r in rule_results if categorize_result(r) == "Removal vs. Removal"]
machine_results = [r for r in rule_results if categorize_result(r) == "Removal vs. Machine Data"]
document_results = [r for r in rule_results if categorize_result(r) == "Removal vs. Documents"]


if menu == "Overview":
    st.subheader("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sink ID", str(ctx.sink_id))
    c2.metric("Carbon Capture Site", str(ctx.carbon_capture_site_id))
    c3.metric("Events", len(ctx.events))

    c4, c5 = st.columns(2)
    c4.metric("Gross Impact (kg)", ctx.gross_impact_kg if ctx.gross_impact_kg is not None else "N/A")
    c5.metric("Net Impact (kg)", ctx.net_impact_kg if ctx.net_impact_kg is not None else "N/A")

    st.write("**Sink Created:**", ctx.sink_created)
    st.write("**Pyrolysis Window:**", ctx.pyrolysis_window)
    st.write("**Sites Loaded:**", len(ctx.sites))
    st.write("**Machine Series Loaded:**", len(ctx.series))
    st.write("**Fetch Errors:**", len(ctx.fetch_errors))

    if ctx.fetch_errors:
        st.subheader("Fetch Errors")
        for err in ctx.fetch_errors:
            st.warning(err)

    st.subheader("Confidence (scoring)")
    band_label = {
        "high": "High (76–100)",
        "medium": "Medium (51–75)",
        "low": "Low (0–50)",
    }.get(verification_report.confidence_band, verification_report.confidence_band)
    s1, s2, s3 = st.columns(3)
    s1.metric("Confidence score", f"{verification_report.confidence_score}/100")
    s2.metric("Band", band_label)
    s3.metric("Non-info checks", verification_report.counts.get("fail", 0) + verification_report.counts.get("warn", 0))
    if verification_report.top_reasons:
        st.write("**Top reasons (by penalty weight):**")
        for line in verification_report.top_reasons:
            st.markdown(f"- {line}")

elif menu == "Verification Summary":
    st.subheader("Verification Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fails", fail_count)
    c2.metric("Warnings", warn_count)
    c3.metric("Info", info_count)

    band_label = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }.get(verification_report.confidence_band, verification_report.confidence_band)
    st.divider()
    s1, s2 = st.columns(2)
    s1.metric("Confidence score", f"{verification_report.confidence_score}/100")
    s2.metric("Confidence band", band_label)

    if not summary_df.empty:
        st.dataframe(summary_df, width="stretch")

elif menu == "Scoring":
    st.subheader("Verification scoring")
    st.markdown(
        "Heuristic score from `cula.verification.scoring`: start at 100, subtract "
        "weighted penalties per check (fails full weight, warns ×0.4), clamp to 0–100."
    )

    band_label = {
        "high": "High — 76–100",
        "medium": "Medium — 51–75",
        "low": "Low — 0–50",
    }.get(verification_report.confidence_band, verification_report.confidence_band)

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence score", f"{verification_report.confidence_score}/100")
    m2.metric("Band", band_label)
    m3.metric("Sink ID", str(verification_report.sink_id)[:8] + "…")

    st.write("**Severity counts**")
    counts_df = pd.DataFrame(
        [{"severity": k, "count": v} for k, v in sorted(verification_report.counts.items())]
    )
    st.dataframe(counts_df, width="stretch", hide_index=True)

    st.write("**Top reasons** (highest penalty first, excluding info-only rows)")
    if verification_report.top_reasons:
        for i, line in enumerate(verification_report.top_reasons, start=1):
            st.markdown(f"{i}. {line}")
    else:
        st.info("No fail/warn checks — nothing to rank.")

    with st.expander("Rule weights (reference)"):
        st.markdown(
            "Tier 1: MACHINE_COVERAGE, TEMP_PLAUSIBLE, PROOF_PRESENCE — "
            "Tier 2: TIMELINE_ORDER, MASS_BALANCE, SITE_CONTINUITY — "
            "Tier 3: DELIVERY_DISTANCE, FILE_REUSE. "
            "Unknown rule codes use default weight 10. See `scoring.py`."
        )

elif menu == "Failures and Warnings":
    st.subheader("Failures and Warnings")

    if fail_count == 0 and warn_count == 0:
        st.success("No failures or warnings found.")
    else:
        if fail_count > 0:
            st.markdown("### Failures")
            for r in grouped["fail"]:
                render_result(r)

        if warn_count > 0:
            st.markdown("### Warnings")
            for r in grouped["warn"]:
                render_result(r)

elif menu == "Removal vs. Removal":
    st.subheader("Removal vs. Removal")

    if not removal_results:
        st.info("No results in this category.")
    else:
        for r in removal_results:
            render_result(r)

elif menu == "Removal vs. Machine Data":
    st.subheader("Removal vs. Machine Data")

    if not machine_results:
        st.info("No results in this category.")
    else:
        for r in machine_results:
            render_result(r)

elif menu == "Removal vs. Documents":
    st.subheader("Removal vs. Documents")

    if not document_results:
        st.info("No results in this category.")
    else:
        for r in document_results:
            render_result(r)

elif menu == "Evidence Table":
    st.subheader("Evidence Table")

    if evidence_df.empty:
        st.info("No evidence data available.")
    else:
        st.dataframe(evidence_df, width="stretch")

elif menu == "Raw Results":
    st.subheader("Raw Rule Results")

    for r in rule_results:
        with st.expander(f"{r.code} | {r.severity}"):
            st.write("**Message:**", r.message)
            st.write("**Category:**", categorize_result(r))
            st.json(r.evidence)

st.markdown("---")
st.caption("Carbon Verification Dashboard – Streamlit frontend")
