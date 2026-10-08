"""
Streamlit Demo — SEC Financial Intelligence System
app/streamlit_app.py
Run: streamlit run app/streamlit_app.py
"""

import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import chromadb

sys.path.insert(0, '.')

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title = "SEC Financial Intelligence",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── CUSTOM STYLES ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .verdict-pass {
        color: #16a34a;
        font-weight: 700;
    }
    .verdict-revise {
        color: #d97706;
        font-weight: 700;
    }
    .verdict-reject {
        color: #dc2626;
        font-weight: 700;
    }
    .source-card {
        background: #f0f9ff;
        border-left: 3px solid #0ea5e9;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        from src.agents.orchestrator import Orchestrator
        return Orchestrator()
    except Exception as e:
        return None

@st.cache_resource
def load_metrics_db():
    try:
        conn = sqlite3.connect(
            "data/processed/metrics.db",
            check_same_thread=False
        )
        return conn
    except Exception:
        return None

@st.cache_data
def get_companies():
    conn = load_metrics_db()
    if not conn:
        return []
    rows = conn.execute("""
        SELECT ticker, name, sector, market_cap
        FROM company_overview
        ORDER BY market_cap DESC
    """).fetchall()
    return [
        {"ticker": r[0], "name": r[1],
         "sector": r[2], "market_cap": r[3]}
        for r in rows
    ]

@st.cache_data
def get_system_stats():
    try:
        client = chromadb.PersistentClient(
            path="data/vectorstore"
        )
        stats = {}
        for name in ["sec_risk_factors", "sec_mda",
                     "sec_financials", "sec_general"]:
            try:
                c = client.get_collection(name)
                stats[name] = c.count()
            except Exception:
                stats[name] = 0
        return stats
    except Exception:
        return {}

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/"
        "commons/thumb/f/f3/SEC_logo.svg/"
        "200px-SEC_logo.svg.png",
        width=80
    )
    st.title("SEC Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Smart Q&A",
         "🏢 Company Deep Dive",
         "⚖️  Compare Companies",
         "📈 System Metrics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    stats = get_system_stats()
    total = sum(stats.values())
    st.metric("Total Chunks", f"{total:,}")
    st.metric("Companies", "100")
    st.caption("S&P 500 | 2022-2025")

# ── PAGE 1: SMART Q&A ─────────────────────────────────────────
if page == "🔍 Smart Q&A":

    st.markdown(
        '<div class="main-header">📊 SEC Financial Q&A</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Ask any question about '
        'S&P 500 companies based on their SEC filings</div>',
        unsafe_allow_html=True
    )

    companies = get_companies()
    company_options = ["All Companies"] + [
        f"{c['ticker']} — {c['name']}"
        for c in companies
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Ask a question",
            placeholder=(
                "e.g. What are NVIDIA main supply "
                "chain risks in 2025?"
            )
        )
    with col2:
        company_select = st.selectbox(
            "Filter by company",
            company_options
        )

    col3, col4 = st.columns([1, 1])
    with col3:
        year_filter = st.selectbox(
            "Year",
            ["All years", "2025", "2024", "2023", "2022"]
        )
    with col4:
        top_k = st.slider("Sources to retrieve", 3, 10, 5)

    # Example questions
    st.markdown("**Example questions:**")
    examples = [
        "What are NVIDIA export control risks?",
        "How has Microsoft Azure revenue grown?",
        "What is JPMorgan credit risk strategy?",
        "Compare ExxonMobil and Chevron strategy",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            question = ex

    if question and st.button(
        "🔍 Search", type="primary",
        use_container_width=True
    ):
        # Extract ticker from selection
        ticker = None
        if company_select != "All Companies":
            ticker = company_select.split(" — ")[0]

        year = None
        if year_filter != "All years":
            year = year_filter

        orchestrator = load_pipeline()

        if not orchestrator:
            st.error(
                "Pipeline not loaded. "
                "Check that all agent files exist."
            )
        else:
            with st.spinner("Analysing SEC filings..."):
                result = orchestrator.run(
                    question = question,
                    ticker   = ticker,
                    year     = year,
                    top_k    = top_k,
                )

            # Display answer
            st.markdown("### Answer")
            st.markdown(result.get("answer", "No answer"))

            # Judge scores
            judgment = result.get("judgment", {})
            verdict  = judgment.get("verdict", "UNKNOWN")
            overall  = judgment.get("overall", 0)

            verdict_class = {
                "PASS":   "verdict-pass",
                "REVISE": "verdict-revise",
                "REJECT": "verdict-reject",
            }.get(verdict, "")

            st.markdown("### Quality Assessment")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Overall",
                f"{overall:.0%}"
            )
            c2.metric(
                "Grounded",
                f"{judgment.get('groundedness',0):.0%}"
            )
            c3.metric(
                "Relevant",
                f"{judgment.get('relevance',0):.0%}"
            )
            c4.metric(
                "Complete",
                f"{judgment.get('completeness',0):.0%}"
            )
            c5.metric(
                "Accurate",
                f"{judgment.get('accuracy',0):.0%}"
            )

            if judgment.get("issues"):
                st.warning(
                    "Issues: " +
                    ", ".join(judgment["issues"])
                )

            if judgment.get("hallucinations"):
                st.error(
                    "⚠️ Possible hallucinations: " +
                    ", ".join(judgment["hallucinations"])
                )

            # Sources
            sources = result.get("sources", [])
            if sources:
                st.markdown("### Sources")
                for s in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>{s["company"]}</b> — '
                        f'{s["form_type"]} {s["year"]} '
                        f'[relevance: {s["score"]:.3f}]'
                        f'<br><small>{s.get("preview","")[:150]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ── PAGE 2: COMPANY DEEP DIVE ─────────────────────────────────
elif page == "🏢 Company Deep Dive":

    st.markdown(
        '<div class="main-header">🏢 Company Deep Dive</div>',
        unsafe_allow_html=True
    )

    companies  = get_companies()
    conn       = load_metrics_db()

    ticker_select = st.selectbox(
        "Select Company",
        [f"{c['ticker']} — {c['name']}"
         for c in companies]
    )
    ticker = ticker_select.split(" — ")[0]

    if conn and ticker:
        # Overview
        overview = conn.execute("""
            SELECT name, sector, industry,
                   market_cap, pe_ratio,
                   revenue_ttm, profit_margin,
                   analyst_target, beta
            FROM company_overview WHERE ticker=?
        """, (ticker,)).fetchone()

        if overview:
            st.subheader(f"{overview[0]} ({ticker})")
            st.caption(
                f"{overview[1]} | {overview[2]}"
            )

            def fmt(v):
                if not v:
                    return "N/A"
                try:
                    v = float(v)
                    if v >= 1e12:
                        return f"${v/1e12:.2f}T"
                    if v >= 1e9:
                        return f"${v/1e9:.1f}B"
                    return f"${v:,.0f}"
                except Exception:
                    return str(v)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap",
                      fmt(overview[3]))
            c2.metric("P/E Ratio",
                      f"{overview[4]:.1f}"
                      if overview[4] else "N/A")
            c3.metric("Revenue TTM",
                      fmt(overview[5]))
            c4.metric("Profit Margin",
                      f"{float(overview[6]):.1%}"
                      if overview[6] else "N/A")

        # Revenue history chart
        rev_rows = conn.execute("""
            SELECT period_date, total_revenue
            FROM financial_metrics
            WHERE ticker=?
            AND total_revenue IS NOT NULL
            ORDER BY period_date
        """, (ticker,)).fetchall()

        if rev_rows:
            rev_df = pd.DataFrame(
                rev_rows,
                columns=["Period", "Revenue"]
            )
            rev_df["Revenue"] = (
                rev_df["Revenue"] / 1e9
            )
            fig = px.bar(
                rev_df,
                x="Period",
                y="Revenue",
                title=f"{ticker} Annual Revenue ($B)",
                color_discrete_sequence=["#3b82f6"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Price history
        price_rows = conn.execute("""
            SELECT date, close_price
            FROM price_history
            WHERE ticker=?
            ORDER BY date
        """, (ticker,)).fetchall()

        if price_rows:
            price_df = pd.DataFrame(
                price_rows,
                columns=["Date", "Price"]
            )
            fig2 = px.line(
                price_df,
                x="Date",
                y="Price",
                title=f"{ticker} 5-Year Price History",
                color_discrete_sequence=["#10b981"]
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 3: COMPARE COMPANIES ─────────────────────────────────
elif page == "⚖️  Compare Companies":

    st.markdown(
        '<div class="main-header">⚖️ Compare Companies</div>',
        unsafe_allow_html=True
    )

    companies = get_companies()
    options   = [
        f"{c['ticker']} — {c['name']}"
        for c in companies
    ]

    col1, col2 = st.columns(2)
    with col1:
        comp_a = st.selectbox("Company A", options, index=0)
    with col2:
        comp_b = st.selectbox("Company B", options, index=1)

    question = st.text_input(
        "Comparison question",
        value="Compare their AI strategy and revenue growth"
    )

    if st.button("⚖️ Compare", type="primary"):
        ticker_a = comp_a.split(" — ")[0]
        ticker_b = comp_b.split(" — ")[0]

        orchestrator = load_pipeline()
        if orchestrator:
            with st.spinner("Comparing companies..."):
                result = orchestrator.run(
                    question = (
                        f"Compare {comp_a.split('—')[1].strip()} "
                        f"and {comp_b.split('—')[1].strip()}: "
                        f"{question}"
                    ),
                    top_k = 5
                )

            st.markdown("### Comparison")
            st.markdown(result.get("answer", ""))

            # Side by side metrics
            conn = load_metrics_db()
            if conn:
                st.markdown("### Key Metrics")
                rows = []
                for ticker in [ticker_a, ticker_b]:
                    r = conn.execute("""
                        SELECT name, market_cap,
                               pe_ratio, revenue_ttm,
                               profit_margin
                        FROM company_overview
                        WHERE ticker=?
                    """, (ticker,)).fetchone()
                    if r:
                        rows.append({
                            "Company": r[0],
                            "Market Cap": (
                                f"${r[1]/1e12:.1f}T"
                                if r[1] and r[1] >= 1e12
                                else f"${r[1]/1e9:.0f}B"
                                if r[1] else "N/A"
                            ),
                            "P/E": (
                                f"{r[2]:.1f}"
                                if r[2] else "N/A"
                            ),
                            "Revenue TTM": (
                                f"${r[3]/1e9:.1f}B"
                                if r[3] else "N/A"
                            ),
                            "Margin": (
                                f"{float(r[4]):.1%}"
                                if r[4] else "N/A"
                            ),
                        })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows)
                        .set_index("Company"),
                        use_container_width=True
                    )

# ── PAGE 4: SYSTEM METRICS ────────────────────────────────────
elif page == "📈 System Metrics":

    st.markdown(
        '<div class="main-header">📈 System Metrics</div>',
        unsafe_allow_html=True
    )

    stats = get_system_stats()
    total = sum(stats.values())

    # Collection stats
    st.subheader("Vector Store Collections")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Risk Factors",
        f"{stats.get('sec_risk_factors',0):,}"
    )
    col2.metric(
        "MD&A",
        f"{stats.get('sec_mda',0):,}"
    )
    col3.metric(
        "Financials",
        f"{stats.get('sec_financials',0):,}"
    )
    col4.metric(
        "General",
        f"{stats.get('sec_general',0):,}"
    )

    # Collection breakdown chart
    if stats:
        fig = px.pie(
            values=list(stats.values()),
            names=[k.replace("sec_", "").replace("_", " ").title()
                   for k in stats.keys()],
            title="Chunks by Collection Type",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    # Company coverage
    conn = load_metrics_db()
    if conn:
        st.subheader("Company Coverage")
        sector_rows = conn.execute("""
            SELECT sector, COUNT(*) as count,
                   AVG(market_cap) as avg_mcap
            FROM company_overview
            WHERE sector IS NOT NULL
            GROUP BY sector
            ORDER BY count DESC
        """).fetchall()

        if sector_rows:
            sector_df = pd.DataFrame(
                sector_rows,
                columns=["Sector", "Companies", "Avg MCap"]
            )
            sector_df["Avg MCap"] = (
                sector_df["Avg MCap"] / 1e9
            ).round(1)
            fig2 = px.bar(
                sector_df,
                x="Sector",
                y="Companies",
                title="Companies by Sector",
                color="Companies",
                color_continuous_scale="blues"
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Architecture")
    st.markdown("""
    | Component | Technology | Details |
    |-----------|-----------|---------|
    | Embeddings | BAAI/bge-base-en-v1.5 | 768 dimensions |
    | Vector Store | ChromaDB + FAISS | 4 collections |
    | Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder |
    | LLM | openai/gpt-oss-20b | Via Groq API |
    | Judge LLM | qwen/qwen3.8-27b | Different model |
    | Data | SEC EDGAR | 100 S&P 500 companies |
    | Filings | 10-K + 10-Q | 2022-2025 |
    """)