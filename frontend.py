import plotly.express as px
import streamlit as st


def configure_page() -> None:
    st.set_page_config(page_title="Forest Cover Trends", page_icon="🔎", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            scroll-behavior: smooth;
        }
        html, body, [class*="css"]  {
            background-color: #ffffff !important;
            color: #000000 !important;
            font-family: "SF Pro Display", "Helvetica Neue", sans-serif;
        }
        .stApp {
            width: 100%;
            max-width: none;
            padding: 2rem 6vw 4rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8f8f8 55%, #ffffff 100%);
        }
        section.main > div {
            padding-top: 0rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.75rem;
            width: 100%;
        }
        .grid--wide {
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
        }
        .hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }
        .hero-title {
            font-size: clamp(2.35rem, 4vw, 3rem);
            font-weight: 600;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            margin-top: 0.35rem;
            color: #4a4a4a;
            font-size: 1.02rem;
        }
        .floating-card {
            background-color: #ffffff;
            border-radius: 20px;
            padding: 1rem 1.2rem;
            box-shadow: 0 32px 60px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            display: inline-flex;
            flex-direction: column;
            gap: 0.35rem;
            animation: floatIn 0.8s ease forwards;
            opacity: 0;
        }
        @keyframes floatIn {
            from { transform: translateY(16px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .select-container {
            width: min(360px, 100%);
        }
        .select-container .stSelectbox div[data-baseweb="select"] {
            border-radius: 18px;
            border: 1px solid #d1d1d1;
            background-color: #ffffff;
            box-shadow: 0 16px 28px rgba(0,0,0,0.08);
        }
        .card {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 1.45rem 1.65rem;
            box-shadow: 0 24px 48px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 30px 60px rgba(0,0,0,0.1);
        }
        .fade-in {
            animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            border: none !important;
            padding: 0.55rem 1.4rem !important;
            font-weight: 600;
            box-shadow: 0 16px 32px rgba(0,0,0,0.12);
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            box-shadow: 0 20px 40px rgba(0,0,0,0.14);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            width: 100%;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 24px 48px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .element-container:has(.stDataFrame) {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 0.5rem;
            box-shadow: 0 18px 36px rgba(0,0,0,0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_selection(entities, default_entity: str) -> str:
    st.markdown(
        """
        <div class="floating-card select-container">
            <span style="text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.12em; color: #7a7a7a;">Select Entity</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = st.selectbox(
        "Entity",
        entities,
        index=entities.index(default_entity),
        label_visibility="collapsed",
        key="primary-entity-select",
    )
    return selected


def render_timeseries_chart(timeseries, entity: str) -> None:
    col_chart, col_summary = st.columns([2.1, 1], gap="large")
    with col_chart:
        st.markdown(
            f"<h2 style='font-weight:600;margin-bottom:0.25rem;'>{entity}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#4a4a4a;margin-bottom:1.5rem;'>Historical forest cover index versus linear projection.</p>",
            unsafe_allow_html=True,
        )

        fig = px.line(
            timeseries,
            x="year",
            y="forest_cover",
            color="phase",
            markers=True,
            template="plotly_white",
            labels={"year": "Year", "forest_cover": "Forest Cover", "phase": ""},
            color_discrete_map={"Historical": "#000000", "Forecast": "#7f7f7f"},
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=9, line=dict(width=1.2, color="#ffffff")))
        fig.update_layout(
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(gridcolor="#f0f0f0"),
            yaxis=dict(gridcolor="#f0f0f0"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_summary:
        st.markdown("<div class='metric-grid fade-in'>", unsafe_allow_html=True)
        historical_latest = timeseries[timeseries["phase"] == "Historical"].tail(1)
        if not historical_latest.empty:
            value = historical_latest.iloc[0]["forest_cover"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="text-transform: uppercase; letter-spacing: 0.12em; color:#808080; font-size:0.8rem;">Latest Observed</div>
                    <div style="font-size:2.1rem;font-weight:600;">{value:,.0f}</div>
                    <div style="color:#9a9a9a;margin-top:0.35rem;font-size:0.95rem;">Most recent reported forest cover value (raw index).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        forecast_latest = timeseries[timeseries["phase"] == "Forecast"].tail(1)
        if not forecast_latest.empty:
            value = forecast_latest.iloc[0]["forest_cover"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="text-transform: uppercase; letter-spacing: 0.12em; color:#808080; font-size:0.8rem;">Projected</div>
                    <div style="font-size:2.1rem;font-weight:600;">{value:,.0f}</div>
                    <div style="color:#9a9a9a;margin-top:0.35rem;font-size:0.95rem;">Forecasted forest cover at projection horizon (raw index).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def render_summary_cards(entity: str, summary_row) -> None:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    if summary_row is not None:
        projected_value = summary_row["predicted_forest_cover"]
        card_html = f"""
            <div class="card" style="margin: 2rem 0;">
                <div style="text-transform: uppercase; letter-spacing: 0.08em; color: #7f7f7f; font-size: 0.8rem; margin-bottom: 0.6rem;">
                    Projected Forest Cover {int(summary_row['year'])}
                </div>
                <div style="font-size: 2.4rem; font-weight: 600;">
                    {projected_value:,.0f}
                </div>
                <div style="color:#8a8a8a;margin-top:0.5rem;font-size:0.98rem;">
                    Raw forecast value derived from the engineered index and linear trend extrapolation.
                </div>
            </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("Projection unavailable for the selected region.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_preview(timeseries) -> None:
    st.markdown(
        "<h3 style='font-weight:500;margin-top:2.25rem;margin-bottom:1rem;'>Latest Data Points</h3>",
        unsafe_allow_html=True,
    )
    display = timeseries.tail(12).copy()
    display["forest_cover"] = display["forest_cover"].round(0)
    display = display[["year", "phase", "forest_cover"]]
    st.dataframe(display, use_container_width=True, height=280)
