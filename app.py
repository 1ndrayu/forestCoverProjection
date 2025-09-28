"""Minimal Streamlit front-end for visualising processed forest cover data."""

import streamlit as st

from backend import (
    build_timeseries,
    list_entities,
    load_forecast_data,
    load_processed_data,
    load_summary_data,
    select_default_entity,
)
from frontend import (
    configure_page,
    render_preview,
    render_selection,
    render_summary_cards,
    render_timeseries_chart,
)


def main() -> None:
    configure_page()

    st.title("Forest Cover Trajectories")
    st.caption("Historic trends and linear projections derived from the annual change dataset.")

    try:
        processed_df = load_processed_data()
        forecast_df = load_forecast_data()
        summary_df = load_summary_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    entities = list_entities(processed_df)
    default_entity = select_default_entity(entities)
    selected_entity = render_selection(entities, default_entity)

    try:
        timeseries = build_timeseries(selected_entity, processed_df, forecast_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    render_timeseries_chart(timeseries, selected_entity)

    summary_row = summary_df[summary_df["entity"] == selected_entity].head(1)
    summary_row = summary_row.iloc[0] if not summary_row.empty else None
    render_summary_cards(selected_entity, summary_row)
    render_preview(timeseries)


if __name__ == "__main__":
    main()
