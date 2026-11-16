"""Stand-alone entry point:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Spatial analysis · BCI course", page_icon="🧠", layout="wide")

from spatial_module.ui import render  # noqa: E402

render()
