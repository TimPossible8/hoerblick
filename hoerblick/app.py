"""Run the shared HörBlick web interface in Streamlit."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from hoerblick.streamlit_bridge import handle_request

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT.parent / "web-prototype"

st.set_page_config(page_title="HörBlick – Audioanalyse", page_icon="👂", layout="wide")
st.markdown("""
<style>
  [data-testid="stHeader"] { display: none; }
  .stMainBlockContainer { padding: 0; max-width: none; }
  [data-testid="stMain"] { overflow: hidden; }
  [data-testid="stVerticalBlock"] { gap: 0; }
  iframe { display: block; }
</style>
""", unsafe_allow_html=True)

frontend = components.declare_component("hoerblick_web", path=str(WEB_ROOT))
reply = st.session_state.get("web_reply")
request = frontend(response=reply, key="hoerblick_web", default=None, height=900)
if request and isinstance(request, dict) and request.get("id") != (reply or {}).get("id"):
    st.session_state.web_reply = handle_request(request, PROJECT_ROOT)
    st.rerun()
