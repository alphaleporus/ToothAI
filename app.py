#!/usr/bin/env python3
"""ToothAI: Streamlit entry point.

    streamlit run app.py                 # full mode (torch + detectron2 + weights in output/)
    TOOTH_AI_DEMO=1 streamlit run app.py # demo mode: bundled sample cases, no Detectron2

Pages live in tooth_ai/views/, the pipeline in tooth_ai/pipeline.py. The previous
single-file app is kept as workspace/app_legacy.py (and at git tag paper-eval-v1).
"""

import streamlit as st

from photo_screening.ui import page as photo_screening
from tooth_ai.ui import components as ui
from tooth_ai.ui import theme
from tooth_ai.views import about, analyze, method, performance

st.set_page_config(
    page_title="ToothAI",
    page_icon=str(theme.BRAND / "favicon-32.png"),
    layout="wide",
    initial_sidebar_state="auto",  # expanded on desktop, collapsed on phones
    menu_items={"About": "ToothAI: research prototype. Not for diagnostic use."},
)
st.logo(str(theme.BRAND / "tooth-ai-lockup-horizontal.svg"), size="large",
        icon_image=str(theme.BRAND / "tooth-ai-mark.svg"))
ui.inject_css()

# Phones open on Photo screening (the tool people use from a phone) and get a bottom tab bar instead of the
# collapsed top menu.
phone = ui.is_phone()
pages = {
    "analyze": st.Page(analyze.render, title="Analyse", icon=":material/radiology:", url_path="analyze",
                       default=not phone),
    "photo-screening": st.Page(photo_screening.render, title="Photo screening", icon=":material/photo_camera:",
                               url_path="photo-screening", default=phone),
    "how-it-works": st.Page(method.render, title="How it works", icon=":material/account_tree:",
                            url_path="how-it-works"),
    "performance": st.Page(performance.render, title="Performance", icon=":material/monitoring:",
                           url_path="performance"),
    "about": st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
}
ui.PAGES.update(pages)
order = ["photo-screening", "analyze", "how-it-works", "performance", "about"] if phone else list(pages)
page = st.navigation([pages[k] for k in order], position="top")
if phone:
    ui.phone_tabbar([(pages["photo-screening"], "Photos", ":material/photo_camera:"),
                     (pages["analyze"], "X-ray", ":material/radiology:"),
                     (pages["how-it-works"], "Method", ":material/account_tree:"),
                     (pages["performance"], "Accuracy", ":material/monitoring:"),
                     (pages["about"], "About", ":material/info:")])
page.run()
