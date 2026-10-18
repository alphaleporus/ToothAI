"""Photo screening page, laid out like Analyze: the case in the sidebar, the answer in the main area."""

from __future__ import annotations

import base64
from contextlib import nullcontext

import pandas as pd
import streamlit as st

from photo_screening.ui import client
from tooth_ai.ui import components as ui

SLOTS = (
    ("front", "Front (required)", "Smile with teeth together, lips pulled back."),
    ("upper", "Upper teeth", "Head tilted back; photograph the biting surfaces, ideally in a mirror."),
    ("lower", "Lower teeth", "Chin down; photograph the biting surfaces, ideally in a mirror."),
    ("left", "Left side", "Pull the left cheek back; side view of your left back teeth."),
    ("right", "Right side", "Pull the right cheek back; side view of your right back teeth."),
)
VIEW_NAMES = {"front": "Front", "upper": "Upper teeth", "lower": "Lower teeth", "left": "Left side",
              "right": "Right side"}
TRIAGE = {
    "ROUTINE": ("ok", "Routine: keep up regular check-ups"),
    "SOON": ("caution", "Book a dental check-up soon"),
    "PROMPT": ("alert", "See a dentist promptly"),
}
LABELS = {"caries": "Possible decay", "cavity": "Possible cavity", "crack": "Possible crack", "wear": "Tooth wear",
          "filling": "Filling", "crown": "Crown"}
TIPS_HTML = """
<ul style="margin:.2rem 0 0 1rem;padding:0;font-size:.88rem;color:#33475B">
<li>Good light: daylight or a bright bathroom, with the phone flash on.</li>
<li>Pull your lips and cheeks back so the teeth and gums are visible. Keep fingertips at the corners
and out of the middle of the photo: the models can mistake a fingernail for a crown.</li>
<li>Hold the camera close and tap the screen on the teeth to focus.</li>
<li>Upper and lower teeth: photograph them in a mirror, or tilt your head back (upper) or down (lower).</li>
<li>Main camera, full resolution: no screenshots, filters or zoom.</li>
</ul>"""


# ------------------------------------------------------------------ helpers

def _status():
    try:
        return True, client.models()
    except client.PhotoApiError as e:
        return False, str(e)


def _banner(level: str, title: str, lines: list[str]) -> None:
    items = "".join(f"<li>{ui.esc(x)}</li>" for x in lines)
    st.html(f'<div class="ta-banner {level}" role="status"><strong>{ui.esc(title)}</strong>'
            f'{"<ul>" + items + "</ul>" if items else ""}</div>')


def _findings_html(findings: list[dict]) -> str:
    """Compact, wrapping findings table; the number badges match the markers on the photo."""
    from photo_screening.palette import COLORS

    cell = 'style="padding:.35rem .25rem;border-bottom:1px solid #E9EDF1;vertical-align:top"'
    rows = []
    for f in findings:
        where = ui.esc(f["region"]) + (f'<br><span class="ta-muted">{ui.esc(f["tooth_hint"])}</span>'
                                       if f.get("tooth_hint") else "")
        rows.append(
            f'<tr><td {cell}><span class="ta-mono" style="display:inline-block;min-width:1.5rem;text-align:center;'
            f'border-radius:999px;color:#fff;background:{COLORS[f["type"]]};font-size:.76rem">{f["id"]}</span></td>'
            f'<td {cell}>{ui.esc(LABELS[f["type"]])}</td><td {cell}>{where}</td>'
            f'<td {cell}>{ui.esc(f["band"])}</td></tr>')
    return ('<table style="width:100%;border-collapse:collapse;font-size:.86rem;line-height:1.3">'
            '<thead><tr style="text-align:left;color:#5B6B78;font-size:.74rem;border-bottom:1px solid #C9D4E0">'
            '<th style="padding:.3rem .25rem">#</th><th>Finding</th><th>Where</th><th>Confidence</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _empty_state(photos: dict) -> None:
    mark = ui.data_uri("assets/brand/tooth-ai-mark.svg")
    st.html(
        f'<div class="ta-hero"><img src="{mark}" alt="">'
        '<h2>Screen your teeth with phone photos</h2>'
        f'<p>Add up to five photos {"above" if ui.is_phone() else "in the panel on the left"}, or try the '
        'sample photos. Photos are analysed in memory and not stored. This is a screening aid, not a '
        'diagnosis.</p>'
        '<div class="ta-steps">'
        '<div><em>01</em><b>Check</b><span>Each photo is checked for blur, light and resolution.</span></div>'
        '<div><em>02</em><b>Detect</b><span>Possible decay, tooth wear, fillings and crowns.</span></div>'
        '<div><em>03</em><b>Triage</b><span>When to see a dentist, and whether an X-ray (OPG) is worth it.</span>'
        '</div></div></div>')
    if photos:
        cols = st.columns(len(photos), gap="small")
        for col, (view, (_, data, _)) in zip(cols, photos.items(), strict=True):
            with col:
                st.image(data, caption=VIEW_NAMES[view], use_container_width=True)
    with st.expander("How to take good photos", icon=":material/photo_camera:"):
        st.html(TIPS_HTML)


def _view_tab(im: dict) -> None:
    if not im["analysed"]:
        fails = [c for c in im["quality"]["checks"] if c["hard"] and not c["passed"]]
        _banner("caution", "This photo couldn't be analysed. Please retake it.",
                [f"{c['name'].capitalize()}: {c['tip']}" for c in fails])
        return
    a, b = st.columns([7, 5], gap="medium")
    with a:
        st.image(base64.b64decode(im["annotated_png_b64"]), use_container_width=True)
    with b:
        strong = [f for f in im["findings"] if f["band"] != "Low"]
        low = [f for f in im["findings"] if f["band"] == "Low"]
        if strong:
            st.html(_findings_html(strong))
        elif not low:
            st.caption("No findings marked in this photo.")
        else:
            st.caption("Only low-confidence marks in this photo.")
        if low:
            with st.expander(f"{len(low)} low-confidence mark{'s' if len(low) > 1 else ''}"):
                st.caption("Faint outlines on the photo. Often stains or lighting; worth a look, not a worry.")
                st.html(_findings_html(low))
        for w in [c for c in im["quality"]["checks"] if not c["hard"] and not c["passed"]]:
            st.caption(f"Photo note: {w['tip']}")


def _results(rep: dict, api_ok: bool) -> None:
    findings = [f for i in rep["images"] for f in i["findings"]]
    strong = [f for f in findings if f["band"] != "Low"]
    analysed = [i for i in rep["images"] if i["analysed"]]
    level, title = TRIAGE[rep["triage"]["level"]]

    _banner(level, title, rep["triage"]["reasons"])
    idx = rep["screening_index"]
    ui.kpi_row([
        (f"{idx if idx is not None else '-'}<small>/100</small>", f"Screening index · {rep['index_band'].lower()}"),
        (f"{len(analysed)}<small>/{len(rep['images'])}</small>", "Photos analysed"),
        (f"{len(strong)}", "Findings to review"),
        (f"{len(findings) - len(strong)}", "Low-confidence marks"),
    ])
    a, b, _ = st.columns([1.1, 1.6, 1.3], gap="small")
    with a:
        key = "ps_pdf_" + rep["report_id"]
        if key not in st.session_state and api_ok:
            try:
                st.session_state[key] = client.report_pdf(rep)
            except client.PhotoApiError:
                st.session_state[key] = None
        if st.session_state.get(key):
            st.download_button("PDF report", st.session_state[key], icon=":material/download:", type="primary",
                               file_name=f"tooth-ai-screening-{rep['report_id']}.pdf", mime="application/pdf",
                               use_container_width=True)
    with b:
        if ui.PAGES.get("analyze"):   # a page link: on phones "/" is this page, not Analyse
            st.page_link(ui.PAGES["analyze"], label="Analyse an X-ray (OPG)", icon=":material/radiology:")

    tabs = st.tabs([f"{VIEW_NAMES[i['view']]}" + (f" ({sum(f['band'] != 'Low' for f in i['findings'])})"
                                                  if i["analysed"] else " (retake)") for i in rep["images"]])
    for tab, im in zip(tabs, rep["images"], strict=True):
        with tab:
            _view_tab(im)

    st.caption(rep["summary_text"] + (" (Rephrased by AI, advisory.)" if rep["summary_source"] == "vlm" else ""))
    if rep["observations"]:
        with st.expander(f"AI observations ({len(rep['observations'])}), advisory", icon=":material/visibility:"):
            st.caption(f"From {rep['vlm'].get('provider_label') or 'an AI model'}, about gums and hygiene only; "
                       "not checked by the detection models.")
            for o in rep["observations"]:
                st.markdown(f"- **{o['type'].replace('_', ' ').capitalize()}** ({o['band'].lower()}): {o['text']}")
    with st.expander("What photos can't show", icon=":material/help:"):
        for n in rep["coverage"]["notes"]:
            st.markdown(f"- {n}")


def _about(models_payload) -> None:
    with st.expander("About these models", icon=":material/info:"):
        if not isinstance(models_payload, dict):
            st.caption("Model details are shown when the service is running.")
            return
        rows = []
        for m in models_payload["models"]:
            metrics = m.get("metrics") or {}
            measured = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in metrics.items() if k != "source")
            rows.append({"Model": m["model_id"], "Status": m["status"], "Version": m.get("version") or "-",
                         "Measured": (f"{measured} ({metrics['source']})" if metrics else "-"),
                         "Licence": m["licence"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for m in models_payload["models"]:
            st.caption(f"**{m['model_id']}**: {m['attribution']}")
        attribution = client.sample_attribution()
        if attribution:
            st.caption("Sample photos: " + attribution)


# ------------------------------------------------------------------ page

def render() -> None:
    api_ok, payload = _status()
    vlm = payload.get("vlm", {}) if api_ok else {}
    samples = st.session_state.get("ps_samples") or {}

    st.html("<style>"
            "section[data-testid='stSidebar'] [data-testid='stFileUploaderDropzone']{padding:.3rem .45rem;}"
            "section[data-testid='stSidebar'] [data-testid='stFileUploaderDropzoneInstructions']{display:none;}"
            "</style>")
    ui.page_header("Photo screening", "Smartphone photos · a screening aid, not a diagnosis", badge="Beta")

    # ---------------- the photos: sidebar on a desktop, a card above the results on a phone
    with ui.controls("Photos and options" if st.session_state.get("ps_report") else None):
        st.html('<div class="ta-eyebrow">Photos</div>')
        consent = st.checkbox("Process my photos for this session only; don't store them", key="ps_consent")
        st.caption("Mouth only: keep your full face, name tags and anything else that identifies you out of "
                   "the photos. Location and camera details are removed before analysis.")
        photos: dict[str, tuple[str, bytes, str]] = {}
        # On a phone the optional views fold away so "Analyse photos" stays near the top of the card.
        more = None
        for view, label, helptext in SLOTS:
            if view != "front" and more is None and ui.is_phone():
                more = st.expander("More views: upper, lower and sides (recommended)", icon=":material/add_a_photo:")
            with more if (more is not None and view != "front") else nullcontext():
                f = st.file_uploader(label, type=["jpg", "jpeg", "png", "heic", "heif"], key=f"ps_up_{view}",
                                     disabled=not consent, help=helptext)
            if f is not None:
                photos[view] = (f.name, f.getvalue(), f.type or "image/jpeg")
            elif view in samples:
                photos[view] = samples[view]
        has_samples = bool(client.sample_photos())
        if st.button("Use sample photos", icon=":material/photo_library:", disabled=not (consent and has_samples),
                     use_container_width=True, help="CC BY photos from the AKU dataset's test split"):
            st.session_state["ps_samples"] = client.sample_photos()
            st.session_state.pop("ps_report", None)
            st.rerun()
        with st.expander("Options", icon=":material/tune:"):
            selfie = st.checkbox("Front and side photos taken with the selfie camera", key="ps_selfie")
            use_vlm = False
            if vlm.get("enabled"):
                use_vlm = st.toggle("AI notes on gums and hygiene (slower)", key="ps_vlm",
                                    help=(f"Photos are sent to {vlm.get('provider_label')}." if vlm.get("third_party")
                                          else f"Runs locally ({vlm.get('provider_label')})."))
        ready = consent and api_ok and "front" in photos
        go = st.button("Analyse photos", type="primary", icon=":material/search:", disabled=not ready,
                       use_container_width=True)
        if consent and "front" not in photos:
            st.caption("Add at least the front photo.")
        if (photos or st.session_state.get("ps_report")) and st.button(
                "Start over", icon=":material/restart_alt:", type="tertiary", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("ps_") and k != "ps_consent"]:
                del st.session_state[k]
            st.rerun()
        names = {"alphadent": "AlphaDent YOLOv8x", "caries_aku": "AKU caries YOLOv8s",
                 "teeth_dentalai": "DentalAI teeth YOLOv8s-seg"}
        loaded = [names.get(m["model_id"], m["model_id"]) for m in payload["models"] if m["status"] == "loaded"] \
            if api_ok else []
        st.caption(" + ".join(loaded) if loaded else "Photo service not running")

    if not api_ok:
        _banner("alert", "The photo screening service isn't running.",
                [payload, "Start it from the repo root: uvicorn photo_screening.app.main:app --port 8001"])

    if go:
        mirrored = ["upper", "lower"] + (["front", "left", "right"] if selfie else [])
        note = " (AI notes take up to a minute)" if use_vlm else ""
        with st.spinner("Checking photo quality and analysing" + note):
            try:
                st.session_state["ps_report"] = client.analyze(photos, use_vlm=use_vlm, mirrored=mirrored)
                if ui.is_phone():
                    st.rerun()   # fold the photo card away so the result sits at the top
            except client.PhotoApiBusy as e:
                st.warning(str(e), icon=":material/hourglass_top:")
            except client.PhotoApiError as e:
                st.error(str(e), icon=":material/error:")

    rep = st.session_state.get("ps_report")
    if rep:
        _results(rep, api_ok)
    else:
        _empty_state(photos)
    _about(payload if api_ok else None)
    ui.footer()
