"""Reusable Streamlit components for ToothAI. HTML goes through ``st.html``."""

import base64
import html
import io
import re
from functools import lru_cache

import streamlit as st
from PIL import Image

import tooth_ai
from tooth_ai.notation import FDI_TO_UNIVERSAL, label as tooth_label, tooth_name, tooth_type
from tooth_ai.ui import theme
from tooth_ai.ui.theme import FDI_ROW_LOWER, FDI_ROW_UPPER, PARTNER_LOGOS, SEVERITY



def esc(text):
    return html.escape(str(text), quote=True)


# ------------------------------------------------------------------ phone layout

_PHONE_UA = re.compile(r"iPhone|iPod|Android.+Mobile|Windows Phone|Mobile Safari", re.I)
PAGES = {}   # url key -> st.Page, filled by app.py (for links between pages and the phone tab bar)


def is_phone() -> bool:
    """Phone layout: controls on the page instead of the sidebar, and a tab bar. Decided from the browser's
    user agent (tablets get the desktop layout); ``?layout=phone`` or ``?layout=desktop`` overrides it for the
    session."""
    forced = st.query_params.get("layout")
    if forced in ("phone", "desktop"):
        st.session_state["ta_layout"] = forced
    if st.session_state.get("ta_layout") in ("phone", "desktop"):
        return st.session_state["ta_layout"] == "phone"
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:  # no request context (tests, bare mode)
        ua = ""
    return bool(_PHONE_UA.search(ua))


def controls(collapsed_label=None):
    """Where a page's inputs go: the sidebar on a desktop; on a phone, a card at the top of the page, folded
    into an expander labelled ``collapsed_label`` once there is a result to show below it."""
    if not is_phone():
        return st.sidebar
    if collapsed_label:
        return st.expander(collapsed_label, icon=":material/tune:")
    return st.container(border=True, key="ta_controls")


def phone_tabbar(items):
    """Bottom tab bar on phones. items: [(page, label, icon)]."""
    with st.container(horizontal=True, key="ta_tabbar", gap=None):
        for page, label, icon in items:
            st.page_link(page, label=label, icon=icon)


@lru_cache(maxsize=1)
def _css():
    return (theme.ROOT / "tooth_ai" / "ui" / "styles.css").read_text(encoding="utf-8")


def inject_css():
    st.html(f"<style>{_css()}</style>")


@lru_cache(maxsize=32)
def data_uri(path, max_height=None):
    """Base64 data URI for an image file, optionally downscaled (cached per process)."""
    path = theme.ROOT / path if not str(path).startswith("/") else path
    if str(path).endswith(".svg"):
        return "data:image/svg+xml;base64," + base64.b64encode(open(path, "rb").read()).decode()
    img = Image.open(path)
    if max_height and img.height > max_height:
        img = img.resize((round(img.width * max_height / img.height), max_height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=1)
def _svg_font_css():
    """@font-face rules with the brand fonts inlined: an SVG shown as <img> can't load the page's fonts."""
    faces = [("Plus Jakarta Sans", "PlusJakartaSans-Regular-latin.ttf", "400"),
             ("Plus Jakarta Sans", "PlusJakartaSans-Bold-latin.ttf", "600 800"),
             ("JetBrains Mono", "JetBrainsMono-Medium-latin.ttf", "400 600")]
    css = []
    for family, file, weight in faces:
        path = theme.FONTS / "web" / file
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            css.append(f"@font-face{{font-family:'{family}';font-weight:{weight};"
                       f"src:url(data:font/ttf;base64,{data}) format('truetype');}}")
    return f"<style>{''.join(css)}</style>" if css else ""


def svg_image(svg, alt, padding=None):
    """Show inline SVG as an <img>: st.html's sanitizer strips <svg> elements. The brand fonts are
    embedded so chart type matches the page."""
    head_end = svg.find(">") + 1
    svg = svg[:head_end] + _svg_font_css() + svg[head_end:]
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()
    style = f' style="padding:{padding}"' if padding else ""
    st.html(f'<div class="ta-chart"{style}><img src="{uri}" alt="{esc(alt)}" '
            f'style="width:100%;height:auto;display:block"></div>')


def page_header(title, subtitle=None, badge=None, demo=False):
    badge_html = ""
    if demo:
        badge_html = '<span class="ta-badge demo">Demo data</span> '
    if badge:
        badge_html += f'<span class="ta-badge">{esc(badge)}</span>'
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.html(f'<div class="ta-pagehead"><div><h1>{esc(title)}</h1>{sub}</div>'
            f'<div>{badge_html}</div></div>')


def kpi_row(items):
    """items: list of (value_html, label)."""
    cells = "".join(f'<div class="ta-kpi"><div class="v">{v}</div><div class="l">{esc(l)}</div></div>'
                    for v, l in items)
    st.html(f'<div class="ta-kpis">{cells}</div>')


def reliability_banner(rel):
    reasons = "".join(f"<li>{esc(r)}</li>" for r in rel.reasons)
    reasons = f"<ul>{reasons}</ul>" if reasons else ""
    st.html(f'<div class="ta-banner {esc(rel.level)}" role="status"><strong>{esc(rel.title)}</strong>'
            f'<p>{esc(rel.detail)}</p>{reasons}</div>')


def legend(findings_present, color_by="Finding"):
    chips = []
    if color_by == "Quadrant":
        names = {1: "Q1 upper right", 2: "Q2 upper left", 3: "Q3 lower left", 4: "Q4 lower right"}
        chips += [f'<span><i style="background:{c}"></i>{names[q]}</span>'
                  for q, c in theme.QUADRANT_COLORS.items()]
    else:
        chips.append(f'<span><i style="background:{theme.TOOTH_FILL}"></i>Tooth</span>')
    for kind in SEVERITY:
        if kind in findings_present:
            chips.append(f'<span><i style="background:{theme.FINDING_COLORS[kind]}"></i>{esc(kind)}</span>')
    chips.append('<span><i class="dash"></i>Low detection score</span>')
    st.html(f'<div class="ta-legend">{"".join(chips)}</div>')


def _tooth_glyph(x, y, w, kind, upper):
    """Simplified crown + roots, crown towards the occlusal plane. Returns (crown, roots) paths."""
    cw = {"molar": w * 0.86, "premolar": w * 0.66, "canine": w * 0.6, "incisor": w * 0.56}[kind]
    ch = 22
    cx = x + w / 2
    s = -1 if upper else 1          # roots grow away from the occlusal plane
    top = y if not upper else y - ch
    crown = (f"M{cx - cw / 2:.1f},{top + 5:.1f} q0,-5 5,-5 h{cw - 10:.1f} q5,0 5,5 "
             f"v{ch - 10:.1f} q0,5 -5,5 h-{cw - 10:.1f} q-5,0 -5,-5 z")
    base_y = y + (ch if not upper else -ch)
    roots = []
    n = 2 if kind == "molar" else 1
    for i in range(n):
        rx = cx + (i - (n - 1) / 2) * cw * 0.42
        roots.append(f"M{rx - 3:.1f},{base_y:.1f} L{rx:.1f},{base_y + s * 24:.1f} L{rx + 3:.1f},{base_y:.1f}")
    return crown, roots


def odontogram_svg(result, selected=None, notation="FDI"):
    """Status chart of all 32 permanent teeth in chart order (patient's right on the left)."""
    present = {t["fdi"]: t for t in result.teeth}
    cell, gap, left = 40, 14, 22
    width = left * 2 + cell * 16 + gap
    parts = [f'<svg viewBox="0 0 {width} 196" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Dental chart: {len(present)} of 32 teeth detected" '
             f'font-family="JetBrains Mono, monospace">']
    parts.append(f'<text x="4" y="110" font-size="10" fill="{theme.INK_500}">R</text>'
                 f'<text x="{width - 12}" y="110" font-size="10" fill="{theme.INK_500}">L</text>'
                 f'<line x1="{left}" y1="106" x2="{width - left}" y2="106" stroke="{theme.SLATE_300}"/>'
                 f'<line x1="{width / 2}" y1="30" x2="{width / 2}" y2="182" stroke="{theme.SLATE_300}" stroke-dasharray="3 3"/>')

    for row, fdis, upper in ((0, FDI_ROW_UPPER, True), (1, FDI_ROW_LOWER, False)):
        occlusal = 100 if upper else 112
        for i, fdi in enumerate(fdis):
            x = left + i * cell + (gap if i >= 8 else 0)
            kind = tooth_type(fdi)
            crown, roots = _tooth_glyph(x, occlusal, cell, kind, upper)
            t = present.get(fdi)
            title = f"{tooth_label(fdi, notation)} · {tooth_name(fdi)}"
            if t is None:
                fill, stroke, dash, extra = "#F2F4F6", theme.MISSING, ' stroke-dasharray="3 2"', ""
                title += " · not detected"
                cxm = x + cell / 2
                cym = occlusal + (-11 if upper else 11)
                extra = (f'<path d="M{cxm - 4},{cym - 4} l8,8 M{cxm + 4},{cym - 4} l-8,8" '
                         f'stroke="{theme.MISSING}" stroke-width="1.4"/>')
            else:
                kinds = [f["type"] for f in t["findings"]]
                top = next((k for k in SEVERITY if k in kinds), None)
                fill = theme.FINDING_COLORS[top] if top else theme.BONE_50
                stroke, dash, extra = theme.INK_900, "", ""
                if t["score"] < theme.LOW_SCORE:
                    stroke, dash = theme.UNCERTAIN, ' stroke-dasharray="3 2"'
                    title += f" · low detection score {t['score']:.2f}"
                if kinds:
                    title += " · " + ", ".join(dict.fromkeys(kinds))
                if len(set(kinds)) > 1:
                    extra = (f'<circle cx="{x + cell - 9}" cy="{occlusal + (-20 if upper else 20)}" r="5" '
                             f'fill="{theme.WHITE}" stroke="{theme.INK_900}"/>'
                             f'<text x="{x + cell - 9}" y="{occlusal + (-17 if upper else 23)}" font-size="8" '
                             f'text-anchor="middle" fill="{theme.INK_900}">{len(set(kinds))}</text>')
            sel = fdi == selected
            sw = 2.6 if sel else 1.2
            if sel:
                parts.append(f'<rect x="{x + 1}" y="{(occlusal - 50) if upper else occlusal - 2}" width="{cell - 2}" '
                             f'height="52" rx="6" fill="none" stroke="{theme.INK_900}" stroke-width="1.5"/>')
            parts.append(f'<g><title>{esc(title)}</title>')
            for r in roots:
                parts.append(f'<path d="{r}" fill="none" stroke="{stroke}" stroke-width="1.2"{dash}/>')
            parts.append(f'<path d="{crown}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>{extra}</g>')
            ly = 44 if upper else 178
            weight = "600" if sel else "400"
            color = theme.INK_900 if t is not None else theme.MISSING
            parts.append(f'<text x="{x + cell / 2}" y="{ly}" font-size="11" font-weight="{weight}" '
                         f'text-anchor="middle" fill="{color}">{tooth_label(fdi, notation)}</text>')
    parts.append("</svg>")
    svg_image("".join(parts), f"Dental chart: {len(present)} of 32 teeth detected")


def why_this_number(tooth, result):
    """Plain-language account of how the geometric engine numbered this tooth."""
    tr = tooth.get("trace") or {}
    jaw = "upper" if tr.get("jaw") == "upper" else "lower"
    if tr.get("engine") == "v2":
        return _why_v2(tooth, tr, jaw)
    steps = [
        f"Its centre lies {'above' if jaw == 'upper' else 'below'} the jaw split line, so it "
        f"belongs to the <b>{jaw} jaw</b>.",
        f"Sorted left to right on the image, it is tooth <b>{tr.get('rank', '?')} of "
        f"{tr.get('jaw_count', '?')}</b> in that jaw. Numbering starts at "
        f"{'18' if jaw == 'upper' else '48'} (the patient's right third molar).",
    ]
    if tr.get("skipped"):
        steps.append(f"The gap to the previous tooth is <b>{tr['gap_ratio']:.1f}×</b> this tooth's "
                     f"width (threshold 1.6×), so <b>{tr['skipped']} number"
                     f"{'s were' if tr['skipped'] > 1 else ' was'} skipped</b> for missing teeth.")
    elif tr.get("gap_ratio") is not None:
        steps.append(f"The gap to the previous tooth is {tr['gap_ratio']:.1f}× its width, below the "
                     "1.6× missing-tooth threshold.")
    if tr.get("reassigned_from"):
        steps.append(f"It first received <b>{tr['reassigned_from']}</b>, which a higher-scoring tooth "
                     f"also claimed; it moved to the nearest free number in the quadrant.")
    if tr.get("overflow"):
        steps.append("The jaw ran out of numbers, so it took the last one; treat it as uncertain.")
    items = "".join(f"<li>{s}</li>" for s in steps)
    return f'<div class="ta-why"><b>Why this number?</b><ol>{items}</ol></div>'


def _why_v2(tooth, tr, jaw):
    """Engine v2: jaws by the bite line, numbers by position outward from the dental midline."""
    side = "right" if tr.get("side") == "right" else "left"
    pos = tooth["fdi"] % 10
    names = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar", 5: "second premolar",
             6: "first molar", 7: "second molar", 8: "third molar"}
    steps = [f"Its centre lies {'above' if jaw == 'upper' else 'below'} the fitted bite line, so it belongs to the "
             f"<b>{jaw} jaw</b>.",
             f"It is on the <b>patient's {side}</b> of the dental midline (the image's {'left' if side == 'right' else 'right'}"
             f"), tooth <b>{tr.get('rank', '?')}</b> counting outward from the midline."]
    if tr.get("template_distance") is not None:
        steps.append(f"Its centre is <b>{tr['template_distance']:.1f} tooth widths</b> from the midline, which best fits "
                     f"position <b>{pos}</b> ({names.get(pos, '')}) in the arch layout learned from the training images.")
    if tr.get("skipped"):
        steps.append(f"<b>{tr['skipped']} position{'s' if tr['skipped'] > 1 else ''}</b> between it and the previous tooth "
                     "stay empty: a missing or undetected tooth, so later numbers do not shift.")
    if tr.get("reassigned_from"):
        steps.append(f"It first received <b>{tr['reassigned_from']}</b>, which a higher-scoring tooth also claimed; it "
                     "moved to the nearest free number in the quadrant.")
    if tr.get("overflow") or tr.get("conflict"):
        steps.append("More teeth than positions were found on this side; treat this number as uncertain.")
    items = "".join(f"<li>{s}</li>" for s in steps)
    return f'<div class="ta-why"><b>Why this number?</b><ol>{items}</ol></div>'


def tooth_detail(tooth, result, notation="FDI"):
    fdi = tooth["fdi"]
    alt = f"Universal {FDI_TO_UNIVERSAL[fdi]}" if notation == "FDI" else f"FDI {fdi}"
    findings = "".join(
        f'<li><span style="display:inline-block;width:.6rem;height:.6rem;border-radius:2px;'
        f'background:{theme.FINDING_COLORS.get(f["type"], theme.MISSING)};margin-right:.4rem"></span>'
        f'{esc(f["type"])} <span class="ta-mono ta-muted">{f["score"]:.2f}</span>'
        f'{" · near metal, lower confidence" if f.get("near_metal") else ""}</li>'
        for f in tooth["findings"]) or '<li class="ta-muted">No findings mapped to this tooth</li>'
    low = (' · <span style="color:#8A5A00">low detection score</span>'
           if tooth["score"] < theme.LOW_SCORE else "")
    st.html(
        f'<div class="ta-detail"><div class="ta-eyebrow">Selected tooth</div>'
        f'<div style="display:flex;gap:.8rem;align-items:baseline">'
        f'<span class="num">{tooth_label(fdi, notation)}</span>'
        f'<div><h3>{esc(tooth_name(fdi))}</h3><div class="ta-muted">{alt} · detection score '
        f'<span class="ta-mono">{tooth["score"]:.2f}</span>{low}</div></div></div>'
        f'<ul style="margin:.5rem 0 0 1rem;padding:0;font-size:.88rem">{findings}</ul>'
        f'{why_this_number(tooth, result)}</div>')


def empty_state():
    where = "Use the controls above" if is_phone() else "Use the panel on the left"
    st.html(
        f'<div class="ta-hero"><img src="{data_uri("assets/brand/tooth-ai-mark.svg")}" alt="">'
        f'<h2>Upload a panoramic radiograph (OPG)</h2>'
        f'<p>JPG, PNG or BMP, up to 15 MB. {where}, or open a sample case. '
        f'Images are analysed in memory and are not stored.</p>'
        f'<div class="ta-steps">'
        f'<div><em>01</em><b>Detect</b><span>Instance masks for every tooth and finding.</span></div>'
        f'<div><em>02</em><b>Number</b><span>Geometric reasoning assigns FDI numbers and shows why.</span></div>'
        f'<div><em>03</em><b>Map</b><span>Findings are attached to teeth and charted.</span></div>'
        f'</div></div>')


def partner_strip():
    imgs = "".join(f'<img src="{data_uri("assets/" + f, 80)}" alt="{esc(alt)}">' for f, alt in PARTNER_LOGOS)
    st.html(f'<div class="ta-partners">{imgs}</div>')


def footer(logos=True):
    """Page footer. Pass logos=False on a page that already shows the partner strip (once per surface)."""
    imgs = "".join(f'<img src="{data_uri("assets/" + f, 80)}" alt="{esc(alt)}">' for f, alt in PARTNER_LOGOS)
    inst = " and ".join(f"<b>{esc(i)}</b>" for i in tooth_ai.INSTITUTIONS)
    strip = f'<div class="ta-partners">{imgs}</div>' if logos else ""
    st.html(f'<div class="ta-footer">{strip}'
            f'<p>{esc(tooth_ai.PROGRAMME)} A collaboration between {inst}.</p>'
            f'<p>{esc(tooth_ai.DISCLAIMER)}</p></div>')
