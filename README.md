<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/tooth-ai-lockup-horizontal-reversed.svg">
    <img src="assets/brand/tooth-ai-lockup-horizontal.svg" alt="ToothAI" width="300">
  </picture>
</h1>

<p align="center">
  <b>Integrated deep learning and geometric reasoning for detection, segmentation, numbering and anatomical
  mapping of teeth in panoramic dental radiographs</b>, plus a smartphone photo screening aid that tells people
  when an X-ray is worth getting.
</p>

<p align="center">
  <a href="https://toothai.duckdns.org"><img alt="Live demo" src="https://img.shields.io/badge/live_demo-open_ToothAI-2D6FDB?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="https://github.com/alphaleporus/ToothAI/actions/workflows/ci.yml"><img alt="ToothAI app CI" src="https://github.com/alphaleporus/ToothAI/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/alphaleporus/ToothAI/actions/workflows/photo-screening.yml"><img alt="Photo screening CI" src="https://github.com/alphaleporus/ToothAI/actions/workflows/photo-screening.yml/badge.svg"></a>
  <img alt="Licence: CC BY-NC 4.0" src="https://img.shields.io/badge/licence-CC_BY--NC_4.0-0E2236">
</p>

<p align="center">
  <img src="docs/images/analyse.jpg" alt="The Analyse page: tooth masks and FDI numbers on a panoramic radiograph, the dental chart and the reliability panel" width="900">
</p>

> **Research prototype. Not a medical device and not for diagnostic use. All findings require review by a
> qualified dentist.**

A collaboration between Bharati Vidyapeeth (Deemed to be University) College of Engineering and Bharati
Vidyapeeth Dental College & Hospital, Pune, developed under the C-CAMP Inter-Institutional Biomedical
Innovations Programme (AICTE–C-CAMP IBIP).

<p align="center">
  <img src="assets/logo_aicte.png" alt="AICTE" height="48">&nbsp;&nbsp;
  <img src="assets/logo_bvu.png" alt="Bharati Vidyapeeth" height="48">&nbsp;&nbsp;
  <img src="assets/logo_ccamp.png" alt="C-CAMP" height="48">&nbsp;&nbsp;
  <img src="assets/logo_cibip.png" alt="C-CAMP IBIP" height="48">
</p>

## What it does

| Page | Input | Output |
| --- | --- | --- |
| **Analyse** | A panoramic radiograph (OPG) | Per-tooth masks, FDI numbers, findings mapped to teeth (caries, restorations, endodontic, prosthetic), a dental chart, PDF / Markdown / JSON reports |
| **Photo screening** | Up to five phone photos of the mouth | Possible decay, wear and restorations marked on each photo, a triage level ("routine", "book a check-up soon", "see a dentist promptly"), a screening index and a PDF. It is a screening aid, not a diagnosis, and recommends an OPG when something is seen |
| **How it works**, **Performance**, **About** | | The method, the measured numbers (read live from `paper_eval/results`), credits and licences |

<p align="center">
  <img src="docs/images/photo_screening.jpg" alt="Photo screening: advice, screening index and findings for a set of intraoral phone photos" width="900">
</p>

## Measured performance

Every number names the configuration it comes from. Nothing here is estimated.

**OPG analysis** (716 held-out radiographs, no train overlap, Tesla T4, bootstrap 95 % CIs;
[`paper_eval/`](paper_eval/README.md)). Published engine (`v1`, `full_fix_routing`: original + CLAHE passes, routing
fix; the configuration in the paper):

| Measure | Value |
| --- | --- |
| Tooth detection recall | 94.9 % [94.0, 95.8] (n = 117) |
| Jaw, given the tooth was detected | 97.9 % |
| FDI numbering, end to end | 61.6 %; complete arches 76.6 % [70.7, 82.2] (n = 67), partial arches 35.4 % [27.7, 42.8] (n = 50) |

The partial-arch collapse is the headline limitation of the published engine: one missing tooth shifts every
later number in that jaw. A single pass with the routing fix (`single_pass_fix_routing`, evaluation only)
measured FDI 64.5 % [58.7, 70.1], +2.88 pp [+0.09, +5.90] paired.

**Improved numbering engine (`v2`, the app's default; `tooth_ai/engine_v2.py`).** Designed on the
validation split, reported once on the same test images ([`paper_eval/results/engine_v2.csv`](paper_eval/results/engine_v2.csv)):

| Measure | Published (`v1`) | Improved (`v2`) | Paired difference |
| --- | --- | --- | --- |
| FDI, 117 canonical images | 61.6 % | 85.8 % | +24.2 pp [+18.6, +29.9] |
| FDI, complete arches (67) | 76.6 % | 90.6 % | +14.0 pp [+7.7, +20.3] |
| FDI, partial arches (50) | 35.4 % | 77.4 % | +42.0 pp [+33.5, +50.6] |
| Tooth detection recall, full arches (93) | 94.3 % | 95.5 % | +1.2 pp [+0.7, +1.7] |

v2 runs one pass, removes duplicate boxes, splits the jaws along a fitted bite line, caps to 32 teeth before
numbering (never deleting a detected one) and numbers each tooth by its position along the arch, using a
template learned on the training split. The Analyse page uses v2 by default; the published engine stays
selectable under Numbering for comparison.
Finding (caries, restoration) accuracy on OPGs has not been validated.

**Photo screening** (AKU official test split, 628 smartphone photos;
[`photo_screening/EVAL.md`](photo_screening/EVAL.md)), service defaults: 90.7 % [86.2, 93.8] of photos with
caries get a check-up or dentist referral; 5.6 % [3.7, 8.3] of caries-free photos are referred anyway. Wear,
filling and crown marks are not validated on phone photos. See
[`photo_screening/MODEL_CARD.md`](photo_screening/MODEL_CARD.md) for the rules and limitations.

## Run it locally

Python 3.11. The repository lives on any disk; run commands from the repository root.

```bash
# The app (Streamlit). Demo mode needs no Detectron2: bundled sample cases run through the real engine.
pip install -r requirements-app.txt
TOOTH_AI_DEMO=1 streamlit run app.py

# Full mode also needs torch + Detectron2 and the weights in output/ (see docker/ui.Dockerfile for pins)
streamlit run app.py

# Photo screening API (the Photo screening page talks to it at PHOTO_API_URL, default localhost:8001)
pip install -r photo_screening/requirements.txt
python photo_screening/scripts/fetch_weights.py
uvicorn photo_screening.app.main:app --port 8001

# Tests
python -m pytest tests/
cd photo_screening && python -m pytest
```

On an Apple Silicon Mac the photo API uses the GPU (MPS) when available: about 1 s per photo.

## Deploy

The public demo runs on an Oracle Cloud **Always Free** ARM VM (2 OCPU, 12 GB) with Docker and Caddy
(automatic HTTPS). The images are built on an arm64 Mac and streamed to the VM; nothing is compiled there.
Step-by-step runbook, costs and limits: [`deploy/README.md`](deploy/README.md).

```bash
deploy/push.sh <vm-ip> --weights     # first deploy; afterwards: deploy/push.sh <vm-ip>
```

## Repository

| Path | What |
| --- | --- |
| `app.py` | Streamlit entry point (navigation, theme) |
| `tooth_ai/` | OPG pipeline (`pipeline.py`), analysis and reliability, rendering, reports, UI theme and pages |
| `photo_screening/` | Photo screening API, rules, PDF, UI page, training notebooks, evaluation ([README](photo_screening/README.md)) |
| `paper_eval/` | Evaluation harness and audit: the source of every OPG number shown |
| `docker/`, `deploy/` | App image, production compose, Caddy, VM setup and deploy scripts |
| `assets/`, `static/fonts/` | Brand assets (generated by `tools/build_brand.py`) and self-hosted fonts |
| `tools/` | Brand builder, demo fixtures, report figures (`report_media/`) and one-off dataset scripts (`dataset/`) |
| `docs/` | Brand and UX spec, product overview, requirements, set-up guides for Mac and Windows (`setup/`) |
| `tests/` | Unit tests for the OPG side (no GPU needed) |
| `workspace/` | Earlier phases, the previous single-file app (`app_legacy.py`) and old scripts, kept for reference |

Model weights are not in git: OPG checkpoints live in `output/<run>/`, photo models in
`photo_screening/weights/` (`fetch_weights.py`, or the training notebooks' artifact zips).

## Privacy

Uploads are processed in memory and never stored. Photo metadata (EXIF location, device, time) and file
names are removed before analysis, and there are no accounts, databases or analytics. Use de-identified
images only: crop or cover any name, date or ID printed on a radiograph.

## Data and licences

- OPG models: trained on the dental radiograph dataset of Adnan et al., *Data in Brief* (2024),
  **CC BY-NC 4.0**, so ToothAI and its OPG weights are for **non-commercial** use.
- Photo models: AlphaDent (Apache-2.0), the AKU intraoral caries dataset (CC BY 4.0) and DentalAI
  (CC BY 4.0); attributions in [`photo_screening/MODEL_CARD.md`](photo_screening/MODEL_CARD.md).
- Fonts: Plus Jakarta Sans and JetBrains Mono, SIL Open Font Licence (`static/fonts/`).
