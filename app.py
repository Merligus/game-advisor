"""Game Advisor — Gradio app (HuggingFace Spaces entry point).

Flow: the user lists games they've played and (optionally) sets year / platform /
language filters; we build a cold-start state, run the offline-RL policy, and show
the top-5 recommendations with cover art + descriptions. "Mark as played" folds a
recommendation back into the history and re-runs.

Run locally:  python app.py   (serves on http://localhost:7860)
On HF Spaces this file is the entry point; ship data/policy.pt,
data/game_embeddings_matrix.npy, data/game_embeddings_index.pkl, and
data/embedding_scalers.pkl alongside it (IGDB creds go in the Space secrets).
"""

import sys
from collections import Counter
from pathlib import Path

# Make the `recommender` package (source/recommender) importable whether run from repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))

import gradio as gr
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from recommender import artifacts
from recommender.inference import recommend
from recommender.state_builder import cold_start_state

load_dotenv()

# --- Catalog-derived UI choices -----------------------------------------------
_idx = artifacts.index_frame()
ALL_NAMES = sorted(_idx["name"].tolist())
PLATFORMS = [p for p, _ in Counter(p for lst in _idx["platforms"] for p in lst).most_common(20)]
ANY_LANG = "(any)"
LANGUAGES = [ANY_LANG] + [l for l, _ in Counter(l for lst in _idx["language_supports"] for l in lst).most_common()]
YEAR_MIN, YEAR_MAX = 1975, 2026

# Placeholder cover for games whose metadata has no image.
_PLACEHOLDER = Image.new("RGB", (264, 374), (38, 38, 46))
ImageDraw.Draw(_PLACEHOLDER).text((132, 187), "no cover art", fill=(150, 150, 160), anchor="mm")


def _make_igdb():
    try:
        from APIs.igdb_api import IGDB

        return IGDB()
    except Exception:
        return None


def _details_md(recs: list[dict]) -> str:
    if not recs:
        return "_No games match those filters. Try widening the year range or platforms._"
    blocks = []
    for i, r in enumerate(recs, 1):
        yr = r["release_year"] or "year unknown"
        plats = ", ".join(r["platforms"][:6]) if r["platforms"] else "platforms unknown"
        desc = (r["description"] or "").strip()
        desc = (desc[:400] + "…") if len(desc) > 400 else desc
        blocks.append(f"**{i}. {r['name']}**  ·  match {r['score']:.3f}  ·  {yr}\n\n" f"<sub>{plats}</sub>\n\n{desc or '_(no description)_'}")
    return "\n\n---\n\n".join(blocks)


def _generate(played_inputs, year_min, year_max, platforms, language, use_igdb):
    state, resolved = cold_start_state(played_inputs)
    filters = {"year_min": int(year_min), "year_max": int(year_max)}
    if platforms:
        filters["platforms"] = platforms
    if language and language != ANY_LANG:
        filters["language"] = language

    igdb = _make_igdb() if use_igdb else None
    recs = recommend(state, filters, played_games=resolved, top_n=5, igdb=igdb)

    gallery = [(r["cover_url"] or _PLACEHOLDER, f"{r['name']} ({r['release_year'] or '?'})") for r in recs]
    history = ", ".join(resolved) if resolved else "_cold start (no recognized history)_"
    info = f"**History:** {history}\n\n**Filters:** {filters}"
    rec_names = [r["name"] for r in recs]
    return gallery, _details_md(recs), info, gr.update(choices=rec_names, value=[])


def on_recommend(played, year_min, year_max, platforms, language, use_igdb):
    return _generate(list(played or []), year_min, year_max, platforms, language, use_igdb)


def on_refine(refine_selected, played, year_min, year_max, platforms, language, use_igdb):
    merged = list(dict.fromkeys(list(played or []) + list(refine_selected or [])))
    gallery, details, info, refine_update = _generate(merged, year_min, year_max, platforms, language, use_igdb)
    # Reflect the folded-in games back into the played dropdown.
    return gr.update(value=merged), gallery, details, info, refine_update


with gr.Blocks(title="Game Advisor") as demo:
    gr.Markdown("# 🎮 Game Advisor\n" "Offline-RL game recommender. List a few games you've enjoyed, set optional filters, " "and get five suggestions. Use **Mark as played → refine** to fold a suggestion into your " "history and recommend again.")

    played_dropdown = gr.Dropdown(
        ALL_NAMES,
        multiselect=True,
        filterable=True,
        label="Games you've played",
        info="Type to search the catalog; the closest titles appear — pick the ones you've enjoyed.",
    )

    with gr.Row():
        year_min = gr.Slider(YEAR_MIN, YEAR_MAX, value=YEAR_MIN, step=1, label="Released from")
        year_max = gr.Slider(YEAR_MIN, YEAR_MAX, value=YEAR_MAX, step=1, label="Released to")
    with gr.Row():
        platforms = gr.CheckboxGroup(PLATFORMS, label="Platforms (any of)")
    with gr.Row():
        language = gr.Dropdown(LANGUAGES, value=ANY_LANG, label="Language")
        use_igdb = gr.Checkbox(value=True, label="Fetch cover art live from IGDB (fills missing covers; slower)")

    recommend_btn = gr.Button("Recommend", variant="primary")
    info_md = gr.Markdown()
    gallery = gr.Gallery(label="Recommendations", columns=5, height=400, object_fit="contain")
    details_md = gr.Markdown()

    with gr.Row():
        refine_dropdown = gr.Dropdown([], multiselect=True, label="Mark recommendations as played")
        refine_btn = gr.Button("Mark as played → refine")

    rec_inputs = [year_min, year_max, platforms, language, use_igdb]
    recommend_btn.click(
        on_recommend,
        [played_dropdown, *rec_inputs],
        [gallery, details_md, info_md, refine_dropdown],
    )
    refine_btn.click(
        on_refine,
        [refine_dropdown, played_dropdown, *rec_inputs],
        [played_dropdown, gallery, details_md, info_md, refine_dropdown],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
