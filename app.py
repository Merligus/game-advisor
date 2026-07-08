"""Game Advisor — Gradio app (HuggingFace Spaces entry point).

Flow: the user searches the catalog for games they've played and (optionally)
sets year / platform / language filters and a ranking mode; we build a state,
rank the catalog, and show the top recommendations with cover art + details.
"Mark as played → refine" folds a suggestion back into the history and re-runs.

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
PLATFORMS = [p for p, _ in Counter(p for lst in _idx["platforms"] for p in lst).most_common(20)]
ANY_LANG = "(any)"
LANGUAGES = [ANY_LANG] + [l for l, _ in Counter(l for lst in _idx["language_supports"] for l in lst).most_common()]
YEAR_MIN, YEAR_MAX = 1975, 2026

# Play-history dropdown choices are preloaded once at startup instead of being
# fetched per keystroke. The old design (key_up -> server fuzzy search -> choices
# update) depended on a queue/SSE round-trip for every key press, and on HF
# Spaces that path silently dropped typing events twice (no suggestions in the
# browser while the same event POSTed straight to /gradio_api worked). Native
# client-side filtering over preloaded choices has no server dependency at all.
# Preloading the FULL ~26k catalog is what made v1 laggy (Gradio renders every
# option in the DOM — no virtualization), so we preload the N games with the
# best source coverage: how many of the five collection APIs knew the game,
# tie-broken by user/critic score — a strong fame proxy already in games.csv.
# The long tail stays reachable by typing a full title + Enter
# (allow_custom_value; state_builder fuzzy-resolves it at recommend time).
N_PRELOAD = 3000


def _preload_names() -> list[str]:
    meta = artifacts.games_metadata()
    df = meta[meta.index.isin(set(_idx["name"]))]
    coverage = (
        (df["cover_url"].fillna("[]") != "[]").astype(int)
        + df["description"].notna().astype(int)
        + (df["metacritic_rating"] > 0).astype(int)
        + (df["user_rating"] > 0).astype(int)
        + (df["rawg_rating"] > 0).astype(int)
        + (df["igdb_rating"] > 0).astype(int)
        + (df["hltb_rating"] > 0).astype(int)
        + (df["main_story"] > 0).astype(int)
    )
    # Blend, not lexicographic: ratings (max 8 pts, matching coverage's max 8)
    # can outvote one missing metadata source. Pure coverage-first ordering cut
    # inside the coverage-7 band and excluded every coverage-6 game — e.g.
    # Elden Ring (cov 6, user .84, critic .96) lost to "Pure Chess" (cov 7,
    # user .71). Blended, acclaimed games clear the bar regardless of one
    # sparse column.
    score = coverage + 4.0 * (df["user_rating"] + df["metacritic_rating"])
    top = sorted(zip(df.index, score.values), key=lambda r: r[1], reverse=True)[:N_PRELOAD]
    return [r[0] for r in top]


PRELOAD_NAMES = _preload_names()

# Ranking modes: UI label -> (inference mode, one-line explanation shown with results).
MODES = {
    "Profile + RL (recommended)": (
        "profile_rl",
        "Kept the games closest to your history, then reranked them with the learned policy.",
    ),
    "RL only (trust the policy)": (
        "rl_only",
        "Ranked all matching games with the learned policy alone (exposes the policy's raw behavior).",
    ),
    "Profile only (no RL)": (
        "profile_only",
        "Ranked purely by similarity to your play history — content-based, the policy is not used.",
    ),
}
MODE_LABELS = list(MODES)

# Theme: keep section titles in the default purple, but make interactive
# *selection* (slider fill, checkbox/radio selected) a brighter purple so it
# stands out from the titles instead of blending in.
SELECT_COLOR = "#7c3aed"  # violet-600, a darker selection accent vs the purple titles
SELECT_TEXT = "#ffffff"  # readable text on the darker chips
THEME = gr.themes.Soft(primary_hue="purple").set(
    # Section/block titles stay at primary_500 (purple). These are the actual
    # variables that color the *selection* surfaces (verified against Gradio's
    # CSS): slider fill, selected dropdown options, selected radio/checkboxes.
    slider_color=SELECT_COLOR,
    slider_color_dark=SELECT_COLOR,
    checkbox_label_background_fill_selected=SELECT_COLOR,
    checkbox_label_background_fill_selected_dark=SELECT_COLOR,
    checkbox_background_color_selected=SELECT_COLOR,
    checkbox_background_color_selected_dark=SELECT_COLOR,
)
# The multiselect chips in the input box use --checkbox-label-background-fill
# (a neutral, not a theme "selected" variable), so recolor them via CSS. Also
# add breathing room above the result-sizing sliders and style the detail panel.
CUSTOM_CSS = f"""
.gradio-container .token {{
  background: {SELECT_COLOR} !important;
  color: {SELECT_TEXT} !important;
}}
#result-sizing {{ margin-top: 22px; }}
#result-sizing .wrap {{ gap: 28px; }}
#detail-panel {{
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  padding: 16px 18px;
  background: var(--background-fill-secondary);
}}
"""

# Gradio's Dropdown doesn't scroll the highlighted option into view during
# keyboard navigation, so arrowing moves the highlight through off-screen items.
# This on-load hook scrolls the active option into view on arrow keys
# (block:'nearest' = minimal scroll, no jarring recenter).
# Scroll-follow for dropdown keyboard navigation.
#
# Verified against Gradio's compiled Dropdown JS (Dropdown-B9BMXBuu.js):
#   - the highlighted <li> gets class "active" (all options are rendered; no
#     virtualization), and
#   - the component's only native scrollTo runs when the list *opens* (to show
#     the selected item) — nothing keeps the highlight visible while arrowing,
#     so it walks off-screen. This shim fills exactly that gap.
#
# The app may mount inside <gradio-app>'s shadow root, where document-level
# querySelector / MutationObserver silently see nothing (why earlier attempts
# did nothing). So we search both document and the shadow root, and attach via
# a cheap poll instead of a DOM-added observer — immune to portal/shadow
# surprises; near-zero work when no dropdown is open. Injected via <head> (not
# launch(js=...)) so execution doesn't depend on Gradio's js hook. Scrolling
# uses block:'nearest' — a no-op when the item is already visible, so mouse
# hover never jiggles the list.
KEYNAV_HEAD = """
<script>
(function () {
  const hook = (list) => {
    if (list.__keynav) return;
    list.__keynav = true;
    const scroll = () =>
      list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
    new MutationObserver(scroll).observe(list, {
      attributes: true, attributeFilter: ['class'], childList: true, subtree: true,
    });
    scroll();
  };
  const roots = () => {
    const r = [document];
    const sr = document.querySelector('gradio-app')?.shadowRoot;
    if (sr) r.push(sr);
    return r;
  };
  setInterval(
    () => roots().forEach((rt) => rt.querySelectorAll('ul.options').forEach(hook)),
    300
  );
})();
</script>
"""

# Placeholder cover for games whose metadata has no image.
_PLACEHOLDER = Image.new("RGB", (264, 374), (38, 38, 46))
ImageDraw.Draw(_PLACEHOLDER).text((132, 187), "no cover art", fill=(150, 150, 160), anchor="mm")


def _dedup(seq):
    return list(dict.fromkeys(x for x in seq if x))


def _pct(x):
    return f"{x * 100:.0f}%" if x and x > 0 else None


def _header_md(mode_label, resolved, filters, n_recs):
    why = MODES[mode_label][1]
    head = f"**Mode:** {mode_label} — {why}\n\n" f"**History:** {', '.join(resolved) if resolved else '_none (cold start)_'}\n\n" f"**Filters:** year {filters['year_min']}–{filters['year_max']}" f"{' · ' + ', '.join(filters['platforms']) if filters.get('platforms') else ''}" f"{' · ' + filters['language'] if filters.get('language') else ''}"
    if not n_recs:
        return head + "\n\n_No games match those filters. Widen the year range or platforms._"
    return head + f"\n\n**{n_recs} recommendations** — click a cover to see its details."


def _card_title(r, rank=None):
    """Title + headline scores (shown above the play-toggle button)."""
    yr = r["release_year"] or (r["release"][:4] if r.get("release") else "year unknown")
    meta_bits = [f"`{r['score']:.0%} match`", str(yr)]
    crit = _pct(r["metacritic_rating"])
    usr = _pct(r["user_rating"])
    if crit:
        meta_bits.append(f"critics {crit}")
    if usr:
        meta_bits.append(f"users {usr}")
    title = f"### {rank}. {r['name']}" if rank else f"### {r['name']}"
    return title + "\n\n" + " · ".join(meta_bits)


def _card_body(r):
    """The detailed fields (shown below the play-toggle button)."""
    parts = []
    if r["genres"]:
        parts.append("**Genres:** " + ", ".join(_dedup(r["genres"])[:8]))
    if r["platforms"]:
        parts.append("**Platforms:** " + ", ".join(_dedup(r["platforms"])[:8]))
    devpub = []
    if r["developers"]:
        devpub.append("**Developer:** " + ", ".join(_dedup(r["developers"])[:3]))
    if r["publishers"]:
        devpub.append("**Publisher:** " + ", ".join(_dedup(r["publishers"])[:3]))
    if devpub:
        parts.append(" · ".join(devpub))
    ttb = []
    if r["main_story"] > 0:
        ttb.append(f"main {r['main_story']:.0f}h")
    if r["main_extra"] > 0:
        ttb.append(f"+extras {r['main_extra']:.0f}h")
    if r["completionist"] > 0:
        ttb.append(f"100% {r['completionist']:.0f}h")
    if ttb:
        parts.append("**Time to beat:** " + " · ".join(ttb))
    if r["language_supports"]:
        langs = _dedup(r["language_supports"])
        extra = f" _(+{len(langs) - 8} more)_" if len(langs) > 8 else ""
        parts.append("**Languages:** " + ", ".join(langs[:8]) + extra)
    if r["keywords"]:
        parts.append("**Tags:** " + ", ".join(_dedup(r["keywords"])[:10]))
    desc = (r["description"] or "").strip()
    parts.append(desc or "_(no description available)_")
    return "\n\n".join(parts)


_CLICK_PROMPT = "_Select a cover from the grid to see the game's details here._"
_ADD_LABEL = "＋ I played this — add to history"
_REMOVE_LABEL = "✓ In your history — click to remove"


def _cover_html(r):
    """Enlarged cover shown at the top of the detail panel (controlled markup, so
    selecting a cover never triggers the gallery preview's page-scroll)."""
    url = r.get("cover_url_display")
    if url:
        return f'<img src="{url}" alt="cover" ' f'style="max-height:300px;max-width:100%;border-radius:8px;display:block;margin:0 auto;">'
    return '<div style="height:160px;display:flex;align-items:center;justify-content:center;' 'color:#888;border:1px dashed var(--border-color-primary);border-radius:8px;">no cover art</div>'


def _generate(played_inputs, year_min, year_max, platforms, language, mode_label, top_n, candidate_k, enrich_live):
    if year_max < year_min:
        year_min, year_max = year_max, year_min
    state, resolved = cold_start_state(played_inputs)
    filters = {"year_min": int(year_min), "year_max": int(year_max)}
    if platforms:
        filters["platforms"] = platforms
    if language and language != ANY_LANG:
        filters["language"] = language

    mode = MODES[mode_label][0]
    recs = recommend(
        state,
        filters,
        played_games=resolved,
        top_n=int(top_n),
        mode=mode,
        candidate_k=int(candidate_k),
        enrich=bool(enrich_live),
    )
    gallery = [(r["cover_url_display"] or _PLACEHOLDER, f"{r['name']} ({r['release_year'] or '?'})") for r in recs]
    header = _header_md(mode_label, resolved, filters, len(recs))
    return gallery, recs, header


def on_recommend(played, year_min, year_max, platforms, language, mode_label, top_n, candidate_k, enrich_live):
    gallery, recs, header = _generate(list(played or []), year_min, year_max, platforms, language, mode_label, top_n, candidate_k, enrich_live)
    # Reset the detail panel: clear cover + title, hide toggle, restore prompt body.
    return gallery, recs, header, "", "", gr.update(visible=False), _CLICK_PROMPT, None


def on_select(recs, played, evt: gr.SelectData):
    """Show the clicked cover (enlarged in the panel) + its info, with the toggle
    reflecting whether the game is already in history."""
    if recs and 0 <= evt.index < len(recs):
        r = recs[evt.index]
        name = r["name"]
        in_hist = name in (played or [])
        label = _REMOVE_LABEL if in_hist else _ADD_LABEL
        return (_cover_html(r), _card_title(r, rank=evt.index + 1), gr.update(value=label, visible=True), _card_body(r), name)
    return "", "", gr.update(visible=False), _CLICK_PROMPT, None


def toggle_played(played, name, current_label):
    """Add or remove the shown game from history, flip the toggle label, and keep
    the main button labelled 'Refine' while there's any history."""
    played = list(played or [])
    if not name:
        return gr.update(), gr.update(), gr.update()
    if name in played:
        played.remove(name)
        label = _ADD_LABEL
    else:
        played.append(name)
        label = _REMOVE_LABEL
    main = "Refine" if played else "Recommend"
    return gr.update(choices=played, value=played), gr.update(value=label), gr.update(value=main)


with gr.Blocks(title="Game Advisor") as demo:
    gr.Markdown("# 🎮 Game Advisor\n" "Offline-RL game recommender. Type games you've enjoyed into the search box and hit " "**Recommend**. Click a cover to enlarge it and see the game's details; the toggle next to " "each title adds/removes it from your history so you can refine.")

    # Choices are preloaded at startup (top N_PRELOAD by source coverage — see
    # _preload_names) and filtered natively in the browser: typing never needs a
    # server round-trip, which repeatedly proved unreliable on HF Spaces.
    # allow_custom_value=True keeps the long tail reachable: type the full title
    # and press Enter — state_builder.cold_start_state fuzzy-resolves it against
    # the whole 26k catalog at recommend time (and drops what it can't match).
    played_dropdown = gr.Dropdown(
        PRELOAD_NAMES,
        multiselect=True,
        filterable=True,
        allow_custom_value=True,
        label="Games you've played",
        info="Type to filter the most popular games. Can't find yours? Type its full name and press Enter — it'll be matched to the catalog.",
    )

    with gr.Accordion("Filters & ranking options", open=False):
        with gr.Row():
            year_min = gr.Slider(YEAR_MIN, YEAR_MAX, value=YEAR_MIN, step=1, label="Released from")
            year_max = gr.Slider(YEAR_MIN, YEAR_MAX, value=YEAR_MAX, step=1, label="Released to")
        platforms = gr.CheckboxGroup(PLATFORMS, label="Platforms (any of)")
        with gr.Row():
            language = gr.Dropdown(LANGUAGES, value=ANY_LANG, label="Language")
            mode_label = gr.Radio(MODE_LABELS, value=MODE_LABELS[0], label="Ranking mode")

        with gr.Group(elem_id="result-sizing"):
            gr.Markdown("#### Result sizing")
            top_n = gr.Slider(3, 12, value=10, step=1, label="Number of recommendations")
            candidate_k = gr.Slider(
                10,
                200,
                value=30,
                step=10,
                label="History anchor size (Profile + RL only)",
                info="How many history-closest games the policy reranks. Larger = looser anchoring.",
            )
        enrich_live = gr.Checkbox(
            value=True,
            label="Fetch cover art + descriptions live (RAWG + IGDB; slower, fills gaps in the local data)",
        )

    recommend_btn = gr.Button("Recommend", variant="primary")

    recs_state = gr.State([])  # full records of the current recommendations
    selected_name = gr.State(None)  # canonical name of the cover currently shown
    header_md = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=3):
            # allow_preview=False: clicking a cover only fires `select` (fills the
            # panel on the right). The built-in preview is off because it scrolls
            # the page to center the enlarged image; we show the enlarged cover in
            # the detail panel instead, so the grid stays put and nothing scrolls.
            gallery = gr.Gallery(label="Recommendations", columns=3, height=460, object_fit="contain", allow_preview=False)
        with gr.Column(scale=2, elem_id="detail-panel"):
            cover_html = gr.HTML("")
            detail_title = gr.Markdown("")
            played_toggle = gr.Button(_ADD_LABEL, visible=False, size="sm")
            detail_body = gr.Markdown(_CLICK_PROMPT)

    rec_inputs = [year_min, year_max, platforms, language, mode_label, top_n, candidate_k, enrich_live]
    recommend_btn.click(
        on_recommend,
        [played_dropdown, *rec_inputs],
        [gallery, recs_state, header_md, cover_html, detail_title, played_toggle, detail_body, selected_name],
    )
    gallery.select(on_select, [recs_state, played_dropdown], [cover_html, detail_title, played_toggle, detail_body, selected_name])
    played_toggle.click(
        toggle_played,
        [played_dropdown, selected_name, played_toggle],
        [played_dropdown, played_toggle, recommend_btn],
    )


if __name__ == "__main__":
    # ssr_mode=False: HF Spaces enables Gradio's Node-SSR by default
    # (GRADIO_SSR_MODE=true). Under SSR the served page never delivered the
    # dropdown's key_up events (typing showed no suggestions), while the same
    # event posted straight to /gradio_api/queue/join returned choices fine —
    # i.e. backend healthy, SSR frontend broken. SSR only speeds first paint,
    # so pin it off; the explicit param overrides the Space's env default.
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CUSTOM_CSS,
                head=KEYNAV_HEAD, ssr_mode=False)
