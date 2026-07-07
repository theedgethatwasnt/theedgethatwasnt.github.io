# Companion website

The companion website for *The Edge That Wasn't* is built into this repository under
[`docs/`](docs/), using GitHub Pages' "deploy from a branch / `/docs` folder" convention.

## Canonical URL

Once Pages is enabled, the site will be served at:

**https://roni762583.github.io/the-edge-that-wasnt/**

- Landing page: `https://roni762583.github.io/the-edge-that-wasnt/`
- Experiment explorer: `https://roni762583.github.io/the-edge-that-wasnt/experiments/`

## What's in `docs/`

| Path | Page |
|------|------|
| `docs/index.html` | Landing page — title, blurb, the three headline numbers, and the CTAs |
| `docs/experiments/index.html` | Interactive drill-down of all 81 experiments (search + verdict filters, per-experiment code links and embedded figures) |
| `docs/experiments/experiments.json` | The experiment catalogue (data payload) |
| `docs/experiments/build_explorer.py` | Regenerates `experiments/index.html` from `experiments.json` (recomputes the repo-local code-link mapping and figure embedding) |
| `docs/experiments/figures/` | The figure PNGs the explorer displays, embedded locally |

Both pages are fully self-contained (system fonts, no external assets/CDN) and render offline
by opening the HTML files directly, as well as when served by Pages.

## Activation steps (repository owner)

Pages is **not** enabled by this commit — the repository is still private and activation is a
deliberate, owner-only step. To publish:

1. Make the repository public (Settings → General → Danger Zone → Change visibility), if/when ready.
2. Go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Set **Branch** to `main` and the folder to **`/docs`**, then **Save**.
5. Wait for the first Pages build to finish; the site appears at the canonical URL above.

After that, running `python3 docs/experiments/build_explorer.py` and pushing to `main`
redeploys the explorer automatically.

## Regenerating the explorer

```bash
python3 docs/experiments/build_explorer.py --check
```

`--check` verifies that all embedded JSON payloads parse and that zero references to the
private source repository leak into the output.
