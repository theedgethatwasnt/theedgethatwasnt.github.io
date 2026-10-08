# theedgethatwasnt.com — after applying R1–R10 (8 October 2026)

Same nine criteria, same 1–5 rubric as `RECOMMENDATIONS.md` / `REVIEW.html`. Measured on the local
build with headless Chrome 151 at 390×844 and 1280×900, light and dark (`after/<page>_<m|d>_<light|dark>.png`
first screen, `…_full.jpg` full page, `after/metrics_after.json` the numbers). Nothing in the manuscript,
the ledger data, the claims, the buy links, or any URL / canonical / OG / JSON-LD / sitemap / robots / CNAME
/ Search Console file changed.

## Scores

| Criterion | Before | After | What moved it |
|---|---|---|---|
| First impression | 3 | 4 | Price and both buy buttons in the first screen on both viewports; cover served at its drawn size; the four numbers as a ruled strip instead of four cards |
| Trust | 4 | 4 | Unchanged content — ledger, retractions, power curve, verify |
| Imagery | 2 | 3 | Cover is now crisp at 240 px / 112 px (WebP 480/960 w, PNG fallback); 154 glossary plates reserve their box (no reflow) at 1.37 MB instead of 8.6 MB. Still a one-image site |
| Typography | 3 | 4 | Georgia for all chrome, mono only on data; tracked-caps eyebrows and arrows gone; measure 71–73 cpl desktop / 43–53 phone (was 79–123) |
| Colour | 4 | 5 | Every text token pair ≥ 4.5:1 in both modes (`--dead` 4.96 / 5.24, chip label 8.0:1); viewer pages follow the theme; light/dark toggle with stored override |
| Navigation | 2 | 4 | One scrollable row, same 8 items on all 97 pages, 44 px targets, header 90 px on phone (was 143–212) / 49 px desktop; explorer no longer auto-scrolls on load; skip link + `aria-label` |
| Buy path | 2 | 4 | "Buy the paperback" at y = 450 px on a phone (was ~3,100) and 387 px on desktop; price stated above the fold; "Read Chapter III free" beside it; second ask kept lower |
| Mobile | 3 | 4 | No horizontal overflow on any page; 2×2 strip; 44 px buttons; cover beside the title |
| Speed | 5 | 5 | Home first view 46 KB (html + site.css + site.js + cover-480.webp) vs 323 KB; two extra same-origin requests, still no fonts / JS frameworks / third parties |
| **Total** | **28** | **37** | |

## Per-recommendation

- **R1** hero: price line + `Buy the paperback` (B0HC5V3YM7) + `Kindle edition` (B0HBT4KW4F) + `Read Chapter III free`; lead CTA demoted to ghost; lower buy section kept as the second ask.
- **R2** nav: `site.css` one-row `nav.sitenav`, 44 px targets, right-edge fade on phones, labels "Viewer" / "Power curve"; viewer chart pages lost their 9-item variant; explorer line 322 scroll now only on a user's pick.
- **R3** `.btn` base style is the ghost look, so a bare `.btn` can never fall through to link blue; "Read an excerpt" is `.btn.ghost`.
- **R4** dark tokens in one place, media query guarded by `:root:not([data-theme="light"])`, repeated under `:root[data-theme="dark"]`; `.vbtn.on` uses `--acc-fg`; `--dead` #6f6d66 / #8c8a81; toggle (`site.js`, localStorage, updates both `theme-color` metas) on every page including the three viewer chart pages; chart canvases stay white panels like the figures.
- **R5** `.numbers` is a 2×2 ruled strip, no cards, left-aligned, on phones too.
- **R6** kicker, Inside heading, retraction `dt`, Verify `h2`, experiment/explorer section heads, excerpt source line all serif; arrows removed from buttons and viewer cards; middle-dot meta strings replaced by spaced spans / sentences.
- **R7** `--measure: 31em` cap on prose everywhere (about/excerpt/retractions at a 610 px column, ledger 860, experiment pages 760, explorer detail, glossary definitions); figures keep their width.
- **R8** `cover-480.webp` 21 KB / `cover-960.webp` 49 KB with `cover.png` fallback (og:image + JSON-LD still `cover.png`); glossary plates → 30 WebP ≤ 1200 px, 1.37 MB total, emitted with `width`/`height`; `build_glossary.py` reads dims via Pillow.
- **R9** skip link, `<nav aria-label="Site">`, `<main id="main">` + `<footer>` on all pages (explorer: compact footer; viewer chart pages: hint + disclaimer as `<footer>`), `scroll-behavior` and the explorer scroll honour `prefers-reduced-motion`.
- **R10** `docs/site.css` (tokens, body, header, nav, toggle, buttons, footer, skip, focus, motion) + `docs/site.js`; every page and all three generators (`scripts/gen_experiment_pages.py`, `experiments/build_explorer.py`, `experiments/build_glossary.py`) emit the same link, the same header and the same 8-item nav. Drifted tokens settled on `--muted:#615f58`, body 17 px/1.65, brand 1 rem/700.

## Checks run

- 96 HTML pages crawled on `python3 -m http.server` from `docs/`: 150 distinct internal hrefs/srcs, all 200.
- `tidy -q -e`: zero errors on every page; warnings are the pre-existing ones (glossary `dl` nesting, JS `&&` in viewer scripts) plus old-tidy not knowing `loading` / `decoding` / `fetchpriority`.
- Contrast: 16 token pairs × 2 modes computed, all ≥ 4.5:1 (lowest: muted on chip, dark, 4.77).
- Horizontal overflow at 390 px: none on any of the 15 page types.
- Theme toggle: click → `data-theme=dark`, stored, body #161614, both metas #161614, label flips; reload on another page keeps it.
- Sizes: `docs/` (excluding this review folder) 18.6 MB → 11.3 MB; glossary plates 8.72 MB → 1.45 MB; 81 experiment pages 910 KB → 829 KB.

## Not done / notes

- The after-screenshots are PNG/JPG and therefore gitignored, exactly like the before-shots in `shots/`; they live on disk in `after/` on fx-workstation.
- The glossary page is long by construction (154 plates now reserve their height instead of popping in); a per-letter lazy section or a plate-per-family dedupe would be the next step, not part of R1–R10.
- Explorer `figures/*.png` (5 MB across 19 figures, loaded only when an experiment with figures is selected) were left as PNG; R8 scoped the glossary.
