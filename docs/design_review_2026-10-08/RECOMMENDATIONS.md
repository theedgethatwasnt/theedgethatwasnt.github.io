# theedgethatwasnt.com — design review, 8 October 2026

Plain-text companion to `REVIEW.html` (same folder; open that for the screenshots, scorecard and side-by-side strips). Nothing outside this folder was changed.

## The dark-mode question

**Yes — what you saw on your phone is the device default.** Every page has

```css
@media (prefers-color-scheme: dark) { :root { --bg:#161614; --fg:#e8e6e1; … } }
```

(docs/index.html lines 29–35) plus two `theme-color` metas, one per scheme. No toggle, no stored preference, no JavaScript: a phone set to dark (or to automatic after sunset) gets the dark site. The screenshots in `shots/` were produced by flipping that one media feature and nothing else.

Faults today: the three viewer chart pages (`viewer/eur_jpy_indicators.html`, `eur_gbp_indicators.html`, `eur_gbp_swing_asi.html`) have no dark block and render cream inside a dark site; the explorer's active filter chip is white on mint at 2.05:1 in dark mode; `--dead` is 3.4:1 on cream.

**Recommendation: keep automatic, add a two-line toggle, do not go light-only.** ThinkDotCo went light-only because framed art must be judged against one wall; this site sells a text. The one image (a cream cover) reads on either ground, the figures sit on bordered white panels, and the longest page is a chapter read in bed on a phone. The dark tokens pass every contrast check (body 14.5:1, muted 6.3:1, accent 8.8:1, button label 8.0:1). A toggle (ten lines of JS, `[data-theme]` selectors) lets a reader override the device and lets the owner check both. It is a convenience, not a priority — it lives inside R4.

## Scores (1–5, nine criteria; same rubric as the ThinkDotCo review)

| Site | First impression | Trust | Imagery | Typography | Colour | Navigation | Buy path | Mobile | Speed | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Crafting Interpreters | 5 | 4 | 5 | 5 | 4 | 3 | 4 | 4 | 4 | **38** |
| Stripe Press (Hamming title page) | 5 | 3 | 5 | 5 | 5 | 3 | 4 | 4 | 2 | **36** |
| Pragmatic Bookshelf (Release It!) | 4 | 4 | 4 | 2 | 2 | 4 | 5 | 3 | 3 | **31** |
| No Starch (Algorithmic Thinking) | 3 | 4 | 4 | 2 | 2 | 4 | 5 | 3 | 3 | **30** |
| **The Edge That Wasn't** | 3 | 4 | 2 | 3 | 4 | 2 | 2 | 3 | 5 | **28** |

O'Reilly was attempted and returned "Access Denied" to the headless browser; excluded.

Last in the set but with the best speed (19 KB HTML, no fonts, no JS, no third parties) and the best trust story (public ledger, reconciled table, four retractions, power curve). The gap is concentrated in buy path, navigation and imagery; R1–R3 move all three.

## Ten changes, ranked by effect ÷ effort

Effort: S under an hour, M an afternoon, L a day or more. Evidence captures are in `shots/` (`<page>_<m|d>_<light|dark>[_full]`).

### R1 — Put the price and the buy buttons in the hero (S)
**Problem.** Nothing to press above the fold on either viewport. First button reached is "Explore the 81 experiments" (1,500 px desktop / 2,800 px phone); the buy block is the sixth section, at 3,100 of 6,284 px on a phone; the price is never stated above the fold. Every benchmark shows a price in the first screen.
**Evidence.** `index_m_light.png`, `index_m_light_full.jpg` (buy block ~3,100 px), `index_d_light_full.jpg`.
**Change (index.html).**
```html
<!-- inside .herotext, after the byline -->
<p class="price">Paperback $29.99, 582 pages. Kindle $9.99.</p>
<div class="cta hero-cta">
  <a class="btn primary" href="https://www.amazon.com/dp/B0HC5V3YM7" rel="noopener">Buy the paperback</a>
  <a class="btn ghost"   href="https://www.amazon.com/dp/B0HBT4KW4F" rel="noopener">Kindle edition</a>
  <a class="buy" href="excerpt.html">Read Chapter III free</a>
</div>
```
```css
.price{margin-top:14px;font-size:.95rem;color:var(--muted)}
.hero .cta{margin:18px 0 0}
.hero .coverimg{width:240px}                       /* was 200 */
@media (max-width:560px){ .hero .coverimg{width:150px;max-width:none} }
.lead .cta .btn.primary{background:transparent;color:var(--fg);border-color:var(--line)}  /* demote */
```
Keep the lower buy section as the second ask.

### R2 — One-row navigation with real tap targets, shared by every page (M)
**Problem.** Eight 12.8 px monospace links wrap to three rows on a phone: header 143 px on content pages, 212 px on explorer/glossary; each link 48×17 px (minimum 44 tall). Viewer chart pages carry a different nine-item nav ("Book", "Charts"). The explorer calls `detail.scrollIntoView()` on first render at ≤760 px (`experiments/index.html` line 322), so phones land below the search box and list.
**Evidence.** `index_m_dark.png`, `experiments_m_light.png`, `viewer-eurjpy_m_light.png`.
**Change (every header; best done once via R10's shared stylesheet).**
```css
nav.sitenav{margin-left:auto;display:flex;gap:2px;overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch}
nav.sitenav::-webkit-scrollbar{display:none}
nav.sitenav a{flex:0 0 auto;display:inline-flex;align-items:center;min-height:44px;padding:0 10px;
  font:600 .9rem/1 Georgia,'Palatino Linotype',serif;letter-spacing:0;color:var(--muted);
  border-bottom:2px solid transparent;text-decoration:none}
@media (max-width:560px){
  header.sitehdr{flex-wrap:nowrap;flex-direction:column;align-items:stretch;gap:0;padding:10px 0 0}
  header.sitehdr .brand{padding:0 24px 8px}
  nav.sitenav{margin:0;padding:0 16px;border-top:1px solid var(--line)}
}
```
Labels: "Indicator Viewer" → "Viewer", "Power Curve" → "Power curve"; use the same eight items on `viewer/*.html`. Explorer line 322: scroll only on a user's selection, not on initial render.

### R3 — Fix the unstyled "Read an excerpt" button (S)
**Problem.** `<a class="btn" href="excerpt.html">` has neither `.primary` nor `.ghost`; index.html has no global `a{color}` rule, so it renders in browser-default link blue (purple once visited), bold, borderless, next to the buy button.
**Evidence.** `index_d_light_full.jpg` at ~1,900 px; `index_m_light_full.jpg` at ~3,500 px.
**Change.** `<a class="btn ghost" href="excerpt.html">Read an excerpt</a>` and add `.btn{color:inherit}` as a guard.

### R4 — Make dark mode consistent, then give it a toggle (S alone, M with toggle on every page)
**Problem.** (a) Viewer chart pages have no dark token block. (b) Explorer `.vbtn.on{color:#fff}` → 2.05:1 on #5ec89f in dark. (c) `--dead:#8a8880` is 3.4:1 on cream for 13 px text. (d) No override for reader or owner.
**Evidence.** `viewer-eurjpy_m_dark.png` (renders light), `viewer_m_dark.png` (parent renders dark).
**Change.**
```css
/* (a) paste into the three viewer chart pages */
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ --bg:#161614; --fg:#e8e6e1; --muted:#9a988f;
  --card:#201f1c; --line:#37352f; --acc:#5ec89f; --acc-fg:#10231b; --chip:#2a2e2a; --rule:#37352f; --neg:#e08b7d; --pos:#5ec89f } }
/* (b) experiments/index.html */
.vbtn.on{background:var(--acc);color:var(--acc-fg);border-color:var(--acc)}   /* define --acc-fg there */
/* (c) experiment pages */
:root{--dead:#6f6d66}   /* 4.6:1 */
```
```html
<!-- (d) toggle -->
<button class="theme" type="button" data-theme-toggle aria-label="Switch between light and dark">&#9681;</button>
<script>
(function(){var k='theme',r=document.documentElement,s=null;try{s=localStorage.getItem(k)}catch(e){}
if(s)r.dataset.theme=s;
document.querySelector('[data-theme-toggle]').addEventListener('click',function(){
  var dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme=dark?'light':'dark';try{localStorage.setItem(k,r.dataset.theme)}catch(e){}});})();
</script>
```
The media block must be guarded with `:root:not([data-theme="light"])` and the dark tokens repeated under `:root[data-theme="dark"]`. Leave the white figure panels as they are.

### R5 — Set the four numbers as a figure strip, not four cards (S)
**Problem.** The −55,000 / ~$200 / +171 / 4 block is wrapped in four identical rounded cards, one per row on a phone (~600 px of scroll before the first explanatory sentence).
**Evidence.** `index_m_light_full.jpg` 1,100–1,750 px.
**Change (index.html).**
```css
.numbers{display:grid;grid-template-columns:repeat(2,1fr);gap:0;margin:26px 0 12px;
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.num{background:none;border:0;border-radius:0;padding:16px 14px 14px;text-align:left}
.num:nth-child(odd){border-right:1px solid var(--line)}
.num:nth-child(-n+2){border-bottom:1px solid var(--line)}
.num .v{font-size:2.1rem}
@media (max-width:560px){ .numbers{grid-template-columns:repeat(2,1fr)} }   /* was 1fr */
a.num:hover{background:var(--chip)}
```

### R6 — Retire the monospace eyebrows and the arrows (S)
**Problem.** Uppercase tracked monospace labels on every page: hero kicker, excerpt source line, "INSIDE", Verify headings, every `dt` in retraction cards, every section heading on 81 experiment pages. "→" on six buttons and three viewer cards; middle-dot meta strings. Keep mono only where it is data (ledger table, code, CSV column list).
**Evidence.** `retractions_m_light.png`, `index_m_light.png`, `verify_d_dark.png`.
**Change.**
```css
.kicker{font:italic 1.05rem/1.4 Georgia,serif;letter-spacing:0;text-transform:none;color:var(--muted)}
.inside h2{font:700 1.35rem/1.25 Georgia,serif;text-transform:none;letter-spacing:0;color:var(--fg)}
.retraction dt{font:italic .95rem/1.4 Georgia,serif;letter-spacing:0;text-transform:none;color:var(--muted)}
/* verify.html */ h2{font:700 1.3rem/1.25 Georgia,serif;text-transform:none;letter-spacing:0;color:var(--fg)}
/* experiment template + explorer */ .sec h2,.sec h3{font:700 .95rem/1.3 Georgia,serif;text-transform:none;letter-spacing:0;color:var(--fg)}
/* excerpt.html */ .srcnote{font:italic 1rem/1.5 Georgia,serif;letter-spacing:0;text-transform:none;color:var(--muted)}
```
HTML: remove " &rarr;" from `.btn` labels and `.card h2 .arrow`; "Get the book — Paperback, 582 pages, $29.99 · Kindle, $9.99" → "Get the book" with the R1 price line beneath.

### R7 — Cap the measure at 66 characters on every text page (S)
**Problem.** Measured at 1280 px: About/Excerpt 88 cpl, Ledger 94, explorer detail 112, experiment pages 123 (`main` widens to 940 px ≥1280). Phone measure (45) is right.
**Evidence.** `exp12_d_light.png`, `about_d_light.png`, `ledger_d_light.png`.
**Change.**
```css
/* about, excerpt, retractions, power-curve, verify */ main{max-width:660px}
/* ledger */ main{max-width:860px} main p{max-width:66ch}
/* experiment template */ .sec p,.oneline,.keynum,.meta,h1{max-width:68ch}   /* keep figures wide */
/* explorer */ #detail>*:not(.figs){max-width:70ch}
```

### R8 — Serve the cover at the size it is drawn; lighten the glossary figures (S cover, M glossary)
**Problem.** cover.png is 700×1050, 304 KB, drawn at 200/240/130 px — the home page's LCP and 15× the HTML. Glossary lazy-loads 154 PNGs at ~2,000 px / ~300 KB (8 MB) with no width/height, so the page reflows as they arrive. og-card.png (1200×630) is correct.
**Change.**
```sh
cwebp -q 82 -resize 480 0 cover.png -o cover-480.webp
cwebp -q 82 -resize 960 0 cover.png -o cover-960.webp
for f in experiments/figures/glossary/*.png; do cwebp -q 80 -resize 1280 0 "$f" -o "${f%.png}.webp"; done
```
```html
<img class="coverimg" src="cover-480.webp" srcset="cover-480.webp 480w, cover-960.webp 960w"
     sizes="(max-width:560px) 150px, 240px" width="480" height="720" fetchpriority="high" decoding="async"
     alt="Front cover of The Edge That Wasn't">
<!-- build_glossary.py: emit width/height and the .webp -->
```
Keep cover.png for og:image and JSON-LD.

### R9 — Landmarks, a skip link and reduced motion (S)
**Problem.** Alt text, table caption, `aria-current` and focus rings are all present. Missing: `<main>` on the explorer and viewer chart pages; `<footer>` on glossary, 404 and viewer chart pages; no skip link (nine links to tab through on every page); `html{scroll-behavior:smooth}` and the explorer's smooth `scrollIntoView` ignore `prefers-reduced-motion`; nav has no accessible name.
**Change.**
```html
<a class="skip" href="#main">Skip to content</a>   <!-- first child of body -->
<nav class="sitenav" aria-label="Site">…</nav>
<main id="main">…</main>
```
```css
.skip{position:absolute;left:-999px;top:8px;background:var(--acc);color:var(--acc-fg);padding:8px 12px;border-radius:6px}
.skip:focus{left:8px;z-index:10}
@media (prefers-reduced-motion:reduce){ html{scroll-behavior:auto} .rp{transition:none} }
```
```js
detail.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
```

### R10 — One shared stylesheet, so the next fix is one edit (M)
**Problem.** Every page carries its own copy of tokens/header/nav/footer CSS and they have drifted: `--muted` #615f58 vs #6b6b6b; brand 1rem/700 vs 1.2rem/600; body 17/1.65 vs 16/1.6 vs 15/1.5; nav gap 16 vs 14; footer styled inline via `style=""` on six pages, inside `main` on the home page and outside elsewhere.
**Change.** `docs/site.css` holding tokens (light + dark + `[data-theme]`), body, `header.sitehdr`, `nav.sitenav`, `.btn`, footer, `.skip`, focus. `<link rel="stylesheet" href="/site.css">` in each page before its own `<style>`; `build_explorer.py`, `build_glossary.py` and the experiment template emit the same link and the same eight-item nav. Pick one value per drifted token (`--muted:#615f58`, body 17px/1.65, brand 1rem/700).

## Checks that passed

- Live site byte-identical to repo `docs/index.html` (md5 match); GitHub Pages, HTTP/2.
- Every relative href/src in the 13 reviewed pages resolves on disk; 98 internal live URLs all 200; missing paths return the custom 404 with status 404.
- Amazon: `/dp/B0HC5V3YM7` 200, `/dp/B0HBT4KW4F` 200; JSON-LD offers correct (price, currency, InStock, URL; ISBN 9798189494147; 582 pages).
- Repository links 200 (one `blob/main/…csv` deep link 429 = GitHub rate-limiting the crawler, not broken). 416 GitHub code links on experiment pages not individually fetched.
- Meta complete on every page (title, description, canonical, OG set incl. image:alt, twitter:card, theme-color ×2, SVG favicon); robots.txt + sitemap; no analytics, cookies or third-party requests.
- Contrast: all body/muted/accent/button/neg/pos pairs pass AA in both schemes (body 16.7/14.5, muted 6.1/6.3, accent 6.2/8.8, button 6.5/8.0, loss red 6.6/6.4). Fails: white-on-mint chip 2.05 (dark), `--dead` 3.4/3.5.
- No horizontal overflow on any page at 390 px.

## Files

- `REVIEW.html` — the review with embedded thumbnails (~0.6 MB of images), light + dark.
- `RECOMMENDATIONS.md` — this file.
- `metrics.json`, `refs_metrics.json` — measured header heights, tap targets, characters per line, fonts, per capture.
- `shots/` — 112 captures: `<page>_<m|d>_<light|dark>.png` (first screen) and `…_full.jpg` (full length); `ref-*` for the benchmarks.
