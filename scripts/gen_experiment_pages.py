#!/usr/bin/env python3
"""Pre-render one static HTML page per experiment + the site sitemap.

Why: docs/experiments/index.html is a client-side drill-down — without JS a
crawler (or reader) sees only the bare shell. This script generates one
crawlable, stable-URL page per experiment at

    docs/experiments/<nn>-<slug>.html

containing the experiment's number, name, hypothesis (what it tests), verdict
(canonical vocabulary: positive / mixed / negative / retracted), key result,
description, and code links where the code is included in this repository.
It also writes docs/sitemap.xml covering every page on the site.

The interactive explorer links to these pages from its server-rendered list
(see build_explorer.py: build_static_list) — progressive enhancement: with JS
the in-page drill-down takes over; without JS the links still work.

Usage (from repo root or anywhere):
    python3 scripts/gen_experiment_pages.py

Repeatable: regenerates all pages + sitemap from docs/experiments/experiments.json.
Run it (followed by build_explorer.py) after any experiments.json edit.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DOCS = REPO_ROOT / "docs"
EXP_DIR = DOCS / "experiments"

sys.path.insert(0, str(EXP_DIR))
import build_explorer  # noqa: E402  (slugify/page_filename/build_pathmap shared)

SITE = "https://theedgethatwasnt.com"

VERDICT_WORD = {
    "✅": "positive",
    "🟡": "mixed",
    "⛔": "negative",
    "🔴": "retracted (one-time survivor, later retracted)",
}

FAVICON = ("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2032%2032%27%3E"
           "%3Crect%20width%3D%2732%27%20height%3D%2732%27%20rx%3D%277%27%20fill%3D%27%230f6b4f%27/%3E"
           "%3Cline%20x1%3D%2711%27%20y1%3D%276%27%20x2%3D%2711%27%20y2%3D%2726%27%20stroke%3D%27%235ec89f%27%20stroke-width%3D%272%27/%3E"
           "%3Crect%20x%3D%278%27%20y%3D%2712%27%20width%3D%276%27%20height%3D%279%27%20rx%3D%271%27%20fill%3D%27%23f2f1ec%27/%3E"
           "%3Cline%20x1%3D%2722%27%20y1%3D%274%27%20x2%3D%2722%27%20y2%3D%2724%27%20stroke%3D%27%23e08b7d%27%20stroke-width%3D%272%27/%3E"
           "%3Crect%20x%3D%2719%27%20y%3D%279%27%20width%3D%276%27%20height%3D%279%27%20rx%3D%271%27%20fill%3D%27%23a23b2c%27/%3E%3C/svg%3E")


def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def verdict_class(verdict: str) -> str:
    m = re.match(r"[⛔✅🟡🔴]", verdict or "")
    return VERDICT_WORD.get(m.group(0), "unrecorded") if m else "unrecorded"


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>#{id} {name} — The Edge That Wasn't</title>
<meta name="description" content="{meta_desc}">
<link rel="canonical" href="{canonical}">
<meta property="og:title" content="#{id} {name} — The Edge That Wasn't">
<meta property="og:description" content="{meta_desc}">
<meta property="og:type" content="article">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="https://theedgethatwasnt.com/og-card.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{favicon}">
<style>
:root {{ --bg:#fafaf8; --fg:#1a1a1a; --muted:#615f58; --card:#fff; --line:#e2e0da; --acc:#0f6b4f; --chip:#eef2ee; --dead:#8a8880; }}
@media (prefers-color-scheme: dark) {{
 :root {{ --bg:#161614; --fg:#e8e6e1; --muted:#9a988f; --card:#201f1c; --line:#37352f; --acc:#5ec89f; --chip:#2a2e2a; --dead:#6f6d66; }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ font:16px/1.6 Georgia,'Palatino Linotype',serif; background:var(--bg); color:var(--fg); }}
header {{ padding:16px 24px 12px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
header .brand {{ font-size:1.2rem; font-weight:600; color:var(--fg); text-decoration:none; }}
header .brand:hover {{ color:var(--acc); }}
nav.sitenav {{ margin-left:auto; display:flex; gap:14px; flex-wrap:wrap; }}
nav.sitenav a {{ color:var(--muted); text-decoration:none; font:600 .82rem/1 ui-monospace,Menlo,monospace; padding-bottom:2px; border-bottom:2px solid transparent; }}
nav.sitenav a:hover {{ color:var(--acc); }}
nav.sitenav a.cur {{ color:var(--acc); border-bottom-color:var(--acc); }}
main {{ max-width:760px; margin:0 auto; padding:26px 24px 80px; }}
footer {{ max-width:760px; margin:0 auto; padding:0 24px 40px; }}
@media (min-width:1280px) {{ main, footer {{ max-width:940px; }} }}
.crumbs {{ font-size:.85rem; color:var(--muted); margin-bottom:16px; }}
.crumbs a {{ color:var(--muted); }}
.crumbs a:hover {{ color:var(--acc); }}
h1 {{ font-size:1.5rem; line-height:1.25; margin-bottom:6px; }}
.meta {{ color:var(--muted); font-size:.9rem; margin-bottom:16px; }}
.verdict {{ display:inline-block; background:var(--chip); border:1px solid var(--line); border-radius:14px; padding:3px 12px; font-size:.9rem; margin-bottom:14px; }}
.oneline {{ font-style:italic; font-size:1.05rem; border-left:3px solid var(--acc); padding:6px 12px; margin:14px 0 20px; background:var(--card); }}
.sec {{ margin:18px 0; }}
.sec h2 {{ font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:6px; }}
.keynum {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 14px; font-size:.95rem; }}
.code a {{ display:table; font:13px/1.6 ui-monospace,Menlo,monospace; color:var(--acc); text-decoration:none; word-break:break-all; }}
.code a:hover {{ text-decoration:underline; }}
.code span.dead {{ display:block; font:13px/1.6 ui-monospace,Menlo,monospace; color:var(--dead); word-break:break-all; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ background:var(--chip); border:1px solid var(--line); padding:2px 10px; border-radius:12px; font-size:.82rem; }}
.pn {{ display:flex; justify-content:space-between; gap:12px; border-top:1px solid var(--line); margin-top:34px; padding-top:16px; font-size:.9rem; }}
.pn a {{ color:var(--acc); text-decoration:none; }}
.pn a:hover {{ text-decoration:underline; }}
.bookline {{ margin-top:26px; padding-top:14px; border-top:1px solid var(--line); font-size:.85rem; color:var(--muted); }}
.bookline a {{ color:var(--acc); text-decoration:none; }}
.bookline a:hover {{ text-decoration:underline; }}
footer {{ border-top:1px solid var(--line); margin-top:40px; padding-top:16px; }}
footer p {{ font-size:.8rem; color:var(--muted); line-height:1.6; margin:0 0 8px; }}
footer a {{ color:var(--muted); }}
</style>
</head>
<body>
<header>
 <a class="brand" href="../index.html">The Edge That Wasn't</a>
 <nav class="sitenav">
  <a href="../verify.html">Verify</a>
  <a href="index.html" class="cur" aria-current="page">Experiments</a>
  <a href="glossary.html">Glossary</a>
  <a href="../viewer/index.html">Indicator Viewer</a>
  <a href="../retractions.html">Retractions</a>
  <a href="../ledger.html">Ledger</a>
  <a href="../about.html">About</a>
 </nav>
</header>
<main>
 <div class="crumbs"><a href="index.html">All 81 experiments</a> &rsaquo; #{id}</div>
 <h1>#{id} — {name}</h1>
 <div class="meta">{period}</div>
 <div class="verdict"><strong>Verdict: {verdict_word}</strong> · {verdict}</div>
 <div class="oneline">{one_line}</div>
 {hypothesis}
 {description}
 {key_result}
 {indicators}
 {algorithms}
 {code}
 <p style="margin-top:22px;font-size:.9rem;color:var(--muted)">Interactive version (search, filters, figures): <a href="index.html" style="color:var(--acc)">the experiment explorer</a>.</p>
 <div class="pn">{prev}{next}</div>
 <div class="bookline">This page is one experiment from the audited record behind the book <em>The Edge That Wasn't</em> — <a href="/index.html#buy">get the book</a> · <a href="/index.html">about the project</a>.</div>
</main>
<footer>
 <p>Nothing here is financial, investment, or trading advice, or a solicitation to trade. Past results do not indicate future performance.</p>
 <p>&copy; 2026 Aharon Zbaida · <a href="https://github.com/theedgethatwasnt/theedgethatwasnt.github.io" target="_blank" rel="noopener">companion repository</a></p>
</footer>
</body>
</html>
"""


def sec(title: str, inner: str) -> str:
    return f'<div class="sec"><h2>{title}</h2>{inner}</div>' if inner else ""


def chips(arr: list | None) -> str:
    if not arr:
        return ""
    return '<div class="chips">' + "".join(f'<span class="chip">{esc(x)}</span>' for x in arr) + "</div>"


def code_block(e: dict, pathmap: dict) -> str:
    parts = []
    for p in e.get("code_paths", []):
        url = pathmap.get(p)
        if url:
            parts.append(f'<a href="{url}" target="_blank" rel="noopener">{esc(p)}</a>')
        else:
            parts.append(f'<span class="dead" title="not included in the curated companion">{esc(p)}</span>')
    return f'<div class="code">{"".join(parts)}</div>' if parts else ""


def meta_description(e: dict, limit: int = 160) -> str:
    """SEO meta description: the one-liner (which front-loads the key finding),
    trimmed to <=160 chars at a word boundary."""
    text = (e.get("one_line", "") or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"


def render(e: dict, pathmap: dict, prev_e: dict | None, next_e: dict | None) -> str:
    fn = build_explorer.page_filename(e)
    vword = verdict_class(e.get("verdict", ""))
    one = esc(e.get("one_line", ""))
    meta_desc = meta_description(e)
    prev_html = (f'<a href="{build_explorer.page_filename(prev_e)}">&larr; #{prev_e["id"]} {esc(prev_e["name"])}</a>'
                 if prev_e else "<span></span>")
    next_html = (f'<a href="{build_explorer.page_filename(next_e)}">#{next_e["id"]} {esc(next_e["name"])} &rarr;</a>'
                 if next_e else "<span></span>")
    period = esc(e.get("period", "")) if e.get("period") and e["period"] != "not recorded" else ""
    return PAGE.format(
        id=e["id"],
        name=esc(e["name"]),
        meta_desc=esc(meta_desc),
        canonical=f"{SITE}/experiments/{fn}",
        favicon=FAVICON,
        period=period,
        verdict_word=vword,
        verdict=esc(e.get("verdict", "")),
        one_line=one,
        hypothesis=sec("Hypothesis — what it tests", f"<p>{esc(e.get('what_it_tests',''))}</p>" if e.get("what_it_tests") else ""),
        description=sec("Description", f"<p>{esc(e.get('description',''))}</p>" if e.get("description") else ""),
        key_result=sec("Key result", f'<div class="keynum">{esc(e.get("key_numbers",""))}</div>' if e.get("key_numbers") else ""),
        indicators=sec("Indicators", chips(e.get("indicators"))),
        algorithms=sec("Algorithms", chips(e.get("algorithms"))),
        code=sec("Code", code_block(e, pathmap)),
        prev=prev_html,
        next=next_html,
    )


def write_sitemap(exp_files: list[str]) -> None:
    today = _dt.date.today().isoformat()
    static_pages = [
        "", "verify.html", "retractions.html", "about.html", "ledger.html",
        "power-curve.html", "excerpt.html",
        "experiments/index.html", "experiments/glossary.html",
        "viewer/index.html", "viewer/eur_jpy_indicators.html",
        "viewer/eur_gbp_indicators.html", "viewer/eur_gbp_swing_asi.html",
    ]
    urls = [f"{SITE}/{p}" for p in static_pages] + [f"{SITE}/experiments/{f}" for f in exp_files]
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for u in urls:
        lines.append(f" <url><loc>{u}</loc><lastmod>{today}</lastmod></url>")
    lines.append("</urlset>")
    (DOCS / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote docs/sitemap.xml: {len(urls)} URLs.")


def write_robots() -> None:
    robots = DOCS / "robots.txt"
    robots.write_text(
        "User-agent: *\nAllow: /\n\nSitemap: %s/sitemap.xml\n" % SITE,
        encoding="utf-8",
    )
    print("Wrote docs/robots.txt (permissive, sitemap referenced).")


def main() -> None:
    data = json.load(EXPERIMENTS_JSON_PATH.open(encoding="utf-8"))
    exps = data["experiments"]
    pathmap = build_explorer.build_pathmap(data)

    # clear previously generated pages (stable pattern NN-*.html)
    for old in EXP_DIR.glob("[0-9][0-9]-*.html"):
        old.unlink()

    files = []
    for i, e in enumerate(exps):
        prev_e = exps[i - 1] if i > 0 else None
        next_e = exps[i + 1] if i < len(exps) - 1 else None
        fn = build_explorer.page_filename(e)
        (EXP_DIR / fn).write_text(render(e, pathmap, prev_e, next_e), encoding="utf-8")
        files.append(fn)
    print(f"Wrote {len(files)} static experiment pages in docs/experiments/.")

    write_sitemap(files)
    write_robots()


EXPERIMENTS_JSON_PATH = EXP_DIR / "experiments.json"

if __name__ == "__main__":
    main()
