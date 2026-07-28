#!/usr/bin/env python3
"""Regenerate docs/experiments/index.html — the 81-experiment drill-down explorer.

Adapted from fx-core's experiment_explorer/build_explorer.py for the *curated
reader repository*. Two differences from the fx-core original:

1. **Code links retarget to THIS repo.** Every `code_paths` entry in
   experiments.json is an fx-core-relative path. This script maps each one to
   its location in the curated companion (`research/experiments/X` ->
   `experiments/X`; `lib/`, `services/`, `docs/` unchanged) and checks whether
   the file/dir actually exists here. Curated files become deep links into
   `github.com/theedgethatwasnt/theedgethatwasnt.github.io`; non-curated paths render as
   plain text with a "not included in the curated companion" tooltip. The
   resulting PATHMAP (orig path -> URL or null) is embedded in the page.

2. **Figures are embedded locally.** The referenced figure PNGs were copied
   into `docs/experiments/figures/` (flattened to basenames). FIGMAP maps each
   original figure path to its local `figures/<basename>` src so the images
   render on GitHub Pages with no reference back to fx-core.

Zero references to the private source repository are emitted.

Usage:
    python3 build_explorer.py            # regenerate index.html
    python3 build_explorer.py --check    # regenerate + verify payloads parse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # docs/experiments/
REPO_ROOT = HERE.parent.parent                   # repo root
EXPERIMENTS_JSON = HERE / "experiments.json"
INDEX_HTML = HERE / "index.html"
FIGURES_DIR = HERE / "figures"

# Public base URLs for the (eventually public) reader repo.
BLOB = "https://github.com/theedgethatwasnt/theedgethatwasnt.github.io/blob/main/"
TREE = "https://github.com/theedgethatwasnt/theedgethatwasnt.github.io/tree/main/"


def slugify(name: str) -> str:
    """Stable URL slug from an experiment name (shared with scripts/gen_experiment_pages.py)."""
    import re
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60].rstrip("-")


def page_filename(e: dict) -> str:
    """Static per-experiment page filename, e.g. 01-project-genesis-architecture.html."""
    return f"{e['id']:02d}-{slugify(e['name'])}.html"


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def build_static_list(data: dict) -> str:
    """Server-rendered experiment list: works with no JS (links to the static
    per-experiment pages); the interactive layer replaces it on load."""
    items = []
    for e in data["experiments"]:
        one = e.get("one_line", "")
        short = one[:110] + ("…" if len(one) > 110 else "")
        items.append(
            f'<a class="item" href="{page_filename(e)}">'
            f'<div class="t">#{e["id"]} {_esc(e["name"])}</div>'
            f'<div class="s">{_esc(e["verdict"])} — {_esc(short)}</div></a>'
        )
    return "\n".join(items)


def map_code_path(p: str) -> str:
    """fx-core-relative path -> reader-repo-relative path."""
    prefix = "research/experiments/"
    if p.startswith(prefix):
        return "experiments/" + p[len(prefix):]
    return p


def build_pathmap(data: dict) -> dict:
    """orig code_path -> deep-link URL if curated here, else None (plain text)."""
    paths = sorted({c for e in data["experiments"] for c in e.get("code_paths", [])})
    pathmap: dict[str, str | None] = {}
    for p in paths:
        mapped = map_code_path(p)
        target = REPO_ROOT / mapped.rstrip("/")
        # trust the filesystem, not the trailing slash: several experiments.json
        # entries list directories without one
        if target.is_dir():
            pathmap[p] = TREE + mapped.rstrip("/") + "/"
        elif target.is_file():
            pathmap[p] = BLOB + mapped
        else:
            pathmap[p] = None
    return pathmap


def build_figmap(data: dict) -> dict:
    """orig figure path -> local relative src under docs/experiments/."""
    figmap: dict[str, str] = {}
    for e in data["experiments"]:
        for f in e.get("figures", []):
            base = Path(f).name
            local = FIGURES_DIR / base
            if local.is_file():
                figmap[f] = f"figures/{base}"
            # if a referenced figure is missing locally we simply omit it from
            # the map; the renderer then shows the caption without a broken img
    return figmap


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Edge That Wasn't — Explore the 81 Experiments</title>
<meta name="description" content="Drill into all 81 documented experiments behind the book — method, data, indicators, algorithms, key numbers, figures, and links to the exact code, positive and negative results alike.">
<link rel="canonical" href="https://theedgethatwasnt.com/experiments/index.html">
<meta property="og:title" content="The Edge That Wasn't — Explore the 81 Experiments">
<meta property="og:description" content="Drill into all 81 documented experiments behind the book — method, data, indicators, algorithms, key numbers, figures, and links to the exact code, positive and negative results alike.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://theedgethatwasnt.com/experiments/index.html">
<meta property="og:image" content="https://theedgethatwasnt.com/og-card.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2032%2032%27%3E%3Crect%20width%3D%2732%27%20height%3D%2732%27%20rx%3D%277%27%20fill%3D%27%230f6b4f%27/%3E%3Cline%20x1%3D%2711%27%20y1%3D%276%27%20x2%3D%2711%27%20y2%3D%2726%27%20stroke%3D%27%235ec89f%27%20stroke-width%3D%272%27/%3E%3Crect%20x%3D%278%27%20y%3D%2712%27%20width%3D%276%27%20height%3D%279%27%20rx%3D%271%27%20fill%3D%27%23f2f1ec%27/%3E%3Cline%20x1%3D%2722%27%20y1%3D%274%27%20x2%3D%2722%27%20y2%3D%2724%27%20stroke%3D%27%23e08b7d%27%20stroke-width%3D%272%27/%3E%3Crect%20x%3D%2719%27%20y%3D%279%27%20width%3D%276%27%20height%3D%279%27%20rx%3D%271%27%20fill%3D%27%23a23b2c%27/%3E%3C/svg%3E">
<style>
:root {{ --bg:#fafaf8; --fg:#1a1a1a; --muted:#6b6b6b; --card:#fff; --line:#e2e0da; --acc:#0f6b4f; --chip:#eef2ee; --dead:#75736b; }}
@media (prefers-color-scheme: dark) {{
 :root {{ --bg:#161614; --fg:#e8e6e1; --muted:#9a988f; --card:#201f1c; --line:#37352f; --acc:#5ec89f; --chip:#2a2e2a; --dead:#8c8a81; }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ font:16px/1.55 Georgia,'Palatino Linotype',serif; background:var(--bg); color:var(--fg); }}
header {{ padding:16px 24px 12px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
h1.brand {{ font-size:1.2rem; font-weight:600; }}
h1.brand a {{ color:var(--fg); text-decoration:none; }}
h1.brand a:hover {{ color:var(--acc); }}
header p {{ color:var(--muted); font-size:.85rem; }}
nav.sitenav {{ margin-left:auto; display:flex; gap:14px; flex-wrap:wrap; }}
nav.sitenav a {{ color:var(--muted); text-decoration:none; font:600 .82rem/1 ui-monospace,Menlo,monospace; padding-bottom:2px; border-bottom:2px solid transparent; }}
nav.sitenav a:hover {{ color:var(--acc); }}
nav.sitenav a.cur {{ color:var(--acc); border-bottom-color:var(--acc); }}
.wrap {{ display:flex; height:calc(100vh - 62px); }}
#side {{ width:400px; min-width:280px; border-right:1px solid var(--line); display:flex; flex-direction:column; }}
#controls {{ padding:10px 12px; border-bottom:1px solid var(--line); }}
#q {{ width:100%; padding:7px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); font:inherit; font-size:1rem; }}
#verds {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:5px; }}
.vbtn {{ font:inherit; font-size:.75rem; padding:2px 9px; border:1px solid var(--line); border-radius:12px; background:var(--card); color:var(--fg); cursor:pointer; }}
.vbtn.on {{ background:var(--acc); color:#fff; border-color:var(--acc); }}
#list {{ overflow-y:auto; flex:1; }}
.item {{ padding:10px 14px; border-bottom:1px solid var(--line); cursor:pointer; }}
a.item {{ display:block; color:inherit; text-decoration:none; }}
.item:hover {{ background:var(--chip); }}
.item.sel {{ background:var(--chip); border-left:3px solid var(--acc); padding-left:11px; }}
.item .t {{ font-weight:600; font-size:.9rem; }}
.item .s {{ color:var(--muted); font-size:.78rem; margin-top:1px; }}
#detail {{ flex:1; overflow-y:auto; padding:26px 34px 60px; }}
#detail h2 {{ font-size:1.45rem; margin-bottom:2px; }}
.meta {{ color:var(--muted); font-size:.85rem; margin-bottom:6px; }}
.permalink {{ color:var(--muted); font-size:.8rem; margin-bottom:14px; }}
.permalink a {{ color:var(--muted); text-decoration:none; border-bottom:1px dotted var(--muted); }}
.permalink a:hover {{ color:var(--acc); border-bottom-color:var(--acc); }}
.oneline {{ font-style:italic; font-size:1.02rem; border-left:3px solid var(--acc); padding:6px 12px; margin:12px 0 18px; background:var(--card); }}
.sec {{ margin:16px 0; }}
.sec h3 {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:5px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ background:var(--chip); border:1px solid var(--line); padding:2px 10px; border-radius:12px; font-size:.8rem; cursor:pointer; user-select:none; display:inline-flex; align-items:center; gap:5px; text-decoration:none; color:inherit; }}
a.chip {{ padding-right:4px; }}
a.chip:hover {{ border-color:var(--acc); color:var(--acc); }}
.chip.alg {{ border-color:var(--acc); }}
.chip.nogloss {{ border-style:dashed; }}
.chip .chipfilter {{ opacity:.5; font-size:.95em; padding:0 2px 0 6px; border-left:1px solid var(--line); cursor:pointer; }}
.chip .chipfilter:hover {{ opacity:1; color:var(--acc); }}
a.term {{ color:inherit; text-decoration:none; border-bottom:1px dotted var(--muted); cursor:help; }}
a.term:hover {{ border-bottom-color:var(--fg); }}
code, .code a, .code span {{ font:13px/1.5 ui-monospace,Menlo,monospace; }}
.code a {{ display:table; color:var(--acc); text-decoration:none; padding:1px 0; word-break:break-all; }}
.code a:hover {{ text-decoration:underline; }}
.code span.dead {{ display:block; color:var(--dead); padding:1px 0; word-break:break-all; cursor:help; }}
.code span.dead::after {{ content:" ·"; }}
.figs {{ display:flex; flex-wrap:wrap; gap:14px; }}
.figs figure {{ max-width:460px; }}
.figs img {{ max-width:100%; border:1px solid var(--line); border-radius:4px; background:#fff; }}
.figs figcaption {{ font-size:.75rem; color:var(--muted); word-break:break-all; }}
.keynum {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 14px; font-size:.92rem; }}
#empty {{ color:var(--muted); padding:40px; }}
@media (min-width:1400px){{ .figs figure{{ max-width:640px; }} #side{{ width:460px; }} }}
@media (max-width:760px){{ .wrap{{flex-direction:column;height:auto}} #side{{width:100%;max-height:45vh}} }}
</style>
</head>
<body>
<header>
 <h1 class="brand"><a href="../index.html">The Edge That Wasn't</a></h1>
 <p>{n} experiments — data, indicators, algorithms, code, figures, verdicts.</p>
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
<div class="wrap">
 <div id="side">
  <div id="controls">
   <input id="q" type="search" placeholder="Search name, description, indicator, algorithm…">
   <div id="verds"></div>
  </div>
  <div id="list">
{staticlist}
  </div>
 </div>
 <div id="detail"><div id="empty">Select an experiment. Every experiment also has a stable static page of its own — the list on the left links to them.</div></div>
</div>
<script>
const DATA = {data};
const PATHMAP = {pathmap};
const FIGMAP = {figmap};
const PAGEMAP = {pagemap};
const exps = DATA.experiments;
const vclass = v => (v.match(/[⛔✅🟡🟢🔴⚫️]+/u)||['·'])[0];
const VLABEL = {{'⛔':'negative','✅':'positive','🟡':'mixed','🔴':'retracted','🟢':'live','⚫️':'off'}};
const verdicts = [...new Set(exps.map(e=>vclass(e.verdict)))];
let sel = null, vFilter = null, q = '';
const list = document.getElementById('list'), detail = document.getElementById('detail');
const vbox = document.getElementById('verds');
verdicts.forEach(v=>{{
  const b=document.createElement('button'); b.className='vbtn';
  b.textContent=v + ' ' + (VLABEL[v]?VLABEL[v]+' ':'') + exps.filter(e=>vclass(e.verdict)===v).length;
  b.onclick=()=>{{ vFilter = vFilter===v?null:v; render(); }};
  b.dataset.v=v; vbox.appendChild(b);
}});
document.getElementById('q').addEventListener('input', e=>{{ q=e.target.value.toLowerCase(); render(); }});
function matches(e){{
  if (vFilter && vclass(e.verdict)!==vFilter) return false;
  if (!q) return true;
  const hay = [e.name,e.one_line,e.description,e.verdict,(e.indicators||[]).join(' '),(e.algorithms||[]).join(' '),e.data_used,e.key_numbers].join(' ').toLowerCase();
  return q.split(/\\s+/).every(w=>hay.includes(w));
}}
function render(){{
  document.querySelectorAll('.vbtn').forEach(b=>b.classList.toggle('on', b.dataset.v===vFilter));
  list.innerHTML='';
  exps.filter(matches).forEach(e=>{{
    // real links to the static per-experiment pages (middle-click / new-tab /
    // keyboard all work); a plain left-click is intercepted for in-pane display
    const d=document.createElement('a'); d.className='item'+(sel===e.id?' sel':'');
    d.href=PAGEMAP[e.id]||'#';
    d.innerHTML=`<div class="t">#${{e.id}} ${{e.name}}</div><div class="s">${{e.verdict}} — ${{e.one_line.slice(0,110)}}${{e.one_line.length>110?'…':''}}</div>`;
    d.onclick=(ev)=>{{ if(ev.metaKey||ev.ctrlKey||ev.shiftKey||ev.altKey) return; ev.preventDefault(); sel=e.id; render(); show(e); }};
    list.appendChild(d);
  }});
}}
function esc(s){{ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;'); }}
function sec(title, inner){{ return inner ? `<div class="sec"><h3>${{title}}</h3>${{inner}}</div>` : ''; }}
// ── Glossary term-marking layer ────────────────────────────────────────
// Loads glossary.json once, builds ONE compiled regex of all surface forms
// (longest-first, word-boundary aware), and marks the FIRST occurrence of
// each term per detail pane. Operates on already-esc()'d prose strings only,
// so it never touches chips, code paths, or figure captions.
let GLOSS = null, GRE = null;
function reEsc(s){{ return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'); }}
function loadGlossary(){{
  fetch('glossary.json').then(r=>r.ok?r.json():null).then(d=>{{
    if(!d||!d.terms) return;
    GLOSS = {{}};
    const surf = [];
    d.terms.forEach(t=>(t.surfaces||[]).forEach(s=>{{
      if(!s || s.length<2) return;                 // skip 1-char / empty surfaces
      const key = esc(s).toLowerCase();            // prose is esc()'d, so key must be too
      if(!(key in GLOSS)){{ GLOSS[key]={{d:t.display,t:t.tooltip,s:t.slug}}; surf.push(esc(s)); }}
    }}));
    surf.sort((a,b)=>b.length-a.length);           // longest match first
    if(surf.length){{
      GRE = new RegExp('(?<![A-Za-z0-9_])(' + surf.map(reEsc).join('|') + ')(?![A-Za-z0-9_])', 'gi');
    }}
    if(sel!=null){{ const e=exps.find(x=>x.id===sel); if(e) show(e); }}  // re-render, now marked
  }}).catch(()=>{{}});
}}
function mark(html, seen){{
  if(!GRE) return html;
  return html.replace(GRE, (m)=>{{
    const info = GLOSS[m.toLowerCase()];
    if(!info || seen.has(info.s)) return m;        // one mark per term per pane
    seen.add(info.s);
    return `<a class="term" href="glossary.html#${{info.s}}" target="_blank" rel="noopener" title="${{esc(info.t)}}">${{m}}</a>`;
  }});
}}
function codeLink(p){{
  const url = PATHMAP[p];
  if (url) return `<a href="${{url}}" target="_blank" rel="noopener">${{esc(p)}}</a>`;
  return `<span class="dead" title="not included in the curated companion">${{esc(p)}}</span>`;
}}
// A chip's PRIMARY action is the glossary entry (if the term resolves there);
// a small trailing ⌕ glyph is the secondary action (filter the experiment
// list by this term). Terms with no glossary entry fall back to filter-only,
// visually marked with a dashed border. Resolution reuses the exact same
// surface-matching data as prose term-marking (GLOSS/GRE): most chip labels
// are full descriptive phrases ("ATR(14, H4) target/stop (2.0x)"), not bare
// glossary headwords, so a whole-chip exact lookup would resolve almost none
// of them — searching for a known surface form ANYWHERE in the label (same
// regex as mark()) is what actually makes the chips useful.
function glossFor(term){{
  if(!GLOSS || !GRE) return null;
  GRE.lastIndex = 0;
  const m = GRE.exec(esc(term));
  return m ? GLOSS[m[1].toLowerCase()] : null;
}}
function chips(arr, cls){{
  if(!arr || !arr.length) return '';
  const parts = arr.map(x=>{{
    const label = esc(x);
    const g = glossFor(x);
    if(g){{
      return `<a class="chip ${{cls||''}}" href="glossary.html#${{g.s}}" target="_blank" rel="noopener" title="${{esc(g.t)}}">${{label}}<span class="chipfilter" data-term="${{label}}" title="Filter experiments by this term" onclick="event.preventDefault();event.stopPropagation();chipFilter(this.dataset.term)">&#8981;</span></a>`;
    }}
    return `<span class="chip ${{cls||''}} nogloss" data-term="${{label}}" title="No glossary entry for this term — click to filter experiments by it" onclick="chipFilter(this.dataset.term)">${{label}}</span>`;
  }});
  return `<div class="chips">${{parts.join('<span style="position:absolute;left:-9999px">, </span>')}}</div>`;
}}
function figBlock(p){{
  const src = FIGMAP[p];
  if (!src) return `<figure><figcaption>${{esc(p)}} (figure not included)</figcaption></figure>`;
  return `<figure><a href="${{src}}" target="_blank" rel="noopener"><img src="${{src}}" loading="lazy" alt="${{esc(p)}}"></a><figcaption>${{esc(p)}}</figcaption></figure>`;
}}
function show(e){{
  const code = (e.code_paths||[]).map(codeLink).join('');
  const figs = (e.figures||[]).map(figBlock).join('');
  const seen = new Set();                          // first-occurrence-per-pane, shared across prose fields
  detail.innerHTML = `
   <h2>#${{e.id}} — ${{esc(e.name)}}</h2>
   <div class="meta">${{esc(e.verdict)}}${{e.period&&e.period!=='not recorded'?' · '+esc(e.period):''}}${{e.journey_anchor?' · JOURNEY: “'+esc(e.journey_anchor)+'”':''}}</div>
   ${{PAGEMAP[e.id]?`<div class="permalink">Permalink: <a href="${{PAGEMAP[e.id]}}">https://theedgethatwasnt.com/experiments/${{PAGEMAP[e.id]}}</a></div>`:''}}
   <div class="oneline">${{mark(esc(e.one_line), seen)}}</div>
   ${{sec('What it tests', `<p>${{mark(esc(e.what_it_tests), seen)}}</p>`)}}
   ${{sec('Description', `<p>${{mark(esc(e.description), seen)}}</p>`)}}
   ${{sec('Method & materials', `<p>${{mark(esc(e.method_materials), seen)}}</p>`)}}
   ${{sec('Data used', `<p>${{esc(e.data_used)}}</p>`)}}
   ${{sec('Key numbers', `<div class="keynum">${{mark(esc(e.key_numbers), seen)}}</div>`)}}
   ${{sec('Indicators', chips(e.indicators))}}
   ${{sec('Algorithms', chips(e.algorithms,'alg'))}}
   ${{sec('Code', code?`<div class="code">${{code}}</div>`:'')}}
   ${{sec('Figures', figs?`<div class="figs">${{figs}}</div>`:'')}}
  `;
  detail.scrollTop=0;
  if(matchMedia('(max-width:760px)').matches) detail.scrollIntoView({{behavior:'smooth'}});
}}
function chipFilter(term){{
  const qEl = document.getElementById('q');
  qEl.value = term.trim(); q = term.trim().toLowerCase(); render();
}}
render();
if (exps.length) {{ sel=exps[0].id; render(); show(exps[0]); }}
loadGlossary();
</script>
<!-- analytics: cookieless counter goes here at hosting step (CF Web Analytics or GoatCounter) — no cookies, no banners -->
</body>
</html>
"""


def build_pagemap(data: dict) -> dict:
    """experiment id -> static per-experiment page filename (for permalinks
    and the real-href list items)."""
    return {e["id"]: page_filename(e) for e in data["experiments"]}


def build() -> str:
    data = json.load(EXPERIMENTS_JSON.open(encoding="utf-8"))
    pathmap = build_pathmap(data)
    figmap = build_figmap(data)
    html = HTML_TEMPLATE.format(
        n=len(data["experiments"]),
        data=json.dumps(data, ensure_ascii=False),
        pathmap=json.dumps(pathmap, ensure_ascii=False),
        figmap=json.dumps(figmap, ensure_ascii=False),
        pagemap=json.dumps(build_pagemap(data), ensure_ascii=False),
        staticlist=build_static_list(data),
    )
    INDEX_HTML.write_text(html, encoding="utf-8")
    n_map = sum(1 for v in pathmap.values() if v)
    print(f"Rebuilt {INDEX_HTML.name}: {len(data['experiments'])} experiments, "
          f"{n_map}/{len(pathmap)} code paths linked ({n_map/len(pathmap)*100:.1f}%), "
          f"{len(figmap)} figures embedded.")
    return html


def check() -> None:
    html = build()
    # Extract and parse each embedded const payload as a smoke test.
    for name in ("DATA", "PATHMAP", "FIGMAP", "PAGEMAP"):
        start = html.index(f"const {name} = ") + len(f"const {name} = ")
        end = html.index(";\n", start)
        json.loads(html[start:end])
    forbidden = "roni762583/" + "fx-core"  # constructed so the literal never appears in source
    assert forbidden not in html, "private source-repo reference leaked into output"
    print("OK: all payloads parse; zero private source-repo references in output.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify payloads parse after build")
    args = ap.parse_args()
    check() if args.check else build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
