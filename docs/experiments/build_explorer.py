#!/usr/bin/env python3
"""Regenerate docs/experiments/index.html — the 81-experiment drill-down explorer.

Adapted from fx-core's experiment_explorer/build_explorer.py for the *curated
reader repository*. Two differences from the fx-core original:

1. **Code links retarget to THIS repo.** Every `code_paths` entry in
   experiments.json is an fx-core-relative path. This script maps each one to
   its location in the curated companion (`research/experiments/X` ->
   `experiments/X`; `lib/`, `services/`, `docs/` unchanged) and checks whether
   the file/dir actually exists here. Curated files become deep links into
   `github.com/roni762583/the-edge-that-wasnt`; non-curated paths render as
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
BLOB = "https://github.com/roni762583/the-edge-that-wasnt/blob/main/"
TREE = "https://github.com/roni762583/the-edge-that-wasnt/tree/main/"


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
        is_dir = p.endswith("/")
        target = REPO_ROOT / mapped.rstrip("/")
        if is_dir:
            pathmap[p] = (TREE + mapped) if target.is_dir() else None
        else:
            pathmap[p] = (BLOB + mapped) if target.is_file() else None
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
<style>
:root {{ --bg:#fafaf8; --fg:#1a1a1a; --muted:#6b6b6b; --card:#fff; --line:#e2e0da; --acc:#0f6b4f; --chip:#eef2ee; --dead:#8a8880; }}
@media (prefers-color-scheme: dark) {{
 :root {{ --bg:#161614; --fg:#e8e6e1; --muted:#9a988f; --card:#201f1c; --line:#37352f; --acc:#5ec89f; --chip:#2a2e2a; --dead:#6f6d66; }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ font:15px/1.55 Georgia,'Palatino Linotype',serif; background:var(--bg); color:var(--fg); }}
header {{ padding:16px 24px 12px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
header h1 {{ font-size:1.2rem; font-weight:600; }}
header p {{ color:var(--muted); font-size:.85rem; }}
header a.home {{ color:var(--acc); text-decoration:none; font-size:.85rem; margin-left:auto; }}
header a.home:hover {{ text-decoration:underline; }}
.wrap {{ display:flex; height:calc(100vh - 62px); }}
#side {{ width:400px; min-width:280px; border-right:1px solid var(--line); display:flex; flex-direction:column; }}
#controls {{ padding:10px 12px; border-bottom:1px solid var(--line); }}
#q {{ width:100%; padding:7px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); font:inherit; font-size:.9rem; }}
#verds {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:5px; }}
.vbtn {{ font:inherit; font-size:.75rem; padding:2px 9px; border:1px solid var(--line); border-radius:12px; background:var(--card); color:var(--fg); cursor:pointer; }}
.vbtn.on {{ background:var(--acc); color:#fff; border-color:var(--acc); }}
#list {{ overflow-y:auto; flex:1; }}
.item {{ padding:10px 14px; border-bottom:1px solid var(--line); cursor:pointer; }}
.item:hover {{ background:var(--chip); }}
.item.sel {{ background:var(--chip); border-left:3px solid var(--acc); padding-left:11px; }}
.item .t {{ font-weight:600; font-size:.9rem; }}
.item .s {{ color:var(--muted); font-size:.78rem; margin-top:1px; }}
#detail {{ flex:1; overflow-y:auto; padding:26px 34px 60px; }}
#detail h2 {{ font-size:1.45rem; margin-bottom:2px; }}
.meta {{ color:var(--muted); font-size:.85rem; margin-bottom:14px; }}
.oneline {{ font-style:italic; font-size:1.02rem; border-left:3px solid var(--acc); padding:6px 12px; margin:12px 0 18px; background:var(--card); }}
.sec {{ margin:16px 0; }}
.sec h3 {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:5px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ background:var(--chip); border:1px solid var(--line); padding:2px 10px; border-radius:12px; font-size:.8rem; }}
.chip.alg {{ border-color:var(--acc); }}
code, .code a, .code span {{ font:13px/1.5 ui-monospace,Menlo,monospace; }}
.code a {{ display:block; color:var(--acc); text-decoration:none; padding:1px 0; word-break:break-all; }}
.code a:hover {{ text-decoration:underline; }}
.code span.dead {{ display:block; color:var(--dead); padding:1px 0; word-break:break-all; cursor:help; }}
.code span.dead::after {{ content:" ·"; }}
.figs {{ display:flex; flex-wrap:wrap; gap:14px; }}
.figs figure {{ max-width:460px; }}
.figs img {{ max-width:100%; border:1px solid var(--line); border-radius:4px; background:#fff; }}
.figs figcaption {{ font-size:.75rem; color:var(--muted); word-break:break-all; }}
.keynum {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 14px; font-size:.92rem; }}
#empty {{ color:var(--muted); padding:40px; }}
@media (max-width:760px){{ .wrap{{flex-direction:column;height:auto}} #side{{width:100%;max-height:45vh}} }}
</style>
</head>
<body>
<header>
 <h1>The Edge That Wasn't — Experiment Explorer</h1>
 <p>{n} experiments — data, indicators, algorithms, code, figures, verdicts.</p>
 <a class="home" href="../index.html">&larr; Back to the book</a>
</header>
<div class="wrap">
 <div id="side">
  <div id="controls">
   <input id="q" type="search" placeholder="Search name, description, indicator, algorithm…">
   <div id="verds"></div>
  </div>
  <div id="list"></div>
 </div>
 <div id="detail"><div id="empty">Select an experiment.</div></div>
</div>
<script>
const DATA = {data};
const PATHMAP = {pathmap};
const FIGMAP = {figmap};
const exps = DATA.experiments;
const vclass = v => (v.match(/[⛔✅🟡🟢🔴⚫️]+/u)||['·'])[0];
const verdicts = [...new Set(exps.map(e=>vclass(e.verdict)))];
let sel = null, vFilter = null, q = '';
const list = document.getElementById('list'), detail = document.getElementById('detail');
const vbox = document.getElementById('verds');
verdicts.forEach(v=>{{
  const b=document.createElement('button'); b.className='vbtn'; b.textContent=v + ' ' + exps.filter(e=>vclass(e.verdict)===v).length;
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
    const d=document.createElement('div'); d.className='item'+(sel===e.id?' sel':'');
    d.innerHTML=`<div class="t">#${{e.id}} ${{e.name}}</div><div class="s">${{e.verdict}} — ${{e.one_line.slice(0,110)}}${{e.one_line.length>110?'…':''}}</div>`;
    d.onclick=()=>{{ sel=e.id; render(); show(e); }};
    list.appendChild(d);
  }});
}}
function esc(s){{ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;'); }}
function sec(title, inner){{ return inner ? `<div class="sec"><h3>${{title}}</h3>${{inner}}</div>` : ''; }}
function codeLink(p){{
  const url = PATHMAP[p];
  if (url) return `<a href="${{url}}" target="_blank" rel="noopener">${{esc(p)}}</a>`;
  return `<span class="dead" title="not included in the curated companion">${{esc(p)}}</span>`;
}}
function figBlock(p){{
  const src = FIGMAP[p];
  if (!src) return `<figure><figcaption>${{esc(p)}} (figure not included)</figcaption></figure>`;
  return `<figure><a href="${{src}}" target="_blank" rel="noopener"><img src="${{src}}" loading="lazy" alt="${{esc(p)}}"></a><figcaption>${{esc(p)}}</figcaption></figure>`;
}}
function show(e){{
  const chips = (arr,cls)=>arr&&arr.length?`<div class="chips">${{arr.map(x=>`<span class="chip ${{cls||''}}">${{esc(x)}}</span>`).join('')}}</div>`:'';
  const code = (e.code_paths||[]).map(codeLink).join('');
  const figs = (e.figures||[]).map(figBlock).join('');
  detail.innerHTML = `
   <h2>#${{e.id}} — ${{esc(e.name)}}</h2>
   <div class="meta">${{esc(e.verdict)}}${{e.period&&e.period!=='not recorded'?' · '+esc(e.period):''}}${{e.journey_anchor?' · JOURNEY: “'+esc(e.journey_anchor)+'”':''}}</div>
   <div class="oneline">${{esc(e.one_line)}}</div>
   ${{sec('What it tests', `<p>${{esc(e.what_it_tests)}}</p>`)}}
   ${{sec('Description', `<p>${{esc(e.description)}}</p>`)}}
   ${{sec('Method & materials', `<p>${{esc(e.method_materials)}}</p>`)}}
   ${{sec('Data used', `<p>${{esc(e.data_used)}}</p>`)}}
   ${{sec('Key numbers', `<div class="keynum">${{esc(e.key_numbers)}}</div>`)}}
   ${{sec('Indicators', chips(e.indicators))}}
   ${{sec('Algorithms', chips(e.algorithms,'alg'))}}
   ${{sec('Code', code?`<div class="code">${{code}}</div>`:'')}}
   ${{sec('Figures', figs?`<div class="figs">${{figs}}</div>`:'')}}
  `;
  detail.scrollTop=0;
}}
render();
if (exps.length) {{ sel=exps[0].id; render(); show(exps[0]); }}
</script>
</body>
</html>
"""


def build() -> str:
    data = json.load(EXPERIMENTS_JSON.open(encoding="utf-8"))
    pathmap = build_pathmap(data)
    figmap = build_figmap(data)
    html = HTML_TEMPLATE.format(
        n=len(data["experiments"]),
        data=json.dumps(data, ensure_ascii=False),
        pathmap=json.dumps(pathmap, ensure_ascii=False),
        figmap=json.dumps(figmap, ensure_ascii=False),
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
    for name in ("DATA", "PATHMAP", "FIGMAP"):
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
