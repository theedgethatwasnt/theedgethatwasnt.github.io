#!/usr/bin/env python3
"""Glossary layer for the experiment explorer.

Two jobs, two subcommands:

    python3 build_glossary.py extract --source PATH_TO_MD [--out glossary.json]
        Parse the book's Part-Two reference (§6 Complete Glossary + the §1
        Master Indicator Encyclopedia) plus a small curated supplement into
        glossary.json. Run this where journey_experiments_break_down.md lives
        (the fx-core repo); commit the resulting glossary.json to both repos.

    python3 build_glossary.py build [--json glossary.json] [--out glossary.html]
        Read glossary.json and emit a self-contained glossary.html (system
        fonts, light/dark, alphabetical, per-term #slug anchors, search box,
        back-to-explorer link). Needs no markdown source, so it runs in the
        curated reader repo too.

glossary.json is the single source of truth for BOTH glossary.html AND the
explorer's hover-tooltip term-marking pass (the explorer fetches it at load).

Schema:
    {
      "generated": "ISO date",
      "terms": [
        {
          "slug": "aroon",
          "display": "Aroon",
          "surfaces": ["Aroon"],          # marking surface forms (display + aliases)
          "tooltip": "first sentence …",   # hover text
          "body": "full definition …",     # glossary-page paragraph
          "source": "glossary|encyclopedia|manual"
        }, ...
      ]
    }
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "glossary.json"
DEFAULT_HTML = HERE / "glossary.html"


# ─────────────────────────────────────────────────────────── helpers ──

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "term"


def first_sentence(body: str, limit: int = 200) -> str:
    """First sentence of a definition, for the tooltip. Falls back to a
    length clip. Strips markdown emphasis/code markers."""
    t = _strip_md(body).strip()
    # Split on sentence end followed by space + capital / end. Keep it simple:
    m = re.search(r"(.+?[.!?])(\s|$)", t)
    sent = m.group(1) if m else t
    if len(sent) > limit:
        sent = sent[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return sent.strip()


def _strip_md(s: str) -> str:
    s = s.replace("\\|", "|")
    s = re.sub(r"`([^`]*)`", r"\1", s)      # inline code
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)  # bold
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", s)  # italic
    return s


def surfaces_from_header(header: str) -> tuple[str, list[str]]:
    """Turn a bold glossary header into (display, surface_forms).

    'Accumulative Swing Index (ASI)'          -> display, [Accumulative Swing Index, ASI]
    'ADX / ADXR'                              -> display, [ADX, ADXR]
    'AMDDP (AMDDP1 / AMDDP5 / AMDDP10)'       -> display, [AMDDP, AMDDP1, AMDDP5, AMDDP10]
    'MFE / MAE'                               -> display, [MFE, MAE]
    """
    display = header.strip()
    parens = re.findall(r"\(([^)]*)\)", header)
    main = re.sub(r"\([^)]*\)", "", header)
    forms: list[str] = []

    def _push(chunk: str) -> None:
        for piece in chunk.split("/"):
            p = piece.strip().strip(".").strip()
            # drop leading enumerators like '(1)' remnants and empties
            if p and not re.fullmatch(r"[0-9]+", p):
                forms.append(p)

    _push(main)
    for pg in parens:
        _push(pg)

    # de-dup preserving order, case-insensitive
    seen, out = set(), []
    for f in forms:
        k = f.lower()
        if k not in seen:
            seen.add(k)
            out.append(f)
    return display, out or [display]


# ─────────────────────────────────────────────────── §6 glossary parse ──

def parse_glossary(md: str) -> list[dict]:
    m = re.search(r"^## 6\. Complete Glossary\b", md, re.M)
    if not m:
        return []
    section = md[m.end():]
    # stop at the next top-level '## ' heading if any
    nxt = re.search(r"^## \d", section, re.M)
    if nxt:
        section = section[: nxt.start()]

    entries: list[dict] = []
    for line in section.splitlines():
        line = line.rstrip()
        if not line.startswith("**"):
            continue
        mm = re.match(r"\*\*(.+?)\*\*(.*)", line)
        if not mm:
            continue
        header, rest = mm.group(1), mm.group(2)
        rest = rest.strip()
        rest = re.sub(r"^\s*[—–-]\s*", "", rest)   # strip leading dash
        # em-dash INSIDE the bold (e.g. 'CSI — two unrelated …')
        if re.search(r"\s[—–]\s", header):
            head_term, extra = re.split(r"\s[—–]\s", header, 1)
            header = head_term
            rest = (extra.strip() + " " + rest).strip()
        display, forms = surfaces_from_header(header)
        body = _strip_md(rest).strip()
        if not body:
            continue
        entries.append({
            "slug": slugify(display),
            "display": display,
            "surfaces": forms,
            "tooltip": first_sentence(body),
            "body": body,
            "source": "glossary",
        })
    return entries


# ──────────────────────────────────────────── §1 encyclopedia parse ──

def _split_pipes(row: str) -> list[str]:
    row = row.replace("\\|", "\x00")
    parts = [p.replace("\x00", "|").strip() for p in row.split("|")]
    return parts


def parse_encyclopedia(md: str) -> list[dict]:
    lines = md.splitlines()
    # locate the header row of the Master Indicator Encyclopedia table
    hdr = None
    for i, ln in enumerate(lines):
        if ln.startswith("| #") and "Indicator" in ln and "What It Does" in ln:
            hdr = i
            break
    if hdr is None:
        return []
    cols = _split_pipes(lines[hdr])
    # find column indices
    try:
        name_i = cols.index("Indicator")
        wid_i = cols.index("What It Does")
    except ValueError:
        return []

    entries: list[dict] = []
    for ln in lines[hdr + 2:]:            # skip header + separator
        if not ln.startswith("|"):
            break
        f = _split_pipes(ln)
        if len(f) <= max(name_i, wid_i):
            continue
        name_raw = f[name_i]
        what = _strip_md(f[wid_i]).strip()
        m = re.search(r"\*\*(.+?)\*\*", name_raw)
        if not m or not what:
            continue
        display, forms = surfaces_from_header(m.group(1))
        entries.append({
            "slug": slugify(display),
            "display": display,
            "surfaces": forms,
            "tooltip": first_sentence(what),
            "body": what,
            "source": "encyclopedia",
        })
    return entries


# ─────────────────────────────────────────────── curated supplement ──

# Extra standalone terms the running prose uses that the reference tables
# don't define as their own row (e.g. SHAP), plus alias injections that map a
# snake_case / abbreviated surface onto an existing entry's slug.
MANUAL_TERMS: list[dict] = [
    {
        "display": "SHAP",
        "surfaces": ["SHAP"],
        "body": "SHapley Additive exPlanations — a game-theoretic feature-importance "
                "score for a fitted model. In this project SHAP importance on zigzag "
                "direction-classification proved orthogonal-to-negatively-correlated "
                "with real trading edge: features ranked #1 by SHAP were dead-last by "
                "out-of-sample P&L, because classifying direction is a different "
                "quantity from knowing when to enter net of spread.",
    },
]

# surface-alias -> the display name of an existing entry it should resolve to.
# Matched case-insensitively against parsed entries by slug of the target.
MANUAL_ALIASES: dict[str, str] = {
    "BB": "Bollinger Bands",
    "bb_width": "BB Width",
    "BBW": "BB Width",
    "BB Width": "BB Width",
    "Bollinger Band Width": "BB Width",
    "TEC_5": "TEC",
    "dTEC": "TEC",
    "sb_a": "SB-A",
    "sb_p": "SB-A",
    "range_pos_30": "Range Position",
    "macd_hist": "MACD",
    "mc_p": "Monte-Carlo (MC) gate",
    "MC p-value": "Monte-Carlo (MC) gate",
    "MC p": "Monte-Carlo (MC) gate",
    "MC gate": "Monte-Carlo (MC) gate",
    "Kalman10": "Kalman filter",
    "compute_mc_on_series": "Lookahead bias",
    "er_norm": "Efficiency Ratio",
    "mc_d": "ASIMC",
    "mc_dd": "ASIMC",
}


# ───────────────────────────────────────────────────────── assemble ──

# Surface forms that must never be a *marking* surface: ubiquitous timeframe
# codes and generic tokens that get scraped out of parentheticals (e.g. a row
# titled "TBP Relative (M5)" would otherwise claim every "M5" in the prose).
# The concepts stay reachable through their real entry ("Timeframe"); we only
# forbid them as hover triggers so the panes don't fill with tooltip soup.
SURFACE_STOPLIST = {
    "m1", "m5", "m15", "m30", "h1", "h4", "s5", "d1", "w1", "mn1", "d",
    "time", "dd", "any", "runtime", "various", "price", "pips",
}


def assemble(md: str) -> dict:
    entries = parse_glossary(md) + parse_encyclopedia(md)
    for e in entries:
        e["surfaces"] = [s for s in e["surfaces"] if s.lower() not in SURFACE_STOPLIST]

    # merge by slug: first writer (glossary) wins the body/tooltip; later
    # entries only contribute extra surface forms.
    by_slug: dict[str, dict] = {}
    for e in entries:
        s = e["slug"]
        if s not in by_slug:
            by_slug[s] = e
        else:
            existing = by_slug[s]
            for f in e["surfaces"]:
                if f.lower() not in {x.lower() for x in existing["surfaces"]}:
                    existing["surfaces"].append(f)

    # manual standalone terms
    for mt in MANUAL_TERMS:
        s = slugify(mt["display"])
        if s not in by_slug:
            by_slug[s] = {
                "slug": s, "display": mt["display"],
                "surfaces": list(mt["surfaces"]),
                "tooltip": first_sentence(mt["body"]),
                "body": mt["body"], "source": "manual",
            }

    # alias injection: attach surface -> target entry. Resolve the target
    # robustly: exact slug, exact display, existing surface, then display
    # prefix — because display names carry parentheticals (e.g. the entry for
    # "TEC" is displayed "TEC (Trend-Efficiency Coefficient)").
    def _resolve(target: str):
        t = target.lower()
        if slugify(target) in by_slug:
            return by_slug[slugify(target)]
        for e in by_slug.values():
            if e["display"].lower() == t:
                return e
        for e in by_slug.values():
            if t in {x.lower() for x in e["surfaces"]}:
                return e
        for e in by_slug.values():
            if e["display"].lower().startswith(t):
                return e
        return None

    for surface, target_display in MANUAL_ALIASES.items():
        tgt = _resolve(target_display)
        if tgt is None:
            continue
        if surface.lower() not in {x.lower() for x in tgt["surfaces"]}:
            tgt["surfaces"].append(surface)

    # global surface de-dup: a surface form may be claimed by only ONE slug for
    # marking. Priority = source (glossary > encyclopedia > manual), then the
    # order terms were added. Keep the surface on the first claimant; strip it
    # from later ones (they keep their remaining surfaces).
    priority = {"glossary": 0, "encyclopedia": 1, "manual": 2}
    ordered = sorted(by_slug.values(), key=lambda e: (priority.get(e["source"], 9),))
    claimed: dict[str, str] = {}       # surface_lower -> slug
    for e in ordered:
        kept = []
        for f in e["surfaces"]:
            k = f.lower()
            if k in claimed and claimed[k] != e["slug"]:
                continue               # already owned by a higher-priority term
            claimed[k] = e["slug"]
            kept.append(f)
        e["surfaces"] = kept

    terms = sorted(by_slug.values(), key=lambda e: e["display"].lower())
    return {
        "generated": _dt.date.today().isoformat(),
        "terms": terms,
    }


# ───────────────────────────────────────────────────── html builder ──

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Edge That Wasn't — Glossary</title>
<style>
:root {{ --bg:#fafaf8; --fg:#1a1a1a; --muted:#6b6b6b; --card:#fff; --line:#e2e0da; --acc:#0f6b4f; --chip:#eef2ee; }}
@media (prefers-color-scheme: dark) {{
 :root {{ --bg:#161614; --fg:#e8e6e1; --muted:#9a988f; --card:#201f1c; --line:#37352f; --acc:#5ec89f; --chip:#2a2e2a; }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ font:16px/1.6 Georgia,'Palatino Linotype',serif; background:var(--bg); color:var(--fg); }}
header {{ padding:16px 24px 12px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; position:sticky; top:0; background:var(--bg); z-index:5; }}
header h1 {{ font-size:1.2rem; font-weight:600; }}
header a.home {{ color:var(--acc); text-decoration:none; font-size:.85rem; }}
header a.home:hover {{ text-decoration:underline; }}
#q {{ margin-left:auto; padding:7px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); font:inherit; font-size:.9rem; min-width:220px; }}
main {{ max-width:820px; margin:0 auto; padding:20px 24px 80px; }}
.count {{ color:var(--muted); font-size:.85rem; margin:6px 0 18px; }}
.jump {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:22px; }}
.jump a {{ font:600 .78rem/1 ui-monospace,Menlo,monospace; color:var(--acc); text-decoration:none; padding:3px 6px; border:1px solid var(--line); border-radius:5px; }}
.jump a:hover {{ background:var(--chip); }}
.term {{ padding:14px 0 12px; border-bottom:1px solid var(--line); scroll-margin-top:70px; }}
.term.hidden, .letter.hidden {{ display:none; }}
.term dt {{ font-weight:700; font-size:1.05rem; }}
.term dt .src {{ font:400 .7rem/1 ui-monospace,Menlo,monospace; color:var(--muted); margin-left:8px; vertical-align:middle; text-transform:uppercase; letter-spacing:.05em; }}
.term dt .alias {{ font:400 .8rem/1 ui-monospace,Menlo,monospace; color:var(--muted); margin-left:8px; }}
.term dd {{ margin:5px 0 0; color:var(--fg); }}
.letter {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:26px 0 2px; border-top:1px solid var(--line); padding-top:10px; }}
#none {{ color:var(--muted); padding:30px 0; display:none; }}
</style>
</head>
<body>
<header>
 <h1>Glossary</h1>
 <a class="home" href="index.html">&larr; Back to the explorer</a>
 <input id="q" type="search" placeholder="Filter terms…" autocomplete="off">
</header>
<main>
 <div class="count">{count} terms — indicators, formulas, reward functions, statistics, and the project's own coinages. Hover a term anywhere in the explorer to preview; click through to here.</div>
 <div class="jump">{jump}</div>
 <dl id="glist">
{items}
 </dl>
 <div id="none">No term matches that filter.</div>
</main>
<script>
const q=document.getElementById('q'), none=document.getElementById('none');
const nodes=[...document.getElementById('glist').children];   // letters + terms in order
q.addEventListener('input',()=>{{
  const v=q.value.trim().toLowerCase(); let shown=0;
  // first pass: show/hide terms
  nodes.forEach(n=>{{
    if(!n.classList.contains('term'))return;
    const hit=!v||n.dataset.k.includes(v);
    n.classList.toggle('hidden',!hit); if(hit)shown++;
  }});
  // second pass: a letter header is visible only if a visible term follows it
  // before the next letter header
  let pendingLetter=null, letterHasVisible=false;
  const flush=()=>{{ if(pendingLetter)pendingLetter.classList.toggle('hidden',!letterHasVisible); }};
  nodes.forEach(n=>{{
    if(n.classList.contains('letter')){{ flush(); pendingLetter=n; letterHasVisible=false; }}
    else if(!n.classList.contains('hidden')){{ letterHasVisible=true; }}
  }});
  flush();
  none.style.display=shown?'none':'block';
}});
// deep-link highlight
if(location.hash){{ const el=document.querySelector(location.hash); if(el){{ el.style.background='var(--chip)'; el.scrollIntoView(); }} }}
</script>
</body>
</html>
"""


def build_html(data: dict) -> str:
    terms = data["terms"]
    # jump bar of first letters present
    firsts = []
    for t in terms:
        c = t["display"][0].upper()
        c = c if c.isalpha() else "#"
        if c not in firsts:
            firsts.append(c)
    firsts_sorted = sorted(firsts, key=lambda c: (c == "#", c))
    jump = "".join(f'<a href="#letter-{c if c!="#" else "sym"}">{c}</a>' for c in firsts_sorted)

    items: list[str] = []
    cur_letter = None
    for t in terms:
        c = t["display"][0].upper()
        c = c if c.isalpha() else "#"
        if c != cur_letter:
            cur_letter = c
            anchor = c if c != "#" else "sym"
            items.append(f'  <div class="letter" id="letter-{anchor}">{c}</div>')
        aliases = [s for s in t["surfaces"] if s.lower() != t["display"].lower()]
        alias_html = (f'<span class="alias">{_h(", ".join(aliases))}</span>'
                      if aliases else "")
        key = _h((t["display"] + " " + " ".join(t["surfaces"]) + " " + t["body"]).lower())
        items.append(
            f'  <div class="term" id="{t["slug"]}" data-k="{key}">'
            f'<dt>{_h(t["display"])}{alias_html}'
            f'<span class="src">{t["source"]}</span></dt>'
            f'<dd>{_h(t["body"])}</dd></div>'
        )
    return HTML.format(count=len(terms), jump=jump, items="\n".join(items))


def _h(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


# ─────────────────────────────────────────────────────────── cli ──

def cmd_extract(args) -> None:
    md = Path(args.source).read_text(encoding="utf-8")
    data = assemble(md)
    Path(args.out).write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    n_surf = sum(len(t["surfaces"]) for t in data["terms"])
    print(f"Wrote {args.out}: {len(data['terms'])} terms, {n_surf} surface forms.")


def cmd_build(args) -> None:
    data = json.load(Path(args.json).open(encoding="utf-8"))
    Path(args.out).write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {args.out}: {len(data['terms'])} terms.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="parse the book md -> glossary.json")
    ex.add_argument("--source", required=True)
    ex.add_argument("--out", default=str(DEFAULT_JSON))
    ex.set_defaults(func=cmd_extract)

    bl = sub.add_parser("build", help="glossary.json -> glossary.html")
    bl.add_argument("--json", default=str(DEFAULT_JSON))
    bl.add_argument("--out", default=str(DEFAULT_HTML))
    bl.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
