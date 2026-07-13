/* ═══════════════════════════════════════════════════════════════════════
   BookChart — the in-house canvas chart engine for The Edge That Wasn't.
   No dependencies. Canvas 2D, devicePixelRatio-aware, light book palette.

   Usage:
     const chart = new BookChart(container, {
       candles: [[unixSec, open, high, low, close, volume], ...],
       priceStyle: 'line',         // optional: close-only line instead of candles
       overlays: [ ... ],          // optional, see setOverlays
       panel:    { ... } | null,   // optional, see setPanel
     });

   Overlay spec (one object per active overlay):
     { name:'SMA(20)', color:'#2f5d8a', values: Float64Array }        line
     { name:'PSAR',    color:'#6d4c9f', values: arr, style:'dots' }   dots
     { name:'Bollinger(20,2)', color:'#2f5d8a',
       bands: { upper: arr, lower: arr, mid: arr|undefined } }        band
   All value arrays are index-aligned with candles; NaN = gap.

   Panel spec:
     { series: [ { name:'RSI(14)', color:'#a8742a', values: arr }, ... ],
       levels: [30, 70] }          // optional dashed horizontal levels

   Marker spec (per-bar glyphs on the main pane — swing points, signals):
     { i: barIndex, price: anchorValue, dy: pxOffset,
       shape: 'circle'|'square'|'triangle-up'|'triangle-down'|'x',
       color:'#a23b2c', size: 4 }

   Methods: setOverlays(list), setPanel(specOrNull), setMarkers(list),
            resize(), destroy()
   ═══════════════════════════════════════════════════════════════════════ */
(function (global) {
"use strict";

const PAL = {
  paneBg:   "#ffffff",
  grid:     "#f0eee8",
  axis:     "#d8d5cd",
  sep:      "#e2e0da",
  text:     "#615f58",
  strong:   "#1a1a1a",
  up:       "#0f6b4f",
  down:     "#a23b2c",
  cross:    "#8b887e",
  level:    "#b9b5aa",
  readBg:   "rgba(255,255,254,0.93)",
  readEdge: "#e2e0da",
};
const MONO  = "ui-monospace,Menlo,Consolas,monospace";
const AXIS_W = 60;          // right price axis width (px)
const TAXIS_H = 24;         // bottom time axis height (px)
const PANEL_FRAC = 0.30;    // panel share of plot height when visible
const MIN_BARS = 12;

let cssInjected = false;
function injectCSS() {
  if (cssInjected) return;
  cssInjected = true;
  const st = document.createElement("style");
  st.textContent =
    ".bookchart{position:relative;overflow:hidden}" +
    ".bookchart canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:crosshair}" +
    ".bc-zoom{position:absolute;right:" + (AXIS_W + 8) + "px;bottom:" + (TAXIS_H + 8) + "px;display:flex;flex-direction:row;gap:6px;z-index:5}" +
    ".bc-zoom button{width:34px;height:34px;border-radius:8px;border:1px solid " + PAL.sep + ";" +
      "background:rgba(255,255,255,.92);color:" + PAL.text + ";font:700 18px/1 Georgia,serif;" +
      "cursor:pointer;padding:0;-webkit-tap-highlight-color:transparent}" +
    ".bc-zoom button:hover{color:" + PAL.up + ";border-color:" + PAL.up + "}";
  document.head.appendChild(st);
}

function niceStep(raw) {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  let s;
  if (n <= 1) s = 1; else if (n <= 2) s = 2; else if (n <= 2.5) s = 2.5;
  else if (n <= 5) s = 5; else s = 10;
  return s * mag;
}
function stepDecimals(step) {
  if (step >= 1) return 0;
  return Math.min(8, Math.max(0, Math.ceil(-Math.log10(step) - 1e-9)));
}
function fmtVal(v) {
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100)  return v.toFixed(1);
  if (a >= 10)   return v.toFixed(2);
  if (a >= 1)    return v.toFixed(3);
  return v.toFixed(4);
}
const p2 = n => (n < 10 ? "0" + n : "" + n);
function fmtTime(sec, withDate, withTime) {
  const d = new Date(sec * 1000);
  const date = p2(d.getUTCMonth() + 1) + "-" + p2(d.getUTCDate());
  const time = p2(d.getUTCHours()) + ":" + p2(d.getUTCMinutes());
  if (withDate && withTime) return date + " " + time;
  return withDate ? date : time;
}
function fmtFull(sec) {
  const d = new Date(sec * 1000);
  return d.getUTCFullYear() + "-" + p2(d.getUTCMonth() + 1) + "-" + p2(d.getUTCDate()) +
    " " + p2(d.getUTCHours()) + ":" + p2(d.getUTCMinutes());
}
function hexA(hex, a) {   // '#rrggbb' + alpha 0..1 → rgba()
  const r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
  return "rgba(" + r + "," + g + "," + b + "," + a + ")";
}

function BookChart(container, opts) {
  if (!(this instanceof BookChart)) return new BookChart(container, opts);
  opts = opts || {};
  injectCSS();
  this.container = container;
  container.classList.add("bookchart");

  this.canvas = document.createElement("canvas");
  container.appendChild(this.canvas);
  this.ctx = this.canvas.getContext("2d");

  this._setCandles(opts.candles || []);
  this.lineMode = opts.priceStyle === "line";
  this.overlays = opts.overlays || [];
  this.panel = opts.panel || null;
  this.markers = opts.markers || [];

  // view: start index (float) + number of visible bars (float)
  const initBars = Math.min(this.N, 260);
  this.view = { start: Math.max(0, this.N - initBars), count: Math.max(MIN_BARS, initBars) };

  this.cross = null;         // {x, y, i} crosshair state
  this._pointers = new Map();
  this._pinch = null;

  this._buildZoomButtons();
  this._bindEvents();

  this._ro = (typeof ResizeObserver !== "undefined")
    ? new ResizeObserver(() => this.resize()) : null;
  if (this._ro) this._ro.observe(container);
  this._onWinResize = () => this.resize();
  window.addEventListener("resize", this._onWinResize);

  this.resize();
}

BookChart.prototype._setCandles = function (rows) {
  const N = rows.length;
  this.N = N;
  this.T = new Float64Array(N); this.O = new Float64Array(N);
  this.H = new Float64Array(N); this.L = new Float64Array(N);
  this.C = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const r = rows[i];
    this.T[i] = r[0]; this.O[i] = r[1]; this.H[i] = r[2];
    this.L[i] = r[3]; this.C[i] = r[4];
  }
  // price decimals for readout (JPY-style vs 5-digit pairs)
  this.priceDec = (N && this.C[N - 1] >= 20) ? 3 : 5;
};

BookChart.prototype.setOverlays = function (list) {
  this.overlays = list || [];
  this.draw();
};
BookChart.prototype.setPanel = function (spec) {
  this.panel = spec || null;
  this.resize();
};
BookChart.prototype.setMarkers = function (list) {
  this.markers = list || [];
  this.draw();
};
BookChart.prototype.destroy = function () {
  if (this._ro) this._ro.disconnect();
  window.removeEventListener("resize", this._onWinResize);
  this.container.innerHTML = "";
};

/* ── layout ─────────────────────────────────────────────────────────── */
BookChart.prototype.resize = function () {
  const dpr = window.devicePixelRatio || 1;
  const w = this.container.clientWidth, h = this.container.clientHeight;
  if (!w || !h) return;
  this.W = w; this.Hh = h;
  this.canvas.width = Math.round(w * dpr);
  this.canvas.height = Math.round(h * dpr);
  this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  this.plotL = 0;
  this.plotR = w - AXIS_W;
  const plotH = h - TAXIS_H;
  if (this.panel) {
    this.mainTop = 0;
    this.mainH = Math.round(plotH * (1 - PANEL_FRAC));
    this.panTop = this.mainH + 1;
    this.panH = plotH - this.mainH - 1;
  } else {
    this.mainTop = 0; this.mainH = plotH;
    this.panTop = 0; this.panH = 0;
  }
  this.draw();
};

/* ── view helpers ───────────────────────────────────────────────────── */
BookChart.prototype._clampView = function () {
  const v = this.view;
  v.count = Math.max(MIN_BARS, Math.min(this.N, v.count));
  const maxStart = this.N - v.count;
  v.start = Math.max(0, Math.min(maxStart, v.start));
};
BookChart.prototype._barW = function () {
  return (this.plotR - this.plotL) / this.view.count;
};
BookChart.prototype._xOf = function (i) {
  return this.plotL + (i - this.view.start + 0.5) * this._barW();
};
BookChart.prototype._iOf = function (x) {
  return this.view.start + (x - this.plotL) / this._barW() - 0.5;
};
BookChart.prototype.zoom = function (factor, anchorX) {
  const v = this.view;
  if (anchorX == null) anchorX = (this.plotL + this.plotR) / 2;
  const anchorBar = this._iOf(anchorX) + 0.5;
  const newCount = Math.max(MIN_BARS, Math.min(this.N, v.count * factor));
  v.start = anchorBar - (anchorBar - v.start) * (newCount / v.count);
  v.count = newCount;
  this._clampView();
  this.draw();
};
BookChart.prototype.panPx = function (dx) {
  this.view.start -= dx / this._barW();
  this._clampView();
  this.draw();
};

/* ── zoom buttons ───────────────────────────────────────────────────── */
BookChart.prototype._buildZoomButtons = function () {
  const box = document.createElement("div");
  box.className = "bc-zoom";
  const mk = (label, f) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-label", label === "+" ? "Zoom in" : "Zoom out");
    b.addEventListener("click", e => { e.preventDefault(); this.zoom(f); });
    box.appendChild(b);
  };
  mk("+", 1 / 1.35);
  mk("−", 1.35);
  this.container.appendChild(box);
};

/* ── events ─────────────────────────────────────────────────────────── */
BookChart.prototype._bindEvents = function () {
  const cv = this.canvas;

  cv.addEventListener("wheel", e => {
    e.preventDefault();
    const rect = cv.getBoundingClientRect();
    this.zoom(e.deltaY > 0 ? 1.12 : 1 / 1.12, e.clientX - rect.left);
  }, { passive: false });

  cv.addEventListener("pointerdown", e => {
    cv.setPointerCapture(e.pointerId);
    const rect = cv.getBoundingClientRect();
    this._pointers.set(e.pointerId, {
      x: e.clientX - rect.left, y: e.clientY - rect.top,
      x0: e.clientX - rect.left, y0: e.clientY - rect.top,
      moved: false, type: e.pointerType,
    });
    if (this._pointers.size === 2) {
      const pts = [...this._pointers.values()];
      this._pinch = { d0: Math.abs(pts[0].x - pts[1].x) || 1, count0: this.view.count };
    }
  });

  cv.addEventListener("pointermove", e => {
    const rect = cv.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const p = this._pointers.get(e.pointerId);

    if (!p) {                                   // hover (mouse, no button)
      this._setCross(x, y);
      return;
    }
    if (this._pointers.size === 2) {            // pinch zoom
      p.x = x; p.y = y;
      const pts = [...this._pointers.values()];
      const d = Math.abs(pts[0].x - pts[1].x) || 1;
      const cx = (pts[0].x + pts[1].x) / 2;
      const v = this.view;
      const anchorBar = this._iOf(cx) + 0.5;
      const newCount = Math.max(MIN_BARS, Math.min(this.N, this._pinch.count0 * (this._pinch.d0 / d)));
      v.start = anchorBar - (anchorBar - v.start) * (newCount / v.count);
      v.count = newCount;
      this._clampView();
      this.draw();
      return;
    }
    const dx = x - p.x;
    p.x = x; p.y = y;
    if (Math.abs(x - p.x0) > 4 || Math.abs(y - p.y0) > 4) p.moved = true;
    if (p.moved) {
      this.cross = null;
      this.panPx(dx);
    }
  });

  const up = e => {
    const p = this._pointers.get(e.pointerId);
    this._pointers.delete(e.pointerId);
    if (this._pointers.size < 2) this._pinch = null;
    if (p && !p.moved && p.type !== "mouse") this._setCross(p.x0, p.y0);  // tap → crosshair
  };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);

  cv.addEventListener("pointerleave", e => {
    // touch pointers "leave" right after every tap — only mouse hides the crosshair
    if (e.pointerType === "mouse" && this._pointers.size === 0) { this.cross = null; this.draw(); }
  });
};

BookChart.prototype._setCross = function (x, y) {
  if (x < this.plotL || x > this.plotR) { this.cross = null; this.draw(); return; }
  let i = Math.round(this._iOf(x));
  i = Math.max(0, Math.min(this.N - 1, i));
  this.cross = { x, y, i };
  this.draw();
};

/* ── scales ─────────────────────────────────────────────────────────── */
BookChart.prototype._visRange = function () {
  const i0 = Math.max(0, Math.floor(this.view.start));
  const i1 = Math.min(this.N - 1, Math.ceil(this.view.start + this.view.count));
  return [i0, i1];
};

function scanMinMax(arr, i0, i1, mm) {
  for (let i = i0; i <= i1; i++) {
    const v = arr[i];
    if (v === v && isFinite(v)) {       // not NaN
      if (v < mm[0]) mm[0] = v;
      if (v > mm[1]) mm[1] = v;
    }
  }
}

BookChart.prototype._mainScale = function () {
  const [i0, i1] = this._visRange();
  const mm = [Infinity, -Infinity];
  scanMinMax(this.L, i0, i1, mm);
  scanMinMax(this.H, i0, i1, mm);
  for (const ov of this.overlays) {
    if (ov.values) scanMinMax(ov.values, i0, i1, mm);
    if (ov.bands) {
      scanMinMax(ov.bands.upper, i0, i1, mm);
      scanMinMax(ov.bands.lower, i0, i1, mm);
    }
  }
  if (!isFinite(mm[0])) { mm[0] = 0; mm[1] = 1; }
  if (mm[0] === mm[1]) { mm[0] -= 0.5; mm[1] += 0.5; }
  const pad = (mm[1] - mm[0]) * 0.05;
  return { lo: mm[0] - pad, hi: mm[1] + pad, top: this.mainTop, h: this.mainH };
};

BookChart.prototype._panelScale = function () {
  const [i0, i1] = this._visRange();
  const mm = [Infinity, -Infinity];
  for (const s of this.panel.series) scanMinMax(s.values, i0, i1, mm);
  if (this.panel.levels) for (const lv of this.panel.levels) {
    if (lv < mm[0]) mm[0] = lv;
    if (lv > mm[1]) mm[1] = lv;
  }
  if (!isFinite(mm[0])) { mm[0] = 0; mm[1] = 1; }
  if (mm[0] === mm[1]) { mm[0] -= 0.5; mm[1] += 0.5; }
  const pad = (mm[1] - mm[0]) * 0.08;
  return { lo: mm[0] - pad, hi: mm[1] + pad, top: this.panTop, h: this.panH };
};

function yOf(scale, v) {
  return scale.top + (scale.hi - v) / (scale.hi - scale.lo) * scale.h;
}

/* ── drawing ────────────────────────────────────────────────────────── */
BookChart.prototype.draw = function () {
  const ctx = this.ctx;
  if (!this.W) return;
  ctx.clearRect(0, 0, this.W, this.Hh);
  ctx.fillStyle = PAL.paneBg;
  ctx.fillRect(0, 0, this.W, this.Hh);
  if (!this.N) return;

  const main = this._mainScale();
  const pan = this.panel ? this._panelScale() : null;

  this._drawTimeAxis(main, pan);
  this._drawPane(main, pan);
  if (pan) this._drawPanel(pan);
  this._drawCrosshair(main, pan);
  this._drawReadout(main, pan);
};

BookChart.prototype._drawYAxis = function (scale) {
  const ctx = this.ctx;
  const targetTicks = Math.max(2, Math.floor(scale.h / 46));
  const step = niceStep((scale.hi - scale.lo) / targetTicks);
  const dec = stepDecimals(step);
  const first = Math.ceil(scale.lo / step) * step;
  ctx.font = "10px " + MONO;
  ctx.fillStyle = PAL.text;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = PAL.grid;
  ctx.lineWidth = 1;
  for (let v = first; v <= scale.hi + 1e-12; v += step) {
    const y = Math.round(yOf(scale, v)) + 0.5;
    if (y < scale.top + 8 || y > scale.top + scale.h - 4) continue;
    ctx.beginPath();
    ctx.moveTo(this.plotL, y);
    ctx.lineTo(this.plotR, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(dec), this.plotR + 6, y);
  }
};

BookChart.prototype._drawTimeAxis = function () {
  const ctx = this.ctx;
  const [i0, i1] = this._visRange();
  const barW = this._barW();
  const span = this.T[i1] - this.T[i0];
  const secPerBar = i1 > i0 ? span / (i1 - i0) : 300;
  const withDate = span > 20 * 3600;
  // decide by tick interval, not total span, so intraday ticks across
  // a multi-day window still show the time instead of a repeated date
  let stepBars = Math.max(1, Math.round(56 / barW));
  const withTime = stepBars * secPerBar < 0.75 * 86400;
  if (withDate && withTime) stepBars = Math.max(1, Math.round(82 / barW));
  const yTop = this.Hh - TAXIS_H;

  ctx.strokeStyle = PAL.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, yTop + 0.5);
  ctx.lineTo(this.W, yTop + 0.5);
  // right price-axis separator
  ctx.moveTo(this.plotR + 0.5, 0);
  ctx.lineTo(this.plotR + 0.5, yTop);
  ctx.stroke();

  ctx.font = "10px " + MONO;
  ctx.fillStyle = PAL.text;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = PAL.grid;
  const startTick = Math.ceil(i0 / stepBars) * stepBars;
  for (let i = startTick; i <= i1; i += stepBars) {
    const x = Math.round(this._xOf(i)) + 0.5;
    if (x < this.plotL || x > this.plotR) continue;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, yTop);
    ctx.stroke();
    ctx.fillText(fmtTime(this.T[i], withDate, withTime), x, yTop + TAXIS_H / 2 + 1);
  }
};

BookChart.prototype._drawPane = function (main) {
  const ctx = this.ctx;
  const [i0, i1] = this._visRange();
  const barW = this._barW();

  this._drawYAxis(main);

  ctx.save();
  ctx.beginPath();
  ctx.rect(this.plotL, main.top, this.plotR - this.plotL, main.h);
  ctx.clip();

  // band fills (behind candles)
  for (const ov of this.overlays) {
    if (!ov.bands) continue;
    const { upper, lower } = ov.bands;
    ctx.fillStyle = hexA(ov.color || "#2f5d8a", 0.08);
    let started = false;
    ctx.beginPath();
    let segStart = -1;
    for (let i = i0; i <= i1 + 1; i++) {
      const ok = i <= i1 && isFinite(upper[i]) && isFinite(lower[i]) && upper[i] === upper[i] && lower[i] === lower[i];
      if (ok && !started) { started = true; segStart = i; ctx.moveTo(this._xOf(i), yOf(main, upper[i])); }
      else if (ok) ctx.lineTo(this._xOf(i), yOf(main, upper[i]));
      if (!ok && started) {  // close segment back along lower
        for (let j = i - 1; j >= segStart; j--) ctx.lineTo(this._xOf(j), yOf(main, lower[j]));
        ctx.closePath();
        started = false;
      }
    }
    ctx.fill();
  }

  // price — close-only line (priceStyle:'line') or candles
  if (this.lineMode) {
    this._strokeLine(this.C, i0, i1, main, PAL.strong, 1.6, null);
  } else {
  const bodyW = Math.min(13, Math.max(1, barW * 0.65));
  const thin = barW < 1.6;
  for (let i = i0; i <= i1; i++) {
    const x = this._xOf(i);
    const up = this.C[i] >= this.O[i];
    ctx.strokeStyle = ctx.fillStyle = up ? PAL.up : PAL.down;
    const yH = yOf(main, this.H[i]), yL = yOf(main, this.L[i]);
    const yO = yOf(main, this.O[i]), yC = yOf(main, this.C[i]);
    if (thin) {
      ctx.beginPath();
      ctx.moveTo(x, yH); ctx.lineTo(x, yL);
      ctx.stroke();
    } else {
      const xw = Math.round(x) + 0.5;
      ctx.beginPath();
      ctx.moveTo(xw, yH); ctx.lineTo(xw, yL);
      ctx.stroke();
      const top = Math.min(yO, yC), hgt = Math.max(1, Math.abs(yC - yO));
      ctx.fillRect(x - bodyW / 2, top, bodyW, hgt);
    }
  }
  }

  // overlay lines / dots / band edges
  for (const ov of this.overlays) {
    const color = ov.color || "#2f5d8a";
    if (ov.bands) {
      this._strokeLine(ov.bands.upper, i0, i1, main, color, 1, null);
      this._strokeLine(ov.bands.lower, i0, i1, main, color, 1, null);
      if (ov.bands.mid) this._strokeLine(ov.bands.mid, i0, i1, main, color, 1, [4, 3]);
    } else if (ov.style === "dots") {
      ctx.fillStyle = color;
      const r = Math.min(2.4, Math.max(1, barW * 0.14));
      for (let i = i0; i <= i1; i++) {
        const v = ov.values[i];
        if (v !== v || !isFinite(v)) continue;
        ctx.beginPath();
        ctx.arc(this._xOf(i), yOf(main, v), r, 0, 6.2832);
        ctx.fill();
      }
    } else if (ov.values) {
      this._strokeLine(ov.values, i0, i1, main, color, 1.5, null);
    }
  }

  // markers (swing points, trade signals) — on top of candles and overlays
  const mkScale = Math.min(1, Math.max(0.45, barW / 7));   // shrink when zoomed out
  for (const mk of this.markers) {
    if (mk.i < i0 || mk.i > i1) continue;
    const v = mk.price;
    if (v !== v || !isFinite(v)) continue;
    const x = this._xOf(mk.i);
    const y = yOf(main, v) + (mk.dy || 0) * mkScale;
    const s = (mk.size || 4) * mkScale;
    ctx.fillStyle = ctx.strokeStyle = mk.color || PAL.strong;
    switch (mk.shape) {
      case "square":
        ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
        break;
      case "triangle-up":
        ctx.beginPath();
        ctx.moveTo(x, y - s); ctx.lineTo(x - s, y + s); ctx.lineTo(x + s, y + s);
        ctx.closePath(); ctx.fill();
        break;
      case "triangle-down":
        ctx.beginPath();
        ctx.moveTo(x, y + s); ctx.lineTo(x - s, y - s); ctx.lineTo(x + s, y - s);
        ctx.closePath(); ctx.fill();
        break;
      case "x":
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
        ctx.stroke();
        break;
      default:  // circle
        ctx.beginPath();
        ctx.arc(x, y, s, 0, 6.2832);
        ctx.fill();
    }
  }
  ctx.restore();
};

BookChart.prototype._strokeLine = function (arr, i0, i1, scale, color, width, dash) {
  const ctx = this.ctx;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  let pen = false;
  for (let i = i0; i <= i1; i++) {
    const v = arr[i];
    if (v !== v || !isFinite(v)) { pen = false; continue; }
    const x = this._xOf(i), y = yOf(scale, v);
    if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
  }
  ctx.stroke();
  ctx.setLineDash([]);
};

BookChart.prototype._drawPanel = function (pan) {
  const ctx = this.ctx;
  const [i0, i1] = this._visRange();

  // separator
  ctx.strokeStyle = PAL.sep;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, pan.top - 0.5);
  ctx.lineTo(this.W, pan.top - 0.5);
  ctx.stroke();

  this._drawYAxis(pan);

  ctx.save();
  ctx.beginPath();
  ctx.rect(this.plotL, pan.top, this.plotR - this.plotL, pan.h);
  ctx.clip();

  if (this.panel.levels) {
    for (const lv of this.panel.levels) {
      const y = Math.round(yOf(pan, lv)) + 0.5;
      ctx.strokeStyle = PAL.level;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(this.plotL, y);
      ctx.lineTo(this.plotR, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  for (const s of this.panel.series) {
    this._strokeLine(s.values, i0, i1, pan, s.color || "#a8742a", s.muted ? 1 : 1.5, s.muted ? [3, 3] : null);
  }
  ctx.restore();
};

BookChart.prototype._drawCrosshair = function (main, pan) {
  if (!this.cross) return;
  const ctx = this.ctx;
  const { y, i } = this.cross;
  const x = Math.round(this._xOf(i)) + 0.5;
  const plotBot = this.Hh - TAXIS_H;
  if (x < this.plotL || x > this.plotR) return;

  ctx.strokeStyle = PAL.cross;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, plotBot);
  ctx.stroke();

  // horizontal line in hovered pane + price tag
  let scale = null;
  if (y >= main.top && y <= main.top + main.h) scale = main;
  else if (pan && y >= pan.top && y <= pan.top + pan.h) scale = pan;
  if (scale) {
    const yy = Math.round(y) + 0.5;
    ctx.beginPath();
    ctx.moveTo(this.plotL, yy);
    ctx.lineTo(this.plotR, yy);
    ctx.stroke();
    ctx.setLineDash([]);
    const v = scale.hi - (y - scale.top) / scale.h * (scale.hi - scale.lo);
    const label = scale === main ? v.toFixed(this.priceDec) : fmtVal(v);
    ctx.font = "10px " + MONO;
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = PAL.strong;
    ctx.fillRect(this.plotR + 1, yy - 8, Math.max(tw, AXIS_W - 2), 16);
    ctx.fillStyle = "#fafaf8";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, this.plotR + 5, yy + 0.5);
  }
  ctx.setLineDash([]);

  // time tag
  const tl = fmtFull(this.T[i]);
  ctx.font = "10px " + MONO;
  const tw2 = ctx.measureText(tl).width + 10;
  let tx = Math.min(Math.max(x - tw2 / 2, 2), this.W - tw2 - 2);
  ctx.fillStyle = PAL.strong;
  ctx.fillRect(tx, plotBot + 1, tw2, TAXIS_H - 3);
  ctx.fillStyle = "#fafaf8";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(tl, tx + tw2 / 2, plotBot + TAXIS_H / 2);
};

BookChart.prototype._drawReadout = function (main, pan) {
  const ctx = this.ctx;
  const i = this.cross ? this.cross.i : this.N - 1;
  const dec = this.priceDec;
  const rows = [];
  rows.push({ text: fmtFull(this.T[i]) + " UTC", color: PAL.text, bold: false });
  if (this.lineMode) {
    rows.push({ text: "Close " + this.C[i].toFixed(dec), color: PAL.strong, bold: true });
  } else {
    const up = this.C[i] >= this.O[i];
    rows.push({
      text: "O " + this.O[i].toFixed(dec) + "  H " + this.H[i].toFixed(dec) +
            "  L " + this.L[i].toFixed(dec) + "  C " + this.C[i].toFixed(dec),
      color: up ? PAL.up : PAL.down, bold: true,
    });
  }
  for (const ov of this.overlays) {
    let txt;
    if (ov.bands) {
      const u = ov.bands.upper[i], l = ov.bands.lower[i];
      txt = (u === u && isFinite(u)) ? (u.toFixed(dec) + " / " + l.toFixed(dec)) : "—";
    } else {
      const v = ov.values[i];
      txt = (v === v && isFinite(v)) ? v.toFixed(dec) : "—";
    }
    rows.push({ text: ov.name + "  " + txt, color: ov.color || PAL.text, dot: true });
  }
  if (pan && this.panel) {
    for (const s of this.panel.series) {
      const v = s.values[i];
      rows.push({
        text: s.name + "  " + ((v === v && isFinite(v)) ? fmtVal(v) : "—"),
        color: s.color || PAL.text, dot: true,
      });
    }
  }

  ctx.font = "11px " + MONO;
  let wMax = 0;
  for (const r of rows) wMax = Math.max(wMax, ctx.measureText(r.text).width + (r.dot ? 12 : 0));
  const pad = 8, lineH = 16;
  const bw = wMax + pad * 2, bh = rows.length * lineH + pad * 2 - 4;
  const bx = this.plotL + 8, by = this.mainTop + 8;

  ctx.fillStyle = PAL.readBg;
  ctx.strokeStyle = PAL.readEdge;
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 6);
  else ctx.rect(bx, by, bw, bh);
  ctx.fill();
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  let yy = by + pad + 5;
  for (const r of rows) {
    let xx = bx + pad;
    if (r.dot) {
      ctx.fillStyle = r.color;
      ctx.beginPath();
      ctx.arc(xx + 3, yy, 3, 0, 6.2832);
      ctx.fill();
      xx += 12;
    }
    ctx.font = (r.bold ? "700 " : "") + "11px " + MONO;
    ctx.fillStyle = r.color;
    ctx.fillText(r.text, xx, yy);
    yy += lineH;
  }
};

global.BookChart = BookChart;
})(typeof window !== "undefined" ? window : this);
