"""Stage 11 - a self-contained dashboard for a reader who will not run the code.

One HTML file with the data inlined as JSON and the charts drawn in vanilla
SVG. No build step, no CDN, no network access at runtime: it opens from a file
path or from GitHub Pages and behaves the same either way.

Colours are the validated default palette, and the same rule the static figures
use applies here: an estimate whose interval covers zero is drawn in grey.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.ROOT / "docs" / "dashboard.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Airline Merger Screens</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --panel: #ffffff; --line: #e6e5e1;
    --ink: #0b0b0b; --ink2: #52514e; --muted: #9aa5ad;
    --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a; --zero: #b0413c;
    --shadow: 0 1px 2px rgba(11,11,11,.06), 0 8px 24px rgba(11,11,11,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --panel: #232322; --line: #383835;
      --ink: #ffffff; --ink2: #c3c2b7; --muted: #8b8a82;
      --blue: #3987e5; --orange: #d95926; --aqua: #199e70; --zero: #e66767;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322; --line: #383835;
    --ink: #ffffff; --ink2: #c3c2b7; --muted: #8b8a82;
    --blue: #3987e5; --orange: #d95926; --aqua: #199e70; --zero: #e66767;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 48px 16px 80px; }
  header { margin-bottom: 36px; }
  .eyebrow { font-size: 12px; letter-spacing: .09em; text-transform: uppercase;
             color: var(--muted); font-weight: 600; }
  h1 { font-size: clamp(26px, 4.4vw, 40px); line-height: 1.16; margin: 10px 0 14px;
       letter-spacing: -.02em; font-weight: 700; }
  .lede { font-size: 17px; color: var(--ink2); max-width: 66ch; margin: 0; }
  h2 { font-size: 21px; margin: 0 0 6px; letter-spacing: -.01em; }
  .sub { color: var(--ink2); font-size: 14px; margin: 0 0 18px; max-width: 72ch; }
  section { margin-top: 44px; }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 14px; padding: 22px; box-shadow: var(--shadow); }
  .kpis { display: grid; gap: 14px; grid-template-columns: repeat(4, 1fr); }
  @media (max-width: 860px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 460px) { .kpis { grid-template-columns: 1fr; } }
  .kpi { background: var(--panel); border: 1px solid var(--line);
         border-radius: 14px; padding: 16px 18px; box-shadow: var(--shadow); }
  .kpi .label { font-size: 12.5px; color: var(--ink2); min-height: 2.9em; }
  .kpi .value { font-size: 30px; font-weight: 700; letter-spacing: -.02em;
                margin-top: 8px; font-variant-numeric: tabular-nums; }
  .kpi .ci { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .pos { color: var(--blue); } .neg { color: var(--orange); } .nil { color: var(--muted); }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .tick text { font-size: 11px; fill: var(--ink2); }
  .tick line { stroke: var(--line); }
  .axis-label { font-size: 12px; fill: var(--ink2); }
  .tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
         background: var(--panel); color: var(--ink); border: 1px solid var(--line);
         border-radius: 9px; padding: 8px 11px; font-size: 12.5px;
         box-shadow: var(--shadow); z-index: 50; max-width: 260px;
         font-variant-numeric: tabular-nums; }
  .tip b { font-weight: 650; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  th { font-weight: 600; color: var(--ink2); font-size: 12px;
       text-transform: uppercase; letter-spacing: .05em; }
  tr.hi td { background: color-mix(in srgb, var(--blue) 9%, transparent); }
  .legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px;
            color: var(--ink2); margin-top: 12px; }
  .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
            margin-right: 6px; vertical-align: -1px; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 800px) { .two { grid-template-columns: 1fr; } }
  footer { margin-top: 56px; padding-top: 22px; border-top: 1px solid var(--line);
           color: var(--muted); font-size: 13px; }
  a { color: var(--blue); }
  .toggle { position: fixed; top: 14px; right: 14px; background: var(--panel);
            border: 1px solid var(--line); border-radius: 9px; padding: 7px 12px;
            font-size: 12.5px; color: var(--ink2); cursor: pointer; z-index: 60; }
  .note { font-size: 12.5px; color: var(--muted); margin-top: 12px; }
</style>
</head>
<body>
<button class="toggle" id="themeBtn">theme</button>
<div class="wrap">

<header>
  <div class="eyebrow">Merger retrospective &middot; DOT DB1B, 2005&ndash;2019</div>
  <h1>The screens that flag airline mergers do not predict which routes got
      more expensive. The ones nobody runs first do.</h1>
  <p class="lede">Five US airline mergers, __NROUTES__ routes where the merging
     carriers overlapped, and __NTICKETS__ ticket records. Every prediction
     below is computed from information available before each deal closed, and
     scored against what fares actually did afterwards.</p>
</header>

<section>
  <div class="kpis" id="kpis"></div>
</section>

<section>
  <h2>Fares rose &mdash; about five quarters late</h2>
  <p class="sub">Effect on the average fare of an overlap route relative to
     matched control routes, by quarter from the merger closing. The quarter
     before closing is the omitted baseline. Grey where the 95% interval
     covers zero.</p>
  <div class="card"><div id="event"></div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--blue)"></i>interval clears zero</span>
      <span><i class="swatch" style="background:var(--orange)"></i>a fall, clearing zero</span>
      <span><i class="swatch" style="background:var(--muted)"></i>cannot be told from zero</span>
    </div>
    <p class="note">Nothing happens for four quarters. The single operating
       certificate &mdash; the point at which two airlines become one for
       regulatory and fare-setting purposes &mdash; lands four to six quarters
       after closing. A retrospective that stopped at one year would have
       found nothing.</p>
  </div>
</section>

<section>
  <h2>The test: do the tools pick the right routes?</h2>
  <p class="sub">Overlap routes sorted into deciles by each prediction, with
     the average realised fare effect in each decile. A tool that works should
     climb from left to right.</p>
  <div class="two">
    <div class="card"><div id="calHHI"></div>
      <p class="note">The increase in HHI is the screen the 2023 Merger
         Guidelines are built on. Across its whole range the realised effect is
         flat.</p></div>
    <div class="card"><div id="calGUPPI"></div>
      <p class="note">GUPPI &mdash; the value of sales diverted to the merger
         partner &mdash; needs demand estimation and is rarely the first thing
         computed. It sorts the routes almost perfectly.</p></div>
  </div>
</section>

<section>
  <h2>How much information each prediction carries</h2>
  <p class="sub">Change in the realised fare effect for a one-standard-deviation
     increase in each prediction, with merger fixed effects and standard errors
     clustered on the route.</p>
  <div class="card"><div id="slopes"></div></div>
</section>

<section>
  <h2>Merger by merger</h2>
  <p class="sub">The same stacked design, one deal at a time, each against its
     own matched controls.</p>
  <div class="card"><div id="mergers"></div></div>
</section>

<section>
  <h2>Where the line is drawn, and where it could be</h2>
  <p class="sub">The Guidelines flag a merger when it raises HHI by more than
     100 points in a concentrated market. This is what other thresholds would
     have done on this data.</p>
  <div class="card"><div id="thresh"></div></div>
</section>

<section>
  <h2>Reading the numbers</h2>
  <div class="card">
    <p class="sub"><b>Overlap route.</b> A city pair where both merging carriers
    each held at least 1% of passengers in the four quarters before the merger
    was announced. Measured before the announcement, because carriers move
    capacity once a deal is public.</p>
    <p class="sub"><b>Control route.</b> A route where exactly one of the two
    carriers was selling. Both groups get whatever the merger did to the
    combined airline's costs and network; only overlap routes lose a
    competitor. Controls are then matched to overlap routes on distance,
    traffic, concentration, fare, nonstop share and pre-merger fare trend.</p>
    <p class="sub"><b>HHI and its increase.</b> The sum of squared market shares
    on a 0&ndash;10,000 scale. A merger of two firms raises it by twice the
    product of their shares. It needs nothing but quantities, which is its
    appeal and its limit.</p>
    <p class="sub"><b>GUPPI.</b> The value of the sales a carrier diverts to its
    merger partner when it raises its fare, as a fraction of that fare. It
    requires a demand system, and the one used here is calibrated rather than
    freely estimated &mdash; see the write-up on demand estimation. Its
    <i>ranking</i> of routes is nearly invariant to that calibration; its level
    is not.</p>
    <p class="sub"><b>Realised effect.</b> The change in a route's average real
    fare from the eight quarters before closing to quarters +5 through +12,
    minus the same change on its matched controls.</p>
  </div>
</section>

<footer>
  Source: US DOT Bureau of Transportation Statistics, Airline Origin and
  Destination Survey (DB1B), Market file, 2005Q1&ndash;2019Q4, with CPI-U and
  Gulf Coast jet fuel from FRED. Fares in constant 2019 dollars.
  <a href="https://github.com/josealemanm/us-airline-merger-price-effects">Code,
  data pipeline and full write-ups on GitHub.</a>
</footer>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA = __DATA__;

const tip = document.getElementById('tip');
function showTip(html, e) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
function hideTip() { tip.style.opacity = 0; }

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
const fmtPct = v => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
/* Grey when the interval covers zero; otherwise the sign picks the hue, so a
   significant fall never reads the same as a significant rise. */
function signColor(v, lo, hi) {
  if (lo <= 0 && hi >= 0) return 'var(--muted)';
  return v > 0 ? 'var(--blue)' : 'var(--orange)';
}

/* ---------------------------------------------------------------- KPIs */
function kpis() {
  const box = document.getElementById('kpis');
  DATA.kpis.forEach(k => {
    const d = document.createElement('div');
    d.className = 'kpi';
    const cls = k.kind || 'nil';
    d.innerHTML = `<div class="label">${k.label}</div>
      <div class="value ${cls}">${k.value}</div>
      <div class="ci">${k.ci || ''}</div>`;
    box.appendChild(d);
  });
}

/* -------------------------------------------------- generic dot-and-CI plot */
function dotPlot(mount, rows, opts) {
  const W = opts.width || 940, H = opts.height || 330;
  const M = opts.margin || { t: 16, r: 18, b: 44, l: 54 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}` });
  mount.innerHTML = ''; mount.appendChild(svg);
  const g = el('g', { transform: `translate(${M.l},${M.t})` }, svg);

  let y0, y1;
  if (opts.domain) { [y0, y1] = opts.domain; }
  else {
    const lo = Math.min(0, ...rows.map(r => r.lo));
    const hi = Math.max(0, ...rows.map(r => r.hi));
    const padY = (hi - lo) * 0.10 || 0.01;
    y0 = lo - padY; y1 = hi + padY;
  }
  const Y = v => ih - (v - y0) / (y1 - y0) * ih;
  const n = rows.length;
  const X = i => (n === 1 ? iw / 2 : (i + 0.5) * iw / n);

  const step = niceStep(y1 - y0);
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
    const gg = el('g', { class: 'tick' }, g);
    el('line', { x1: 0, x2: iw, y1: Y(v), y2: Y(v) }, gg);
    el('text', { x: -10, y: Y(v) + 4, 'text-anchor': 'end' }, gg)
      .textContent = (v >= 0 ? '+' : '') + (v * 100).toFixed(0) + '%';
  }
  el('line', { x1: 0, x2: iw, y1: Y(0), y2: Y(0),
               stroke: 'var(--zero)', 'stroke-width': 1.3 }, g);
  if (opts.band) {
    const a = X(opts.band[0]) - iw / n / 2, b = X(opts.band[1]) + iw / n / 2;
    el('rect', { x: a, y: 0, width: b - a, height: ih,
                 fill: 'var(--blue)', opacity: .07 }, g);
  }

  rows.forEach((r, i) => {
    const col = r.color || signColor(r.v, r.lo, r.hi);
    el('line', { x1: X(i), x2: X(i), y1: Y(r.lo), y2: Y(r.hi), stroke: col,
                 'stroke-width': 2.4, 'stroke-linecap': 'round' }, g);
    const c = el('circle', { cx: X(i), cy: Y(r.v), r: 5, fill: col,
                             stroke: 'var(--panel)', 'stroke-width': 1.6 }, g);
    const hit = el('rect', { x: X(i) - iw / n / 2, y: 0, width: iw / n,
                             height: ih, fill: 'transparent' }, g);
    const html = `<b>${r.label}</b><br>${fmtPct(r.v)}
        <span style="color:var(--muted)">[${fmtPct(r.lo)}, ${fmtPct(r.hi)}]</span>
        ${r.note ? '<br>' + r.note : ''}`;
    [c, hit].forEach(node => {
      node.addEventListener('mousemove', e => showTip(html, e));
      node.addEventListener('mouseleave', hideTip);
    });
    if (opts.everyTick ? i % opts.everyTick === 0 : true) {
      el('text', { x: X(i), y: ih + 20, 'text-anchor': 'middle',
                   class: 'axis-label' }, g).textContent = r.tick ?? r.label;
    }
  });
  if (opts.xlab)
    el('text', { x: iw / 2, y: ih + 40, 'text-anchor': 'middle',
                 class: 'axis-label' }, g).textContent = opts.xlab;
  return svg;
}

function niceStep(range) {
  const raw = range / 5, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  return (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
}

/* ---------------------------------------------------- horizontal bar plot */
function barPlot(mount, rows, opts) {
  const W = opts.width || 940, rowH = 34;
  const M = { t: 10, r: 120, b: 40, l: opts.labelWidth || 250 };
  const ih = rows.length * rowH, H = ih + M.t + M.b;
  const iw = W - M.l - M.r;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}` });
  mount.innerHTML = ''; mount.appendChild(svg);
  const g = el('g', { transform: `translate(${M.l},${M.t})` }, svg);

  const lo = Math.min(0, ...rows.map(r => r.lo ?? r.v));
  const hi = Math.max(0, ...rows.map(r => r.hi ?? r.v));
  const pad = (hi - lo) * 0.08 || 0.01;
  const X = v => (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * iw;

  el('line', { x1: X(0), x2: X(0), y1: 0, y2: ih,
               stroke: 'var(--zero)', 'stroke-width': 1.3 }, g);

  rows.forEach((r, i) => {
    const y = i * rowH + rowH / 2;
    const col = r.color || signColor(r.v, r.lo ?? r.v, r.hi ?? r.v);
    if (r.lo !== undefined)
      el('line', { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: col,
                   'stroke-width': 2.6, 'stroke-linecap': 'round' }, g);
    el('circle', { cx: X(r.v), cy: y, r: 5.5, fill: col,
                   stroke: 'var(--panel)', 'stroke-width': 1.6 }, g);
    el('text', { x: -12, y: y + 4, 'text-anchor': 'end', class: 'axis-label' }, g)
      .textContent = r.label;
    el('text', { x: iw + 12, y: y + 4, class: 'axis-label',
                 fill: 'var(--ink)' }, g).textContent = r.right ?? fmtPct(r.v);
    const hit = el('rect', { x: 0, y: i * rowH, width: iw, height: rowH,
                             fill: 'transparent' }, g);
    const html = `<b>${r.label}</b><br>${r.tipValue ?? fmtPct(r.v)}
      ${r.lo !== undefined ? `<span style="color:var(--muted)"><br>[${fmtPct(r.lo)}, ${fmtPct(r.hi)}]</span>` : ''}
      ${r.note ? '<br>' + r.note : ''}`;
    hit.addEventListener('mousemove', e => showTip(html, e));
    hit.addEventListener('mouseleave', hideTip);
  });
  if (opts.xlab)
    el('text', { x: iw / 2, y: ih + 28, 'text-anchor': 'middle',
                 class: 'axis-label' }, g).textContent = opts.xlab;
}

/* ------------------------------------------------------------------ table */
function tableView(mount, cols, rows, highlight) {
  const t = document.createElement('table');
  t.innerHTML = '<thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') +
    '</tr></thead><tbody>' + rows.map(r =>
      `<tr class="${highlight && highlight(r) ? 'hi' : ''}">` +
      r.map(v => `<td>${v}</td>`).join('') + '</tr>').join('') + '</tbody>';
  mount.innerHTML = ''; mount.appendChild(t);
}

/* ------------------------------------------------------------------- draw */
kpis();

dotPlot(document.getElementById('event'), DATA.event.map(d => ({
  label: `quarter ${d.k >= 0 ? '+' : ''}${d.k}`, tick: d.k,
  v: d.coef, lo: d.ci_lo, hi: d.ci_hi,
  note: d.k === -1 ? 'omitted baseline' : null,
})), { xlab: 'quarters from the merger closing', everyTick: 2,
       band: [DATA.bandStart, DATA.event.length - 1] });

/* One shared y-range across both panels. Two charts meant to be compared must
   not be drawn on different scales - the flat one would otherwise look as
   dramatic as the climbing one. */
const CAL_DOMAIN = (() => {
  const all = ['delta_hhi', 'guppi_wavg'].flatMap(k => DATA.bins[k]
    .flatMap(b => [b.realised - 1.96 * b.se, b.realised + 1.96 * b.se]));
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
})();
function calChart(mountId, key, label) {
  const rows = DATA.bins[key].map(b => ({
    label: `decile ${b.bin}`, tick: b.bin, v: b.realised,
    lo: b.realised - 1.96 * b.se, hi: b.realised + 1.96 * b.se,
    note: `${b.n.toLocaleString()} routes<br>${label}: ${b.range}`,
  }));
  dotPlot(document.getElementById(mountId), rows,
          { xlab: `decile of ${label}`, width: 470, height: 320,
            margin: { t: 16, r: 14, b: 44, l: 48 }, everyTick: 3,
            domain: CAL_DOMAIN });
}
calChart('calHHI', 'delta_hhi', 'the HHI increase');
calChart('calGUPPI', 'guppi_wavg', 'GUPPI');

barPlot(document.getElementById('slopes'), DATA.slopes.map(s => ({
  label: s.label, v: s.per_sd, lo: s.per_sd - 1.96 * s.se,
  hi: s.per_sd + 1.96 * s.se,
  right: fmtPct(s.per_sd), note: `rank correlation ${s.rho.toFixed(3)}`,
})), { xlab: 'realised fare effect per standard deviation of the prediction' });

barPlot(document.getElementById('mergers'), DATA.mergers.map(m => ({
  label: m.label, v: m.coef, lo: m.ci_lo, hi: m.ci_hi,
  right: fmtPct(m.coef), note: `${m.n.toLocaleString()} overlap routes`,
})), { xlab: 'effect on overlap-route fares' });

tableView(document.getElementById('thresh'),
  ['HHI increase above', 'routes flagged', 'share of routes',
   'flagged', 'not flagged', 'difference'],
  DATA.thresholds.map(t => [
    t.threshold.toLocaleString() + (t.threshold === 100 ? ' (the Guidelines)' : ''),
    t.flagged.toLocaleString(),
    (t.share * 100).toFixed(0) + '%',
    fmtPct(t.mean_in), fmtPct(t.mean_out), fmtPct(t.diff),
  ]), r => String(r[0]).includes('Guidelines'));

/* ------------------------------------------------------------------ theme */
const btn = document.getElementById('themeBtn');
btn.addEventListener('click', () => {
  const now = document.documentElement.getAttribute('data-theme');
  const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
  const next = now ? (now === 'dark' ? 'light' : 'dark')
                   : (prefersDark ? 'light' : 'dark');
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('theme', next); } catch (e) {}
});
try {
  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
} catch (e) {}
</script>
</body>
</html>
"""

LABELS = {
    "delta_hhi": "increase in HHI",
    "combined_share": "combined share of the parties",
    "diversion_to_partner": "diversion to the partner",
    "guppi_wavg": "GUPPI",
    "sim_dp_market": "simulated fare change",
}


def main() -> int:
    did = json.loads((C.DATA_PROCESSED / "did_results.json").read_text())
    val = json.loads((C.DATA_PROCESSED / "validation.json").read_text())
    man = json.loads((C.DATA_RAW / "manifest.json").read_text())
    ev = pd.read_parquet(C.DATA_PROCESSED / "event_study.parquet").sort_values("k")

    tickets = sum(q.get("rows_in", 0) for q in man["quarters"])
    n_routes = val["n_routes"]

    def kind(lo, hi):
        if lo > 0:
            return "pos"
        if hi < 0:
            return "neg"
        return "nil"

    st, lr, pl = did["stacked_did"], did["longrun"], did.get("placebo_one_party")
    flag = val["flag"]
    kpis = [
        {"label": "Pooled two-way fixed effects, the estimator that should "
                  "not be used here",
         "value": f"{did['naive_twfe']['coef']*100:+.2f}%",
         "ci": "wrong sign; 14% of it uses already-merged routes as controls",
         "kind": "neg"},
        {"label": "Stacked difference-in-differences, all five mergers",
         "value": f"{st['coef']*100:+.2f}%",
         "ci": f"95% [{st['ci_lo']*100:+.2f}%, {st['ci_hi']*100:+.2f}%]",
         "kind": kind(st["ci_lo"], st["ci_hi"])},
        {"label": "Once the carriers actually operate as one, quarters +5 to +12",
         "value": f"{lr['coef']*100:+.2f}%",
         "ci": f"95% [{lr['ci_lo']*100:+.2f}%, {lr['ci_hi']*100:+.2f}%]",
         "kind": kind(lr["ci_lo"], lr["ci_hi"])},
        {"label": "Routes the structural presumption flags, against the routes "
                  "it does not",
         "value": f"{flag['diff']*100:+.2f}%",
         "ci": f"95% [{flag['diff_lo']*100:+.2f}%, {flag['diff_hi']*100:+.2f}%]",
         "kind": kind(flag["diff_lo"], flag["diff_hi"])},
    ]

    bins = {}
    for key in ("delta_hhi", "guppi_wavg"):
        rows = val["bins"].get(key, [])
        bins[key] = [{
            "bin": r["bin"], "n": r["n"], "realised": r["realised"], "se": r["se"],
            "range": (f"{r['pred_lo']:,.0f} to {r['pred_hi']:,.0f}"
                      if key == "delta_hhi"
                      else f"{r['pred_lo']*100:.2f}% to {r['pred_hi']*100:.2f}%"),
        } for r in rows]

    slopes = [{"label": LABELS.get(k, k), "per_sd": v["slope_per_sd"],
               "se": v["se_per_sd"], "rho": v["spearman"]}
              for k, v in val["slopes"].items()]
    slopes.sort(key=lambda s: s["per_sd"])

    mergers = [{"label": v["label"], "coef": v["coef"], "ci_lo": v["ci_lo"],
                "ci_hi": v["ci_hi"], "n": v["n_treated"]}
               for v in did["by_merger"].values()]
    mergers.sort(key=lambda m: m["coef"])

    thresholds = [{"threshold": t["threshold"], "flagged": t["flagged"],
                   "share": t["share_flagged"], "mean_in": t["mean_in"],
                   "mean_out": t["mean_out"], "diff": t["diff"]}
                  for t in val["thresholds"]]

    ks = ev["k"].tolist()
    payload = {
        "kpis": kpis,
        "event": ev[["k", "coef", "ci_lo", "ci_hi"]].to_dict("records"),
        "bandStart": ks.index(5) if 5 in ks else 0,
        "bins": bins, "slopes": slopes, "mergers": mergers,
        "thresholds": thresholds,
    }

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__NROUTES__", f"{n_routes:,}")
            .replace("__NTICKETS__", f"{tickets/1e6:.0f} million"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"  {OUT.relative_to(C.ROOT)}  ({len(html)/1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
