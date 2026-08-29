"""The dashboard's design system: palette, type, motion.

This is a trading terminal, and it is built like one rather than like a web
page with financial data on it. Four decisions carry the whole thing.

**It is set in monospace, all of it.** Not the numbers only, but the labels,
the headings and the prose. A terminal reads in one width because every character
occupying the same cell is what lets a column of figures be scanned rather than
read, and because the moment a proportional face appears next to a fixed one
the whole surface stops looking like an instrument and starts looking like a
website about an instrument.

**Nothing is rounded.** Panels meet at hairlines. A terminal is a grid of cells
and the rules between them are the structure, so borders do the work that
shadows and corner radii would do elsewhere, and they cost nothing to paint.

**The palette is semantic before it is decorative.** Green is a gain and red is
a loss and nothing else is allowed to use them. Amber belongs to execution
cost, because cost is the antagonist of this entire project and it should be
the same colour every time it appears: in the waterfall, in a table, in a
refusal. That leaves cyan for the interface itself: keys, rules, the things you
can click. Four colours, each meaning exactly one thing.

**Density is the point.** Terminals are dense because a trader wants the whole
state at once. Rows are tight, padding is small, the type is small, and the
information is allowed to be close together.

The light theme is a paper terminal, warm ground and dark ink on the same
grid, rather than the dark theme with the colours flipped.
"""

from __future__ import annotations

# --------------------------------------------------------------------- tokens

TOKENS = """
:root {
  color-scheme: dark light;

  /* Black, with the faintest warm cast so amber sits on it rather than
     glowing off it. The Bloomberg ground is the reference. */
  --ground:   #07070a;
  --panel:    #0c0c11;
  --raised:   #121218;
  --sunk:     #050507;

  --rule:     #1b1b24;
  --rule-mid: #262632;
  --rule-hi:  #3a3a4a;

  --ink:      #e2ded6;
  --ink-hi:   #fbf7ef;
  --ink-dim:  #a8a49c;
  --ink-faint:#7d7973;

  /* Amber is the terminal itself: field names, rules, headings, the chrome.
     It is the colour you remember the machine by, and it is used everywhere
     the interface speaks in its own voice rather than reporting a number. */
  --key:      #ffb020;
  --key-hot:  #ffd166;

  /* Phosphor green for a gain and a hot red for a loss, and nothing else may
     use either. They are the two colours a trader reads before any text. */
  --up:       #00e08a;
  --down:     #ff4d5a;

  /* Cost gets magenta of its own. It used to share amber with the chrome,
     which meant the antagonist of this whole project looked like furniture.
     It is now the one colour on the page that appears nowhere else, so a
     cost bar is identifiable at a glance in any chart it turns up in. */
  --cost:     #ff4fd8;

  --up-wash:   rgba(0, 224, 138, 0.12);
  --down-wash: rgba(255, 77, 90, 0.12);
  --cost-wash: rgba(255, 79, 216, 0.12);
  --key-wash:  rgba(255, 176, 32, 0.10);

  --mono: ui-monospace, "SF Mono", "JetBrains Mono", "IBM Plex Mono",
          "Roboto Mono", Menlo, Consolas, "Liberation Mono", monospace;

  --t-micro: 11.5px;
  --t-small: 13px;
  --t-base:  15px;
  --t-mid:   17px;
  --t-lg:    clamp(19px, 1rem + 1.1vw, 28px);
  --t-xl:    clamp(28px, 1.2rem + 3.1vw, 56px);

  --row: 36px;
  --pad: 18px;

  --ease: cubic-bezier(0.2, 0.8, 0.2, 1);
  --fast: 120ms;
  --mid: 280ms;
  --slow: 900ms;
}

:root[data-theme="light"] {
  color-scheme: light;
  --ground:   #eae7df;
  --panel:    #f5f2ea;
  --raised:   #fffdf7;
  --sunk:     #ded9cf;
  --rule:     #d3cec2;
  --rule-mid: #b9b3a4;
  --rule-hi:  #938d7e;
  --ink:      #1b1a17;
  --ink-hi:   #07070a;
  --ink-dim:  #5a564e;
  --ink-faint:#8a857b;
  --key:      #9a5b00;
  --key-hot:  #c47800;
  --up:       #00734a;
  --down:     #c0202e;
  --cost:     #a3007f;
  --up-wash:   rgba(0, 115, 74, 0.10);
  --down-wash: rgba(192, 32, 46, 0.09);
  --cost-wash: rgba(163, 0, 127, 0.10);
  --key-wash:  rgba(154, 91, 0, 0.10);
}
"""

# ----------------------------------------------------------------------- base

BASE = """
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--mono);
  font-size: var(--t-base);
  line-height: 1.62;
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1, "zero" 1;
  -webkit-font-smoothing: antialiased;
}

/* The faintest grid, fixed behind everything. It is barely visible and that
   is the intent: it gives the ground a texture that reads as ruled paper
   rather than as a flat fill, and it costs one paint. */
body::before {
  content: "";
  position: fixed; inset: 0;
  background-image:
    linear-gradient(to right, var(--rule) 1px, transparent 1px),
    linear-gradient(to bottom, var(--rule) 1px, transparent 1px);
  background-size: 96px 96px;
  opacity: 0.35;
  pointer-events: none;
  z-index: 0;
}

.wrap { position: relative; z-index: 1; max-width: 1280px;
        margin: 0 auto; padding: 0 18px 80px; }

h1, h2, h3 { margin: 0; font-weight: 600; line-height: 1.12; color: var(--ink-hi); }
h1 { font-size: var(--t-xl); letter-spacing: -0.03em; }
h2 { font-size: var(--t-lg); letter-spacing: -0.015em; }
h3 { font-size: var(--t-mid); }
p  { margin: 0 0 0.9em; color: var(--ink-dim); }
a  { color: var(--key); text-underline-offset: 2px; }
code { color: var(--key); }
strong { color: var(--ink-hi); font-weight: 600; }
em { color: var(--ink); font-style: normal; text-decoration: underline;
     text-decoration-color: var(--rule-mid); text-underline-offset: 3px; }

.up { color: var(--up); } .down { color: var(--down); }
.cost { color: var(--cost); } .key { color: var(--key); }
.dim { color: var(--ink-dim); } .faint { color: var(--ink-faint); }
.good { color: var(--up); } .bad { color: var(--down); }

/* Uppercase micro-labels, the terminal's own voice for a field name. */
.eyebrow, .tile-key, th, .badge, .stat-key {
  font-size: var(--t-micro);
  text-transform: uppercase;
  letter-spacing: 0.13em;
  color: var(--ink-dim);
  font-weight: 600;
}

/* --------------------------------------------------------------- status bar */

.statusbar {
  position: sticky; top: 0; z-index: 20;
  display: flex; align-items: stretch; flex-wrap: wrap;
  margin: 0 -18px 26px;
  border-bottom: 1px solid var(--rule-mid);
  background: color-mix(in srgb, var(--ground) 86%, transparent);
  backdrop-filter: blur(8px);
}
.statusbar > * {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 14px;
  border-right: 1px solid var(--rule);
  white-space: nowrap;
}
.statusbar .mark {
  font-weight: 700; letter-spacing: 0.24em; color: var(--ink-hi);
  font-size: var(--t-small);
}
.statusbar .mark b { color: var(--key); font-weight: 700; }
.stat-key { color: var(--ink-faint); }
.stat-val { color: var(--ink-hi); font-size: var(--t-small); }
.statusbar .spacer { flex: 1; border-right: 0; }

.led { width: 7px; height: 7px; background: var(--up); flex: none; }
.led.off { background: var(--ink-faint); }
.led.live { animation: blink 2s steps(1, end) infinite; }
@keyframes blink { 0%, 60% { opacity: 1; } 61%, 100% { opacity: 0.25; } }

/* ------------------------------------------------------------------ panels */

.panel {
  background: var(--panel);
  border: 1px solid var(--rule-mid);
  padding: 0;
  margin-top: 14px;
}
.panel-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 14px; flex-wrap: wrap;
  padding: 9px var(--pad);
  border-bottom: 1px solid var(--rule-mid);
  background: var(--raised);
}
.panel-head h3 { font-size: var(--t-small); letter-spacing: 0.14em;
                 text-transform: uppercase; color: var(--ink); }
.panel-body { padding: var(--pad); }

/* A section rule that reads like a terminal divider: label, then the line
   running out to the edge of the column. */
section { margin: 44px 0 0; scroll-margin-top: 60px; }
.rule {
  display: flex; align-items: center; gap: 12px;
  margin: 0 0 12px;
  color: var(--key);
  font-size: var(--t-micro); text-transform: uppercase; letter-spacing: 0.22em;
}
.rule::after {
  content: ""; flex: 1; height: 1px; background: var(--rule-mid);
}

.grid { display: grid; gap: 1px; background: var(--rule-mid);
        border: 1px solid var(--rule-mid); }
.grid.tiles { grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); }
.grid.two   { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }

/* ------------------------------------------------------------------- tiles */

.tile { background: var(--panel); padding: 11px var(--pad) 12px;
        transition: background var(--fast) var(--ease); }
.tile:hover { background: var(--raised); }
.tile-value {
  font-size: 24px; font-weight: 600; color: var(--ink-hi);
  margin-top: 5px; letter-spacing: -0.02em; line-height: 1.1;
}
.tile-note { font-size: var(--t-micro); color: var(--ink-faint); margin-top: 3px;
             letter-spacing: 0.04em; }

/* ------------------------------------------------------------------ tables */

table { width: 100%; border-collapse: collapse; font-size: var(--t-base); }
th {
  text-align: left; padding: 7px var(--pad);
  border-bottom: 1px solid var(--rule-mid);
  background: var(--raised);
  position: sticky; top: 0;
}
td { padding: 0 var(--pad); height: var(--row);
     border-bottom: 1px solid var(--rule); vertical-align: middle; }
tbody tr { transition: background var(--fast) var(--ease); }
tbody tr:hover { background: var(--raised); }
tbody tr:hover td:first-child { box-shadow: inset 2px 0 0 var(--key); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: 0; }
.scroll-x { overflow-x: auto; }

/* ------------------------------------------------------------------ badges */

.badge {
  display: inline-block; padding: 2px 7px;
  border: 1px solid currentColor;
  letter-spacing: 0.14em;
}
.badge.opened  { color: var(--up);   background: var(--up-wash); }
.badge.refused { color: var(--down); background: var(--down-wash); }
.badge.stood   { color: var(--ink-dim); }
.badge.halt    { color: var(--cost); background: var(--cost-wash); }

/* ------------------------------------------------------------------- notes */

.note {
  border-left: 2px solid var(--cost);
  background: var(--cost-wash);
  padding: 10px 14px;
  color: var(--ink-dim);
  font-size: var(--t-small);
  max-width: 90ch;
}
.note strong { color: var(--ink); }

.lede { font-size: var(--t-mid); color: var(--ink); }

footer.foot {
  margin-top: 56px; padding: 14px 0; border-top: 1px solid var(--rule-mid);
  color: var(--ink-faint); font-size: var(--t-micro);
  letter-spacing: 0.08em; text-transform: uppercase;
  display: flex; justify-content: space-between; gap: 18px; flex-wrap: wrap;
}

button.theme-toggle {
  background: transparent; color: var(--ink-dim);
  border: 0; border-left: 1px solid var(--rule);
  font: inherit; font-size: var(--t-micro); letter-spacing: 0.16em;
  text-transform: uppercase; cursor: pointer; padding: 8px 14px;
  transition: color var(--fast) var(--ease), background var(--fast) var(--ease);
}
button.theme-toggle:hover { color: var(--key); background: var(--key-wash); }
button.theme-toggle:focus-visible { outline: 1px solid var(--key); outline-offset: -1px; }

/* ------------------------------------------------------------------ layout */

/* The page reads across, not down one edge. A single measured column with the
   rest of the viewport left blank is a document, and this is an instrument:
   prose sits in one field and the figures it is talking about sit beside it,
   so the eye never has to travel to an empty half.

   The ratio is deliberate. Reading text wants a bounded measure, so the prose
   column is the narrower of the two and the data takes the rest. */
.split {
  display: grid;
  gap: 1px;
  background: var(--rule-mid);
  border: 1px solid var(--rule-mid);
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.split > * { background: var(--panel); padding: 22px; min-width: 0; }
.split.even { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.split .prose { max-width: 62ch; }

@media (max-width: 900px) {
  .split { grid-template-columns: 1fr; }
}

/* A field: label above, value below, sitting in a run across the panel. */
.readout { display: flex; flex-wrap: wrap; gap: 1px; background: var(--rule); }
.readout > div { background: var(--panel); padding: 8px 14px; flex: 1 1 120px; }
.readout .k { font-size: var(--t-micro); text-transform: uppercase;
              letter-spacing: 0.16em; color: var(--ink-faint); }
.readout .v { font-size: 19px; color: var(--ink-hi); margin-top: 2px; }

/* --------------------------------------------------------------------- log */

/* The decision log. It is written the way a log is written: oldest at the
   top, newest at the bottom, days marked as they turn. It scrolls inside its
   own frame rather than growing the page. That is the whole point: another
   session appends to the bottom and the panel above it does not move, so a
   week of trading is a scroll rather than a redesign. */
.log {
  max-height: 62vh;
  min-height: 260px;
  overflow-y: auto;
  overscroll-behavior: contain;
  background: var(--sunk);
  font-size: var(--t-small);
  line-height: 1.5;
  scrollbar-width: thin;
  scrollbar-color: var(--rule-hi) transparent;
}
.log::-webkit-scrollbar { width: 10px; }
.log::-webkit-scrollbar-track { background: transparent; }
.log::-webkit-scrollbar-thumb { background: var(--rule-mid); border: 3px solid var(--sunk); }
.log::-webkit-scrollbar-thumb:hover { background: var(--rule-hi); }

/* The day turning over. Sticky, so while you are inside a session you can
   always see which one you are inside. */
.log-day {
  position: sticky; top: 0; z-index: 2;
  display: flex; align-items: center; gap: 12px;
  padding: 7px 14px;
  background: var(--raised);
  border-top: 1px solid var(--rule-mid);
  border-bottom: 1px solid var(--rule-mid);
  color: var(--key);
  font-size: var(--t-micro); letter-spacing: 0.18em; text-transform: uppercase;
}
.log-day::after { content: ""; flex: 1; height: 1px; background: var(--rule-mid); }
.log-day .count { color: var(--ink-dim); letter-spacing: 0.1em; }

.log-line {
  display: grid;
  grid-template-columns: 46px 68px 96px minmax(96px, 1fr) repeat(3, minmax(72px, auto));
  gap: 10px;
  padding: 5px 14px;
  border-bottom: 1px solid var(--rule);
  align-items: baseline;
}
.log-line:hover { background: var(--panel); }
.log-line .seq { color: var(--ink-faint); }
.log-line .at { color: var(--ink-dim); }
.log-line .what { color: var(--ink-hi); }
.log-line .fig { text-align: right; font-variant-numeric: tabular-nums; }
.log-line .fig .k { color: var(--ink-faint); }

/* The reason the agent gave, carried under the line it belongs to and indented
   the way a continuation is indented in a terminal. */
.log-note {
  padding: 2px 14px 9px 60px;
  border-bottom: 1px solid var(--rule);
  color: var(--ink-dim);
  max-width: 110ch;
}
.log-note::before { content: "> "; color: var(--key); }

.log-empty { padding: 22px 14px; color: var(--ink-dim); }

@media (max-width: 780px) {
  .log-line { grid-template-columns: 44px 64px 92px 1fr; }
  .log-line .fig { grid-column: span 1; text-align: left; }
}

/* ------------------------------------------------------------------ motion */

.reveal { opacity: 0; transform: translateY(8px); }
.reveal.in { opacity: 1; transform: none;
  transition: opacity var(--slow) var(--ease), transform var(--slow) var(--ease); }

.stagger > * { opacity: 0; }
.stagger.in > * { opacity: 1; transition: opacity var(--mid) var(--ease); }
.stagger.in > *:nth-child(1) { transition-delay: 0ms; }
.stagger.in > *:nth-child(2) { transition-delay: 40ms; }
.stagger.in > *:nth-child(3) { transition-delay: 80ms; }
.stagger.in > *:nth-child(4) { transition-delay: 120ms; }
.stagger.in > *:nth-child(5) { transition-delay: 160ms; }
.stagger.in > *:nth-child(6) { transition-delay: 200ms; }
.stagger.in > *:nth-child(7) { transition-delay: 240ms; }
.stagger.in > *:nth-child(8) { transition-delay: 280ms; }

/* A rule that a light travels along once as it comes into view, the way a
   terminal draws a divider rather than simply having one. */
.rule { position: relative; overflow: hidden; }
.rule::before {
  content: ""; position: absolute; inset: auto 0 0 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--key-hot) 45%,
              var(--key-hot) 55%, transparent);
  transform: translateX(-100%);
  opacity: 0;
}
.rule.in::before { animation: streak 1.5s var(--ease) 200ms both; }
@keyframes streak {
  0%   { transform: translateX(-100%); opacity: 0; }
  15%  { opacity: 1; }
  85%  { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}

/* One pass of a scanline down a panel as it arrives. It runs once, never on a
   loop: a permanently sweeping CRT line is a costume, and this is meant to
   read as the panel being drawn. */
.scan { position: relative; overflow: hidden; }
.scan::after {
  content: ""; position: absolute; left: 0; right: 0; top: -40%; height: 40%;
  background: linear-gradient(180deg, transparent,
              rgba(255, 176, 32, 0.22), transparent);
  opacity: 0; pointer-events: none;
}
.scan.in::after { animation: scan 1.4s var(--ease) 150ms both; }
@keyframes scan {
  0% { top: -30%; opacity: 0.9; } 100% { top: 100%; opacity: 0; }
}

/* The headline carries a single chromatic split on arrival, then settles.
   Held to one pass and a fraction of a pixel: enough to read as a CRT
   struggling into focus, not enough to look broken. */
.shift.in { animation: shift 620ms steps(2, end) both; }
@keyframes shift {
  0%   { text-shadow: -1.5px 0 var(--cost), 1.5px 0 var(--key); }
  60%  { text-shadow: -0.5px 0 var(--cost), 0.5px 0 var(--key); }
  100% { text-shadow: none; }
}

/* A light runs the length of the plotted series, over and over. This is the
   one thing on the page allowed to loop, and it is allowed because it carries
   no information: the series underneath it is already drawn and static, and
   the pulse only says the instrument is live. pathLength is set to 100 on the
   element so the dash figures are percentages of the line however wide the
   chart is drawn. */
.pulse {
  stroke-dasharray: 8 92;
  animation: pulse-run 3.4s linear infinite;
  opacity: 0;
}
.pulse.on { opacity: 0.95; }
@keyframes pulse-run { from { stroke-dashoffset: 100; } to { stroke-dashoffset: 0; } }

/* The scrubber walks the measured points and fades in once it starts. */
.scrub { transition: opacity 400ms var(--ease); }
.scrub-line, .scrub-dot { transition: transform 900ms var(--ease); }

.scrub-readout { display: flex; flex-wrap: wrap; gap: 1px; background: var(--rule);
                 margin-top: 14px; border: 1px solid var(--rule-mid); }
.scrub-readout > div { background: var(--panel); padding: 9px 14px; flex: 1 1 110px; }
.scrub-readout .k { font-size: var(--t-micro); text-transform: uppercase;
                    letter-spacing: 0.13em; color: var(--ink-dim); }
.scrub-readout .v { font-size: 20px; color: var(--ink-hi); margin-top: 2px;
                    font-variant-numeric: tabular-nums; }

.draw { stroke-dasharray: var(--len); stroke-dashoffset: var(--len); }
.draw.in { transition: stroke-dashoffset 1s var(--ease); stroke-dashoffset: 0; }

.grow { transform: scaleY(0); transform-origin: var(--origin, bottom); }
.grow.in { transition: transform var(--slow) var(--ease); transform: scaleY(1); }

/* The headline types itself once, in the terminal's own idiom, with a cursor
   that stops blinking when the line is finished. Purely presentational: the
   text is in the markup and is revealed by clipping, never by scripting
   characters into an empty element. */
.type { display: inline-block; overflow: hidden; white-space: nowrap;
        border-right: 0.5ch solid var(--key); }
.type.in { animation: type 1.3s steps(30, end) both, caret 1.3s steps(1) 4; }
@keyframes type { from { max-width: 0; } to { max-width: 100%; } }
@keyframes caret { 0%, 49% { border-color: var(--key); }
                   50%, 100% { border-color: transparent; } }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
  }
  .reveal, .stagger > * { opacity: 1; transform: none; }
  .draw { stroke-dashoffset: 0; }
  .grow { transform: none; }
  .type { border-right: 0; max-width: none; }
  .rule::before, .scan::after { display: none; }
  .shift.in { animation: none; text-shadow: none; }
  /* The pulse and the scrubber are the only looping things here, so under a
     reduced-motion preference they stop entirely rather than slowing down. */
  .pulse { animation: none; opacity: 0; }
  .scrub { opacity: 0 !important; }
}

@media (max-width: 640px) {
  .statusbar { position: static; }
  .statusbar > * { padding: 7px 10px; }
  :root { --pad: 11px; }
}
"""


# ----------------------------------------------------------------- behaviour

SCRIPT = """
(function () {
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("convex-theme", next); } catch (e) {}
      toggle.textContent = next === "light" ? "DARK" : "LIGHT";
    });
  }

  var watched = document.querySelectorAll(
    ".reveal, .stagger, .draw, .grow, .type, .rule, .scan, .shift"
  );
  if (!("IntersectionObserver" in window) || reduced) {
    watched.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        io.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.08 });
    watched.forEach(function (el) { io.observe(el); });
  }

  /* A figure counts up once, the way a terminal cell settles. The markup
     already holds the final value, so no JavaScript and no motion both leave
     the correct number on screen rather than a zero. */
  function countUp(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    var places = parseInt(el.getAttribute("data-places") || "0", 10);
    var prefix = el.getAttribute("data-prefix") || "";
    var signed = el.hasAttribute("data-signed");
    var started = null;
    function frame(now) {
      if (started === null) started = now;
      var t = Math.min((now - started) / 780, 1);
      var value = target * (1 - Math.pow(1 - t, 3));
      var text = Math.abs(value).toLocaleString(undefined, {
        minimumFractionDigits: places, maximumFractionDigits: places
      });
      el.textContent = (value < 0 ? "-" : (signed && value > 0 ? "+" : "")) + prefix + text;
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  if (!reduced && "IntersectionObserver" in window) {
    var co = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        countUp(entry.target);
        co.unobserve(entry.target);
      });
    }, { threshold: 0.5 });
    document.querySelectorAll("[data-count]").forEach(function (el) { co.observe(el); });
  }

  /* The chart plays itself. A crosshair walks the points the sweep actually
     measured, pausing on each one while the readout under the chart shows what
     was recorded there, and the value flips colour as it crosses zero. It
     stops on nothing that was not measured: between two points it is in
     transit, and the readout still shows the point it is heading for rather
     than a number interpolated out of the gap. */
  var chart = document.querySelector("svg[data-scrub]");
  if (chart && !reduced) {
    var points = [];
    try { points = JSON.parse(chart.getAttribute("data-scrub")); } catch (e) {}
    var scrub = chart.querySelector(".scrub");
    var line = chart.querySelector(".scrub-line");
    var dot = chart.querySelector(".scrub-dot");
    var pulse = chart.querySelector(".pulse");
    var out = document.querySelector("[data-scrub-readout]");

    if (points.length && scrub && line && dot) {
      var index = 0;
      var running = false;

      function show(point) {
        var dx = point.x - parseFloat(line.getAttribute("x1"));
        line.setAttribute("transform", "translate(" + dx + ",0)");
        dot.setAttribute("transform",
          "translate(" + dx + "," + (point.y - parseFloat(dot.getAttribute("cy"))) + ")");
        if (!out) return;
        var sign = point.n > 0 ? "up" : "down";
        out.querySelector("[data-f=spread]").textContent =
          (point.s * 100).toFixed(1) + "%";
        out.querySelector("[data-f=gross]").textContent = point.g.toFixed(2);
        var net = out.querySelector("[data-f=net]");
        net.textContent = (point.n > 0 ? "+" : "") + point.n.toFixed(2);
        net.className = "v " + sign;
        out.querySelector("[data-f=trades]").textContent = point.t;
      }

      function step() {
        show(points[index]);
        index = (index + 1) % points.length;
        setTimeout(step, 1700);
      }

      var start = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting || running) return;
          running = true;
          scrub.setAttribute("opacity", "1");
          if (pulse) pulse.classList.add("on");
          setTimeout(step, 1200);
          start.disconnect();
        });
      }, { threshold: 0.25 });
      start.observe(chart);
    }
  }

  /* A log opens where a log is read from: the end. Jumped rather than
     smooth-scrolled, because animating to the bottom of something the reader
     has not looked at yet is motion for its own sake. */
  var log = document.querySelector("[data-log]");
  if (log) log.scrollTop = log.scrollHeight;

  /* The clock in the status bar. A terminal that does not tick looks frozen,
     and this is the only thing on the page that changes without a reload. */
  var clock = document.querySelector("[data-clock]");
  if (clock) {
    (function tick() {
      var now = new Date();
      var parts = now.toLocaleTimeString("en-GB", {
        timeZone: "America/New_York", hour12: false
      });
      clock.textContent = parts + " ET";
      setTimeout(tick, 1000 - (now.getTime() % 1000));
    })();
  }
})();
"""

HEAD_SCRIPT = """
try {
  var t = localStorage.getItem("convex-theme");
  if (t) document.documentElement.setAttribute("data-theme", t);
} catch (e) {}
"""


def stylesheet() -> str:
    return TOKENS + BASE
