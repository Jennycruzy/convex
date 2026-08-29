"""The dashboard's design system: tokens, type, motion.

Kept apart from app.py so that the page can be read as structure and this can
be read as craft. There is no build step, no framework and no external request
— the stylesheet and the behaviour ship inside the response, which is also why
the page survives being loaded from a phone on a conference wifi.

Three decisions run through all of it.

**Numbers are the subject.** Every figure is set in tabular numerals so columns
align down the page and a changing value does not make the row twitch. Money
and Sharpe ratios get the same treatment as a trading terminal gives them:
monospaced digits, tight tracking, the sign carried in colour as well as glyph.

**Motion explains rather than decorates.** Things enter in the order they were
computed, a bar grows from the baseline it is measured against, a curve draws
left to right the way it was swept. Nothing loops, nothing bounces, and every
transition is cancelled outright under prefers-reduced-motion — a trader with
vestibular sensitivity should get the same page, still.

**The dark theme is the default** because that is where this kind of instrument
lives, but the light theme is a real design and not an inversion.
"""

from __future__ import annotations

# --------------------------------------------------------------------- tokens

TOKENS = """
:root {
  color-scheme: dark light;

  /* Ground up: the page is layered, not flat. Each surface sits a measured
     step above the one behind it so depth reads without a single shadow. */
  --ground:  #07090d;
  --sunk:    #0a0d13;
  --panel:   #0e1219;
  --raised:  #141924;
  --line:    #1e2530;
  --line-bright: #2a3342;

  --ink:     #e8edf5;
  --ink-dim: #9aa7ba;
  --ink-faint: #61708a;

  /* One accent, used sparingly enough that it still means something. */
  --accent:  #4c8dff;
  --accent-dim: #1d3a6b;
  --accent-glow: rgba(76, 141, 255, 0.16);

  --good:    #35d0a5;
  --bad:     #ff6b6b;
  --cost:    #f5a524;
  --neutral: #7c8ba1;

  --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
          Roboto, "Helvetica Neue", sans-serif;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;

  /* A scale, not a pile of arbitrary pixel values. */
  --step--1: clamp(0.75rem, 0.73rem + 0.1vw, 0.8rem);
  --step-0:  clamp(0.875rem, 0.85rem + 0.12vw, 0.94rem);
  --step-1:  clamp(1.05rem, 0.99rem + 0.3vw, 1.25rem);
  --step-2:  clamp(1.4rem, 1.25rem + 0.75vw, 1.9rem);
  --step-3:  clamp(2rem, 1.6rem + 2vw, 3.2rem);
  --step-4:  clamp(2.6rem, 1.9rem + 3.4vw, 4.6rem);

  --gap: 20px;
  --radius: 10px;
  --radius-sm: 6px;

  /* Motion. One easing curve for entrances, one for anything the pointer
     drives, and durations short enough that nobody waits on the interface. */
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-io:  cubic-bezier(0.65, 0, 0.35, 1);
  --fast: 140ms;
  --mid: 320ms;
  --slow: 720ms;
}

:root[data-theme="light"] {
  color-scheme: light;
  --ground:  #f2f4f8;
  --sunk:    #e9edf3;
  --panel:   #ffffff;
  --raised:  #f7f9fc;
  --line:    #dde3ec;
  --line-bright: #c6cfdd;
  --ink:     #0d1219;
  --ink-dim: #4d5a6d;
  --ink-faint: #7d8a9e;
  --accent:  #1f6feb;
  --accent-glow: rgba(31, 111, 235, 0.1);
  --good:    #0a7f68;
  --bad:     #c0392b;
  --cost:    #b9741a;
  --neutral: #64748b;
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
  font-family: var(--font);
  font-size: var(--step-0);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* The ground is not a flat fill. A single very soft wash behind the masthead
   gives the page a horizon without costing a request or a repaint. */
body::before {
  content: "";
  position: fixed;
  inset: 0 0 auto 0;
  height: 60vh;
  background:
    radial-gradient(90ch 40vh at 18% -10%, var(--accent-glow), transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.wrap { position: relative; z-index: 1; max-width: 1180px;
        margin: 0 auto; padding: 0 24px 96px; }

h1, h2, h3 { margin: 0; font-weight: 600; letter-spacing: -0.022em; line-height: 1.15; }
h1 { font-size: var(--step-4); letter-spacing: -0.04em; }
h2 { font-size: var(--step-2); }
h3 { font-size: var(--step-1); }
p { margin: 0 0 1em; color: var(--ink-dim); max-width: 68ch; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }

/* Everything numeric. Tabular figures keep columns honest and stop a
   counting animation from shifting the layout under itself. */
.num, td.num, .tile-value, .stat, .metric {
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1, "ss01" 1;
}

.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

.eyebrow {
  font-size: var(--step--1);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--ink-faint);
  font-weight: 600;
}

.good { color: var(--good); }
.bad { color: var(--bad); }
.cost { color: var(--cost); }
.dim { color: var(--ink-dim); }
.faint { color: var(--ink-faint); }

/* ------------------------------------------------------------- masthead */

header.top {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 24px; flex-wrap: wrap;
  padding: 28px 0 20px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 40px;
}
.brand { display: flex; align-items: baseline; gap: 14px; }
.brand .mark {
  font-size: var(--step-1); font-weight: 700; letter-spacing: -0.04em;
}
/* The mark carries a hairline underscore that draws itself once on load —
   the only ornament on the page, and it earns its place by being the thing
   that says the page is live. */
.brand .mark::after {
  content: ""; display: block; height: 2px; margin-top: 4px;
  background: var(--accent); border-radius: 2px;
  transform-origin: left;
  animation: draw var(--slow) var(--ease-out) both;
}
@keyframes draw { from { transform: scaleX(0); } to { transform: scaleX(1); } }

.pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 4px 11px; border-radius: 999px;
  border: 1px solid var(--line-bright); background: var(--panel);
  font-size: var(--step--1); color: var(--ink-dim);
}
.pill .dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--good);
  box-shadow: 0 0 0 0 var(--good);
  animation: pulse 2.4s var(--ease-io) infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(53, 208, 165, 0.5); }
  70%  { box-shadow: 0 0 0 7px rgba(53, 208, 165, 0); }
  100% { box-shadow: 0 0 0 0 rgba(53, 208, 165, 0); }
}

/* ---------------------------------------------------------------- panels */

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 24px;
}
.panel + .panel { margin-top: var(--gap); }
.panel-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 16px; flex-wrap: wrap; margin-bottom: 18px;
}

section { margin: 56px 0 0; scroll-margin-top: 24px; }

.grid { display: grid; gap: var(--gap); }
.grid.two { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.grid.tiles { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }

/* ----------------------------------------------------------------- tiles */

.tile {
  background: var(--raised);
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 16px 16px 14px;
  transition: border-color var(--fast) var(--ease-io),
              transform var(--fast) var(--ease-out);
}
.tile:hover { border-color: var(--line-bright); transform: translateY(-1px); }
.tile-key {
  font-size: var(--step--1); color: var(--ink-faint);
  text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600;
}
.tile-value {
  font-size: var(--step-2); font-weight: 600; letter-spacing: -0.03em;
  margin-top: 6px; line-height: 1.1;
}
.tile-note { font-size: var(--step--1); color: var(--ink-faint); margin-top: 4px; }

/* ---------------------------------------------------------------- tables */

table { width: 100%; border-collapse: collapse; font-size: var(--step-0); }
th {
  text-align: left; font-size: var(--step--1); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.09em; color: var(--ink-faint);
  padding: 0 12px 10px; border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
td { padding: 11px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
tbody tr { transition: background var(--fast) var(--ease-io); }
tbody tr:hover { background: var(--raised); }
td.num, th.num { text-align: right; }
.scroll-x { overflow-x: auto; margin: 0 -8px; padding: 0 8px; }

/* --------------------------------------------------------------- badges */

.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: var(--step--1); font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid currentColor; white-space: nowrap;
}
.badge.refused { color: var(--bad); }
.badge.opened { color: var(--good); }
.badge.stood { color: var(--neutral); }
.badge.halt { color: var(--cost); }

/* ----------------------------------------------------------------- motion */

/* Entrance. Children of a .stagger reveal in document order, which is the
   order the agent computed them in. */
.reveal { opacity: 0; transform: translateY(14px); }
.reveal.in {
  opacity: 1; transform: none;
  transition: opacity var(--slow) var(--ease-out),
              transform var(--slow) var(--ease-out);
}
.stagger > * { opacity: 0; transform: translateY(10px); }
.stagger.in > * {
  opacity: 1; transform: none;
  transition: opacity var(--mid) var(--ease-out), transform var(--mid) var(--ease-out);
}
.stagger.in > *:nth-child(1) { transition-delay: 0ms; }
.stagger.in > *:nth-child(2) { transition-delay: 55ms; }
.stagger.in > *:nth-child(3) { transition-delay: 110ms; }
.stagger.in > *:nth-child(4) { transition-delay: 165ms; }
.stagger.in > *:nth-child(5) { transition-delay: 220ms; }
.stagger.in > *:nth-child(6) { transition-delay: 275ms; }
.stagger.in > *:nth-child(7) { transition-delay: 330ms; }
.stagger.in > *:nth-child(8) { transition-delay: 385ms; }

/* An SVG series draws itself in the direction it was swept. */
.draw { stroke-dasharray: var(--len); stroke-dashoffset: var(--len); }
.draw.in { transition: stroke-dashoffset 1.1s var(--ease-io); stroke-dashoffset: 0; }

/* A waterfall bar grows from the baseline it is measured against. */
.grow { transform: scaleY(0); transform-origin: var(--origin, bottom); }
.grow.in { transition: transform var(--slow) var(--ease-out); transform: scaleY(1); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
  }
  .reveal, .stagger > * { opacity: 1; transform: none; }
  .draw { stroke-dashoffset: 0; }
  .grow { transform: none; }
}

/* ------------------------------------------------------------------ notes */

.note {
  border-left: 2px solid var(--line-bright);
  padding: 2px 0 2px 14px;
  color: var(--ink-faint);
  font-size: var(--step--1);
  max-width: 74ch;
}
.note strong { color: var(--ink-dim); font-weight: 600; }

footer.foot {
  margin-top: 72px; padding-top: 24px; border-top: 1px solid var(--line);
  color: var(--ink-faint); font-size: var(--step--1);
  display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap;
}

.theme-toggle {
  background: var(--panel); color: var(--ink-dim);
  border: 1px solid var(--line-bright); border-radius: var(--radius-sm);
  padding: 5px 11px; font: inherit; font-size: var(--step--1); cursor: pointer;
  transition: color var(--fast) var(--ease-io), border-color var(--fast) var(--ease-io);
}
.theme-toggle:hover { color: var(--ink); border-color: var(--ink-faint); }
.theme-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""


# ----------------------------------------------------------------- behaviour

SCRIPT = """
(function () {
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Theme. Remembered per viewer, and it never flashes: the attribute is set
     from storage before paint by the inline head script. */
  var toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("convex-theme", next); } catch (e) {}
      toggle.textContent = next === "light" ? "Dark" : "Light";
    });
  }

  /* Reveal on scroll. Anything marked reveals once, in document order, and
     is then left alone — nothing re-animates when you scroll back up. */
  var watched = document.querySelectorAll(".reveal, .stagger, .draw, .grow");
  if (!("IntersectionObserver" in window) || reduced) {
    watched.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        io.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    watched.forEach(function (el) { io.observe(el); });
  }

  /* Count a figure up to its value once, when it first appears. The text is
     already correct in the HTML, so a reader without JavaScript, or one who
     asked for less motion, sees the final number and never a zero. */
  function countUp(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    var places = parseInt(el.getAttribute("data-places") || "0", 10);
    var prefix = el.getAttribute("data-prefix") || "";
    var suffix = el.getAttribute("data-suffix") || "";
    var signed = el.hasAttribute("data-signed");
    var started = null;
    var span = 900;
    function frame(now) {
      if (started === null) started = now;
      var t = Math.min((now - started) / span, 1);
      var eased = 1 - Math.pow(1 - t, 3);
      var value = target * eased;
      var text = Math.abs(value).toLocaleString(undefined, {
        minimumFractionDigits: places, maximumFractionDigits: places
      });
      var sign = value < 0 ? "-" : (signed && value > 0 ? "+" : "");
      el.textContent = sign + prefix + text + suffix;
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  var counters = document.querySelectorAll("[data-count]");
  if (reduced || !("IntersectionObserver" in window)) {
    /* leave the server-rendered text exactly as it is */
  } else {
    var co = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        countUp(entry.target);
        co.unobserve(entry.target);
      });
    }, { threshold: 0.5 });
    counters.forEach(function (el) { co.observe(el); });
  }
})();
"""

# Set before first paint so a remembered light theme never flashes dark.
HEAD_SCRIPT = """
try {
  var t = localStorage.getItem("convex-theme");
  if (t) document.documentElement.setAttribute("data-theme", t);
} catch (e) {}
"""


def stylesheet() -> str:
    return TOKENS + BASE
