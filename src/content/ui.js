"use strict";

(() => {
  const core = globalThis.AIScreenerCore;
  if (!core) {
    throw new Error("AIScreenerCore is required before AIScreenerUI");
  }

  const { CARD_CLASS, clamp, escapeHtml } = core;
  const INLINE_CARD_BREAKPOINT = 1180;
  const CARD_TONE_CLASSES = [
    "ai-screener-card-ai",
    "ai-screener-card-human",
    "ai-screener-card-unknown",
    "ai-screener-card-checking",
    "ai-screener-card-error"
  ];

  function ensureResultCard(container) {
    let card = container.querySelector(`.${CARD_CLASS}`);
    if (!card) {
      card = document.createElement("div");
      card.className = `${CARD_CLASS} ai-screener-card-checking`;
      container.prepend(card);
    }

    const computed = window.getComputedStyle(container);
    if (computed.position === "static") {
      container.style.position = "relative";
    }

    applyResultCardLayout(card);
    return card;
  }

  function applyResultCardLayout(card) {
    if (window.innerWidth < INLINE_CARD_BREAKPOINT) {
      card.classList.add("ai-screener-card-inline");
      return;
    }
    card.classList.remove("ai-screener-card-inline");
  }

  function renderResultCardChecking(card) {
    setResultCardTone(card, "checking");
    card.innerHTML = createResultCardMarkup("--", "Checking");
    card.title = "判定中";
  }

  function renderResultCard(card, record) {
    const tone = judgeToTone(record.judge);
    setResultCardTone(card, tone);
    card.innerHTML = createResultCardMarkup(record.displayScore, record.judge);
    card.title = `Score: ${record.displayScore} / Judge: ${record.judge}`;
  }

  function renderResultCardError(card) {
    setResultCardTone(card, "error");
    card.innerHTML = createResultCardMarkup("--", "N/A");
    card.title = "Score: -- / Judge: N/A";
  }

  function setResultCardTone(card, tone) {
    card.classList.remove(...CARD_TONE_CLASSES);
    card.classList.add(`ai-screener-card-${tone}`);
  }

  function judgeToTone(judge) {
    if (judge === "AI") {
      return "ai";
    }
    if (judge === "Human") {
      return "human";
    }
    return "unknown";
  }

  function createResultCardMarkup(displayScore, judge) {
    return `
      <div class="ai-screener-mini-row">
        <span class="ai-screener-mini-k">Score</span>
        <strong class="ai-screener-mini-v">${escapeHtml(displayScore)}</strong>
      </div>
      <div class="ai-screener-mini-row">
        <span class="ai-screener-mini-k">Judge</span>
        <strong class="ai-screener-mini-v ai-screener-mini-judge">${escapeHtml(judge)}</strong>
      </div>
    `;
  }

  function injectStyles() {
    if (document.getElementById("ai-screener-style")) {
      return;
    }

    const style = document.createElement("style");
    style.id = "ai-screener-style";
    style.textContent = `
      :root {
        --ais-text: #1a1f2b;
        --ais-muted: #586174;
        --ais-surface: rgba(255, 255, 255, 0.96);
        --ais-border: rgba(38, 49, 66, 0.2);
        --ais-shadow: rgba(14, 20, 29, 0.15);
        --ais-tone-neutral: #6b7688;
        --ais-tone-unknown: #5f739d;
        --ais-tone-human: #1f8f67;
        --ais-tone-ai: #cf405d;
        --ais-tone-error: #c08a3f;
        --ais-judge-unknown: #38507a;
        --ais-judge-human: #166b4d;
        --ais-judge-ai: #9b1f38;
        --ais-judge-error: #8f6226;
      }
      :root.ai-screener-theme-dark {
        --ais-text: #e6edf7;
        --ais-muted: #9ca8bc;
        --ais-surface: rgba(19, 24, 33, 0.95);
        --ais-border: rgba(161, 182, 210, 0.28);
        --ais-shadow: rgba(0, 0, 0, 0.42);
        --ais-tone-neutral: #a3b2c8;
        --ais-tone-unknown: #b4c7ea;
        --ais-tone-human: #5bd1ab;
        --ais-tone-ai: #ff8fa6;
        --ais-tone-error: #f5bc74;
        --ais-judge-unknown: #d3e0fb;
        --ais-judge-human: #a7edd8;
        --ais-judge-ai: #ffc0cd;
        --ais-judge-error: #ffe0b8;
      }
      .ai-screener-card {
        --ais-accent: var(--ais-tone-neutral);
        --ais-judge: var(--ais-muted);
        box-sizing: border-box;
        border: 1px solid var(--ais-border);
        border-left: 3px solid var(--ais-accent);
        border-radius: 8px;
        background: var(--ais-surface);
        color: var(--ais-text);
        font: 600 13px/1.3 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing: 0;
        box-shadow: 0 2px 8px var(--ais-shadow);
        pointer-events: none;
        position: absolute;
        left: -138px;
        top: 0;
        width: 128px;
        padding: 8px 9px;
        z-index: 2;
      }
      .ai-screener-mini-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
      }
      .ai-screener-mini-row + .ai-screener-mini-row {
        margin-top: 4px;
      }
      .ai-screener-mini-k {
        color: var(--ais-muted);
        font-size: 12px;
      }
      .ai-screener-mini-v {
        font-size: 13px;
        font-weight: 700;
      }
      .ai-screener-mini-judge {
        color: var(--ais-judge);
      }
      .ai-screener-card-checking {
        --ais-accent: var(--ais-tone-neutral);
        --ais-judge: var(--ais-muted);
      }
      .ai-screener-card-human {
        --ais-accent: var(--ais-tone-human);
        --ais-judge: var(--ais-judge-human);
      }
      .ai-screener-card-unknown {
        --ais-accent: var(--ais-tone-unknown);
        --ais-judge: var(--ais-judge-unknown);
      }
      .ai-screener-card-ai {
        --ais-accent: var(--ais-tone-ai);
        --ais-judge: var(--ais-judge-ai);
      }
      .ai-screener-card-error {
        --ais-accent: var(--ais-tone-error);
        --ais-judge: var(--ais-judge-error);
      }
      .ai-screener-card-inline {
        position: static;
        left: auto;
        top: auto;
        width: 128px;
        margin: 0 0 8px;
      }
      @media (max-width: 1179px) {
        .ai-screener-card {
          position: static;
          left: auto;
          top: auto;
          width: 128px;
          margin: 0 0 8px;
        }
      }
    `;

    document.documentElement.appendChild(style);
  }

  function setupThemeWatcher(onChange) {
    if (typeof window.matchMedia !== "function") {
      return null;
    }

    const prefersDarkQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => onChange();

    if (typeof prefersDarkQuery.addEventListener === "function") {
      prefersDarkQuery.addEventListener("change", handler);
    } else if (typeof prefersDarkQuery.addListener === "function") {
      prefersDarkQuery.addListener(handler);
    }

    return prefersDarkQuery;
  }

  function applyThemeMode(prefersDarkQuery) {
    const root = document.documentElement;
    const mode = detectThemeMode(prefersDarkQuery);
    root.classList.remove("ai-screener-theme-light", "ai-screener-theme-dark");
    root.classList.add(`ai-screener-theme-${mode}`);
  }

  function detectThemeMode(prefersDarkQuery) {
    const bodyBg = readNodeBackgroundColor(document.body);
    const rootBg = readNodeBackgroundColor(document.documentElement);
    const background = bodyBg || rootBg;

    if (background && isDarkColor(background)) {
      return "dark";
    }
    if (prefersDarkQuery && prefersDarkQuery.matches) {
      return "dark";
    }
    return "light";
  }

  function readNodeBackgroundColor(node) {
    if (!node) {
      return "";
    }

    const value = window.getComputedStyle(node).backgroundColor || "";
    if (!value || value === "transparent" || value === "rgba(0, 0, 0, 0)") {
      return "";
    }
    return value;
  }

  function isDarkColor(color) {
    const rgb = parseColorToRgb(color);
    if (!rgb) {
      return false;
    }

    const [r, g, b] = rgb.map((value) => value / 255);
    const luminance = 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
    return luminance < 0.48;
  }

  function parseColorToRgb(color) {
    const rgbMatch = color.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
    if (rgbMatch) {
      return [
        clamp(Number(rgbMatch[1]) || 0, 0, 255),
        clamp(Number(rgbMatch[2]) || 0, 0, 255),
        clamp(Number(rgbMatch[3]) || 0, 0, 255)
      ];
    }

    const hex = color.trim().replace("#", "");
    if (/^[a-f\d]{3}$/i.test(hex)) {
      return [
        parseInt(hex[0] + hex[0], 16),
        parseInt(hex[1] + hex[1], 16),
        parseInt(hex[2] + hex[2], 16)
      ];
    }
    if (/^[a-f\d]{6}$/i.test(hex)) {
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16)
      ];
    }
    return null;
  }

  function toLinear(channel) {
    if (channel <= 0.03928) {
      return channel / 12.92;
    }
    return ((channel + 0.055) / 1.055) ** 2.4;
  }

  globalThis.AIScreenerUI = {
    ensureResultCard,
    applyResultCardLayout,
    renderResultCardChecking,
    renderResultCard,
    renderResultCardError,
    injectStyles,
    setupThemeWatcher,
    applyThemeMode
  };
})();
