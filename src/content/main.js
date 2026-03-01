"use strict";

(() => {
  const SETTINGS_KEY = "settings";
  const CACHE_PREFIX = "cache:";
  const CACHE_VERSION = 3;
  const DEFAULT_CACHE_TTL_HOURS = 24 * 7;
  const SEARCH_SCAN_DEBOUNCE_MS = 250;
  const ANALYSIS_CONCURRENCY = 2;
  const MIN_ANALYZABLE_TEXT_LENGTH = 180;
  const RESULT_H3_SELECTOR = "#search a h3";
  const RESULT_CONTAINER_SELECTOR = "div.MjjYud, div.g";
  const RESULT_SNIPPET_SELECTOR = ".VwiC3b, .IsZvec, .s3v9rd";
  const INLINE_CARD_BREAKPOINT = 1180;
  const CARD_CLASS = "ai-screener-card";
  const CARD_STATE_LOADING = "loading";
  const CARD_STATE_DONE = "done";
  const CARD_STATE_ERROR = "error";
  const CARD_TONE_CLASSES = [
    "ai-screener-card-ai",
    "ai-screener-card-human",
    "ai-screener-card-unknown",
    "ai-screener-card-checking",
    "ai-screener-card-error"
  ];
  const ARTICLE_CANDIDATE_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    "[itemprop='articleBody']",
    ".post",
    ".entry-content",
    ".article-body"
  ];

  const DEFAULT_SETTINGS = {
    enabled: true,
    cacheTTLHours: DEFAULT_CACHE_TTL_HOURS
  };

  const HASH_MODEL_DEFAULT = resolveHashModel(globalThis.AI_SCREENER_HASH_MODEL);
  const HASH_MODEL_JA = globalThis.AI_SCREENER_HASH_MODEL_JA
    ? resolveHashModel(globalThis.AI_SCREENER_HASH_MODEL_JA)
    : null;

  const analysisLimiter = createLimiter(ANALYSIS_CONCURRENCY);
  const pendingByUrl = new Map();

  let prefersDarkQuery = null;
  let settings = DEFAULT_SETTINGS;
  let observer = null;
  let scanTimer = null;

  init().catch((error) => {
    console.error("[ai-screener] initialization failed", error);
  });

  async function init() {
    if (!isGoogleSearchPage()) {
      return;
    }

    setupThemeWatcher();
    applyThemeMode();
    injectStyles();

    settings = await getSettings();
    await cleanupExpiredCache(settings.cacheTTLHours);

    chrome.storage.onChanged.addListener((changes, areaName) => {
      if (areaName !== "local" || !changes[SETTINGS_KEY]) {
        return;
      }

      settings = mergeSettings(changes[SETTINGS_KEY].newValue || {});
      if (!settings.enabled) {
        clearResultCards();
        return;
      }
      scanGoogleResults();
    });

    if (!settings.enabled) {
      clearResultCards();
      return;
    }

    startGoogleSearchMode();
  }

  function resolveHashModel(rawModel) {
    const fallback = {
      name: "fallback",
      dim: 4096,
      max_chars: 1200,
      prior_logit: 0,
      delta: new Array(4096).fill(0),
      thresholds: {
        human_max: 0.45,
        ai_min: 0.55
      }
    };

    if (!rawModel || typeof rawModel !== "object") {
      return fallback;
    }

    const dim = Math.max(1, Number(rawModel.dim) || fallback.dim);
    const maxChars = Math.max(200, Number(rawModel.max_chars) || fallback.max_chars);
    const priorLogit = Number(rawModel.prior_logit) || 0;

    let delta = Array.isArray(rawModel.delta) ? rawModel.delta.slice(0, dim) : [];
    if (delta.length < dim) {
      delta = delta.concat(new Array(dim - delta.length).fill(0));
    }

    const humanMax = clamp(Number(rawModel.thresholds?.human_max) || fallback.thresholds.human_max, 0, 1);
    const aiMin = clamp(Number(rawModel.thresholds?.ai_min) || fallback.thresholds.ai_min, 0, 1);

    return {
      name: String(rawModel.name || "hash_nb_char3"),
      dim,
      max_chars: maxChars,
      prior_logit: priorLogit,
      delta,
      thresholds: {
        human_max: Math.min(humanMax, aiMin),
        ai_min: Math.max(aiMin, humanMax)
      }
    };
  }

  function isGoogleSearchPage() {
    const host = location.hostname.toLowerCase();
    return host.includes("google.") && location.pathname === "/search";
  }

  function startGoogleSearchMode() {
    applyThemeMode();
    scanGoogleResults();

    observer = new MutationObserver(scheduleSearchScan);
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    window.addEventListener("resize", handleWindowResize);
  }

  function scheduleSearchScan() {
    if (scanTimer) {
      clearTimeout(scanTimer);
    }

    scanTimer = setTimeout(() => {
      applyThemeMode();
      scanGoogleResults();
    }, SEARCH_SCAN_DEBOUNCE_MS);
  }

  function handleWindowResize() {
    applyThemeMode();
    for (const card of document.querySelectorAll(`.${CARD_CLASS}`)) {
      applyResultCardLayout(card);
    }
  }

  function clearResultCards() {
    for (const card of document.querySelectorAll(`.${CARD_CLASS}`)) {
      card.remove();
    }
  }

  function scanGoogleResults() {
    if (!settings.enabled) {
      clearResultCards();
      return;
    }

    const results = collectGoogleResults();
    for (const result of results) {
      const card = ensureResultCard(result.container);
      if (isCompletedCardForUrl(card, result.url)) {
        continue;
      }

      setCardLoadingState(card, result.url);

      analyzeResult(result)
        .then((record) => {
          if (!isCardForUrl(card, result.url)) {
            return;
          }

          card.dataset.state = CARD_STATE_DONE;
          renderResultCard(card, record);
        })
        .catch((error) => {
          console.warn("[ai-screener] result analysis failed", error);
          if (!isCardForUrl(card, result.url)) {
            return;
          }

          card.dataset.state = CARD_STATE_ERROR;
          renderResultCardError(card);
        });
    }
  }

  function isCompletedCardForUrl(card, url) {
    return card.dataset.url === url && card.dataset.state === CARD_STATE_DONE;
  }

  function isCardForUrl(card, url) {
    return card.dataset.url === url;
  }

  function setCardLoadingState(card, url) {
    card.dataset.url = url;
    card.dataset.state = CARD_STATE_LOADING;
    renderResultCardChecking(card);
  }

  function collectGoogleResults() {
    const h3Nodes = document.querySelectorAll(RESULT_H3_SELECTOR);
    const seenContainers = new Set();
    const results = [];

    for (const h3 of h3Nodes) {
      const result = buildResultFromHeading(h3, seenContainers);
      if (result) {
        results.push(result);
      }
    }

    return results;
  }

  function buildResultFromHeading(h3, seenContainers) {
    const anchor = h3.closest("a[href]");
    if (!anchor) {
      return null;
    }

    const url = normalizeResultUrl(anchor.href);
    if (!/^https?:\/\//i.test(url)) {
      return null;
    }

    const container = anchor.closest(RESULT_CONTAINER_SELECTOR);
    if (!container || seenContainers.has(container)) {
      return null;
    }

    seenContainers.add(container);

    const snippetNode = container.querySelector(RESULT_SNIPPET_SELECTOR);
    const snippet = normalizeText(snippetNode ? snippetNode.innerText : "");
    const title = normalizeText(h3.innerText || "");

    return {
      container,
      url,
      title,
      snippet
    };
  }

  async function analyzeResult(result) {
    const memoized = pendingByUrl.get(result.url);
    if (memoized) {
      return memoized;
    }

    const task = analysisLimiter(async () => {
      const cacheKey = makeCacheKey(result.url);
      const cached = await getValidCache(cacheKey);
      if (cached) {
        return cached;
      }

      const payload = await resolvePayloadForResult(result);
      const record = createAnalysisRecord(payload);
      await setToStorage({ [cacheKey]: record });
      return record;
    });

    pendingByUrl.set(result.url, task);
    try {
      return await task;
    } finally {
      pendingByUrl.delete(result.url);
    }
  }

  async function resolvePayloadForResult(result) {
    const fetched = await fetchAndExtractPayload(result.url);
    if (shouldFallbackToSnippet(fetched)) {
      return buildPayloadFromSnippet(result);
    }
    return fetched;
  }

  function shouldFallbackToSnippet(payload) {
    return !payload.text || payload.text.length < MIN_ANALYZABLE_TEXT_LENGTH;
  }

  function createAnalysisRecord(payload) {
    const analyzed = analyzePayload(payload);
    return {
      ...analyzed,
      contentHash: hashString(payload.text || ""),
      cacheVersion: CACHE_VERSION,
      updatedAt: Date.now(),
      source: payload.source || "snippet"
    };
  }

  async function getValidCache(cacheKey) {
    const cacheMap = await getFromStorage(cacheKey);
    const cached = cacheMap ? cacheMap[cacheKey] : null;
    if (
      !cached ||
      Number(cached.cacheVersion) !== CACHE_VERSION ||
      isExpired(cached.updatedAt, settings.cacheTTLHours)
    ) {
      return null;
    }
    return cached;
  }

  async function fetchAndExtractPayload(url) {
    try {
      const response = await sendMessage({
        type: "FETCH_HTML",
        url
      });
      if (!response || !response.ok || !response.html) {
        return createEmptyPayload("fetched");
      }
      return extractFromHtml(response.html, response.finalUrl || url);
    } catch (error) {
      console.debug("[ai-screener] fetch fallback to snippet", error);
      return createEmptyPayload("fetched");
    }
  }

  function buildPayloadFromSnippet(result) {
    const text = normalizeText(`${result.title || ""}\n${result.snippet || ""}`);
    return {
      text,
      headingsText: result.title || "",
      externalLinkCount: 0,
      source: "snippet"
    };
  }

  function normalizeResultUrl(href) {
    try {
      const parsed = new URL(href, location.href);
      const isGoogleRedirect = parsed.hostname.includes("google.") && parsed.pathname === "/url";
      if (isGoogleRedirect) {
        const target = parsed.searchParams.get("url") || parsed.searchParams.get("q");
        if (target && /^https?:\/\//i.test(target)) {
          return target;
        }
      }
      return parsed.href;
    } catch (_error) {
      return href;
    }
  }

  function analyzePayload(payload) {
    const selectedModel = selectHashModel(payload.text || payload.headingsText || "");
    const score = computeHashScore(payload.text || "", selectedModel);
    const judge = toJudge(score, selectedModel);

    return {
      score,
      judge,
      displayScore: score.toFixed(2)
    };
  }

  function selectHashModel(text) {
    if (HASH_MODEL_JA && isLikelyJapaneseText(text)) {
      return HASH_MODEL_JA;
    }
    return HASH_MODEL_DEFAULT;
  }

  function isLikelyJapaneseText(text) {
    const sample = String(text || "").slice(0, 2400);
    if (!sample) {
      return false;
    }

    const matches = sample.match(/[\u3040-\u30ff\u4e00-\u9fff]/g);
    const jpCount = matches ? matches.length : 0;
    if (jpCount >= 40) {
      return true;
    }
    return jpCount >= 8 && jpCount / Math.max(1, sample.length) >= 0.03;
  }

  function computeHashScore(text, model) {
    const activeModel = model || HASH_MODEL_DEFAULT;
    const normalized = normalizeHashText(text);
    if (normalized.length < 3) {
      return 0.5;
    }

    const limit = Math.min(normalized.length, activeModel.max_chars);
    let logit = activeModel.prior_logit;

    for (let i = 0; i < limit - 2; i += 1) {
      const bin = hashTrigram(normalized, i, activeModel.dim);
      logit += activeModel.delta[bin] || 0;
    }

    return sigmoid(logit);
  }

  function normalizeHashText(value) {
    return normalizeText(value || "")
      .toLowerCase()
      .replace(/\u00a0/g, " ");
  }

  function hashTrigram(text, start, dim) {
    let h = 2166136261;
    for (let i = 0; i < 3; i += 1) {
      h ^= text.charCodeAt(start + i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h % dim;
  }

  function toJudge(score, model) {
    const activeModel = model || HASH_MODEL_DEFAULT;
    if (score < activeModel.thresholds.human_max) {
      return "Human";
    }
    if (score < activeModel.thresholds.ai_min) {
      return "Unknown";
    }
    return "AI";
  }

  function sigmoid(value) {
    const x = Number(value) || 0;
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
  }

  function extractFromHtml(html, url) {
    try {
      const sanitized = sanitizeFetchedHtml(html);
      const doc = new DOMParser().parseFromString(sanitized, "text/html");
      return extractFromDocument(doc, url, "fetched");
    } catch (_error) {
      return createEmptyPayload("fetched");
    }
  }

  function sanitizeFetchedHtml(html) {
    return String(html || "").replace(/<base\b[^>]*>/gi, "");
  }

  function createEmptyPayload(source) {
    return {
      text: "",
      headingsText: "",
      externalLinkCount: 0,
      source
    };
  }

  function extractFromDocument(doc, url, source = "document") {
    const body = doc.body;
    if (!body) {
      return createEmptyPayload(source);
    }

    const clone = body.cloneNode(true);
    for (const node of clone.querySelectorAll("script,style,noscript,svg,canvas,iframe,form")) {
      node.remove();
    }
    for (const node of clone.querySelectorAll("nav,footer,header,aside")) {
      node.remove();
    }

    const mainElement = pickMainTextElement(clone);
    const text = normalizeText(mainElement.innerText || clone.innerText || "");
    const headingsText = Array.from(mainElement.querySelectorAll("h1,h2,h3"))
      .map((node) => normalizeText(node.innerText || ""))
      .filter(Boolean)
      .join(" ");

    return {
      text,
      headingsText,
      externalLinkCount: countExternalLinks(mainElement, url),
      source
    };
  }

  function pickMainTextElement(root) {
    let best = null;
    let bestLength = 0;

    for (const selector of ARTICLE_CANDIDATE_SELECTORS) {
      for (const element of root.querySelectorAll(selector)) {
        const length = normalizeText(element.innerText || "").length;
        if (length > bestLength) {
          best = element;
          bestLength = length;
        }
      }
    }

    return best || root;
  }

  function countExternalLinks(root, pageUrl) {
    let host = "";
    try {
      host = new URL(pageUrl).host;
    } catch (_error) {
      host = "";
    }

    let count = 0;
    for (const anchor of root.querySelectorAll("a[href]")) {
      const href = anchor.getAttribute("href");
      if (!href || !/^https?:\/\//i.test(href)) {
        continue;
      }

      try {
        const linked = new URL(href);
        if (!host || linked.host !== host) {
          count += 1;
        }
      } catch (_error) {
        continue;
      }
    }

    return count;
  }

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

  async function cleanupExpiredCache(ttlHours) {
    const all = await getFromStorage(null);
    const now = Date.now();
    const ttlMs = getCacheTtlMs(ttlHours);
    const keysToDelete = [];

    for (const [key, value] of Object.entries(all || {})) {
      if (!key.startsWith(CACHE_PREFIX) || !value || typeof value !== "object") {
        continue;
      }

      const updatedAt = Number(value.updatedAt) || 0;
      if (!updatedAt || now - updatedAt > ttlMs) {
        keysToDelete.push(key);
      }
    }

    if (keysToDelete.length > 0) {
      await removeFromStorage(keysToDelete);
    }
  }

  function isExpired(updatedAt, ttlHours) {
    if (!updatedAt) {
      return true;
    }
    return Date.now() - Number(updatedAt) > getCacheTtlMs(ttlHours);
  }

  function getCacheTtlMs(ttlHours) {
    const hours = Math.max(1, Number(ttlHours) || DEFAULT_CACHE_TTL_HOURS);
    return hours * 60 * 60 * 1000;
  }

  function makeCacheKey(url) {
    return `${CACHE_PREFIX}${hashString(url)}`;
  }

  async function getSettings() {
    const result = await getFromStorage(SETTINGS_KEY);
    return mergeSettings(result[SETTINGS_KEY] || {});
  }

  function mergeSettings(raw) {
    const merged = {
      ...DEFAULT_SETTINGS,
      ...(raw || {})
    };

    return {
      enabled: merged.enabled !== false,
      cacheTTLHours: Math.max(1, Number(merged.cacheTTLHours) || DEFAULT_CACHE_TTL_HOURS)
    };
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

  function setupThemeWatcher() {
    if (typeof window.matchMedia !== "function") {
      return;
    }

    prefersDarkQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyThemeMode();

    if (typeof prefersDarkQuery.addEventListener === "function") {
      prefersDarkQuery.addEventListener("change", handler);
      return;
    }
    if (typeof prefersDarkQuery.addListener === "function") {
      prefersDarkQuery.addListener(handler);
    }
  }

  function applyThemeMode() {
    const root = document.documentElement;
    const mode = detectThemeMode();
    root.classList.remove("ai-screener-theme-light", "ai-screener-theme-dark");
    root.classList.add(`ai-screener-theme-${mode}`);
  }

  function detectThemeMode() {
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

  async function sendMessage(message) {
    return callChromeApi((callback) => chrome.runtime.sendMessage(message, callback));
  }

  function getFromStorage(key) {
    return callChromeApi((callback) => chrome.storage.local.get(key, callback));
  }

  function setToStorage(items) {
    return callChromeApi((callback) => {
      chrome.storage.local.set(items, () => callback());
    });
  }

  function removeFromStorage(keys) {
    return callChromeApi((callback) => {
      chrome.storage.local.remove(keys, () => callback());
    });
  }

  function callChromeApi(executor) {
    return new Promise((resolve, reject) => {
      executor((value) => {
        const error = chrome.runtime.lastError;
        if (error) {
          reject(new Error(error.message));
          return;
        }
        resolve(value);
      });
    });
  }

  function createLimiter(limit) {
    let active = 0;
    const queue = [];

    const pump = () => {
      if (active >= limit || queue.length === 0) {
        return;
      }

      const entry = queue.shift();
      active += 1;

      Promise.resolve()
        .then(entry.task)
        .then(entry.resolve, entry.reject)
        .finally(() => {
          active -= 1;
          pump();
        });
    };

    return (task) =>
      new Promise((resolve, reject) => {
        queue.push({ task, resolve, reject });
        pump();
      });
  }

  function normalizeText(value) {
    return String(value || "")
      .replace(/\u00a0/g, " ")
      .replace(/[ \t\f\v]+/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function clamp(value, min, max) {
    const safeMin = Number.isFinite(min) ? min : 0;
    const safeMax = Number.isFinite(max) ? max : 1;
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return safeMin;
    }
    return Math.min(safeMax, Math.max(safeMin, num));
  }

  function hashString(input) {
    let hash = 5381;
    const str = String(input || "");
    for (let i = 0; i < str.length; i += 1) {
      hash = (hash * 33) ^ str.charCodeAt(i);
    }
    return (hash >>> 0).toString(36);
  }
})();
