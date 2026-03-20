"use strict";

(() => {
  const core = globalThis.AIScreenerCore;
  const analysis = globalThis.AIScreenerAnalysis;
  const ui = globalThis.AIScreenerUI;
  if (!core || !analysis || !ui) {
    throw new Error("AIScreener content dependencies are missing");
  }

  const {
    ANALYSIS_CONCURRENCY,
    CACHE_PREFIX,
    CACHE_VERSION,
    CARD_CLASS,
    CARD_STATE_DONE,
    CARD_STATE_ERROR,
    CARD_STATE_LOADING,
    SEARCH_SCAN_DEBOUNCE_MS,
    getSettings,
    getFromStorage,
    isExpired,
    isGoogleSearchPage,
    makeCacheKey,
    removeFromStorage,
    sendMessage,
    setToStorage,
    createLimiter,
    normalizeText,
    hashString
  } = core;
  const {
    resolveFetchUrl,
    buildPayloadFromSnippet,
    buildPayloadFromMetaDescription,
    chooseBestPayload,
    createEmptyPayload,
    extractFromHtml,
    analyzePayload
  } = analysis;

  const RESULT_H3_SELECTOR = "#search a h3";
  const RESULT_CONTAINER_SELECTOR = "div.MjjYud, div.g";
  const RESULT_SNIPPET_SELECTOR = ".VwiC3b, .IsZvec, .s3v9rd";

  const analysisLimiter = createLimiter(ANALYSIS_CONCURRENCY);
  const pendingByUrl = new Map();

  let prefersDarkQuery = null;
  let settings = core.DEFAULT_SETTINGS;
  let scanTimer = null;

  init().catch((error) => {
    console.error("[ai-screener] initialization failed", error);
  });

  async function init() {
    if (!isGoogleSearchPage()) {
      return;
    }

    prefersDarkQuery = ui.setupThemeWatcher(applyThemeMode);
    applyThemeMode();
    ui.injectStyles();

    settings = await getSettings();
    await cleanupExpiredCache(settings.cacheTTLHours);

    chrome.storage.onChanged.addListener((changes, areaName) => {
      if (areaName !== "local" || !changes[core.SETTINGS_KEY]) {
        return;
      }

      settings = core.mergeSettings(changes[core.SETTINGS_KEY].newValue || {});
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

  function startGoogleSearchMode() {
    applyThemeMode();
    scanGoogleResults();

    const observer = new MutationObserver(scheduleSearchScan);
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    window.addEventListener("resize", handleWindowResize);
  }

  function applyThemeMode() {
    ui.applyThemeMode(prefersDarkQuery);
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
      ui.applyResultCardLayout(card);
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

    for (const result of collectGoogleResults()) {
      const card = ui.ensureResultCard(result.container);
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
          ui.renderResultCard(card, record);
        })
        .catch((error) => {
          console.warn("[ai-screener] result analysis failed", error);
          if (!isCardForUrl(card, result.url)) {
            return;
          }
          card.dataset.state = CARD_STATE_ERROR;
          ui.renderResultCardError(card);
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
    ui.renderResultCardChecking(card);
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
    const descriptionPayload = buildPayloadFromMetaDescription(result, fetched);
    const snippetPayload = buildPayloadFromSnippet(result);
    return chooseBestPayload(fetched, descriptionPayload, snippetPayload);
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
    const fetchUrl = resolveFetchUrl(url);
    if (!fetchUrl) {
      return createEmptyPayload("body");
    }

    try {
      const response = await sendMessage({
        type: "FETCH_HTML",
        url: fetchUrl
      });
      if (!response || !response.ok || !response.html) {
        return createEmptyPayload("body");
      }
      return extractFromHtml(response.html, response.finalUrl || fetchUrl);
    } catch (error) {
      console.debug("[ai-screener] fetch fallback to snippet", error);
      return createEmptyPayload("body");
    }
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

  async function cleanupExpiredCache(ttlHours) {
    const all = await getFromStorage(null);
    const now = Date.now();
    const ttlMs = core.getCacheTtlMs(ttlHours);
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
})();
