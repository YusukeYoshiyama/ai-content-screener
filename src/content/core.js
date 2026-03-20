"use strict";

(() => {
  const SETTINGS_KEY = "settings";
  const CACHE_PREFIX = "cache:";
  const CACHE_VERSION = 13;
  const DEFAULT_CACHE_TTL_HOURS = 24 * 7;
  const SEARCH_SCAN_DEBOUNCE_MS = 250;
  const ANALYSIS_CONCURRENCY = 2;
  const FETCH_SUPPORTED_PROTOCOL = "https:";
  const FETCH_UPGRADE_PROTOCOL = "http:";
  const CARD_CLASS = "ai-screener-card";
  const CARD_STATE_LOADING = "loading";
  const CARD_STATE_DONE = "done";
  const CARD_STATE_ERROR = "error";

  const DEFAULT_SETTINGS = {
    enabled: true,
    cacheTTLHours: DEFAULT_CACHE_TTL_HOURS
  };

  const HASH_MODEL_DEFAULT = resolveHashModel(globalThis.AI_SCREENER_HASH_MODEL);
  const HASH_MODEL_JA = globalThis.AI_SCREENER_HASH_MODEL_JA
    ? resolveHashModel(globalThis.AI_SCREENER_HASH_MODEL_JA)
    : null;

  function resolveHashModel(rawModel) {
    const fallback = {
      name: "fallback",
      type: "naive_bayes_hash3",
      dim: 4096,
      max_chars: 1200,
      prior_logit: 0,
      delta: new Array(4096).fill(0),
      calibration: null,
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
      type: String(rawModel.type || "naive_bayes_hash3"),
      dim,
      max_chars: maxChars,
      prior_logit: priorLogit,
      delta,
      calibration: resolveCalibrationModel(rawModel.calibration),
      thresholds: {
        human_max: Math.min(humanMax, aiMin),
        ai_min: Math.max(aiMin, humanMax)
      }
    };
  }

  function resolveCalibrationModel(rawCalibration) {
    if (!rawCalibration || typeof rawCalibration !== "object") {
      return null;
    }

    const weights = Array.isArray(rawCalibration.weights) ? rawCalibration.weights.map((value) => Number(value) || 0) : [];
    const means = Array.isArray(rawCalibration.means) ? rawCalibration.means.map((value) => Number(value) || 0) : [];
    const scales = Array.isArray(rawCalibration.scales) ? rawCalibration.scales.map((value) => Number(value) || 1) : [];
    if (weights.length === 0 || weights.length !== means.length || weights.length !== scales.length) {
      return null;
    }

    return {
      bias: Number(rawCalibration.bias) || 0,
      weights,
      means,
      scales
    };
  }

  function isGoogleSearchPage() {
    const host = location.hostname.toLowerCase();
    return host.includes("google.") && location.pathname === "/search";
  }

  function getSettings() {
    return getFromStorage(SETTINGS_KEY).then((result) => mergeSettings(result[SETTINGS_KEY] || {}));
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

  function makeCacheKey(url) {
    return `${CACHE_PREFIX}${hashString(url)}`;
  }

  function getCacheTtlMs(ttlHours) {
    const hours = Math.max(1, Number(ttlHours) || DEFAULT_CACHE_TTL_HOURS);
    return hours * 60 * 60 * 1000;
  }

  function isExpired(updatedAt, ttlHours) {
    if (!updatedAt) {
      return true;
    }
    return Date.now() - Number(updatedAt) > getCacheTtlMs(ttlHours);
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
      .replace(/\u3000/g, " ")
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

  function sigmoid(value) {
    const x = Number(value) || 0;
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
  }

  function hashString(input) {
    let hash = 5381;
    const str = String(input || "");
    for (let i = 0; i < str.length; i += 1) {
      hash = (hash * 33) ^ str.charCodeAt(i);
    }
    return (hash >>> 0).toString(36);
  }

  globalThis.AIScreenerCore = {
    SETTINGS_KEY,
    CACHE_PREFIX,
    CACHE_VERSION,
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_SETTINGS,
    SEARCH_SCAN_DEBOUNCE_MS,
    ANALYSIS_CONCURRENCY,
    FETCH_SUPPORTED_PROTOCOL,
    FETCH_UPGRADE_PROTOCOL,
    CARD_CLASS,
    CARD_STATE_LOADING,
    CARD_STATE_DONE,
    CARD_STATE_ERROR,
    HASH_MODEL_DEFAULT,
    HASH_MODEL_JA,
    isGoogleSearchPage,
    getSettings,
    mergeSettings,
    makeCacheKey,
    getCacheTtlMs,
    isExpired,
    sendMessage,
    getFromStorage,
    setToStorage,
    removeFromStorage,
    createLimiter,
    normalizeText,
    escapeHtml,
    clamp,
    sigmoid,
    hashString
  };
})();
