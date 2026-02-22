"use strict";

(() => {
  const SETTINGS_KEY = "settings";
  const DEFAULT_SETTINGS = {
    enabled: true,
    cacheTTLHours: 168
  };

  const enabled = document.getElementById("enabled");
  const cacheTTLHours = document.getElementById("cacheTTLHours");
  const status = document.getElementById("status");
  const saveButton = document.getElementById("save");
  const resetButton = document.getElementById("reset");

  saveButton.addEventListener("click", saveSettings);
  resetButton.addEventListener("click", resetSettings);

  loadSettings().catch((error) => {
    showStatus(`読込に失敗しました: ${error.message}`, true);
  });

  async function loadSettings() {
    const stored = await storageGet(SETTINGS_KEY);
    const merged = mergeSettings(stored[SETTINGS_KEY] || {});
    applyToForm(merged);
    showStatus("現在の設定を読み込みました");
  }

  async function saveSettings() {
    try {
      const settings = readFromForm();
      await storageSet({ [SETTINGS_KEY]: settings });
      showStatus("保存しました");
    } catch (error) {
      showStatus(`保存に失敗しました: ${error.message}`, true);
    }
  }

  function resetSettings() {
    applyToForm(DEFAULT_SETTINGS);
    showStatus("初期値をフォームに反映しました。保存を押すと確定します。");
  }

  function applyToForm(settings) {
    enabled.checked = settings.enabled !== false;
    cacheTTLHours.value = String(settings.cacheTTLHours);
  }

  function readFromForm() {
    return {
      enabled: Boolean(enabled.checked),
      cacheTTLHours: Math.max(1, Math.round(toNumber(cacheTTLHours.value) || 168))
    };
  }

  function mergeSettings(raw) {
    const merged = {
      ...DEFAULT_SETTINGS,
      ...(raw || {})
    };
    merged.cacheTTLHours = Math.max(1, Math.round(toNumber(merged.cacheTTLHours) || 168));
    merged.enabled = merged.enabled !== false;
    return merged;
  }

  function showStatus(message, isError = false) {
    status.textContent = message;
    status.style.color = isError ? "#b00020" : "#555";
  }

  function toNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  }

  function storageGet(key) {
    return new Promise((resolve, reject) => {
      chrome.storage.local.get(key, (items) => {
        const error = chrome.runtime.lastError;
        if (error) {
          reject(new Error(error.message));
          return;
        }
        resolve(items);
      });
    });
  }

  function storageSet(items) {
    return new Promise((resolve, reject) => {
      chrome.storage.local.set(items, () => {
        const error = chrome.runtime.lastError;
        if (error) {
          reject(new Error(error.message));
          return;
        }
        resolve();
      });
    });
  }
})();
