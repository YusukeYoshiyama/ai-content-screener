"use strict";

const FETCH_TIMEOUT_MS = 8000;
const MAX_HTML_BYTES = 1024 * 1024 * 2;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "FETCH_HTML" || !message.url) {
    return undefined;
  }

  fetchHtml(message.url)
    .then((result) => {
      sendResponse({ ok: true, ...result });
    })
    .catch((error) => {
      sendResponse({
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      });
    });

  return true;
});

async function fetchHtml(url) {
  if (!/^https?:\/\//i.test(url)) {
    throw new Error("Unsupported URL");
  }

  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: "GET",
      redirect: "follow",
      signal: controller.signal
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("text/html")) {
      throw new Error("Non-HTML response");
    }

    let html = await response.text();
    if (html.length > MAX_HTML_BYTES) {
      html = html.slice(0, MAX_HTML_BYTES);
    }

    return {
      html,
      finalUrl: response.url,
      status: response.status
    };
  } finally {
    clearTimeout(timerId);
  }
}
