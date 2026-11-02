import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const MAX_HTML_BYTES = 100_000;

function reject(code) {
  const error = new Error(code);
  error.code = code;
  throw error;
}

function validateStatic(html) {
  if (typeof html !== "string" || Buffer.byteLength(html, "utf8") > MAX_HTML_BYTES) {
    reject("output-size");
  }
  if (!/^\s*<!doctype html\s*>/i.test(html)) reject("doctype");
  if (/<(?:script|link|iframe|img|audio|video|source|object|embed)\b[^>]*(?:src|href|data)\s*=/i.test(html)) {
    reject("external-dependency");
  }
}

export async function validateHtml(html) {
  validateStatic(html);

  const csp = [
    "default-src 'self'",
    "script-src 'unsafe-inline'",
    "style-src 'unsafe-inline'",
    "connect-src 'none'",
    "img-src data:",
    "font-src 'none'",
    "media-src 'none'",
    "object-src 'none'",
    "frame-src 'none'",
    "form-action 'none'",
    "base-uri 'none'"
  ].join("; ");
  const server = createServer((request, response) => {
    if (request.method === "GET" && request.url === "/favicon.ico") {
      response.writeHead(204);
      response.end();
      return;
    }
    if (request.method !== "GET" || request.url !== "/") {
      response.writeHead(404);
      response.end();
      return;
    }
    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Content-Security-Policy": csp,
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    });
    response.end(html);
  });

  let browser;
  try {
    await new Promise((resolve, rejectPromise) => {
      server.once("error", rejectPromise);
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    const origin = "http://127.0.0.1:" + address.port;

    // Explicitly enable Chromium's process sandbox. If it cannot start safely,
    // fail closed; never fall back to --no-sandbox or Node's vm module.
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: true,
      args: ["--disable-background-networking", "--disable-sync", "--no-first-run"]
    });
    const context = await browser.newContext({
      serviceWorkers: "block",
      acceptDownloads: false
    });

    let networkAttempts = 0;
    let policyViolations = 0;
    let pageErrors = 0;
    let popups = 0;

    await context.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.origin === origin) {
        await route.continue();
        return;
      }
      if (["data:", "blob:", "about:"].includes(url.protocol)) {
        await route.continue();
        return;
      }
      networkAttempts += 1;
      await route.abort("blockedbyclient");
    });
    await context.routeWebSocket("**/*", (socket) => {
      networkAttempts += 1;
      socket.close();
    });

    const page = await context.newPage();
    page.setDefaultTimeout(5_000);
    page.on("pageerror", () => { pageErrors += 1; });
    page.on("console", (message) => {
      if (message.type() === "error") pageErrors += 1;
    });
    page.on("popup", (popup) => {
      popups += 1;
      popup.close().catch(() => {});
    });
    await page.addInitScript(() => {
      let count = 0;
      window.addEventListener("securitypolicyviolation", () => { count += 1; });
      Object.defineProperty(window, "__adosCspViolationCount", {
        value: () => count,
        configurable: false,
        writable: false
      });
      for (const api of ["RTCPeerConnection", "webkitRTCPeerConnection", "WebTransport"]) {
        try {
          Object.defineProperty(window, api, {
            value: undefined,
            configurable: false,
            writable: false
          });
        } catch {}
      }
    });

    await page.goto(origin + "/", { waitUntil: "load", timeout: 10_000 });
    const structure = await page.evaluate(() => ({
      lang: document.documentElement.getAttribute("lang"),
      title: document.title.trim(),
      doctype: document.doctype && document.doctype.name.toLowerCase(),
      externalAssets: document.querySelectorAll(
        "script[src], link[href], iframe[src], img[src], audio[src], video[src], source[src], object[data], embed[src]"
      ).length,
      cspViolations: window.__adosCspViolationCount()
    }));
    if (!structure.doctype || !structure.lang || !structure.title) reject("document-structure");
    if (structure.externalAssets !== 0) reject("external-dependency");
    if (await page.getByRole("heading", { level: 1, name: "Daily Practice", exact: true }).count() !== 1) {
      reject("heading");
    }
    const button = page.getByRole("button", { name: "Mark Complete", exact: true });
    if (await button.count() !== 1 || !(await button.isVisible()) || !(await button.isEnabled())) {
      reject("accessible-button");
    }
    const status = page.getByRole("status");
    if (await status.count() !== 1 || (await status.getAttribute("aria-live")) !== "polite") {
      reject("accessible-status");
    }
    if ((await status.innerText()).trim() !== "Not completed") reject("initial-status");

    await button.click();
    if ((await status.innerText()).trim() !== "Completed") reject("completion");
    const afterClickViolations = await page.evaluate(() => window.__adosCspViolationCount());
    if (networkAttempts || afterClickViolations) reject("network-attempt");
    await page.reload({ waitUntil: "load", timeout: 10_000 });
    const reloadedStatus = page.getByRole("status");
    if ((await reloadedStatus.innerText()).trim() !== "Not completed") reject("reload-reset");

    if (networkAttempts || structure.cspViolations) reject("network-attempt");
    if (pageErrors) reject("browser-error");
    if (popups) reject("popup");
    await context.close();
    return {
      html_structure: "PASS",
      accessibility: "PASS",
      button_behavior: "PASS",
      reload_reset: "PASS",
      external_network: "PASS",
      sandbox: "Chromium process sandbox enabled"
    };
  } catch (error) {
    if (error && error.code) throw error;
    if (process.env.GEMINI_VALIDATOR_DEBUG === "1") {
      console.error("VALIDATOR_DEBUG:", error && error.name, error && error.message);
    }
    reject("browser-isolation-or-runtime");
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (server.listening) {
      await new Promise((resolve) => server.close(() => resolve()));
    }
  }
}

if (process.argv[1] && import.meta.url === new URL("file://" + process.argv[1]).href) {
  try {
    const path = process.argv[2];
    if (!path) reject("missing-input");
    const html = await readFile(path, "utf8");
    const result = await validateHtml(html);
    console.log("FUNCTIONAL_VALIDATION: PASS");
    for (const [key, value] of Object.entries(result)) {
      console.log(key.toUpperCase() + "=" + value);
    }
  } catch (error) {
    const code = error && typeof error.code === "string" ? error.code : "unknown";
    console.error("FUNCTIONAL_VALIDATION: FAIL; code=" + code);
    process.exitCode = 1;
  }
}
