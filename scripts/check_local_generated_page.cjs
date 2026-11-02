"use strict";

// This trusted harness is run only inside a disposable, network-disabled
// Playwright container. Generated page JavaScript is never evaluated in the
// orchestrator process.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const { chromium } = require("playwright");

(async () => {
  const candidate = process.argv[2] || "/candidate/daily-practice.html";
  const html = fs.readFileSync(candidate);
  const server = http.createServer((request, response) => {
    if (request.method !== "GET" || request.url !== "/") {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Content-Length": html.length,
      "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src data:; form-action 'none'; base-uri 'none'",
      "X-Content-Type-Options": "nosniff",
    });
    response.end(html);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const url = `http://127.0.0.1:${address.port}/`;
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  const unexpectedRequests = [];
  const pageErrors = [];

  page.on("request", (request) => {
    if (request.url() !== url) unexpectedRequests.push(request.url());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  try {
    await page.goto(url, { waitUntil: "load", timeout: 10000 });
    assert.equal(await page.locator("html").getAttribute("lang"), "en");
    assert.equal(await page.getByRole("heading", { name: "Daily Practice", exact: true }).count(), 1);
    assert.equal(await page.getByRole("button", { name: "Mark Complete", exact: true }).count(), 1);
    const status = page.getByRole("status");
    assert.equal(await status.count(), 1);
    assert.equal(await status.getAttribute("aria-live"), "polite");
    assert.equal((await status.innerText()).trim(), "Not completed");
    assert.deepEqual(await page.evaluate(() => ({
      local: Object.keys(localStorage),
      session: Object.keys(sessionStorage),
      cookies: document.cookie,
    })), { local: [], session: [], cookies: "" });

    await page.getByRole("button", { name: "Mark Complete", exact: true }).click();
    assert.equal((await status.innerText()).trim(), "Completed");
    await page.reload({ waitUntil: "load", timeout: 10000 });
    assert.equal((await page.getByRole("status").innerText()).trim(), "Not completed");
    await page.getByRole("button", { name: "Mark Complete", exact: true }).focus();
    await page.keyboard.press("Enter");
    assert.equal((await page.getByRole("status").innerText()).trim(), "Completed");
    assert.deepEqual(await page.evaluate(() => ({
      local: Object.keys(localStorage),
      session: Object.keys(sessionStorage),
      cookies: document.cookie,
    })), { local: [], session: [], cookies: "" });
    assert.deepEqual(unexpectedRequests, [], "generated page requested an external resource");
    assert.deepEqual(pageErrors, [], "generated page raised a JavaScript error");
    console.log(JSON.stringify({
      status: "PASS",
      checks: ["accessible controls", "initial state", "click", "keyboard", "reload reset", "no persistence", "no network", "no page errors"],
      external_requests: 0,
    }));
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => {
  console.error(`LOCAL_PAGE_REJECTED: ${error.name}: ${error.message}`);
  process.exitCode = 1;
});
