import test from "node:test";
import assert from "node:assert/strict";
import { validateHtml } from "./validate_gemini_html.mjs";

const good = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Daily Practice</title></head>
<body>
  <main>
    <h1>Daily Practice</h1>
    <button type="button" id="complete">Mark Complete</button>
    <p id="status" role="status" aria-live="polite">Not completed</p>
  </main>
  <script>
    document.getElementById("complete").addEventListener("click", () => {
      document.getElementById("status").textContent = "Completed";
    });
  </script>
</body>
</html>`;

test("valid accessible page works and resets on reload", async () => {
  const result = await validateHtml(good);
  assert.equal(result.button_behavior, "PASS");
  assert.equal(result.reload_reset, "PASS");
  assert.match(result.sandbox, /enabled/);
});

test("rejects a no-op button", async () => {
  const html = good.replace(
    'document.getElementById("status").textContent = "Completed";',
    ""
  );
  await assert.rejects(validateHtml(html), (error) => error.code === "completion");
});

test("rejects completion that persists after reload", async () => {
  const html = good.replace(
    'document.getElementById("status").textContent = "Completed";',
    'localStorage.setItem("done", "1"); document.getElementById("status").textContent = "Completed";'
  ).replace(
    '<p id="status" role="status" aria-live="polite">Not completed</p>',
    '<p id="status" role="status" aria-live="polite">Not completed</p><script>if (localStorage.getItem("done")) document.getElementById("status").textContent = "Completed";</script>'
  );
  await assert.rejects(validateHtml(html), (error) => error.code === "reload-reset");
});

test("rejects external assets before execution", async () => {
  const html = good.replace("</head>", '<script src="https://example.invalid/worker.js"></script></head>');
  await assert.rejects(validateHtml(html), (error) => error.code === "external-dependency");
});

test("rejects external network attempts during page behavior", async () => {
  const html = good.replace(
    'document.getElementById("status").textContent = "Completed";',
    'fetch("https://example.invalid/collect"); document.getElementById("status").textContent = "Completed";'
  );
  await assert.rejects(validateHtml(html), (error) =>
    ["network-attempt", "browser-error"].includes(error.code)
  );
});

test("rejects inaccessible document structure", async () => {
  const html = good.replace('role="status" aria-live="polite"', "");
  await assert.rejects(validateHtml(html), (error) => error.code === "accessible-status");
});
