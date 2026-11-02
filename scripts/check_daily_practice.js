#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const vm = require("node:vm");

function fail(message) {
  console.error("REJECTED: " + message);
  process.exit(1);
}
function textOnly(html) {
  return html.replace(/<[^>]*>/g, " ").replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&").replace(/&lt;/gi, "<").replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"').replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ").trim();
}
function hasAttr(tag, name, value) {
  return new RegExp("\\b" + name + "\\s*=\\s*(?:\"" + value + "\"|'"+ value +"'|\\b" + value + "\\b)", "i").test(tag);
}
function getElements(html) {
  const h1 = html.match(/<h1\b[^>]*>([\s\S]*?)<\/h1\s*>/i);
  if (!h1 || textOnly(h1[1]) !== "Daily Practice") fail("missing Daily Practice heading");

  const button = html.match(/<button\b[^>]*\bid\s*=\s*(?:"completeButton"|'completeButton'|completeButton\b)[^>]*>([\s\S]*?)<\/button\s*>/i);
  if (!button || !hasAttr(button[0].slice(0, button[0].indexOf(">") + 1), "type", "button")) {
    fail("missing type=button completion control with required ID");
  }
  if (textOnly(button[1]) !== "Mark Complete") fail("completion button label is invalid");

  const status = html.match(/<(\w+)\b[^>]*\bid\s*=\s*(?:"completionStatus"|'completionStatus'|completionStatus\b)[^>]*>([\s\S]*?)<\/\1\s*>/i);
  if (!status) fail("missing completion status element");
  const open = status[0].slice(0, status[0].indexOf(">") + 1);
  if (!hasAttr(open, "role", "status") || !hasAttr(open, "aria-live", "polite")) {
    fail("completion status must expose role=status and aria-live=polite");
  }
  if (textOnly(status[2]) !== "Not completed") fail("initial status must be Not completed");
  return { buttonId: "completeButton", statusId: "completionStatus", initial: textOnly(status[2]) };
}
function runPage(html, storage) {
  const ids = getElements(html);
  const listeners = new Map();
  const elements = {
    completeButton: {
      textContent: "Mark Complete",
      addEventListener(type, handler) {
        if (type !== "click" || typeof handler !== "function") throw new Error("unsupported event listener");
        listeners.set(type, handler);
      },
      click() {
        const handler = listeners.get("click");
        if (handler) handler({ target: this });
      }
    },
    completionStatus: { textContent: ids.initial }
  };
  const document = {
    getElementById(id) {
      if (!Object.prototype.hasOwnProperty.call(elements, id)) throw new Error("unknown DOM id: " + id);
      return elements[id];
    }
  };
  const scripts = Array.from(html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script\s*>/gi))
    .map(match => match[1]).filter(script => script.trim());
  if (scripts.length === 0) fail("missing executable inline behavior");
  const context = {
    document,
    localStorage: storage,
    sessionStorage: storage,
    window: { document, localStorage: storage, sessionStorage: storage }
  };
  for (const script of scripts) {
    try {
      vm.runInNewContext(script, context, { timeout: 1000, codeGeneration: { strings: false, wasm: false } });
    } catch (error) {
      fail("page script failed in isolated DOM: " + error.message);
    }
  }
  return { elements, context };
}
const file = process.argv[2];
if (!file) fail("usage: node scripts/check_daily_practice.js <html-file>");
let html;
try {
  html = fs.readFileSync(file, "utf8");
} catch (error) {
  fail("cannot read page: " + error.message);
}
if (!html.trim()) fail("page is empty");

const values = new Map();
const storage = {
  getItem(key) { return values.has(String(key)) ? values.get(String(key)) : null; },
  setItem(key, value) { values.set(String(key), String(value)); },
  removeItem(key) { values.delete(String(key)); },
  clear() { values.clear(); }
};

const firstLoad = runPage(html, storage);
if (firstLoad.elements.completionStatus.textContent !== "Not completed") fail("page did not start in Not completed state");
vm.runInNewContext("document.getElementById('completeButton').click()", firstLoad.context, { timeout: 1000, codeGeneration: { strings: false, wasm: false } });
if (firstLoad.elements.completionStatus.textContent !== "Completed") fail("clicking Mark Complete did not change status to Completed");

const reloaded = runPage(html, storage);
if (reloaded.elements.completionStatus.textContent !== "Not completed") fail("completion state did not reset after reload");
console.log("VERIFIED: initial state, click transition, and reset on reload");
