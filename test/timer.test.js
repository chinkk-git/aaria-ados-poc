import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(resolve(root, "index.html"), "utf8");
const script = html.match(/<script>([\\s\\S]*?)<\\/script>/)?.[1];
assert.ok(script, "timer script is present");

function loadTimer() {
  const timer = { textContent: "" };
  const buttons = Object.fromEntries(
    ["start", "pause", "reset"].map((id) => [
      id,
      {
        listener: null,
        addEventListener(event, listener) {
          assert.equal(event, "click");
          this.listener = listener;
        },
      },
    ]),
  );
  const intervals = new Map();
  let nextInterval = 1;

  const document = {
    getElementById(id) {
      if (id === "timer") return timer;
      return buttons[id];
    },
  };

  runInNewContext(script, {
    document,
    setInterval(callback, delay) {
      assert.equal(delay, 1000);
      const id = nextInterval++;
      intervals.set(id, callback);
      return id;
    },
    clearInterval(id) {
      intervals.delete(id);
    },
  });

  return {
    timer,
    intervals,
    click(id) {
      buttons[id].listener();
    },
    tick() {
      for (const callback of [...intervals.values()]) callback();
    },
  };
}

test("starts, pauses, resumes, and resets the 15-minute timer", () => {
  const app = loadTimer();
  assert.equal(app.timer.textContent, "15:00");

  app.click("start");
  app.click("start");
  assert.equal(app.intervals.size, 1, "repeated Start does not create another interval");
  app.tick();
  assert.equal(app.timer.textContent, "14:59");

  app.click("pause");
  assert.equal(app.intervals.size, 0);
  app.tick();
  assert.equal(app.timer.textContent, "14:59", "Pause freezes the countdown");

  app.click("start");
  app.tick();
  assert.equal(app.timer.textContent, "14:58", "Start resumes from the paused value");

  app.click("reset");
  assert.equal(app.timer.textContent, "15:00");
  assert.equal(app.intervals.size, 0, "Reset stops the countdown");
});

test("stops cleanly at zero", () => {
  const app = loadTimer();
  app.click("start");
  for (let second = 0; second < 900; second += 1) app.tick();
  assert.equal(app.timer.textContent, "0:00");
  assert.equal(app.intervals.size, 0);
});
