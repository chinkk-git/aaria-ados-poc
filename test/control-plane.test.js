import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateManifest, transitionTask, executionKey, claimWriteLease, releaseWriteLease, authorizeWrite, assertRetryAvailable, validateEvidence } from "../src/control-plane.mjs";

const manifest = JSON.parse(readFileSync(new URL("../task.json", import.meta.url), "utf8"));

test("validates manifest and rejects retry limits above two", () => {
  assert.equal(validateManifest(manifest), true);
  assert.throws(()=>validateManifest({...manifest,maxRetries:3}),/maxRetries/);
});

test("allows only declared task-state transitions", () => {
  assert.equal(transitionTask("READY","CLAIMED"),"CLAIMED");
  assert.throws(()=>transitionTask("DEFINED","COMPLETE"),/invalid task-state transition/);
});

test("execution key is stable and scoped to task phase and checkpoint", () => {
  assert.equal(executionKey("ADOS-POC-001","build","abc"),executionKey("ADOS-POC-001","build","abc"));
  assert.notEqual(executionKey("ADOS-POC-001","build","abc"),executionKey("ADOS-POC-001","review","abc"));
});

test("lease permits one writer, same-run replay is idempotent, duplicates are rejected", () => {
  const first=claimWriteLease({active:null,completed:[]},{taskId:"ADOS-POC-001",branch:"feature/ados-control-plane-poc",worker:"worker-a",runId:"run-1",writeScope:["src/**"]});
  assert.equal(first.claimed,true);
  assert.equal(claimWriteLease(first.ledger,{taskId:"ADOS-POC-001",branch:"feature/ados-control-plane-poc",worker:"worker-a",runId:"run-1",writeScope:["src/**"]}).idempotent,true);
  assert.throws(()=>claimWriteLease(first.ledger,{taskId:"ADOS-POC-002",branch:"feature/other",worker:"worker-b",runId:"run-2",writeScope:["src/**"]}),/already leased/);
  const released=releaseWriteLease(first.ledger,"run-1");
  assert.throws(()=>claimWriteLease(released,{taskId:"ADOS-POC-001",branch:"feature/ados-control-plane-poc",worker:"worker-a",runId:"run-3",writeScope:["src/**"]}),/duplicate task execution/);
});

test("rejects unauthorized identity, path, and traversal", () => {
  const access={worker:"ados-builder",accountableWorker:"ados-builder",writeScope:["index.html","src/**"]};
  assert.equal(authorizeWrite({...access,path:"src/control-plane.mjs"}),true);
  assert.throws(()=>authorizeWrite({...access,worker:"other",path:"index.html"}),/unauthorized worker/);
  assert.throws(()=>authorizeWrite({...access,path:"secrets.txt"}),/outside authorized/);
  assert.throws(()=>authorizeWrite({...access,path:"src/../secrets.txt"}),/unauthorized path/);
});

test("bounds retries at two", () => {
  assert.equal(assertRetryAvailable(0,2),1);
  assert.equal(assertRetryAvailable(1,2),2);
  assert.throws(()=>assertRetryAvailable(2,2),/retry limit/);
});

test("requires exact-SHA CI evidence and independent review", () => {
  const good={testedSha:"a".repeat(40),headSha:"a".repeat(40),ciResult:"PASS",reviewResult:"PASS",reviewer:"reviewer",worker:"builder"};
  assert.equal(validateEvidence(good),true);
  assert.throws(()=>validateEvidence({...good,headSha:"b".repeat(40)}),/exact task head/);
  assert.throws(()=>validateEvidence({...good,reviewer:"builder"}),/independent/);
});
