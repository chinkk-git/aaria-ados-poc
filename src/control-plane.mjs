const STATES = ["DEFINED","APPROVED","READY","CLAIMED","BUILDING","PR_OPEN","VERIFYING","REVIEWING","ELIGIBLE","COMPLETE","BLOCKED","FAILED","CANCELLED","SUPERSEDED"];
const TRANSITIONS = {
  DEFINED:["APPROVED","CANCELLED"], APPROVED:["READY","CANCELLED"], READY:["CLAIMED","CANCELLED"],
  CLAIMED:["BUILDING","BLOCKED","FAILED"], BUILDING:["PR_OPEN","BLOCKED","FAILED"],
  PR_OPEN:["VERIFYING","BLOCKED"], VERIFYING:["REVIEWING","BLOCKED","FAILED"],
  REVIEWING:["ELIGIBLE","BUILDING","BLOCKED","FAILED"], ELIGIBLE:["COMPLETE","BLOCKED"],
  BLOCKED:["READY","CANCELLED"], FAILED:["READY","CANCELLED"],
  COMPLETE:[], CANCELLED:[], SUPERSEDED:[]
};
const requiredStrings = ["id","title","branch","accountableWorker"];
const fail = (message) => { throw new Error(message); };

export function validateManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) fail("manifest must be an object");
  for (const key of requiredStrings) if (typeof manifest[key] !== "string" || !manifest[key].trim()) fail(key+" is required");
  if (!/^ADOS-POC-\\d{3}$/.test(manifest.id)) fail("invalid task id");
  if (!/^feature\\/[a-zA-Z0-9._/-]+$/.test(manifest.branch) || manifest.branch.includes("..")) fail("invalid branch");
  if (!Array.isArray(manifest.writeScope) || !manifest.writeScope.length || manifest.writeScope.some((x)=>typeof x!=="string"||!x.trim())) fail("writeScope must contain paths");
  if (!Number.isInteger(manifest.maxRetries) || manifest.maxRetries < 0 || manifest.maxRetries > 2) fail("maxRetries must be an integer from 0 to 2");
  if (!Array.isArray(manifest.acceptanceCriteria) || !manifest.acceptanceCriteria.length) fail("acceptanceCriteria must be nonempty");
  if (!manifest.experiments || !["E01","E02","E03","E04"].every((x)=>x in manifest.experiments)) fail("all experiment states are required");
  if (!STATES.includes(manifest.state)) fail("invalid task state");
  return true;
}

export function transitionTask(from, to) {
  if (!STATES.includes(from) || !TRANSITIONS[from]?.includes(to)) fail("invalid task-state transition: "+from+" -> "+to);
  return to;
}

export function executionKey(taskId, phase, commitSha) {
  if (![taskId,phase,commitSha].every((x)=>typeof x==="string"&&x.length)) fail("execution key fields are required");
  return [taskId,phase,commitSha].join(":");
}

export function claimWriteLease(ledger, {taskId,branch,worker,runId,writeScope}) {
  const key = executionKey(taskId, "build", branch);
  const current = ledger.active;
  if (current?.key === key && current.runId === runId) return {ledger, claimed:false, idempotent:true};
  if (ledger.completed?.includes(key)) fail("duplicate task execution rejected");
  if (current) fail("write scope already leased by "+current.worker);
  const next = {...ledger, active:{key,taskId,branch,worker,runId,writeScope:[...writeScope]}};
  return {ledger:next,claimed:true,idempotent:false};
}

export function releaseWriteLease(ledger, runId) {
  if (!ledger.active || ledger.active.runId !== runId) fail("only lease owner may release the write scope");
  return {...ledger,active:null,completed:[...(ledger.completed||[]),ledger.active.key]};
}

export function authorizeWrite({worker,accountableWorker,writeScope,path}) {
  if (worker !== accountableWorker) fail("unauthorized worker");
  if (typeof path !== "string" || path.startsWith("/") || path.split("/").includes("..")) fail("unauthorized path");
  const allowed = writeScope.some((scope)=>scope.endsWith("/**") ? path.startsWith(scope.slice(0,-2)) : scope===path);
  if (!allowed) fail("path outside authorized write scope");
  return true;
}

export function assertRetryAvailable(retries,maxRetries=2) {
  if (!Number.isInteger(retries)||!Number.isInteger(maxRetries)||maxRetries<0||maxRetries>2||retries>=maxRetries) fail("retry limit reached");
  return retries+1;
}

export function validateEvidence({testedSha,headSha,ciResult,reviewResult,reviewer,worker}) {
  if (!/^[0-9a-f]{40}$/.test(testedSha||"") || testedSha!==headSha) fail("evidence SHA does not match exact task head");
  if (ciResult!=="PASS") fail("focused CI has not passed");
  if (reviewResult!=="PASS") fail("independent review has not passed");
  if (!reviewer || reviewer===worker) fail("reviewer must be independent of the implementer");
  return true;
}
