"use strict";

const OPERATOR_KEY = "inverse-agent.operator-token";
const WORKSPACE_KEY = "inverse-agent.workspace";
const TRANSIENT_STATUSES = new Set(["queued", "starting", "running", "approving", "cancel_requested"]);
const TERMINAL_STATUSES = new Set(["succeeded", "incomplete", "cancelled", "failed", "refused"]);
const STATUS_LABELS = {
  planned: "Planned",
  queued: "Queued",
  starting: "Starting",
  running: "Running",
  waiting_for_approval: "Waiting for approval",
  approving: "Executing approved action",
  cancel_requested: "Cancellation requested",
  succeeded: "Succeeded",
  incomplete: "Incomplete",
  cancelled: "Cancelled",
  failed: "Failed",
  refused: "Declined"
};

const SCOPE_LABELS = {
  source_read: "Local model source access",
  code_execution: "Approved command execution"
};

function runStatusLabel(run) {
  if (run.autonomy_level === 0 && run.status === "succeeded") {
    return "Plan ready";
  }
  return STATUS_LABELS[run.status] || run.status;
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const state = {
  operatorToken: sessionStorage.getItem(OPERATOR_KEY) || "",
  approverToken: "",
  approverIdentity: "",
  workspace: sessionStorage.getItem(WORKSPACE_KEY) || "",
  runtime: null,
  profile: null,
  runs: [],
  selectedRun: null,
  trace: null,
  events: [],
  eventCursor: 0,
  eventRunId: "",
  mode: 1,
  kind: "verification",
  domain: "",
  busy: false,
  pendingGoal: "",
  pendingRunId: "",
  pendingTrustScope: "",
  pendingRevokeScope: "",
  navigationEpoch: 0,
  pollTimer: 0
};

const elements = {
  shell: document.getElementById("app-shell"),
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  mobileBackdrop: document.getElementById("mobile-backdrop"),
  inspectorToggle: document.getElementById("inspector-toggle"),
  newTask: document.getElementById("new-task-button"),
  runList: document.getElementById("run-list"),
  connectionDot: document.getElementById("connection-dot"),
  connectionLabel: document.getElementById("connection-label"),
  settingsButton: document.getElementById("settings-button"),
  workspaceButton: document.getElementById("workspace-button"),
  workspaceName: document.getElementById("workspace-name"),
  workspacePath: document.getElementById("workspace-path"),
  kindSelect: document.getElementById("kind-select"),
  domainSelect: document.getElementById("domain-select"),
  modeButtons: Array.from(document.querySelectorAll("[data-mode]")),
  conversation: document.getElementById("conversation"),
  goalInput: document.getElementById("goal-input"),
  plannerNote: document.getElementById("planner-note"),
  sendButton: document.getElementById("send-button"),
  runtimeDetails: document.getElementById("runtime-details"),
  workspaceDetails: document.getElementById("workspace-details"),
  consentList: document.getElementById("consent-list"),
  budgetDetails: document.getElementById("budget-details"),
  toolList: document.getElementById("tool-list"),
  settingsDialog: document.getElementById("settings-dialog"),
  settingsClose: document.getElementById("settings-close"),
  settingsCancel: document.getElementById("settings-cancel"),
  settingsStatus: document.getElementById("settings-status"),
  operatorInput: document.getElementById("operator-token-input"),
  approverInput: document.getElementById("approver-token-input"),
  workspaceInput: document.getElementById("workspace-input"),
  connectButton: document.getElementById("connect-button"),
  forgetButton: document.getElementById("forget-button"),
  trustDialog: document.getElementById("trust-dialog"),
  trustTitle: document.getElementById("trust-title"),
  trustPath: document.getElementById("trust-path"),
  trustCopy: document.getElementById("trust-copy"),
  trustStatus: document.getElementById("trust-status"),
  trustCancel: document.getElementById("trust-cancel"),
  trustConfirm: document.getElementById("trust-confirm"),
  toastRegion: document.getElementById("toast-region")
};
const mobileInspectorQuery = window.matchMedia("(max-width: 760px)");

function authScope(method, path) {
  const pathname = path.split("?", 1)[0];
  if (pathname === "/approver/session") {
    return "approver";
  }
  if ((method === "POST" || method === "DELETE") && pathname === "/workspaces/trust") {
    return "approver";
  }
  if (method === "POST" && /\/runs\/[^/]+\/(approvals|decline)$/.test(pathname)) {
    return "approver";
  }
  return "operator";
}

async function request(path, options = {}) {
  const method = options.method || "GET";
  const scope = authScope(method, path);
  const token = scope === "approver" ? state.approverToken : state.operatorToken;
  if (!token) {
    throw new ApiError(scope === "approver" ? "Approver access is required" : "Connect the workbench first", 401);
  }

  const headers = new Headers();
  if (scope === "approver") {
    headers.set("X-Inverse-Agent-Approval-Token", token);
  } else {
    headers.set("X-Inverse-Agent-Token", token);
  }
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, {
    method,
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    cache: "no-store",
    credentials: "omit"
  });
  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    payload = await response.json();
  }
  if (!response.ok) {
    if (response.status === 401) {
      if (scope === "approver") {
        state.approverToken = "";
        state.approverIdentity = "";
      } else {
        state.operatorToken = "";
        sessionStorage.removeItem(OPERATOR_KEY);
      }
      updateConnectionState();
    }
    throw new ApiError(errorMessage(payload, response.status), response.status);
  }
  return payload;
}

function errorMessage(payload, status) {
  if (payload && typeof payload.detail === "string") {
    return payload.detail;
  }
  if (payload && Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || "Invalid value").join("; ");
  }
  return `Request failed with status ${status}`;
}

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) {
    value.className = className;
  }
  if (text !== undefined) {
    value.textContent = String(text);
  }
  return value;
}

function setBusy(busy) {
  state.busy = busy;
  elements.sendButton.disabled = busy;
  elements.connectButton.disabled = busy;
  elements.trustConfirm.disabled = busy;
}

function updateConnectionState() {
  const connected = Boolean(state.runtime && state.operatorToken);
  elements.connectionDot.classList.toggle("connected", connected);
  elements.connectionLabel.textContent = connected ? "Connected" : "Not connected";
}

function showToast(message) {
  const toast = node("div", "toast", message);
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function openSettings(message = "", approvalFocus = false) {
  elements.operatorInput.value = state.operatorToken;
  elements.approverInput.value = "";
  elements.workspaceInput.value = state.workspace;
  elements.settingsStatus.textContent = message;
  if (!elements.settingsDialog.open) {
    elements.settingsDialog.showModal();
  }
  window.setTimeout(() => {
    (approvalFocus ? elements.approverInput : elements.operatorInput).focus();
  }, 0);
}

function closeSettings() {
  if (elements.settingsDialog.open) {
    elements.settingsDialog.close();
  }
  elements.settingsStatus.textContent = "";
  elements.operatorInput.value = "";
  elements.approverInput.value = "";
}

function cancelSettings() {
  state.pendingGoal = "";
  state.pendingRunId = "";
  state.pendingTrustScope = "";
  state.pendingRevokeScope = "";
  closeSettings();
}

async function connectFromSettings() {
  const operatorCandidate = elements.operatorInput.value.trim() || state.operatorToken;
  const approverCandidate = elements.approverInput.value.trim();
  const workspaceCandidate = elements.workspaceInput.value.trim();
  const pendingRevokeCandidate = state.pendingRevokeScope;
  if (!operatorCandidate) {
    elements.settingsStatus.textContent = "Operator token is required";
    return;
  }

  setBusy(true);
  const previousOperator = state.operatorToken;
  const previousApprover = state.approverToken;
  let revokeAfterConnect = "";
  state.operatorToken = operatorCandidate;
  try {
    const runtime = await request("/runtime");
    let approverIdentity = state.approverIdentity;
    if (approverCandidate) {
      state.approverToken = approverCandidate;
      const approver = await request("/approver/session");
      approverIdentity = approver.approver;
    }
    state.runtime = runtime;
    state.approverIdentity = approverIdentity;
    sessionStorage.setItem(OPERATOR_KEY, operatorCandidate);
    if (workspaceCandidate) {
      state.workspace = workspaceCandidate;
      sessionStorage.setItem(WORKSPACE_KEY, workspaceCandidate);
    }
    closeSettings();
    await refreshWorkspace();
    await refreshRuns();
    chooseRuntimeDefaultKind();
    renderAll();
    if (pendingRevokeCandidate) {
      revokeAfterConnect = pendingRevokeCandidate;
      state.pendingRevokeScope = "";
    } else if (state.pendingGoal || state.pendingRunId) {
      await continuePendingStart();
    } else if (state.pendingTrustScope) {
      openTrustDialog(state.pendingTrustScope);
    }
  } catch (error) {
    state.operatorToken = previousOperator;
    state.approverToken = previousApprover;
    elements.settingsStatus.textContent = readableError(error);
  } finally {
    setBusy(false);
    updateConnectionState();
    if (state.runtime) {
      renderAll();
    }
  }
  if (revokeAfterConnect) {
    await revokeWorkspaceScope(revokeAfterConnect);
  }
}

function forgetAccess() {
  sessionStorage.removeItem(OPERATOR_KEY);
  sessionStorage.removeItem(WORKSPACE_KEY);
  state.operatorToken = "";
  state.approverToken = "";
  state.approverIdentity = "";
  state.workspace = "";
  state.runtime = null;
  state.profile = null;
  state.runs = [];
  state.selectedRun = null;
  state.trace = null;
  resetEvents();
  state.pendingGoal = "";
  state.pendingRunId = "";
  state.pendingTrustScope = "";
  state.pendingRevokeScope = "";
  state.navigationEpoch += 1;
  closeSettings();
  updateConnectionState();
  renderAll();
  openSettings("Access cleared for this tab");
}

async function refreshWorkspace(expectedEpoch = null) {
  if (!state.workspace || !state.operatorToken) {
    state.profile = null;
    renderWorkspace();
    return false;
  }
  const requestedWorkspace = state.workspace;
  try {
    const profile = await request(`/profile?path=${encodeURIComponent(requestedWorkspace)}`);
    if (
      requestedWorkspace !== state.workspace ||
      (expectedEpoch !== null && expectedEpoch !== state.navigationEpoch)
    ) {
      return false;
    }
    state.profile = profile;
    const domains = Array.isArray(state.profile.domains) ? state.profile.domains : [];
    if (!domains.includes(state.domain)) {
      state.domain = domains[0] || "";
    }
    renderWorkspace();
    return true;
  } catch (error) {
    if (
      requestedWorkspace !== state.workspace ||
      (expectedEpoch !== null && expectedEpoch !== state.navigationEpoch)
    ) {
      return false;
    }
    state.profile = null;
    renderWorkspace();
    showToast(readableError(error));
    return false;
  }
}

async function refreshRuns() {
  if (!state.operatorToken) {
    state.runs = [];
    return;
  }
  try {
    state.runs = await request("/runs?limit=100&offset=0");
  } catch (error) {
    handleRequestError(error);
  }
}

async function selectRun(runId, propagateError = false, navigate = true) {
  clearPoll();
  const epoch = navigate ? ++state.navigationEpoch : state.navigationEpoch;
  if (navigate || state.eventRunId !== runId) {
    resetEvents(runId);
  }
  try {
    if (!navigate && (!state.selectedRun || state.selectedRun.run_id !== runId)) {
      return false;
    }
    const selectedRun = await request(`/runs/${encodeURIComponent(runId)}`);
    if (epoch !== state.navigationEpoch) {
      return false;
    }
    let trace = null;
    if (selectedRun.has_trace) {
      try {
        trace = await request(`/runs/${encodeURIComponent(runId)}/trace`);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) {
          throw error;
        }
      }
    }
    if (epoch !== state.navigationEpoch) {
      return false;
    }
    state.selectedRun = selectedRun;
    state.runs = state.runs.map((run) => run.run_id === runId ? selectedRun : run);
    state.trace = trace;
    state.workspace = selectedRun.workspace;
    sessionStorage.setItem(WORKSPACE_KEY, state.workspace);
    if (!await refreshWorkspace(epoch) || epoch !== state.navigationEpoch) {
      return false;
    }
    await refreshEvents(runId, epoch);
    if (epoch !== state.navigationEpoch) {
      return false;
    }
    renderAll();
    schedulePoll();
    closeMobileSidebar();
    return true;
  } catch (error) {
    if (epoch !== state.navigationEpoch) {
      return false;
    }
    if (propagateError) {
      throw error;
    }
    handleRequestError(error);
    return false;
  }
}

function newTask() {
  clearPoll();
  state.navigationEpoch += 1;
  state.selectedRun = null;
  state.trace = null;
  resetEvents();
  state.pendingGoal = "";
  state.pendingRunId = "";
  elements.goalInput.value = "";
  resizeGoalInput();
  renderAll();
  elements.goalInput.focus();
  closeMobileSidebar();
}

function resetEvents(runId = "") {
  state.events = [];
  state.eventCursor = 0;
  state.eventRunId = runId;
}

async function refreshEvents(runId, epoch) {
  if (state.eventRunId !== runId) {
    resetEvents(runId);
  }
  let hasMore = true;
  let pages = 0;
  while (hasMore && pages < 20) {
    const payload = await request(
      `/runs/${encodeURIComponent(runId)}/events?after=${state.eventCursor}&wait_seconds=0&limit=200`
    );
    if (epoch !== state.navigationEpoch || state.eventRunId !== runId) {
      return false;
    }
    const events = Array.isArray(payload.events) ? payload.events : [];
    state.events.push(...events);
    if (state.events.length > 400) {
      state.events = state.events.slice(-400);
    }
    state.eventCursor = Number.isInteger(payload.next_cursor)
      ? payload.next_cursor
      : state.eventCursor;
    hasMore = payload.has_more === true;
    pages += 1;
  }
  return true;
}

async function startGoal() {
  if (state.busy) {
    return;
  }
  const goal = elements.goalInput.value.trim();
  if (!goal) {
    elements.goalInput.focus();
    return;
  }
  if (!state.runtime) {
    openSettings("Connect before starting a task");
    return;
  }
  if (!state.profile || !state.domain) {
    openSettings("Choose a valid workspace before starting a task");
    return;
  }
  if (
    state.kind === "investigation" &&
    (!state.runtime.planner || state.runtime.planner.investigation_available !== true)
  ) {
    showToast("Configure a calibrated loopback model before starting an investigation");
    return;
  }
  state.pendingGoal = goal;
  state.pendingRunId = "";
  await continuePendingStart();
}

async function createAndStartRun(goal) {
  clearPoll();
  const epoch = ++state.navigationEpoch;
  let createdRunId = "";
  setBusy(true);
  try {
    const created = await request("/runs", {
      method: "POST",
      body: {
        goal,
        workspace: state.workspace,
        domain: state.domain,
        kind: state.kind,
        autonomy_level: state.mode
      }
    });
    createdRunId = created.run_id;
    state.runs = [created, ...state.runs.filter((run) => run.run_id !== created.run_id)];
    elements.goalInput.value = "";
    resizeGoalInput();
    if (epoch === state.navigationEpoch) {
      state.selectedRun = created;
      state.trace = null;
      renderAll();
    } else {
      renderRuns();
    }

    const started = await request(`/runs/${encodeURIComponent(created.run_id)}/start`, {
      method: "POST"
    });
    state.pendingGoal = "";
    await refreshRuns();
    if (
      epoch === state.navigationEpoch &&
      state.selectedRun &&
      state.selectedRun.run_id === created.run_id
    ) {
      state.selectedRun = started;
      renderAll();
      schedulePoll();
    } else {
      renderRuns();
    }
  } catch (error) {
    if (createdRunId && state.operatorToken && epoch === state.navigationEpoch) {
      try {
        await selectRun(createdRunId, true, false);
      } catch (_refreshError) {
        // Preserve the original start failure for the operator.
      }
    }
    handleRequestError(error);
  } finally {
    setBusy(false);
    if (selectionMatches(createdRunId, epoch)) {
      renderAll();
    }
  }
}

function selectionMatches(runId, epoch) {
  return Boolean(
    epoch === state.navigationEpoch &&
    state.selectedRun &&
    state.selectedRun.run_id === runId
  );
}

async function startExistingRun(runId) {
  const run = state.selectedRun;
  if (state.busy || !run || run.run_id !== runId || run.status !== "planned") {
    return;
  }
  const missing = firstMissingScope(run.kind || "verification", run.autonomy_level);
  if (missing) {
    state.pendingRunId = runId;
    state.pendingGoal = "";
    openTrustDialog(missing);
    return;
  }

  state.pendingRunId = "";
  clearPoll();
  const epoch = ++state.navigationEpoch;
  state.selectedRun = {...run, status: "starting"};
  setBusy(true);
  renderAll();
  try {
    const started = await request(`/runs/${encodeURIComponent(runId)}/start`, {
      method: "POST"
    });
    await refreshRuns();
    if (selectionMatches(runId, epoch)) {
      state.selectedRun = started;
      state.trace = null;
      schedulePoll();
    }
  } catch (error) {
    if (state.operatorToken && selectionMatches(runId, epoch)) {
      try {
        await selectRun(runId, true, false);
      } catch (_refreshError) {
        // Preserve the original start failure for the operator.
      }
    }
    handleRequestError(error);
  } finally {
    setBusy(false);
    if (selectionMatches(runId, epoch)) {
      renderAll();
      schedulePoll();
    } else {
      renderRuns();
    }
  }
}

function requiredScopes(kind, mode) {
  const scopes = [];
  if (kind === "investigation") {
    scopes.push("source_read");
  }
  if (Number(mode) !== 0) {
    scopes.push("code_execution");
  }
  return scopes;
}

function hasScope(scope) {
  return Boolean(
    state.profile &&
    state.profile.trust &&
    state.profile.trust.scopes &&
    state.profile.trust.scopes[scope]
  );
}

function firstMissingScope(kind, mode) {
  return requiredScopes(kind, mode).find((scope) => !hasScope(scope)) || "";
}

async function continuePendingStart() {
  if (!state.profile) {
    return;
  }
  const run = state.pendingRunId && state.selectedRun && state.selectedRun.run_id === state.pendingRunId
    ? state.selectedRun
    : null;
  const kind = run ? run.kind || "verification" : state.kind;
  const mode = run ? run.autonomy_level : state.mode;
  const missing = firstMissingScope(kind, mode);
  if (missing) {
    openTrustDialog(missing);
    return;
  }
  if (run) {
    const runId = state.pendingRunId;
    state.pendingRunId = "";
    await startExistingRun(runId);
  } else if (state.pendingGoal) {
    await createAndStartRun(state.pendingGoal);
  }
}

function openTrustDialog(scope) {
  if (!state.profile) {
    return;
  }
  state.pendingTrustScope = scope;
  if (!state.approverToken) {
    openSettings(`Enter the approver token to grant ${SCOPE_LABELS[scope] || scope}`, true);
    return;
  }
  elements.trustPath.textContent = state.profile.root || state.workspace;
  if (scope === "source_read") {
    elements.trustTitle.textContent = "Allow local model source access?";
    elements.trustCopy.textContent = "This permits bounded, redacted source from this workspace to be disclosed to the configured loopback model process. Loopback is not containment: the model server remains an untrusted peer. You can revoke this consent here.";
  } else {
    elements.trustTitle.textContent = "Allow approved command execution?";
    elements.trustCopy.textContent = "This permits separately approved commands to execute workspace code with your user permissions. Every approval remains action-bound. You can revoke this consent here.";
  }
  elements.trustStatus.textContent = state.approverIdentity ? `Approver: ${state.approverIdentity}` : "";
  if (!elements.trustDialog.open) {
    elements.trustDialog.showModal();
  }
  window.setTimeout(() => elements.trustCancel.focus(), 0);
}

async function trustWorkspace() {
  const scope = state.pendingTrustScope;
  if (!state.profile || !state.approverToken || !scope) {
    return;
  }
  setBusy(true);
  try {
    await request("/workspaces/trust", {
      method: "POST",
      body: {workspace: state.profile.root || state.workspace, scope}
    });
    elements.trustDialog.close();
    state.pendingTrustScope = "";
    await refreshWorkspace();
    renderAll();
  } catch (error) {
    elements.trustStatus.textContent = readableError(error);
  } finally {
    setBusy(false);
    renderAll();
  }
  if (!elements.trustDialog.open && (state.pendingRunId || state.pendingGoal)) {
    await continuePendingStart();
  }
}

async function revokeWorkspaceScope(scope) {
  if (!state.profile) {
    return;
  }
  if (!state.approverToken) {
    state.pendingRevokeScope = scope;
    openSettings(`Enter the approver token to revoke ${SCOPE_LABELS[scope] || scope}`, true);
    return;
  }
  setBusy(true);
  try {
    await request("/workspaces/trust", {
      method: "DELETE",
      body: {workspace: state.profile.root || state.workspace, scope}
    });
    state.pendingRevokeScope = "";
    await refreshWorkspace();
    if (state.selectedRun) {
      await selectRun(state.selectedRun.run_id, false, false);
    } else {
      renderAll();
    }
    showToast(`${SCOPE_LABELS[scope] || scope} revoked`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      state.pendingRevokeScope = scope;
    }
    handleRequestError(error);
  } finally {
    setBusy(false);
    renderAll();
  }
}

async function resolveApproval(action) {
  const run = state.selectedRun;
  if (!run || !run.pending_approval) {
    return;
  }
  if (!state.approverToken) {
    openSettings("Enter the approver token to continue", true);
    return;
  }
  const digest = run.pending_approval.action_digest;
  const challengeId = run.pending_approval.challenge_id;
  const runId = run.run_id;
  clearPoll();
  const epoch = ++state.navigationEpoch;
  state.selectedRun = {...run, status: "approving"};
  renderAll();
  setBusy(true);
  try {
    const resolved = await request(`/runs/${encodeURIComponent(runId)}/${action}`, {
      method: "POST",
      body: {action_digest: digest, challenge_id: challengeId}
    });
    await refreshRuns();
    if (selectionMatches(runId, epoch)) {
      let trace = null;
      if (resolved.has_trace) {
        trace = await request(`/runs/${encodeURIComponent(runId)}/trace`);
      }
      if (selectionMatches(runId, epoch)) {
        state.selectedRun = resolved;
        state.trace = trace;
        renderAll();
        schedulePoll();
      }
    } else {
      renderRuns();
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) {
      if (selectionMatches(runId, epoch)) {
        await selectRun(runId, false, false);
      } else {
        await refreshRuns();
        renderRuns();
      }
      showToast("The task changed while that decision was being processed");
    } else {
      handleRequestError(error);
      if (selectionMatches(runId, epoch)) {
        schedulePoll(runId);
      }
    }
  } finally {
    setBusy(false);
    if (selectionMatches(runId, epoch)) {
      renderAll();
    }
  }
}

function schedulePoll(runId = "") {
  clearPoll();
  const activeRun = state.selectedRun;
  const targetId = runId || (activeRun && activeRun.run_id);
  if (!state.operatorToken || !targetId || !activeRun || !TRANSIENT_STATUSES.has(activeRun.status)) {
    return;
  }
  state.pollTimer = window.setTimeout(() => pollRun(targetId), 1400);
}

async function pollRun(runId) {
  try {
    await selectRun(runId, true, false);
  } catch (error) {
    handleRequestError(error);
    schedulePoll(runId);
  }
}

function clearPoll() {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = 0;
  }
}

function renderAll() {
  renderConnection();
  renderRuns();
  renderWorkspace();
  renderConversation();
  renderInspector();
  renderPlannerNote();
}

function renderConnection() {
  updateConnectionState();
}

function renderRuns() {
  const fragment = document.createDocumentFragment();
  for (const run of state.runs) {
    const button = node("button", "run-item");
    button.type = "button";
    button.classList.toggle("active", Boolean(state.selectedRun && state.selectedRun.run_id === run.run_id));
    button.addEventListener("click", () => selectRun(run.run_id));
    button.append(node("span", "run-title", run.goal || "Untitled task"));
    const meta = node("span", "run-meta");
    meta.append(node("span", `run-status-dot ${run.status}`));
    meta.append(node("span", "", `${run.domain} / ${runStatusLabel(run)}`));
    button.append(meta);
    fragment.append(button);
  }
  if (!state.runs.length && state.runtime) {
    fragment.append(node("div", "run-item", "No tasks yet"));
  }
  elements.runList.replaceChildren(fragment);
}

function renderWorkspace() {
  const path = state.profile ? state.profile.root : state.workspace;
  elements.workspacePath.textContent = path || "";
  elements.workspaceName.textContent = path ? basename(path) : "Choose workspace";

  const domains = state.profile && Array.isArray(state.profile.domains) ? state.profile.domains : [];
  const options = document.createDocumentFragment();
  if (!domains.length) {
    const option = node("option", "", "No domain");
    option.value = "";
    options.append(option);
  }
  for (const domain of domains) {
    const option = node("option", "", domainLabel(domain));
    option.value = domain;
    option.selected = domain === state.domain;
    options.append(option);
  }
  elements.domainSelect.replaceChildren(options);
  elements.domainSelect.disabled = domains.length === 0;
  elements.kindSelect.value = state.selectedRun
    ? state.selectedRun.kind || "verification"
    : state.kind;
  elements.kindSelect.disabled = Boolean(state.selectedRun);
}

function renderPlannerNote() {
  const planner = state.runtime && state.runtime.planner ? state.runtime.planner : null;
  if (!planner) {
    elements.plannerNote.textContent = "Connect to inspect the planner";
    return;
  }
  if (state.kind === "investigation" && planner.investigation_available !== true) {
    elements.plannerNote.textContent = "Investigation requires loopback model context calibration";
    return;
  }
  if (planner.kind === "deterministic") {
    elements.plannerNote.textContent = state.kind === "investigation"
      ? "Investigation requires a configured loopback model"
      : `Standard ${state.domain || "domain"} verification sequence; the goal is recorded`;
  } else {
    elements.plannerNote.textContent = state.kind === "investigation"
      ? `${planner.model || "Local model"} investigates with registered read-only tools`
      : `${planner.model || "Local model"} selects registered verification tools`;
  }
}

function renderConversation() {
  const inner = node("div", "conversation-inner");
  const run = state.selectedRun;
  if (!run) {
    const empty = node("div", "empty-state");
    empty.append(node("div", "empty-mark", "IA"));
    const investigation = state.kind === "investigation";
    empty.append(node("h1", "", investigation ? "Start an investigation" : "Start a verification task"));
    empty.append(node("p", "", investigation
      ? "Investigate with bounded read-only tools, explicit local-model source consent, durable evidence, and human-approved commands."
      : "Run typed engineering checks with explicit workspace consent and human approval. This workbench does not edit source code."));
    inner.append(empty);
    elements.conversation.replaceChildren(inner);
    return;
  }

  inner.append(message("user", "You", run.goal));
  const runMessage = node("article", "message run-message");
  runMessage.append(node("div", "message-avatar", "IA"));
  const content = node("div", "message-content");
  content.append(node("div", "message-label", run.kind === "investigation" ? "Investigation run" : "Verification run"));
  const heading = node("div", "run-heading");
  heading.append(node("span", `status-badge ${run.status}`, runStatusLabel(run)));
  heading.append(node("span", "message-text", domainLabel(run.domain)));
  content.append(heading);

  if (run.plan_rationale) {
    content.append(node("p", "message-text", run.plan_rationale));
  } else if (run.status === "queued") {
    content.append(node("p", "message-text", "Queued for the local worker. Activity will reconnect from the durable event cursor."));
  } else if (run.status === "starting") {
    content.append(node("p", "message-text", run.kind === "investigation" ? "Starting the investigation..." : "Preparing the verification plan..."));
  } else if (run.status === "cancel_requested") {
    content.append(node("p", "message-text", "Cancellation will take effect at the next safe workflow boundary."));
  } else if (run.status === "planned") {
    content.append(node("p", "message-text", "This task is saved and ready to start."));
  }
  if (Array.isArray(run.plan) && run.plan.length) {
    content.append(planView(run));
  }
  if (run.pending_approval && run.status === "waiting_for_approval") {
    content.append(approvalView(run.pending_approval));
  }
  if (run.status === "planned") {
    content.append(plannedRunView(run));
  }
  if (!TERMINAL_STATUSES.has(run.status) && run.status !== "planned") {
    content.append(activeRunActions(run));
  }
  if (run.answer) {
    content.append(answerView(run.answer));
  }
  if (run.error) {
    const errorPanel = node("section", "error-panel");
    errorPanel.append(node("h3", "", run.status === "refused" ? "Task declined" : "Task stopped"));
    errorPanel.append(node("p", "", run.error));
    content.append(errorPanel);
  }
  if (state.trace && Array.isArray(state.trace.actions)) {
    content.append(traceView(state.trace));
  }
  if (run.kind === "investigation" && (run.budget || run.usage)) {
    content.append(budgetView(run));
  }
  if (state.eventRunId === run.run_id && state.events.length) {
    content.append(activityView(state.events));
  }
  runMessage.append(content);
  inner.append(runMessage);
  elements.conversation.replaceChildren(inner);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function message(role, label, text) {
  const article = node("article", `message ${role}`);
  article.append(node("div", "message-avatar", role === "user" ? "YOU" : "IA"));
  const content = node("div", "message-content");
  content.append(node("div", "message-label", label));
  content.append(node("p", "message-text", text));
  article.append(content);
  return article;
}

function planView(run) {
  const list = node("div", "plan-list");
  const advisoryPlan = run.autonomy_level === 0 && run.status === "succeeded";
  run.plan.forEach((tool, index) => {
    const step = node("div", "plan-step");
    const complete = index < (run.completed_actions || 0);
    if (complete) {
      step.classList.add("complete");
    }
    step.append(node("span", "step-index", complete ? "OK" : index + 1));
    step.append(node("code", "", tool));
    let stateLabel = "queued";
    if (complete) {
      stateLabel = "complete";
    } else if (advisoryPlan) {
      stateLabel = "planned";
    } else if (run.status === "refused") {
      stateLabel = "declined";
    } else if (run.status === "failed") {
      stateLabel = "not run";
    }
    step.append(node("span", "run-meta", stateLabel));
    list.append(step);
  });
  return list;
}

function plannedRunView(run) {
  const actions = node("div", "run-actions");
  const start = node("button", "primary-button", "Start run");
  start.type = "button";
  start.disabled = state.busy;
  start.addEventListener("click", () => startExistingRun(run.run_id));
  const cancel = node("button", "quiet-button", "Cancel run");
  cancel.type = "button";
  cancel.disabled = state.busy;
  cancel.addEventListener("click", () => cancelRun(run.run_id));
  actions.append(cancel, start);
  return actions;
}

function activeRunActions(run) {
  const actions = node("div", "run-actions");
  const cancel = node("button", "danger-quiet-button", run.status === "cancel_requested" ? "Cancellation requested" : "Cancel run");
  cancel.type = "button";
  cancel.disabled = state.busy || run.status === "cancel_requested";
  cancel.addEventListener("click", () => cancelRun(run.run_id));
  actions.append(cancel);
  return actions;
}

async function cancelRun(runId) {
  if (state.busy || !state.selectedRun || state.selectedRun.run_id !== runId) {
    return;
  }
  clearPoll();
  const epoch = state.navigationEpoch;
  setBusy(true);
  try {
    const cancelled = await request(`/runs/${encodeURIComponent(runId)}/cancel`, {method: "POST"});
    await refreshRuns();
    if (selectionMatches(runId, epoch)) {
      state.selectedRun = cancelled;
      await refreshEvents(runId, epoch);
      renderAll();
      schedulePoll();
    } else {
      renderRuns();
    }
  } catch (error) {
    handleRequestError(error);
    schedulePoll(runId);
  } finally {
    setBusy(false);
    if (selectionMatches(runId, epoch)) {
      renderAll();
    }
  }
}

function answerView(answer) {
  const panel = node("section", "answer-panel");
  panel.append(node("h3", "", answer.complete === false ? "Investigation answer / incomplete" : "Investigation answer"));
  panel.append(node("p", "message-text", answer.summary || "No summary was recorded."));
  const findings = Array.isArray(answer.findings) ? answer.findings : [];
  const citations = Array.isArray(answer.citations) ? answer.citations : [];
  if (findings.length) {
    const list = node("ol", "finding-list");
    findings.forEach((finding, index) => {
      const item = node("li", "", finding);
      if (citations[index]) {
        item.append(citationView(citations[index]));
      }
      list.append(item);
    });
    panel.append(list);
  }
  const nextActions = Array.isArray(answer.next_actions) ? answer.next_actions : [];
  if (nextActions.length) {
    panel.append(node("h4", "", "Next actions"));
    const list = node("ul", "next-action-list");
    nextActions.forEach((item) => list.append(node("li", "", item)));
    panel.append(list);
  }
  return panel;
}

function citationView(citation) {
  const start = Number(citation.start_line) || 1;
  const end = Number(citation.end_line) || start;
  const suffix = start === end ? `:${start}` : `:${start}-${end}`;
  const view = node("div", "citation");
  view.append(node("code", "", `${citation.path || "source"}${suffix}`));
  if (citation.note) {
    view.append(node("span", "", citation.note));
  }
  return view;
}

function budgetView(run) {
  const panel = node("section", "budget-panel");
  panel.append(node("h3", "", "Budget usage"));
  const grid = node("div", "budget-grid");
  const metrics = [
    ["Decisions", "decisions_used", "max_decisions"],
    ["Tool calls", "tool_calls_used", "max_tool_calls"],
    ["Commands", "command_calls_used", "max_command_calls"],
    ["Model requests", "physical_requests_used", "max_physical_requests"],
    ["Completion tokens", "completion_tokens_charged", "max_completion_tokens"],
    ["Evidence bytes", "observation_bytes_used", "max_observation_bytes"],
    ["Active seconds", "active_seconds", "max_active_seconds"]
  ];
  metrics.forEach(([label, usedKey, maxKey]) => {
    const item = node("div", "budget-item");
    item.append(node("span", "", label));
    item.append(node("strong", "", `${formatMetric((run.usage || {})[usedKey])} / ${formatMetric((run.budget || {})[maxKey])}`));
    grid.append(item);
  });
  panel.append(grid);
  return panel;
}

function formatMetric(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "0";
  }
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(1);
}

function activityView(events) {
  const panel = node("section", "activity-panel");
  panel.append(node("h3", "", "Durable activity"));
  const list = node("div", "activity-list");
  events.slice(-100).forEach((event) => {
    const item = node("div", "activity-item");
    item.append(node("span", "activity-sequence", `#${event.sequence}`));
    const detail = node("div", "activity-detail");
    detail.append(node("strong", "", eventLabel(event)));
    const summary = eventSummary(event);
    if (summary) {
      detail.append(node("span", "", summary));
    }
    item.append(detail);
    list.append(item);
  });
  panel.append(list);
  return panel;
}

function eventLabel(event) {
  const labels = {
    "run.created": "Run created",
    "run.queued": "Worker queued",
    "run.status": "Status changed",
    "run.recovered": "Run recovered",
    "investigation.model_request_started": "Local model request",
    "investigation.model_request_abandoned": "Model request recovered",
    "investigation.decision": "Investigation decision",
    "investigation.observation": "Evidence observed",
    "investigation.compaction": "History compacted",
    "investigation.result": "Result recorded",
    "investigation.finished": "Investigation finished",
    "approval.dequeued": "Approval claimed",
    "approval.refreshed": "Approval refreshed"
  };
  return labels[event.kind] || String(event.kind || "Activity");
}

function eventSummary(event) {
  const payload = event && event.payload && typeof event.payload === "object" ? event.payload : {};
  if (event.kind === "run.status") {
    return runStatusLabel({status: payload.status || "unknown", autonomy_level: 1});
  }
  if (event.kind === "run.queued") {
    return payload.work_kind === "resume" ? "Approved action resume" : "Run start";
  }
  if (event.kind === "investigation.decision") {
    const decision = payload.decision && typeof payload.decision === "object" ? payload.decision : {};
    return payload.decision_kind === "answer"
      ? "Final answer selected"
      : `${decision.tool || "tool"}${decision.path ? ` / ${decision.path}` : ""}`;
  }
  if (event.kind === "investigation.observation") {
    const observation = payload.observation && typeof payload.observation === "object" ? payload.observation : {};
    const flags = [observation.truncated ? "truncated" : "", observation.incomplete ? "incomplete" : "", observation.redacted ? "redacted" : ""].filter(Boolean);
    return `${observation.tool || "tool"}${observation.path ? ` / ${observation.path}` : ""}${flags.length ? ` / ${flags.join(", ")}` : ""}`;
  }
  if (event.kind === "investigation.result" || event.kind === "investigation.finished") {
    return payload.stop_reason || payload.status || payload.verdict || "Complete";
  }
  return "";
}

function approvalView(challenge) {
  const panel = node("section", "approval-panel");
  panel.append(node("h3", "", "Review command before execution"));
  panel.append(node("p", "", challenge.reason || "This command executes workspace code."));
  const facts = node("dl", "approval-facts");
  appendFact(facts, "Rule", challenge.rule);
  appendFact(facts, "Workspace", challenge.workspace);
  appendFact(facts, "Domain", challenge.domain);
  panel.append(facts);
  const command = node("pre", "command-preview", JSON.stringify(challenge.argv, null, 2));
  panel.append(command);
  const actions = node("div", "approval-actions");
  const decline = node("button", "quiet-button", "Decline");
  decline.type = "button";
  decline.addEventListener("click", () => resolveApproval("decline"));
  const approve = node("button", "warning-button", "Approve and run");
  approve.type = "button";
  approve.disabled = true;
  approve.addEventListener("click", () => resolveApproval("approvals"));
  actions.append(decline, approve);
  panel.append(actions);

  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.intersectionRatio >= 0.95)) {
      window.setTimeout(() => {
        approve.disabled = state.busy;
      }, 350);
      observer.disconnect();
    }
  }, {threshold: [0.95]});
  window.setTimeout(() => observer.observe(panel), 0);
  return panel;
}

function traceView(trace) {
  const wrapper = node("section", "trace-list");
  trace.actions.forEach((action) => {
    const item = node("div", "trace-action");
    const header = node("div", "trace-action-header");
    header.append(node("code", "", action.name || "action"));
    header.append(node("span", "", action.status || "complete"));
    item.append(header);
    if (action.reason) {
      item.append(node("p", "trace-reason", action.reason));
    }
    appendOutput(item, "stdout", action.stdout, action.stdout_truncated);
    appendOutput(item, "stderr", action.stderr, action.stderr_truncated);
    wrapper.append(item);
  });
  if (trace.actions_truncated || trace.output_truncated) {
    wrapper.append(node("p", "trace-reason", "Output preview was truncated by the server."));
  }
  return wrapper;
}

function appendOutput(parent, label, text, truncated) {
  if (!text) {
    return;
  }
  parent.append(node("span", "output-label", truncated ? `${label} / truncated` : label));
  parent.append(node("pre", "output-block", text));
}

function renderInspector() {
  const runtime = state.runtime;
  const profile = state.profile;
  elements.runtimeDetails.replaceChildren();
  if (runtime) {
    appendFact(elements.runtimeDetails, "Planner", runtime.planner.kind || "unknown");
    appendFact(elements.runtimeDetails, "Model", runtime.planner.model || "deterministic");
    appendFact(elements.runtimeDetails, "Endpoint", runtime.planner.base_url || "offline");
    appendFact(elements.runtimeDetails, "API", runtime.api_version);
  } else {
    appendFact(elements.runtimeDetails, "Status", "Not connected");
  }

  elements.workspaceDetails.replaceChildren();
  if (profile) {
    appendFact(elements.workspaceDetails, "Path", profile.root);
    appendFact(elements.workspaceDetails, "Code consent", hasScope("code_execution") ? "Granted" : "Not granted");
    appendFact(elements.workspaceDetails, "Source consent", hasScope("source_read") ? "Granted" : "Not granted");
    appendFact(elements.workspaceDetails, "Domains", (profile.domains || []).join(", "));
    const toolchain = profile.toolchain || {};
    Object.keys(toolchain).slice(0, 6).forEach((key) => appendFact(elements.workspaceDetails, key, toolchain[key]));
  } else {
    appendFact(elements.workspaceDetails, "Status", "No workspace selected");
  }

  const consent = document.createDocumentFragment();
  for (const scope of ["source_read", "code_execution"]) {
    const granted = hasScope(scope);
    const row = node("div", "consent-row");
    const copy = node("div", "consent-row-copy");
    copy.append(node("strong", "", SCOPE_LABELS[scope]));
    copy.append(node("span", "", granted ? "Granted / revocable" : "Not granted"));
    const button = node("button", granted ? "danger-quiet-button" : "quiet-button", granted ? "Revoke" : "Grant");
    button.type = "button";
    button.disabled = !profile || state.busy;
    button.addEventListener("click", () => {
      if (granted) {
        revokeWorkspaceScope(scope);
      } else {
        openTrustDialog(scope);
      }
    });
    row.append(copy, button);
    consent.append(row);
  }
  elements.consentList.replaceChildren(consent);

  elements.budgetDetails.replaceChildren();
  const run = state.selectedRun;
  if (run && run.kind === "investigation") {
    appendFact(elements.budgetDetails, "Decisions", `${formatMetric((run.usage || {}).decisions_used)} / ${formatMetric((run.budget || {}).max_decisions)}`);
    appendFact(elements.budgetDetails, "Tools", `${formatMetric((run.usage || {}).tool_calls_used)} / ${formatMetric((run.budget || {}).max_tool_calls)}`);
    appendFact(elements.budgetDetails, "Requests", `${formatMetric((run.usage || {}).physical_requests_used)} / ${formatMetric((run.budget || {}).max_physical_requests)}`);
    appendFact(elements.budgetDetails, "Tokens", `${formatMetric((run.usage || {}).completion_tokens_charged)} / ${formatMetric((run.budget || {}).max_completion_tokens)}`);
    appendFact(elements.budgetDetails, "Stop reason", run.stop_reason || "Active");
  } else {
    appendFact(elements.budgetDetails, "Status", "No investigation selected");
  }

  const tools = document.createDocumentFragment();
  if (profile && profile.commands) {
    Object.keys(profile.commands).sort().forEach((name) => tools.append(node("div", "tool-item", name)));
  }
  if (!tools.childNodes.length) {
    tools.append(node("div", "tool-item", "No runnable tools detected"));
  }
  elements.toolList.replaceChildren(tools);
}

function appendFact(list, label, value) {
  list.append(node("dt", "", label));
  list.append(node("dd", "", value === undefined || value === null ? "" : value));
}

function readableError(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error";
}

function handleRequestError(error) {
  const message = readableError(error);
  showToast(message);
  if (error instanceof ApiError && error.status === 401) {
    openSettings(message, message.toLowerCase().includes("approver"));
  }
}

function domainLabel(domain) {
  const labels = {
    django: "Django",
    pytorch: "PyTorch",
    android: "Android",
    android_ndk: "Android NDK",
    ios: "iOS",
    generic: "Generic"
  };
  return labels[domain] || domain || "Unknown";
}

function basename(path) {
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || path;
}

function resizeGoalInput() {
  elements.goalInput.style.height = "auto";
  elements.goalInput.style.height = `${Math.min(elements.goalInput.scrollHeight, 180)}px`;
}

function closeMobileSidebar() {
  elements.shell.classList.remove("sidebar-open");
}

function setInspectorHidden(hidden) {
  elements.shell.classList.toggle("inspector-hidden", hidden);
  elements.inspectorToggle.setAttribute("aria-expanded", String(!hidden));
}

function syncResponsiveInspector(event = mobileInspectorQuery) {
  if (event.matches) {
    setInspectorHidden(true);
  } else {
    elements.inspectorToggle.setAttribute(
      "aria-expanded",
      String(!elements.shell.classList.contains("inspector-hidden"))
    );
  }
}

function setMode(mode) {
  state.mode = mode;
  elements.modeButtons.forEach((button) => {
    const active = Number(button.dataset.mode) === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  renderPlannerNote();
  renderConversation();
}

function setKind(kind) {
  state.kind = kind === "investigation" ? "investigation" : "verification";
  elements.kindSelect.value = state.kind;
  elements.goalInput.placeholder = state.kind === "investigation"
    ? "Describe an investigation"
    : "Describe a verification task";
  renderPlannerNote();
  renderConversation();
}

function chooseRuntimeDefaultKind() {
  if (state.selectedRun || state.pendingGoal || state.pendingRunId) {
    return;
  }
  const planner = state.runtime && state.runtime.planner ? state.runtime.planner : null;
  setKind(planner && planner.investigation_available === true ? "investigation" : "verification");
}

elements.newTask.addEventListener("click", newTask);
elements.settingsButton.addEventListener("click", () => openSettings());
elements.workspaceButton.addEventListener("click", () => openSettings("", false));
elements.settingsClose.addEventListener("click", cancelSettings);
elements.settingsCancel.addEventListener("click", cancelSettings);
elements.settingsDialog.addEventListener("cancel", () => {
  state.pendingGoal = "";
  state.pendingRunId = "";
  state.pendingTrustScope = "";
  state.pendingRevokeScope = "";
});
elements.connectButton.addEventListener("click", connectFromSettings);
elements.forgetButton.addEventListener("click", forgetAccess);
elements.sendButton.addEventListener("click", startGoal);
elements.goalInput.addEventListener("input", resizeGoalInput);
elements.goalInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    startGoal();
  }
});
elements.domainSelect.addEventListener("change", () => {
  state.domain = elements.domainSelect.value;
  renderPlannerNote();
});
elements.kindSelect.addEventListener("change", () => setKind(elements.kindSelect.value));
elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(Number(button.dataset.mode)));
});
function cancelPendingTrust() {
  state.pendingGoal = "";
  state.pendingRunId = "";
  state.pendingTrustScope = "";
}

elements.trustCancel.addEventListener("click", () => {
  cancelPendingTrust();
  elements.trustDialog.close();
});
elements.trustDialog.addEventListener("cancel", cancelPendingTrust);
elements.trustConfirm.addEventListener("click", trustWorkspace);
elements.sidebarToggle.addEventListener("click", () => elements.shell.classList.add("sidebar-open"));
elements.mobileBackdrop.addEventListener("click", closeMobileSidebar);
elements.inspectorToggle.addEventListener("click", () => {
  setInspectorHidden(!elements.shell.classList.contains("inspector-hidden"));
});
mobileInspectorQuery.addEventListener("change", syncResponsiveInspector);

window.addEventListener("beforeunload", clearPoll);

async function bootstrap() {
  syncResponsiveInspector();
  setMode(1);
  renderAll();
  if (!new Set(["127.0.0.1", "localhost", "::1", "[::1]"]).has(window.location.hostname)) {
    showToast("This workbench is intended for a loopback address");
  }
  if (!state.operatorToken) {
    openSettings();
    return;
  }
  try {
    state.runtime = await request("/runtime");
    await refreshWorkspace();
    await refreshRuns();
    chooseRuntimeDefaultKind();
    renderAll();
  } catch (error) {
    handleRequestError(error);
  }
}

bootstrap();
