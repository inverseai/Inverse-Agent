"use strict";

const OPERATOR_KEY = "inverse-agent.operator-token";
const WORKSPACE_KEY = "inverse-agent.workspace";
const TRANSIENT_STATUSES = new Set(["starting", "running", "approving"]);
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "refused"]);
const STATUS_LABELS = {
  planned: "Planned",
  starting: "Starting",
  running: "Running",
  waiting_for_approval: "Waiting for approval",
  approving: "Executing approved action",
  succeeded: "Succeeded",
  failed: "Failed",
  refused: "Declined"
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
  mode: 1,
  domain: "",
  busy: false,
  pendingGoal: "",
  pendingRunId: "",
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
  domainSelect: document.getElementById("domain-select"),
  modeButtons: Array.from(document.querySelectorAll("[data-mode]")),
  conversation: document.getElementById("conversation"),
  goalInput: document.getElementById("goal-input"),
  plannerNote: document.getElementById("planner-note"),
  sendButton: document.getElementById("send-button"),
  runtimeDetails: document.getElementById("runtime-details"),
  workspaceDetails: document.getElementById("workspace-details"),
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
  trustPath: document.getElementById("trust-path"),
  trustStatus: document.getElementById("trust-status"),
  trustCancel: document.getElementById("trust-cancel"),
  trustConfirm: document.getElementById("trust-confirm"),
  toastRegion: document.getElementById("toast-region")
};

function authScope(method, path) {
  const pathname = path.split("?", 1)[0];
  if (pathname === "/approver/session") {
    return "approver";
  }
  if (method === "POST" && pathname === "/workspaces/trust") {
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

async function connectFromSettings() {
  const operatorCandidate = elements.operatorInput.value.trim() || state.operatorToken;
  const approverCandidate = elements.approverInput.value.trim();
  const workspaceCandidate = elements.workspaceInput.value.trim();
  if (!operatorCandidate) {
    elements.settingsStatus.textContent = "Operator token is required";
    return;
  }

  setBusy(true);
  const previousOperator = state.operatorToken;
  const previousApprover = state.approverToken;
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
    renderAll();
    if ((state.pendingGoal || state.pendingRunId) && state.profile && !state.profile.trust.trusted) {
      openTrustDialog();
    }
  } catch (error) {
    state.operatorToken = previousOperator;
    state.approverToken = previousApprover;
    elements.settingsStatus.textContent = readableError(error);
  } finally {
    setBusy(false);
    updateConnectionState();
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
  state.pendingGoal = "";
  state.pendingRunId = "";
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
    state.trace = trace;
    state.workspace = selectedRun.workspace;
    sessionStorage.setItem(WORKSPACE_KEY, state.workspace);
    if (!await refreshWorkspace(epoch) || epoch !== state.navigationEpoch) {
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
  state.pendingGoal = "";
  state.pendingRunId = "";
  elements.goalInput.value = "";
  resizeGoalInput();
  renderAll();
  elements.goalInput.focus();
  closeMobileSidebar();
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
  state.pendingGoal = goal;
  state.pendingRunId = "";
  if (state.mode === 1 && !state.profile.trust.trusted) {
    openTrustDialog();
    return;
  }
  await createAndStartRun(goal);
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
  if (run.autonomy_level !== 0 && state.profile && !state.profile.trust.trusted) {
    state.pendingRunId = runId;
    openTrustDialog();
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

function openTrustDialog() {
  if (!state.profile) {
    return;
  }
  if (!state.approverToken) {
    openSettings("Enter the approver token to trust this workspace", true);
    return;
  }
  elements.trustPath.textContent = state.profile.root || state.workspace;
  elements.trustStatus.textContent = state.approverIdentity ? `Approver: ${state.approverIdentity}` : "";
  if (!elements.trustDialog.open) {
    elements.trustDialog.showModal();
  }
  window.setTimeout(() => elements.trustCancel.focus(), 0);
}

async function trustWorkspace() {
  if (!state.profile || !state.approverToken) {
    return;
  }
  let runToStart = "";
  let goalToStart = "";
  setBusy(true);
  try {
    await request("/workspaces/trust", {
      method: "POST",
      body: {workspace: state.profile.root || state.workspace}
    });
    elements.trustDialog.close();
    await refreshWorkspace();
    renderAll();
    if (state.pendingRunId) {
      runToStart = state.pendingRunId;
      state.pendingRunId = "";
    } else if (state.pendingGoal) {
      goalToStart = state.pendingGoal;
    }
  } catch (error) {
    elements.trustStatus.textContent = readableError(error);
  } finally {
    setBusy(false);
  }
  if (runToStart) {
    await startExistingRun(runToStart);
  } else if (goalToStart) {
    await createAndStartRun(goalToStart);
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
}

function renderPlannerNote() {
  const planner = state.runtime && state.runtime.planner ? state.runtime.planner : null;
  if (!planner) {
    elements.plannerNote.textContent = "Connect to inspect the planner";
    return;
  }
  if (planner.kind === "deterministic") {
    elements.plannerNote.textContent = `Standard ${state.domain || "domain"} verification sequence; the goal is recorded`;
  } else {
    elements.plannerNote.textContent = `${planner.model || "Local model"} selects registered verification tools`;
  }
}

function renderConversation() {
  const inner = node("div", "conversation-inner");
  const run = state.selectedRun;
  if (!run) {
    const empty = node("div", "empty-state");
    empty.append(node("div", "empty-mark", "IA"));
    empty.append(node("h1", "", "Start a verification task"));
    empty.append(node("p", "", "Run typed engineering checks with explicit workspace trust and human approval. This workbench does not edit source code."));
    inner.append(empty);
    elements.conversation.replaceChildren(inner);
    return;
  }

  inner.append(message("user", "You", run.goal));
  const runMessage = node("article", "message run-message");
  runMessage.append(node("div", "message-avatar", "IA"));
  const content = node("div", "message-content");
  content.append(node("div", "message-label", "Verification run"));
  const heading = node("div", "run-heading");
  heading.append(node("span", `status-badge ${run.status}`, runStatusLabel(run)));
  heading.append(node("span", "message-text", domainLabel(run.domain)));
  content.append(heading);

  if (run.plan_rationale) {
    content.append(node("p", "message-text", run.plan_rationale));
  } else if (run.status === "starting") {
    content.append(node("p", "message-text", "Preparing the verification plan..."));
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
  if (run.error) {
    const errorPanel = node("section", "error-panel");
    errorPanel.append(node("h3", "", run.status === "refused" ? "Task declined" : "Task stopped"));
    errorPanel.append(node("p", "", run.error));
    content.append(errorPanel);
  }
  if (state.trace && Array.isArray(state.trace.actions)) {
    content.append(traceView(state.trace));
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
  actions.append(start);
  return actions;
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
    appendFact(elements.workspaceDetails, "Trusted", profile.trust.trusted ? "Yes" : "No");
    appendFact(elements.workspaceDetails, "Domains", (profile.domains || []).join(", "));
    const toolchain = profile.toolchain || {};
    Object.keys(toolchain).slice(0, 6).forEach((key) => appendFact(elements.workspaceDetails, key, toolchain[key]));
  } else {
    appendFact(elements.workspaceDetails, "Status", "No workspace selected");
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

function setMode(mode) {
  state.mode = mode;
  elements.modeButtons.forEach((button) => {
    const active = Number(button.dataset.mode) === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  renderPlannerNote();
}

elements.newTask.addEventListener("click", newTask);
elements.settingsButton.addEventListener("click", () => openSettings());
elements.workspaceButton.addEventListener("click", () => openSettings("", false));
elements.settingsClose.addEventListener("click", closeSettings);
elements.settingsCancel.addEventListener("click", closeSettings);
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
elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(Number(button.dataset.mode)));
});
function cancelPendingTrust() {
  state.pendingGoal = "";
  state.pendingRunId = "";
}

elements.trustCancel.addEventListener("click", () => {
  cancelPendingTrust();
  elements.trustDialog.close();
});
elements.trustDialog.addEventListener("cancel", cancelPendingTrust);
elements.trustConfirm.addEventListener("click", trustWorkspace);
elements.sidebarToggle.addEventListener("click", () => elements.shell.classList.add("sidebar-open"));
elements.mobileBackdrop.addEventListener("click", closeMobileSidebar);
elements.inspectorToggle.addEventListener("click", () => elements.shell.classList.toggle("inspector-hidden"));

window.addEventListener("beforeunload", clearPoll);

async function bootstrap() {
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
    renderAll();
  } catch (error) {
    handleRequestError(error);
  }
}

bootstrap();
