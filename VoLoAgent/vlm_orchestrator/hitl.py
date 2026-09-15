# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human-in-the-loop (HITL) controller for the VLM orchestrator.

Provides a web UI where a human operator can take the role of the VLM:
- At episode start: view the scene and write/rewrite the task decomposition
- During execution: monitor progress, pause, and intervene
- On failure: write specific recovery instructions
- At any time: mark subgoal complete, mark failure, skip, or abort

Architecture:
  The HITL controller runs a lightweight web server (FastAPI + WebSocket)
  alongside the proxy. The proxy strategy checks a shared state object
  for human decisions and blocks on pending actions when in HITL mode.

  Browser  <──WebSocket──>  HITLServer  <──shared state──>  Strategy
     │                         │                               │
     │  show image, buttons    │  HITLState (thread-safe)      │
     │  receive human input    │                               │
     └─────────────────────────┘                               │
                                                               │
     VLA  <──actions──  Proxy  <──obs/instruction──  Strategy ─┘
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Sentinel for "not provided" in update_status (distinct from None)
class _Sentinel:
    pass


_UNSET = _Sentinel()


# ======================================================================
# Shared state (thread-safe bridge between UI and strategy)
# ======================================================================

class HITLAction(str, Enum):
    """Actions the human can take."""
    NONE = "none"                        # No pending action
    REWRITE_INSTRUCTION = "rewrite"      # Rewrite task/subgoal instruction
    START_SUBGOAL = "start"              # Approve and start a subgoal
    SUBGOAL_DONE = "done"               # Mark current subgoal as complete
    FAILURE = "failure"                  # Mark current subgoal as failed
    RECOVERY = "recovery"               # Provide recovery instruction
    SKIP = "skip"                        # Skip current subgoal
    PAUSE = "pause"                      # Pause execution
    RESUME = "resume"                    # Resume execution
    ABORT = "abort"                      # Abort episode
    GRASP_WITH_TOOL = "grasp_tool"       # Trigger planned grasp for a named object
    PLACE_WITH_TOOL = "place_tool"       # Trigger planned placement at a target


@dataclass
class HITLState:
    """Thread-safe shared state between the web UI and the strategy.

    The strategy polls ``pending_action`` each step. The UI thread sets
    it when the human makes a decision.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Current state (read by UI to render)
    episode_id: int = 0
    task_instruction: str = ""
    subgoals: list[str] = field(default_factory=list)
    current_subgoal_idx: int = 0
    current_instruction: str = ""
    step_count: int = 0
    is_paused: bool = False
    is_active: bool = False       # True when episode is running
    last_failure_type: str = ""   # Set by detector, read by UI
    status_message: str = "Waiting for episode..."
    gt_failure: dict | None = None  # GT failure context (set by GT detector)

    # Image (updated every N steps for UI display)
    _current_image: np.ndarray | None = field(default=None, repr=False)
    _image_updated: bool = False

    # Persistent settings (toggled from UI, read by grasp tool etc.)
    skip_pre_grasp_lift: bool = False

    # Pending human decision (written by UI, consumed by strategy)
    pending_action: HITLAction = HITLAction.NONE
    pending_data: dict = field(default_factory=dict)  # action-specific payload

    # Event for blocking strategy until human responds
    _decision_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

    def set_action(self, action: HITLAction, data: dict | None = None):
        """Called by UI thread when human makes a decision."""
        with self._lock:
            self.pending_action = action
            self.pending_data = data or {}
            self._decision_event.set()

    def consume_action(self) -> tuple[HITLAction, dict]:
        """Called by strategy thread to get and clear pending action."""
        with self._lock:
            action = self.pending_action
            data = self.pending_data.copy()
            self.pending_action = HITLAction.NONE
            self.pending_data = {}
            self._decision_event.clear()
            return action, data

    def has_pending_action(self) -> bool:
        with self._lock:
            return self.pending_action != HITLAction.NONE

    def wait_for_decision(self, timeout: float | None = None) -> bool:
        """Block until human makes a decision. Returns True if decision arrived."""
        return self._decision_event.wait(timeout=timeout)

    def update_image(self, image: np.ndarray):
        """Update the current scene image (for UI display)."""
        with self._lock:
            self._current_image = image.copy()
            self._image_updated = True

    def get_image(self) -> tuple[np.ndarray | None, bool]:
        """Get current image. Returns (image, was_updated)."""
        with self._lock:
            updated = self._image_updated
            self._image_updated = False
            return self._current_image, updated

    def update_status(
        self,
        step_count: int | None = None,
        instruction: str | None = None,
        subgoals: list[str] | None = None,
        subgoal_idx: int | None = None,
        failure_type: str | None = None,
        status_message: str | None = None,
        is_paused: bool | None = None,
        is_active: bool | None = None,
        gt_failure: "dict | None | _Sentinel" = _UNSET,
    ):
        """Update display state (called by strategy thread)."""
        with self._lock:
            if step_count is not None:
                self.step_count = step_count
            if instruction is not None:
                self.current_instruction = instruction
            if subgoals is not None:
                self.subgoals = subgoals
            if subgoal_idx is not None:
                self.current_subgoal_idx = subgoal_idx
            if failure_type is not None:
                self.last_failure_type = failure_type
            if status_message is not None:
                self.status_message = status_message
            if is_paused is not None:
                self.is_paused = is_paused
            if is_active is not None:
                self.is_active = is_active
            if gt_failure is not _UNSET:
                self.gt_failure = gt_failure

    def get_display_state(self) -> dict:
        """Snapshot of state for UI rendering."""
        with self._lock:
            return {
                "episode_id": self.episode_id,
                "task_instruction": self.task_instruction,
                "subgoals": list(self.subgoals),
                "current_subgoal_idx": self.current_subgoal_idx,
                "current_instruction": self.current_instruction,
                "step_count": self.step_count,
                "is_paused": self.is_paused,
                "is_active": self.is_active,
                "last_failure_type": self.last_failure_type,
                "status_message": self.status_message,
                "skip_pre_grasp_lift": self.skip_pre_grasp_lift,
                "gt_failure": self.gt_failure,
            }


# ======================================================================
# Web UI server
# ======================================================================

# HTML/JS for the HITL interface (served as a single page)
HITL_HTML = """\
<!DOCTYPE html>
<html>
<head>
<title>VLM Orchestrator — Human-in-the-Loop</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  h1 { color: #4fc3f7; margin-bottom: 10px; font-size: 1.4em; }
  .container { display: grid; grid-template-columns: 1fr 400px; gap: 20px;
               max-width: 1400px; margin: 0 auto; }
  .panel { background: #16213e; border-radius: 8px; padding: 16px;
           border: 1px solid #333; }
  .status-bar { background: #0f3460; padding: 12px 16px; border-radius: 8px;
                margin-bottom: 16px; display: flex; justify-content: space-between;
                align-items: center; }
  .status-bar .badge { background: #4fc3f7; color: #1a1a2e; padding: 4px 12px;
                       border-radius: 12px; font-weight: bold; font-size: 0.85em; }
  .status-bar .badge.paused { background: #ffa726; }
  .status-bar .badge.failure { background: #ef5350; }
  .status-bar .badge.idle { background: #666; }
  #scene-image { width: 100%; border-radius: 6px; background: #111;
                 min-height: 300px; object-fit: contain; }
  .subgoal-list { list-style: none; margin: 10px 0; }
  .subgoal-list li { padding: 8px 12px; margin: 4px 0; border-radius: 6px;
                     background: #1a1a2e; border: 1px solid #333; font-size: 0.9em; }
  .subgoal-list li.active { border-color: #4fc3f7; background: #0f3460; }
  .subgoal-list li.done { border-color: #66bb6a; opacity: 0.6; }
  .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer;
         font-size: 0.95em; font-weight: 600; margin: 4px; transition: 0.2s; }
  .btn:hover { filter: brightness(1.2); }
  .btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .btn-primary { background: #4fc3f7; color: #1a1a2e; }
  .btn-success { background: #66bb6a; color: #1a1a2e; }
  .btn-danger { background: #ef5350; color: white; }
  .btn-warning { background: #ffa726; color: #1a1a2e; }
  .btn-secondary { background: #555; color: white; }
  .btn-group { margin: 12px 0; display: flex; flex-wrap: wrap; gap: 4px; }
  textarea { width: 100%; background: #1a1a2e; color: #e0e0e0; border: 1px solid #444;
             border-radius: 6px; padding: 10px; font-family: inherit; font-size: 0.9em;
             resize: vertical; }
  textarea:focus { border-color: #4fc3f7; outline: none; }
  .section-title { color: #4fc3f7; font-size: 0.85em; text-transform: uppercase;
                   letter-spacing: 1px; margin: 16px 0 8px; }
  .info-row { display: flex; justify-content: space-between; padding: 4px 0;
              font-size: 0.9em; border-bottom: 1px solid #222; }
  .info-row .label { color: #888; }
  #log { background: #111; padding: 10px; border-radius: 6px; font-family: monospace;
         font-size: 0.8em; max-height: 200px; overflow-y: auto; margin-top: 10px;
         white-space: pre-wrap; }
  #plan-editor { border: 2px solid #4fc3f7; }
  #plan-editor .hint { font-size: 0.82em; color: #888; margin-bottom: 8px;
                        line-height: 1.4; }
  #start-exec-btn { width: 100%; padding: 14px; font-size: 1.1em; margin-top: 8px; }
  #sg-line-count { font-size: 0.8em; color: #4fc3f7; float: right; }
</style>
</head>
<body>
<h1>🤖 VLM Orchestrator — Human-in-the-Loop</h1>

<div class="status-bar">
  <span id="status-text">Connecting...</span>
  <span id="status-badge" class="badge idle">IDLE</span>
</div>

<div class="container">
  <!-- Left: Scene view -->
  <div>
    <div class="panel">
      <img id="scene-image" src="" alt="Scene camera" />
    </div>
    <div class="panel" style="margin-top: 12px;">
      <div class="section-title">Event Log</div>
      <div id="log"></div>
    </div>
  </div>

  <!-- Right: Controls -->
  <div>
    <div class="panel">
      <div class="section-title">Task</div>
      <div class="info-row">
        <span class="label">Instruction:</span>
        <span id="task-instruction" style="text-align:right; max-width:250px;">—</span>
      </div>
      <div class="info-row">
        <span class="label">Episode:</span>
        <span id="episode-id">—</span>
      </div>
      <div class="info-row">
        <span class="label">Step:</span>
        <span id="step-count">—</span>
      </div>
    </div>

    <!-- ======== Subgoal Plan Editor (episode start + re-plan) ======== -->
    <div class="panel" style="margin-top: 12px; display:none;" id="plan-editor">
      <div class="section-title">📋 Plan Subgoals <span id="sg-line-count"></span></div>
      <p class="hint">
        Look at the scene, then type one subgoal per line.<br>
        The robot will execute them top-to-bottom in order.
      </p>
      <textarea id="sg-plan-input" rows="6"
                placeholder="Pick up the red block and place it in the bin&#10;Pick up the blue block and place it in the bin&#10;Pick up the green block and place it in the bin"
                oninput="sgUpdateCount()"></textarea>
      <button class="btn btn-success" id="start-exec-btn"
              onclick="sgSubmit()">▶ Start Execution</button>
    </div>

    <!-- ======== Live Subgoal List (during execution) ======== -->
    <div class="panel" style="margin-top: 12px;" id="live-subgoals">
      <div class="section-title">Subgoals</div>
      <ul id="subgoal-list" class="subgoal-list">
        <li>No subgoals yet</li>
      </ul>
    </div>

    <!-- ======== Execution Controls ======== -->
    <div class="panel" style="margin-top: 12px;" id="exec-controls">
      <div class="section-title">Actions</div>

      <div class="btn-group">
        <button class="btn btn-success" onclick="sendAction('done')">✓ Subgoal Done</button>
        <button class="btn btn-danger"  onclick="sendAction('failure')">✗ Failure</button>
        <button class="btn btn-secondary" onclick="sendAction('skip')">⏭ Skip</button>
      </div>

      <div class="btn-group">
        <button class="btn btn-warning" onclick="sendAction('pause')">⏸ Pause</button>
        <button class="btn btn-primary" onclick="sendAction('resume')">▶ Resume</button>
        <button class="btn btn-danger"  onclick="sendAction('abort')">⏹ Abort</button>
      </div>

      <div class="section-title">Instruction</div>
      <textarea id="instruction-input" rows="2"
                placeholder="Recovery instruction or rewritten subgoal..."></textarea>
      <div class="btn-group">
        <button class="btn btn-primary"
                onclick="sendInstruction('recovery')">Send as Recovery</button>
        <button class="btn btn-secondary"
                onclick="sendInstruction('rewrite')">Rewrite Subgoal</button>
        <button class="btn btn-warning"
                onclick="openReplan()">✎ Re-plan…</button>
      </div>

      <div class="section-title">🤏 Planned Grasp</div>
      <div style="display:flex; gap:4px; align-items:center;">
        <input id="grasp-target" type="text" style="flex:1; background:#1a1a2e; color:#e0e0e0;
               border:1px solid #444; border-radius:6px; padding:8px; font-family:inherit; font-size:0.9em;"
               placeholder="Object name (e.g. red block)..." />
        <button class="btn btn-primary" onclick="sendGraspTool()"
                style="white-space:nowrap;">Execute Grasp</button>
      </div>

      <div class="section-title" style="margin-top:10px;">📍 Planned Place</div>
      <div style="display:flex; gap:4px; align-items:center;">
        <input id="place-target" type="text" style="flex:1; background:#1a1a2e; color:#e0e0e0;
               border:1px solid #444; border-radius:6px; padding:8px; font-family:inherit; font-size:0.9em;"
               placeholder="Destination (e.g. red bowl)..." />
        <button class="btn btn-primary" onclick="sendPlaceTool()"
                style="white-space:nowrap;">Execute Place</button>
      </div>
      <div style="display:flex; gap:4px; align-items:center; margin-top:6px;">
        <select id="place-relation" style="background:#1a1a2e; color:#e0e0e0;
                border:1px solid #444; border-radius:6px; padding:6px; font-size:0.85em;">
          <option value="in" selected>in</option>
          <option value="on">on</option>
          <option value="on_top_of">on_top_of</option>
        </select>
        <input id="place-held-hint" type="text" style="flex:1; background:#1a1a2e; color:#e0e0e0;
               border:1px solid #444; border-radius:6px; padding:6px; font-size:0.85em;"
               placeholder="Held object hint (optional, e.g. cube)" />
      </div>
      <div style="display:flex; align-items:center; gap:8px; margin-top:8px;">
        <label style="display:flex; align-items:center; gap:6px; cursor:pointer;
                      font-size:0.88em; color:#ccc; user-select:none;">
          <span style="position:relative; display:inline-block; width:38px; height:20px;">
            <input type="checkbox" id="skip-lift-toggle" onchange="toggleSkipLift()"
                   style="opacity:0; width:0; height:0;" checked />
            <span id="skip-lift-slider" style="position:absolute; cursor:pointer;
                  top:0; left:0; right:0; bottom:0; background:#4fc3f7; border-radius:10px;
                  transition:.3s;"></span>
            <span id="skip-lift-knob" style="position:absolute; height:16px; width:16px;
                  left:20px; bottom:2px; background:white; border-radius:50%;
                  transition:.3s;"></span>
          </span>
          Skip pre-grasp lift
        </label>
        <span id="skip-lift-status" style="font-size:0.78em; color:#4fc3f7;">(lift OFF — skip)</span>
      </div>
    </div>

    <!-- ======== GT Failure Info Panel ======== -->
    <div class="panel" style="margin-top: 12px; display:none; border-color:#ef5350;" id="gt-failure-panel">
      <div class="section-title" style="color:#ef5350;">⚠ GT Failure Detected</div>
      <div id="gt-failure-type" style="font-weight:bold; color:#ffa726; margin:6px 0;"></div>
      <div id="gt-failure-reason" style="font-size:0.9em; margin-bottom:8px;"></div>
      <div class="info-row"><span class="label">Grasped:</span><span id="gt-grasped">—</span></div>
      <div class="info-row"><span class="label">Target(s):</span><span id="gt-targets">—</span></div>
      <div class="info-row"><span class="label">Container:</span><span id="gt-container">—</span></div>
      <div class="info-row"><span class="label">Remaining:</span><span id="gt-remaining">—</span></div>
      <div class="info-row"><span class="label">Completed:</span><span id="gt-completed">—</span></div>
      <div id="gt-near-miss" style="display:none;" class="info-row">
        <span class="label">Near miss:</span><span id="gt-near-miss-val">—</span>
      </div>
      <div class="section-title" style="margin-top:10px;">Suggested Actions</div>
      <div id="gt-suggested-actions" class="btn-group"></div>
    </div>

  </div>
</div>

<script>
let ws;
let prevWaitingForPlan = false;
let manualPlanMode = false;  // true while the user has the re-plan editor open

/* ============================================================
   Subgoal Plan Editor (multi-line textarea)
   ============================================================ */

function sgUpdateCount() {
  const lines = sgNonEmptyLines();
  document.getElementById('sg-line-count').textContent =
    lines.length ? lines.length + ' subgoal' + (lines.length > 1 ? 's' : '') : '';
}

function sgNonEmptyLines() {
  const raw = document.getElementById('sg-plan-input').value;
  return raw.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
}

function sgSubmit() {
  const lines = sgNonEmptyLines();
  if (lines.length === 0) { alert('Type at least one subgoal (one per line).'); return; }
  const instruction = lines.join('\\n');
  if (ws && ws.readyState === 1) {
    const msg = JSON.stringify({action: 'start', instruction: instruction});
    ws.send(msg);
    log('Sent START_SUBGOAL: ' + lines.length + ' subgoal(s)');
    console.log('[HITL] sgSubmit sent:', msg);
  } else {
    log('ERROR: WebSocket not connected, cannot send subgoals');
    console.error('[HITL] sgSubmit: ws not ready, state=' + (ws ? ws.readyState : 'null'));
    return;
  }
  manualPlanMode = false;
  setPanelMode('exec');
}

/** Open the plan editor mid-execution to let the human re-plan.
 *  Pre-fills with remaining (not-yet-done) subgoals. */
function openReplan() {
  const list = document.getElementById('subgoal-list');
  const items = list.querySelectorAll('li:not(.done)');
  const remaining = Array.from(items)
    .map(li => li.textContent.replace(/^\\[\\d+\\]\\s*/, ''))
    .filter(t => t && t !== 'No subgoals yet');
  const ta = document.getElementById('sg-plan-input');
  ta.value = remaining.join('\\n');
  sgUpdateCount();
  document.getElementById('start-exec-btn').textContent = '\\u25b6 Replace Remaining Subgoals';
  manualPlanMode = true;
  setPanelMode('plan');
  ta.focus();
}

/* Show plan editor or execution controls based on state. */
function setPanelMode(mode) {
  document.getElementById('plan-editor').style.display =
    mode === 'plan' ? 'block' : 'none';
  document.getElementById('live-subgoals').style.display =
    mode === 'plan' ? 'none' : 'block';
  document.getElementById('exec-controls').style.display =
    mode === 'plan' ? 'none' : 'block';
}

/* ============================================================
   WebSocket + state updates
   ============================================================ */

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    log('Connected to orchestrator');
    document.getElementById('status-text').textContent = 'Connected';
  };
  ws.onclose = (e) => {
    log('Disconnected (code=' + e.code + ') — reconnecting in 2s...');
    document.getElementById('status-text').textContent = 'Disconnected — reconnecting...';
    setTimeout(connect, 2000);
  };
  ws.onerror = (e) => { log('WebSocket error: ' + (e.message || 'unknown')); };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'state') updateState(msg.data);
    else if (msg.type === 'image') updateImage(msg.data);
    else if (msg.type === 'log') log(msg.data);
  };
}

function updateState(s) {
  document.getElementById('task-instruction').textContent = s.task_instruction || '—';
  document.getElementById('episode-id').textContent = s.episode_id;
  document.getElementById('step-count').textContent = s.step_count;
  document.getElementById('status-text').textContent = s.status_message;

  const badge = document.getElementById('status-badge');
  if (s.is_paused) { badge.textContent = 'PAUSED'; badge.className = 'badge paused'; }
  else if (s.last_failure_type) { badge.textContent = 'FAILURE: ' + s.last_failure_type; badge.className = 'badge failure'; }
  else if (s.is_active) { badge.textContent = 'RUNNING'; badge.className = 'badge'; }
  else { badge.textContent = 'IDLE'; badge.className = 'badge idle'; }

  /* Switch to plan editor when strategy is waiting for initial subgoals
     (active + paused + empty subgoal list).
     Don't touch the panel while the user has the re-plan editor open
     (manualPlanMode) — they'll close it themselves via Submit. */
  const waitingForPlan = s.is_active && s.is_paused && s.subgoals.length === 0;
  if (!manualPlanMode) {
    if (waitingForPlan && !prevWaitingForPlan) {
      document.getElementById('sg-plan-input').value = '';
      document.getElementById('start-exec-btn').textContent = '\\u25b6 Start Execution';
      sgUpdateCount();
      setPanelMode('plan');
      document.getElementById('sg-plan-input').focus();
      log('Episode started \\u2014 type your subgoals and click Start Execution.');
    } else if (!waitingForPlan && prevWaitingForPlan) {
      setPanelMode('exec');
    }
  }
  prevWaitingForPlan = waitingForPlan;

  const list = document.getElementById('subgoal-list');
  list.innerHTML = '';
  s.subgoals.forEach((sg, i) => {
    const li = document.createElement('li');
    li.textContent = '[' + (i + 1) + '] ' + sg;
    if (i === s.current_subgoal_idx) li.className = 'active';
    else if (i < s.current_subgoal_idx) li.className = 'done';
    list.appendChild(li);
  });
  if (!s.subgoals.length) list.innerHTML = '<li>No subgoals yet</li>';

  /* Sync toggle state from server (in case another client changed it) */
  const skipCb = document.getElementById('skip-lift-toggle');
  if (skipCb.checked !== s.skip_pre_grasp_lift) {
    skipCb.checked = s.skip_pre_grasp_lift;
    updateSkipLiftUI(s.skip_pre_grasp_lift);
  }

  /* ---- GT Failure Panel ---- */
  const gtPanel = document.getElementById('gt-failure-panel');
  if (s.gt_failure && s.gt_failure.type) {
    gtPanel.style.display = 'block';
    document.getElementById('gt-failure-type').textContent = s.gt_failure.type.replace(/_/g, ' ').toUpperCase();
    document.getElementById('gt-failure-reason').textContent = s.gt_failure.reason || '';
    document.getElementById('gt-grasped').textContent = s.gt_failure.grasped || '—';
    document.getElementById('gt-targets').textContent = (s.gt_failure.targets || []).join(', ') || '—';
    document.getElementById('gt-container').textContent = s.gt_failure.container || '—';
    document.getElementById('gt-remaining').textContent = (s.gt_failure.remaining || []).join(', ') || '—';
    document.getElementById('gt-completed').textContent = (s.gt_failure.completed || []).join(', ') || 'none';
    if (s.gt_failure.near_miss_distance != null) {
      document.getElementById('gt-near-miss').style.display = 'flex';
      document.getElementById('gt-near-miss-val').textContent =
        s.gt_failure.near_miss_object + ' @ ' + (s.gt_failure.near_miss_distance * 100).toFixed(1) + 'cm';
    } else {
      document.getElementById('gt-near-miss').style.display = 'none';
    }
    /* Render suggested action buttons */
    const actDiv = document.getElementById('gt-suggested-actions');
    actDiv.innerHTML = '';
    (s.gt_failure.suggested_actions || []).forEach(act => {
      const btn = document.createElement('button');
      btn.className = 'btn btn-primary';
      btn.style.fontSize = '0.85em';
      if (act.startsWith('grasp_tool(')) {
        const target = act.match(/grasp_tool\\((.+)\\)/)[1];
        btn.textContent = '\\ud83e\\udd0f Grasp: ' + target;
        btn.onclick = () => {
          document.getElementById('grasp-target').value = target;
          sendGraspTool();
        };
      } else if (act === 'retry') {
        btn.textContent = '\\u21bb Retry';
        btn.className = 'btn btn-warning';
        btn.onclick = () => {
          /* Retry = re-attempt the current subgoal: send a recovery
             action with the current subgoal instruction so counters
             reset and the VLA restarts from scratch. */
          const instr = (s.subgoals && s.subgoals[s.current_subgoal_idx]) || '';
          if (instr && ws && ws.readyState === 1) {
            ws.send(JSON.stringify({action: 'recovery', instruction: instr}));
          } else {
            sendAction('resume');
          }
        };
      } else if (act === 'resume') {
        btn.textContent = '\\u25b6 Resume VLA';
        btn.className = 'btn btn-success';
        btn.onclick = () => sendAction('resume');
      } else if (act === 'replan') {
        btn.textContent = '\\ud83d\\udd04 Replan';
        btn.className = 'btn btn-info';
        btn.onclick = () => openReplan();
      } else if (act === 'skip') {
        btn.textContent = '\\u23ed Skip';
        btn.className = 'btn btn-secondary';
        btn.onclick = () => sendAction('skip');
      } else {
        btn.textContent = act;
        btn.onclick = () => sendAction('resume');
      }
      actDiv.appendChild(btn);
    });
  } else {
    gtPanel.style.display = 'none';
  }
}

function updateImage(b64) {
  document.getElementById('scene-image').src = 'data:image/jpeg;base64,' + b64;
}

function sendGraspTool() {
  const target = document.getElementById('grasp-target').value.trim();
  if (!target) { alert('Type an object name first.'); return; }
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({action: 'grasp_tool', instruction: target}));
  log('Requested planned grasp for: ' + target);
}

function sendPlaceTool() {
  const target = document.getElementById('place-target').value.trim();
  if (!target) { alert('Type a destination first.'); return; }
  const relation = document.getElementById('place-relation').value;
  const heldHint = document.getElementById('place-held-hint').value.trim();
  /* Pack destination + relation + held-object hint into the
     `instruction` field separated by `|` so the strategy can parse
     them.  Format: `<relation>|<target>|<held_hint>` */
  const payload = relation + '|' + target + '|' + heldHint;
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({action: 'place_tool', instruction: payload}));
  log('Requested planned place: ' + relation + ' ' + target
      + (heldHint ? ' (held=' + heldHint + ')' : ''));
}

function toggleSkipLift() {
  const cb = document.getElementById('skip-lift-toggle');
  const on = cb.checked;
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({type: 'setting', key: 'skip_pre_grasp_lift', value: on}));
  updateSkipLiftUI(on);
  log('Skip pre-grasp lift: ' + (on ? 'ON (skip)' : 'OFF (lift)'));
}

function updateSkipLiftUI(on) {
  const slider = document.getElementById('skip-lift-slider');
  const knob = document.getElementById('skip-lift-knob');
  const label = document.getElementById('skip-lift-status');
  slider.style.background = on ? '#4fc3f7' : '#555';
  knob.style.left = on ? '20px' : '2px';
  label.textContent = on ? '(lift OFF — skip)' : '(lift ON)';
  label.style.color = on ? '#4fc3f7' : '#888';
}

function sendAction(action) {
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({action: action}));
}

function sendInstruction(action) {
  const text = document.getElementById('instruction-input').value.trim();
  if (!text) return;
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({action: action, instruction: text}));
  document.getElementById('instruction-input').value = '';
}

function log(msg) {
  const el = document.getElementById('log');
  const ts = new Date().toLocaleTimeString();
  el.textContent += '[' + ts + '] ' + msg + '\\n';
  el.scrollTop = el.scrollHeight;
}

connect();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# Module-level ASGI app factory (used by uvicorn import-string mode)
# ------------------------------------------------------------------

_hitl_app_state: dict[str, Any] = {}


def create_hitl_app():
    """Factory called by uvicorn to build the HITL FastAPI application.

    Uses ``_hitl_app_state["server"]`` (set by :meth:`HITLServer._run`
    before ``uvicorn.run``).
    """
    import asyncio

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    from vlm_orchestrator.vlm import encode_image_b64

    server: "HITLServer" = _hitl_app_state["server"]

    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(HITL_HTML)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        server._clients.append(websocket)
        logger.info("HITL client connected")

        _closed: tuple = (WebSocketDisconnect, RuntimeError)
        try:
            import websockets.exceptions as _ws_exc
            _closed = (WebSocketDisconnect, RuntimeError,
                       _ws_exc.ConnectionClosed)
        except ImportError:
            pass

        async def push_updates():
            while True:
                try:
                    display = server.state.get_display_state()
                    await websocket.send_json(
                        {"type": "state", "data": display})
                except _closed:
                    break
                except Exception:
                    logger.warning("HITL push: state send failed",
                                   exc_info=True)

                try:
                    img, updated = server.state.get_image()
                    if updated and img is not None:
                        b64 = encode_image_b64(img)
                        await websocket.send_json(
                            {"type": "image", "data": b64})
                except _closed:
                    break
                except Exception:
                    logger.warning("HITL push: image send failed",
                                   exc_info=True)

                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    break

        push_task = asyncio.ensure_future(push_updates())

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                # ---- Persistent settings (not one-shot actions) ----
                if msg.get("type") == "setting":
                    key = msg.get("key", "")
                    value = msg.get("value")
                    if key == "skip_pre_grasp_lift":
                        with server.state._lock:
                            server.state.skip_pre_grasp_lift = bool(value)
                        logger.info(
                            f"HITL setting: skip_pre_grasp_lift = {value}"
                        )
                        await websocket.send_json({
                            "type": "log",
                            "data": (
                                f"Setting: skip_pre_grasp_lift = {value}"
                            ),
                        })
                    else:
                        logger.warning(f"HITL: unknown setting key '{key}'")
                    continue

                action_str = msg.get("action", "none")
                instruction = msg.get("instruction", "")

                logger.info(f"HITL action: {action_str} "
                            f"instruction={instruction[:60]}")

                action_map = {
                    "done": HITLAction.SUBGOAL_DONE,
                    "failure": HITLAction.FAILURE,
                    "skip": HITLAction.SKIP,
                    "pause": HITLAction.PAUSE,
                    "resume": HITLAction.RESUME,
                    "abort": HITLAction.ABORT,
                    "recovery": HITLAction.RECOVERY,
                    "rewrite": HITLAction.REWRITE_INSTRUCTION,
                    "start": HITLAction.START_SUBGOAL,
                    "grasp_tool": HITLAction.GRASP_WITH_TOOL,
                    "place_tool": HITLAction.PLACE_WITH_TOOL,
                }
                hitl_action = action_map.get(action_str, HITLAction.NONE)
                data = {}
                if instruction:
                    data["instruction"] = instruction

                server.state.set_action(hitl_action, data)

                await websocket.send_json({
                    "type": "log",
                    "data": (f"Action: {action_str}"
                             + (f' → "{instruction.split(chr(10))[0][:50]}"'
                                if instruction else "")),
                })

        except WebSocketDisconnect:
            logger.info("HITL client disconnected")
        finally:
            push_task.cancel()
            if websocket in server._clients:
                server._clients.remove(websocket)

    return app


class HITLServer:
    """Lightweight web server for HITL interface.

    Runs in a background thread, communicates with the strategy via
    ``HITLState``.
    """

    def __init__(self, state: HITLState, port: int = 8002):
        self.state = state
        self.port = port
        self._thread: threading.Thread | None = None
        self._clients: list = []

    def start(self):
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="hitl-server",
        )
        self._thread.start()
        logger.info(f"HITL UI available at http://localhost:{self.port}")

    def _run(self):
        try:
            import uvicorn
        except ImportError:
            logger.error(
                "HITL mode requires: pip install fastapi uvicorn"
            )
            return

        _hitl_app_state["server"] = self
        app = create_hitl_app()
        uvicorn.run(app, host="0.0.0.0", port=self.port, log_level="warning")
