"use strict";
const $ = (id) => document.getElementById(id);
let current = null,
  filter = "all",
  history = false,
  lastJournal = "",
  connectionLost = false;
const labels = {
  rapid_posture_change: "Rapid posture change",
  slow_posture_descent: "Gradual loss of posture",
  support_pattern_lost: "New descent after supported posture",
  persistent_low_posture: "Persistent low posture",
  view_lost_after_suspicion: "View lost after concern",
  visible_smoke: "Visible smoke",
  visible_fire: "Visible flame",
  scheduled_scene_overview: "Scheduled scene overview",
  overview_detected_concern: "Concern found in scene overview",
};
const medicalObservations = {
  recovered: "Visible recovery",
  no_concern: "No ongoing concern visible",
  persistent_concern: "No recovery visible",
  view_lost: "Person lost from view",
  uncertain: "Visible outcome unclear",
  immediate_danger: "Immediate danger visible",
};
const limitations = {
  mock_has_no_semantic_evaluation: "Mock: cloud interpretation disabled",
  video_not_configured: "No video source configured",
  video_starting: "Opening video source",
  video_ended: "Video ended — last frame is not live",
  video_disconnected: "Video disconnected",
  video_unavailable: "Video unavailable",
  stale_video: "Video frame is stale",
  analysis_stale: "Analysis is not current",
  pose_missing_weights: "Medical model unavailable: missing weights",
  fire_missing_weights: "Fire model unavailable: missing weights",
  pose_disabled: "Medical route disabled",
  fire_disabled: "Fire route disabled",
  pose_unavailable: "Medical model unavailable",
  fire_unavailable: "Fire model unavailable",
  pose_runtime_error: "Medical inference failed",
  fire_runtime_error: "Fire inference failed",
  dark: "Low light limits visibility",
  blurred: "Blur limits visibility",
  overexposed: "Exposure limits visibility",
  cloud_unavailable: "Cloud assessment unavailable",
  analysis_failed: "Local processing stopped",
  metadata_write_failed: "Metadata storage failed",
  evidence_write_failed: "Evidence storage failed",
};
const escapeHTML = (text) =>
  String(text ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const seconds = (value) => (value == null ? "—" : `${value.toFixed(1)} s`);
const sourceTime = (t) =>
  t == null
    ? "—"
    : `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const localTime = (t) =>
  new Date(t * 1000).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
const human = (s) => String(s ?? "").replaceAll("_", " ");
const observingDevelopment = (e) =>
  e.queue_reason === "awaiting_event_development" && !e.assessment_stale;
function assessmentLabel(e) {
  if (e.superseded_by && e.analysis_status !== "URGENT")
    return "Continued in a later episode";
  if (e.queue_reason === "medical_observation_started")
    return "Observation started in the medical event";
  if (e.assessment_stale)
    return e.request_count
      ? "Previous assessment is stale"
      : "Previous observation is stale";
  if (observingDevelopment(e)) {
    const progress = e.observation_seconds_required
      ? `${e.observation_seconds_seen.toFixed(1)}/${e.observation_seconds_required.toFixed(0)} s`
      : `${e.confirmation_frames_seen}/${e.confirmation_frames_required} new frames`;
    return current.capture_state === "streaming" && !connectionLost
      ? `Observing event development · ${progress}`
      : `Observation incomplete · ${progress}`;
  }
  if (e.analysis_status === "PENDING") return "Assessment pending";
  if (e.policy_reason === "review_pending_resolution") return "Review remains open";
  if (e.policy_reason === "conflicting_medical_answers")
    return "Conflicting assessments";
  if (e.policy_reason === "recovery_not_supported_by_latest_posture")
    return "Recovery unconfirmed";
  if (
    e.policy_reason?.startsWith("local_") ||
    e.policy_reason === "new_transition_after_assessment"
  )
    return "Episode unresolved";
  if (e.ordinary_activity_confirmed)
    return "Ordinary activity · continuity held";
  if (e.category === "medical" && e.medical_observation)
    return (
      medicalObservations[e.medical_observation] || human(e.medical_observation)
    );
  return current.mode === "mock"
    ? "Local suspicion · mock assessment"
    : "Visible evidence assessment";
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  setTimeout(() => ($("toast").hidden = true), 3500);
}
function renderRoute(category, state) {
  const events = current.events.filter(
    (e) =>
      e.category === category &&
      e.lifecycle !== "CLOSED" &&
      e.epoch === current.source?.epoch &&
      !e.assessment_stale,
  );
  const urgent = events.some(
    (e) => e.analysis_status === "URGENT" && !e.assessment_stale,
  );
  const review = events.filter((e) => e.analysis_status === "REVIEW").length;
  const uncertain = events.filter((e) => ["UNKNOWN", "UNAVAILABLE"].includes(e.analysis_status)).length;
  const pending = events.some((e) => e.analysis_status === "PENDING");
  const developing = events.some(observingDevelopment);
  const ready =
    state === "ready" &&
    current.capture_state === "streaming" &&
    !connectionLost;
  $(category + "-status").textContent = urgent
    ? "Immediate review required"
    : review
      ? `${review} event${review === 1 ? "" : "s"} to review`
      : uncertain
        ? `${uncertain} unresolved`
      : pending
        ? developing && ready
          ? "Observing event development"
          : "Assessment pending"
        : ready
          ? "No local candidates"
          : "Observation limited";
  $(category + "-detail").textContent = urgent
    ? "Review the visible evidence now"
    : review
      ? "Assess the visible evidence"
      : pending
        ? developing && ready
          ? "Waiting for new frames after suspicion"
          : "No current assessment available"
        : uncertain ? "Insufficient evidence" : "";
}
function renderJournal() {
  const events = current.events.filter(
    (e) =>
      (history ||
        (e.lifecycle !== "CLOSED" &&
          e.epoch === current.source?.epoch &&
          !e.assessment_stale)) &&
      (filter === "all" || e.category === filter),
  );
  $("event-count").textContent = events.length;
  const historical = current.events.filter(
    (e) =>
      e.lifecycle === "CLOSED" ||
      e.epoch !== current.source?.epoch ||
      e.assessment_stale,
  ).length;
  $("history-toggle").textContent = history
    ? "Hide history"
    : `History${historical ? ` (${historical})` : ""}`;
  const key =
    JSON.stringify(events) +
    history +
    filter +
    current.capture_state +
    connectionLost;
  if (key === lastJournal) return;
  lastJournal = key;
  $("event-list").innerHTML = events.length
    ? events
        .map(
          (e) => `<article class="event ${e.analysis_status.toLowerCase()}">
    <div class="event-top"><span class="category">${escapeHTML(e.category)}</span><time>${localTime(e.created_at)}</time></div>
    <h3>${escapeHTML(labels[e.reason] || human(e.reason))}</h3>
    <div class="event-state"><span class="status">${escapeHTML(e.analysis_status)}</span><span>${escapeHTML(assessmentLabel(e))}</span></div>
    <div class="event-sub">Source ${sourceTime(e.first_ts)} · ${escapeHTML(e.lifecycle.toLowerCase())} · ${e.request_count} check${e.request_count === 1 ? "" : "s"}${e.limited ? " · limited view" : ""}${e.queue_reason ? `<br>${escapeHTML(human(e.queue_reason))}` : ""}${e.evidence_missing ? "<br>Some evidence is unavailable" : ""}</div>
    ${e.feedback ? `<span class="feedback-tag">Operator marked ${escapeHTML(human(e.feedback))}</span>` : ""}
    <div class="event-actions"><button class="primary" data-evidence="${e.incident_id}">View evidence</button>${e.lifecycle !== "CLOSED" ? `<button data-action="confirmed" data-id="${e.incident_id}">Confirm</button><button data-action="false_alarm" data-id="${e.incident_id}">False alarm</button><button data-action="close" data-id="${e.incident_id}">Close</button>` : ""}</div></article>`,
        )
        .join("")
    : `<div class="empty-events"><h3>No events</h3></div>`;
}
function drawOverlay() {
  const canvas = $("overlay"),
    stage = $("video-stage"),
    image = $("video");
  canvas.width = stage.clientWidth * devicePixelRatio;
  canvas.height = stage.clientHeight * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);
  if (!$("show-boxes").checked || !current?.frame || connectionLost) return;
  const { width: w, height: h } = current.frame;
  const scale = Math.min(stage.clientWidth / w, stage.clientHeight / h),
    ox = (stage.clientWidth - w * scale) / 2,
    oy = (stage.clientHeight - h * scale) / 2;
  if (!image.naturalWidth) return;
  for (const obs of current.overlays) {
    const b = obs.box;
    ctx.strokeStyle = obs.label === "person" ? "#eeeeee" : "#e7c68c";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      ox + b.x1 * scale,
      oy + b.y1 * scale,
      (b.x2 - b.x1) * scale,
      (b.y2 - b.y1) * scale,
    );
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "10px Segoe UI";
    ctx.fillText(
      obs.label + (obs.track_id == null ? "" : ` ${obs.track_id}`),
      ox + b.x1 * scale + 3,
      Math.max(12, oy + b.y1 * scale - 5),
    );
  }
}
function render(snapshot) {
  snapshot.events ??= current?.events || [];
  snapshot.overlays ??= [];
  current = snapshot;
  const source = snapshot.source;
  $("source-name").textContent = source?.label || "No source selected";
  $("source-description").textContent =
    source?.kind !== "none"
      ? `${human(source?.state)} · ${human(source?.kind)}${source?.loop ? " · looping" : ""}`
      : "Choose a video, camera or network stream.";
  $("choose-source").textContent =
    source?.kind && source.kind !== "none" ? "Change source" : "Choose source";
  $("stop-source").disabled = !source?.kind || source.kind === "none";
  $("motion-summary").innerHTML = (snapshot.motion || [])
    .filter((m) => m.phase !== "observing")
    .slice(0, 4)
    .map(
      (m) =>
        `<div class="motion-chip"><strong>Track ${escapeHTML(m.track_id)}</strong> · ${escapeHTML(human(m.phase))}<br>${escapeHTML(human(m.transition))}${m.recovery_seconds ? ` · ${m.recovery_seconds.toFixed(1)} s of recovery` : ""}${m.identity_uncertain ? " · association uncertain" : ""}</div>`,
    )
    .join("");
  $("camera").textContent = snapshot.camera_id;
  $("mode").textContent = snapshot.mode === "mock" ? "MOCK / LOCAL" : "OPENAI";
  $("source-privacy").textContent =
    snapshot.mode === "mock"
      ? "Processing stays local in mock mode."
      : "Selected frames from this source will be sent to OpenAI for assessment.";
  $("capture-state").textContent = human(snapshot.capture_state).toUpperCase();
  $("empty-source").classList.toggle("hidden", !!snapshot.frame);
  $("video").classList.toggle("no-source", !snapshot.frame);
  $("source-time").textContent = sourceTime(snapshot.source_ts);
  $("video-resolution").textContent = snapshot.frame
    ? `${snapshot.frame.width} × ${snapshot.frame.height} · EPOCH ${snapshot.frame.epoch}`
    : "";
  $("buffer-depth").textContent =
    `Buffer ${seconds(snapshot.buffer.depth_seconds)}`;
  $("pose-state").textContent = human(snapshot.routes.pose);
  $("fire-state").textContent = human(snapshot.routes.fire);
  renderRoute("medical", snapshot.routes.pose);
  renderRoute("fire", snapshot.routes.fire);
  $("limitations").innerHTML = snapshot.limitations
    .map(
      (s) =>
        `<span class="limitation">${escapeHTML(limitations[s] || human(s))}</span>`,
    )
    .join("");
  $("system-state").textContent = snapshot.limitations.length
    ? "LIMITED"
    : "OBSERVING";
  $("frame-age").textContent = seconds(snapshot.frame_age_seconds);
  $("route-fps").textContent =
    `${snapshot.performance.fps.pose.toFixed(1)} / ${snapshot.performance.fps.fire.toFixed(1)} Hz`;
  $("overview-age").textContent = seconds(
    snapshot.scheduler.overview_age_seconds,
  );
  $("queue").textContent =
    `${snapshot.scheduler.queue_size} / ${snapshot.scheduler.capacity}${snapshot.scheduler.inflight ? " + 1" : ""}`;
  $("requests").textContent =
    snapshot.scheduler.budget.requests.toLocaleString("en-US");
  $("tokens").textContent =
    snapshot.scheduler.budget.input_tokens.toLocaleString("en-US") +
    (snapshot.scheduler.budget.unknown_usage_requests ? "*" : "");
  $("storage").textContent =
    `${(snapshot.storage.evidence_bytes / 1048576).toFixed(1)} / ${(snapshot.storage.quota_bytes / 1048576).toFixed(0)} MB`;
  $("usage-label").textContent =
    snapshot.mode === "mock"
      ? "Mock mode does not interpret images. Usage is simulated."
      : "Usage includes reservations for unknown responses. Quotas use UTC.";
  renderJournal();
  drawOverlay();
}
async function act(id, action, button) {
  button.disabled = true;
  try {
    const response = await fetch(
      `/api/v1/events/${id}/${action === "close" ? "close" : "feedback"}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Stillwatch-Action": "operator",
        },
        body: action === "close" ? undefined : JSON.stringify({ action }),
      },
    );
    if (!response.ok) throw Error("Action could not be saved");
    toast(action === "close" ? "Event closed." : "Operator feedback saved.");
    const item = await response.json();
    if (current) {
      const index = current.events.findIndex((e) => e.incident_id === id);
      if (index >= 0) current.events[index] = item;
      renderJournal();
    }
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}
async function showEvidence(id) {
  try {
    const response = await fetch(`/api/v1/events/${id}`);
    if (!response.ok) throw Error("Event unavailable");
    const data = await response.json(),
      e = data.event;
    $("detail-title").textContent = labels[e.reason] || human(e.reason);
    let body = `<p class="detail-meta">${escapeHTML(e.category)} · ${escapeHTML(e.analysis_status)} · ${escapeHTML(e.lifecycle)}<br>Source ${sourceTime(e.first_ts)} — ${sourceTime(e.last_ts)} · ${escapeHTML(e.incident_id)}<br>Operator feedback: ${escapeHTML(human(e.feedback || "none"))}</p>`;
    if (e.motion)
      body += `<h3 class="detail-section">Episode context</h3><p class="detail-meta">${escapeHTML(human(e.motion.phase))} · ${escapeHTML(human(e.motion.transition))}<br>Last observed posture: ${escapeHTML(human(e.motion.posture))} at ${sourceTime(e.motion.last_seen_ts)}<br>Recovery observed: ${e.motion.recovery_seconds.toFixed(1)} s${e.motion.recovery_basis ? ` · ${escapeHTML(human(e.motion.recovery_basis))}` : ""}<br>Association: ${e.motion.identity_uncertain ? "uncertain" : "continuous local track"}<br>Ordinary activity remembered: ${e.ordinary_activity_confirmed ? "yes" : "no"}</p>`;
    if (e.confirmation_frames_required)
      body += `<p class="detail-meta">Initial observation: ${e.confirmation_frames_seen}/${e.confirmation_frames_required} new frames after suspicion at ${sourceTime(e.detected_ts)}.</p>`;
    if (e.observation_seconds_required)
      body += `<p class="detail-meta">Observation window: ${e.observation_seconds_seen.toFixed(1)}/${e.observation_seconds_required.toFixed(0)} seconds of source time after the first suspicion.</p>`;
    if (!data.packets.length)
      body +=
        '<p class="evidence-unavailable">No packet has been sent. Local metadata is available; cloud interpretation may be queued, unavailable, or disabled.</p>';
    for (const p of data.packets) {
      const kind = p.temporal_context?.archived_without_assessment ? "Local snapshot" : "Packet";
      body += `<h3 class="detail-section">${kind} ${p.version} · ${p.images.length} images · ${escapeHTML(p.packet_id.slice(0, 10))}</h3><div class="evidence-grid">`;
      p.images.forEach((im, i) => {
        const file = p.files[i];
        body += `<figure class="evidence-frame">${file?.available ? `<img src="${escapeHTML(file.url)}" alt="${escapeHTML(im.kind)} at source time ${im.capture_ts.toFixed(2)} seconds" loading="lazy">` : '<p class="evidence-unavailable">Evidence expired or unavailable</p>'}<figcaption>${sourceTime(im.capture_ts)} · ${escapeHTML(im.kind)} · ${im.width} × ${im.height}</figcaption></figure>`;
      });
      body += "</div>";
    }
    body += '<h3 class="detail-section">Assessment history</h3>';
    for (const a of data.assessments) {
      body += `<p class="detail-meta">${escapeHTML(a.provider)} / ${escapeHTML(a.model)} · ${a.latency_ms.toFixed(0)} ms · input ${a.input_tokens ?? "unknown"}${a.stale ? " · stale response" : ""}<br>Medical: ${escapeHTML(a.medical_status)} · Fire: ${escapeHTML(a.fire_status)}${a.error ? `<br>Error: ${escapeHTML(a.error)}` : ""}</p><details><summary class="detail-meta">Raw answer</summary><pre class="raw-result">${escapeHTML(JSON.stringify(a.raw_answer, null, 2))}</pre></details>`;
      if (a.medical_observation)
        body += `<p class="detail-meta">Visible outcome: ${escapeHTML(medicalObservations[a.medical_observation] || human(a.medical_observation))}${a.policy_reason ? `<br>Assessment rule: ${escapeHTML(human(a.policy_reason))}` : ""}.</p>`;
      if (a.latest_posture)
        body += `<p class="detail-meta">Latest visible posture: ${escapeHTML(human(a.latest_posture))}.</p>`;
    }
    $("detail-body").innerHTML = body;
    $("detail-dialog").showModal();
  } catch (error) {
    toast(error.message);
  }
}
$("event-list").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.evidence) showEvidence(button.dataset.evidence);
  else if (button.dataset.action)
    act(button.dataset.id, button.dataset.action, button);
});
document.querySelectorAll("[data-filter]").forEach((button) =>
  button.addEventListener("click", () => {
    filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((b) => {
      b.classList.toggle("active", b === button);
      b.setAttribute("aria-pressed", String(b === button));
    });
    if (current) renderJournal();
  }),
);
$("history-toggle").addEventListener("click", () => {
  history = !history;
  $("history-toggle").textContent = history ? "Hide history" : "Show history";
  $("history-toggle").setAttribute("aria-pressed", String(history));
  if (current) renderJournal();
});
$("dialog-close").addEventListener("click", () => $("detail-dialog").close());
$("show-boxes").addEventListener("change", drawOverlay);
window.addEventListener("resize", drawOverlay);
setInterval(
  () =>
    ($("clock").textContent = new Date().toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })),
  1000,
);
let updates;
function connectUpdates() {
  updates?.close();
  $("video").src = "/api/v1/stream";
  updates = new EventSource("/api/v1/updates");
  updates.addEventListener("snapshot", (event) => {
    connectionLost = false;
    $("connection").textContent = "Desk connected";
    $("connection-dot").classList.remove("offline");
    render(JSON.parse(event.data));
  });
  updates.onerror = async () => {
    connectionLost = true;
    $("connection").textContent = "Connection lost";
    $("connection-dot").classList.add("offline");
    $("system-state").textContent = "DISCONNECTED";
    if (current) {
      renderRoute("medical", "unavailable");
      renderRoute("fire", "unavailable");
      renderJournal();
    }
    drawOverlay();
    try {
      const response = await fetch("/api/v1/telemetry");
      if (response.status === 401) {
        updates.close();
        if (!$("access-dialog").open) $("access-dialog").showModal();
      }
    } catch {
      /* The connection indicator already describes an unreachable service. */
    }
  };
}

async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, options);
  if (!response.ok) {
    if (response.status === 401 && !$("access-dialog").open)
      $("access-dialog").showModal();
    const data = await response.json().catch(() => ({}));
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`,
    );
  }
  return response.json();
}

async function openSource() {
  $("source-error").hidden = true;
  try {
    const videos = await api("/videos");
    $("saved-video").innerHTML =
      '<option value="">Select a video</option>' +
      videos
        .map(
          (v) =>
            `<option value="${escapeHTML(v.file_id)}">${escapeHTML(v.name)} · ${(v.bytes / 1048576).toFixed(1)} MB</option>`,
        )
        .join("");
    if (current?.source?.file_id)
      $("saved-video").value = current.source.file_id;
    if (!$("source-dialog").open) $("source-dialog").showModal();
  } catch (error) {
    toast(error.message);
  }
}
$("choose-source").addEventListener("click", openSource);
$("source-close").addEventListener("click", () => $("source-dialog").close());
$("source-kind").addEventListener("change", () => {
  for (const kind of ["file", "camera", "rtsp"])
    $(kind + "-fields").hidden = $("source-kind").value !== kind;
  $("source-error").hidden = true;
});
$("video-file").addEventListener("change", () => {
  $("file-label").textContent =
    $("video-file").files[0]?.name ||
    "MP4, AVI, MOV, MKV or WebM · up to 512 MB";
  $("saved-video").value = "";
  $("server-path").value = "";
});
$("saved-video").addEventListener("change", () => {
  $("video-file").value = "";
  $("server-path").value = "";
  $("file-label").textContent = "MP4, AVI, MOV, MKV or WebM · up to 512 MB";
});
$("server-path").addEventListener("input", () => {
  $("video-file").value = "";
  $("saved-video").value = "";
});
const actionHeaders = {
  "Content-Type": "application/json",
  "X-Stillwatch-Action": "operator",
};
$("source-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("source-submit");
  button.disabled = true;
  $("source-error").hidden = true;
  try {
    const kind = $("source-kind").value;
    const selection = { kind };
    if (kind === "file") {
      selection.loop = $("source-loop").checked;
      const file = $("video-file").files[0];
      if (file) {
        button.textContent = "Uploading…";
        const saved = await api(
          `/videos?name=${encodeURIComponent(file.name)}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/octet-stream",
              "X-Stillwatch-Action": "operator",
            },
            body: file,
          },
        );
        selection.file_id = saved.file_id;
      } else if ($("server-path").value.trim())
        selection.path = $("server-path").value.trim();
      else if ($("saved-video").value)
        selection.file_id = $("saved-video").value;
      else throw Error("Choose a video file or a saved video.");
    } else if (kind === "camera")
      selection.camera_index = Number($("camera-index").value);
    else {
      selection.url = $("rtsp-url").value.trim();
      if (!selection.url) throw Error("Enter an RTSP stream URL.");
    }
    button.textContent = "Connecting…";
    await api("/source", {
      method: "PUT",
      headers: actionHeaders,
      body: JSON.stringify(selection),
    });
    $("source-dialog").close();
    $("rtsp-url").value = "";
    $("video-file").value = "";
    $("file-label").textContent = "MP4, AVI, MOV, MKV or WebM · up to 512 MB";
    render(await api("/telemetry"));
    toast("Source changed. A fresh observation has started.");
  } catch (error) {
    $("source-error").textContent = error.message;
    $("source-error").hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Use source";
  }
});
$("stop-source").addEventListener("click", async () => {
  $("stop-source").disabled = true;
  try {
    await api("/source", {
      method: "PUT",
      headers: actionHeaders,
      body: JSON.stringify({ kind: "none" }),
    });
    render(await api("/telemetry"));
    toast("Source stopped. Events are preserved.");
  } catch (error) {
    toast(error.message);
    $("stop-source").disabled = false;
  }
});
$("access-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/session", {
      method: "POST",
      headers: { Authorization: `Bearer ${$("access-token").value}` },
    });
    $("access-token").value = "";
    $("access-dialog").close();
    $("access-error").hidden = true;
    connectUpdates();
  } catch (error) {
    $("access-error").textContent = error.message;
    $("access-error").hidden = false;
  }
});
api("/telemetry")
  .then((snapshot) => {
    render(snapshot);
    connectUpdates();
  })
  .catch((error) => toast(error.message));
