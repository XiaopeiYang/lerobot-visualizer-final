import { createViewerState } from "./viewerState.js";
import { createVideoPanel } from "./videoPanel.js";
import { createScenePanel } from "./scenePanel.js";
import { createChartsPanel } from "./chartsPanel.js";
import { createFrameInspector } from "./frameInspector.js";
import { HAND_SKELETON_LABEL } from "./handSkeleton.js";
import { initTabs } from "./tabs.js";
import * as api from "./api.js";
import * as evidence from "./evidence.js";

const viewerState = createViewerState();

const videoElement = document.getElementById("video");
const videoPanel = createVideoPanel({ videoElement, viewerState });

const scenePanel = createScenePanel({ container: document.getElementById("scene3d"), viewerState });

const chartsPanel = createChartsPanel({
  canvases: {
    handCentroidSpeed: document.getElementById("chartHandCentroidSpeed"),
    handArticulation: document.getElementById("chartHandArticulation"),
    handSpan: document.getElementById("chartHandSpan"),
    cameraSpeed: document.getElementById("chartCameraSpeed"),
  },
  qaContainer: document.getElementById("qaReadout"),
  viewerState,
});

const frameInspector = createFrameInspector({ container: document.getElementById("frameInspector"), viewerState });

const episodeSelect = document.getElementById("episodeSelect");
const episodeSidebarHeader = document.getElementById("episodeSidebarHeader");
const episodeList = document.getElementById("episodeList");

const viewerTabs = initTabs({
  nav: document.getElementById("viewerTabs"),
  getPanel: (target) => document.querySelector(`[data-tab-panel="${target}"]`),
});
const playPauseButton = document.getElementById("playPause");
const stepBackButton = document.getElementById("stepBack");
const stepForwardButton = document.getElementById("stepForward");
const skeletonToggle = document.getElementById("skeletonToggle");
const cameraToggle = document.getElementById("cameraToggle");
const currentTrackToggle = document.getElementById("currentTrackToggle");
const actionTargetToggle = document.getElementById("actionTargetToggle");
const nextTrackToggle = document.getElementById("nextTrackToggle");
const scrubber = document.getElementById("scrubber");
const scrubberTooltip = document.getElementById("scrubberTooltip");
const frameInput = document.getElementById("frameInput");
const frameTotal = document.getElementById("frameTotal");
const timestampDisplay = document.getElementById("timestampDisplay");

skeletonToggle.nextElementSibling.textContent = HAND_SKELETON_LABEL;
skeletonToggle.addEventListener("change", () => scenePanel.setSkeletonEnabled(skeletonToggle.checked));
cameraToggle.addEventListener("change", () => scenePanel.setCameraGizmoEnabled(cameraToggle.checked));
currentTrackToggle.addEventListener("change", () => scenePanel.setShowCurrent(currentTrackToggle.checked));
actionTargetToggle.addEventListener("change", () => scenePanel.setShowTarget(actionTargetToggle.checked));
nextTrackToggle.addEventListener("change", () => scenePanel.setShowNext(nextTrackToggle.checked));

playPauseButton.addEventListener("click", () => {
  if (videoPanel.isPaused()) {
    videoPanel.play();
  } else {
    videoPanel.pause();
  }
});
stepBackButton.addEventListener("click", () => videoPanel.stepFrame(-1));
stepForwardButton.addEventListener("click", () => videoPanel.stepFrame(1));

videoElement.addEventListener("play", () => {
  playPauseButton.textContent = "⏸ Pause";
});
videoElement.addEventListener("pause", () => {
  playPauseButton.textContent = "▶ Play";
});

let currentFrameIndices = [];
let currentTimestamps = [];

scrubber.addEventListener("input", () => {
  const position = Number(scrubber.value);
  const frameIndex = currentFrameIndices[position];
  const timestamp = currentTimestamps[position];
  if (frameIndex === undefined) return;
  videoElement.pause();
  viewerState.setFrame({ frameIndex, timestamp, source: "scrub" });
});

// Hover preview: shows the timestamp at the cursor's position along the track,
// independent of the scrubber's own value -- purely a preview, doesn't seek.
scrubber.addEventListener("mousemove", (event) => {
  if (currentTimestamps.length === 0) return;
  const rect = scrubber.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  const position = Math.round(fraction * (currentTimestamps.length - 1));
  const timestamp = currentTimestamps[position];
  if (timestamp === undefined) return;
  scrubberTooltip.textContent = `${timestamp.toFixed(3)} s`;
  scrubberTooltip.style.left = `${event.clientX}px`;
  scrubberTooltip.style.top = `${rect.top - 6}px`;
  scrubberTooltip.hidden = false;
});

scrubber.addEventListener("mouseleave", () => {
  scrubberTooltip.hidden = true;
});

viewerState.subscribe(({ frameIndex, timestamp }) => {
  const position = currentFrameIndices.indexOf(frameIndex);
  if (position !== -1) scrubber.value = String(position);
  // Don't overwrite the frame input while the user is actively typing into it.
  if (frameIndex !== null && frameIndex !== undefined && document.activeElement !== frameInput) {
    frameInput.value = String(frameIndex);
    frameTotal.textContent = String(currentFrameIndices[currentFrameIndices.length - 1] ?? "—");
  }
  if (timestamp !== null && timestamp !== undefined) {
    timestampDisplay.textContent = `Timestamp ${timestamp.toFixed(3)} s`;
  }
});

// Nearest-by-value fallback for a typed frame index that isn't in
// currentFrameIndices (e.g. a gap) -- mirrors the "closest available" spirit of
// the dataset access layer's own nearest-frame lookup, just over frame index
// instead of timestamp.
function nearestFrameIndexByValue(target) {
  let nearest = currentFrameIndices[0];
  let nearestDistance = Infinity;
  for (const frameIndex of currentFrameIndices) {
    const distance = Math.abs(frameIndex - target);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = frameIndex;
    }
  }
  return nearest;
}

frameInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  const requested = Number(frameInput.value);
  if (!Number.isFinite(requested) || currentFrameIndices.length === 0) return;
  const frameIndex = currentFrameIndices.includes(requested) ? requested : nearestFrameIndexByValue(requested);
  const position = currentFrameIndices.indexOf(frameIndex);
  videoElement.pause();
  viewerState.setFrame({ frameIndex, timestamp: currentTimestamps[position], source: "frameInput" });
  frameInput.blur();
});

// Best-effort: the Dataset Analysis artifact may not have been computed yet
// (`scripts/analyze_dataset.py`), in which case motion-event/evidence data is
// simply unavailable and every consumer below degrades gracefully (empty
// arrays / null baselines), matching analysis.html's own "not computed yet"
// behavior.
let analysisReport = null;
async function loadAnalysisReport() {
  try {
    analysisReport = await api.getAnalysis();
  } catch {
    analysisReport = null;
  }
}

async function loadEpisode(episodeIndex, initialFrameIndex) {
  viewerState.setEpisode(episodeIndex);
  videoElement.src = api.videoUrl(episodeIndex);

  const metrics = await chartsPanel.loadEpisode(episodeIndex);
  currentFrameIndices = metrics.frame_indices;
  currentTimestamps = metrics.timestamps;
  videoPanel.setTimeline(currentFrameIndices, currentTimestamps);
  frameInspector.setMetrics(metrics);

  const motionEventsForEpisode = evidence.filterMotionEventsByEpisode(analysisReport, episodeIndex);
  frameInspector.setMotionEvents(motionEventsForEpisode, analysisReport?.global ?? null);
  scrubber.min = "0";
  scrubber.max = String(currentFrameIndices.length - 1);
  scrubber.value = "0";

  const startPosition =
    initialFrameIndex !== undefined && currentFrameIndices.includes(initialFrameIndex)
      ? currentFrameIndices.indexOf(initialFrameIndex)
      : 0;
  viewerState.setFrame({
    frameIndex: currentFrameIndices[startPosition],
    timestamp: currentTimestamps[startPosition],
    source: "episode",
  });
}

// Renders the persistent episode sidebar from the same in-memory episodes array
// used to populate the header <select> -- no second fetch. Active-row highlighting
// is driven entirely by viewerState (see subscribe() below) so it stays correct
// no matter which entry point (sidebar click, <select>, or an evidence.js deep
// link) actually changed the episode.
function renderEpisodeSidebar(episodes) {
  episodeSidebarHeader.textContent = `${episodes.length} episodes`;
  episodeList.innerHTML = episodes
    .map(
      (episode) => `
        <li class="episodeRow" data-episode-index="${episode.episode_index}" title="${episode.tasks.join(", ")}">
          <div class="episodeRowTitle">Episode ${episode.episode_index}</div>
          <div class="episodeRowTasks">${episode.tasks.join(", ") || "—"}</div>
        </li>`
    )
    .join("");
  episodeList.addEventListener("click", (event) => {
    const row = event.target.closest(".episodeRow");
    if (!row) return;
    loadEpisode(Number(row.dataset.episodeIndex));
  });
}

viewerState.subscribe(({ episodeIndex }) => {
  if (episodeIndex === null || episodeIndex === undefined) return;
  for (const row of episodeList.querySelectorAll(".episodeRow")) {
    row.classList.toggle("isActive", Number(row.dataset.episodeIndex) === episodeIndex);
  }
});

async function init() {
  await loadAnalysisReport();

  const { episodes } = await api.listEpisodes();
  episodeSelect.innerHTML = episodes
    .map((episode) => `<option value="${episode.episode_index}">episode ${episode.episode_index} (${episode.tasks.join(", ")})</option>`)
    .join("");
  episodeSelect.addEventListener("change", () => loadEpisode(Number(episodeSelect.value)));
  renderEpisodeSidebar(episodes);

  const { episodeIndex, frameIndex, metric } = evidence.parseParams();
  const requestedEpisode = episodeIndex ?? episodes[0]?.episode_index;
  episodeSelect.value = String(requestedEpisode);
  if (metric) {
    frameInspector.focusMetric(metric);
    viewerTabs.activate("inspector");
  }
  await loadEpisode(requestedEpisode, frameIndex);
}

init();
