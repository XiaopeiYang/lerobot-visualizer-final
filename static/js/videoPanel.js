// Owns the native <video> element: transport controls, and bidirectional sync with
// viewerState. During native playback it derives the nearest frame_index client-side
// from the already-fetched metrics timestamp array (binary search) rather than calling
// the per-frame API at video framerate -- see docs/phase-4-visualizer.md.

export function createVideoPanel({ videoElement, viewerState }) {
  let frameIndices = [];
  let timestamps = [];

  function setTimeline(newFrameIndices, newTimestamps) {
    frameIndices = newFrameIndices;
    timestamps = newTimestamps;
  }

  // Nearest-neighbor binary search over the (sorted, strictly increasing) timestamps
  // array -- mirrors the access layer's get_nearest_frame() logic, but client-side so
  // it can run once per timeupdate tick without a network round trip.
  function nearestFrameIndexForTime(t) {
    const n = timestamps.length;
    if (n === 0) return null;
    if (t <= timestamps[0]) return frameIndices[0];
    if (t >= timestamps[n - 1]) return frameIndices[n - 1];
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (timestamps[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    const before = lo > 0 ? lo - 1 : lo;
    const distBefore = Math.abs(timestamps[before] - t);
    const distAt = Math.abs(timestamps[lo] - t);
    return distBefore <= distAt ? frameIndices[before] : frameIndices[lo];
  }

  function currentPosition() {
    const frameIndex = viewerState.get().frameIndex;
    const position = frameIndices.indexOf(frameIndex);
    return position === -1 ? 0 : position;
  }

  videoElement.addEventListener("timeupdate", () => {
    const t = videoElement.currentTime;
    const frameIndex = nearestFrameIndexForTime(t);
    if (frameIndex === null) return;
    viewerState.setFrame({ frameIndex, timestamp: t, source: "video" });
  });

  viewerState.subscribe(({ timestamp, source }) => {
    if (source === "video" || source === "subscribe" || timestamp === null || timestamp === undefined) return;
    videoElement.currentTime = timestamp;
  });

  return {
    setTimeline,
    play: () => videoElement.play(),
    pause: () => videoElement.pause(),
    isPaused: () => videoElement.paused,
    stepFrame(delta) {
      const position = currentPosition();
      const nextPosition = Math.min(Math.max(position + delta, 0), frameIndices.length - 1);
      const frameIndex = frameIndices[nextPosition];
      const timestamp = timestamps[nextPosition];
      videoElement.pause();
      viewerState.setFrame({ frameIndex, timestamp, source: "stepper" });
    },
  };
}
