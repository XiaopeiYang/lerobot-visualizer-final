// Shared viewer state: the single source of truth for episode -> frame_index -> timestamp.
// All panels (video, 3D, charts, inspector) subscribe here and may independently call
// setFrame()/setEpisode() to drive navigation; the state module fans changes out to
// every other panel so they stay in lockstep regardless of which one initiated the change.

export function createViewerState() {
  const state = { episodeIndex: null, frameIndex: null, timestamp: null };
  const listeners = new Set();

  function notify(source) {
    const snapshot = { ...state, source };
    for (const listener of listeners) listener(snapshot);
  }

  return {
    get() {
      return { ...state };
    },

    subscribe(listener) {
      listeners.add(listener);
      listener({ ...state, source: "subscribe" });
      return () => listeners.delete(listener);
    },

    // The single mutation entry point for frame navigation. `source` records which
    // panel triggered the change ("video", "chart", "scrub", "stepper", "nearest") for
    // debugging and future Phase 4.4 evidence-linkage -- it does not change behavior,
    // except that repeated "video" ticks landing on the same frame_index (the native
    // <video> element's timeupdate event fires far more often than the frame rate) are
    // coalesced so downstream panels re-render at most once per actual frame change.
    setFrame({ frameIndex, timestamp, source }) {
      const frameChanged = frameIndex !== state.frameIndex;
      state.frameIndex = frameIndex;
      state.timestamp = timestamp;
      if (!frameChanged && source === "video") return;
      notify(source);
    },

    setEpisode(episodeIndex) {
      state.episodeIndex = episodeIndex;
      state.frameIndex = null;
      state.timestamp = null;
      notify("episode");
    },
  };
}
