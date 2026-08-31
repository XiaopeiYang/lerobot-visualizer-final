// Bounded async queue for panels that load per-frame data over HTTP.
//
// Native video playback may publish frame changes faster than the frame API can
// respond. Keeping one request in flight and one latest pending request avoids
// both an unbounded request backlog and the starvation caused by discarding every
// response whenever a newer frame has already been requested.

export function createLatestFrameQueue({ loadFrame, onFrame, onError = console.error }) {
  let activeEpisodeIndex = null;
  let inFlight = null;
  let pending = null;

  function sameFrame(left, right) {
    return (
      left !== null &&
      right !== null &&
      left.episodeIndex === right.episodeIndex &&
      left.frameIndex === right.frameIndex
    );
  }

  async function drain() {
    if (inFlight !== null || pending === null) return;

    const request = pending;
    pending = null;
    inFlight = request;

    try {
      const frame = await loadFrame(request.episodeIndex, request.frameIndex);
      // A completed frame from the active episode is useful even if playback has
      // already advanced: displaying it gives steady, bounded-lag progress. The
      // newest pending frame is loaded immediately afterward.
      if (request.episodeIndex === activeEpisodeIndex) onFrame(frame);
    } catch (error) {
      if (request.episodeIndex === activeEpisodeIndex) onError(error, request);
    } finally {
      inFlight = null;
      void drain();
    }
  }

  function requestFrame(episodeIndex, frameIndex) {
    if (episodeIndex !== activeEpisodeIndex) {
      activeEpisodeIndex = episodeIndex;
      pending = null;
    }

    if (episodeIndex === null || episodeIndex === undefined || frameIndex === null || frameIndex === undefined) {
      pending = null;
      return;
    }

    const request = { episodeIndex, frameIndex };
    if (sameFrame(request, inFlight)) {
      // The latest desired state is already loading, so an older pending frame is
      // no longer relevant.
      pending = null;
      return;
    }
    if (sameFrame(request, pending)) return;

    pending = request;
    void drain();
  }

  return { requestFrame };
}
