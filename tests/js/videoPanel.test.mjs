import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// videoPanel.js is an ES module in the browser. Loading its source through a data
// URL keeps this test independent of npm/package.json module settings.
const source = await readFile(new URL("../../static/js/videoPanel.js", import.meta.url), "utf8");
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { createVideoPanel } = await import(moduleUrl);

class FakeVideoElement extends EventTarget {
  constructor() {
    super();
    this.currentTime = 0;
    this.paused = true;
  }

  play() {
    this.paused = false;
    return Promise.resolve();
  }

  pause() {
    this.paused = true;
  }
}

function makeViewerState(initialFrameIndex = 0) {
  const published = [];
  let listener = null;
  return {
    published,
    get: () => ({ frameIndex: initialFrameIndex }),
    setFrame(frame) {
      published.push(frame);
    },
    subscribe(newListener) {
      listener = newListener;
      newListener({ timestamp: null, source: "subscribe" });
    },
    publish(frame) {
      listener(frame);
    },
  };
}

test("playback publishes frame changes after an initial zero-time seek", () => {
  const videoElement = new FakeVideoElement();
  const viewerState = makeViewerState();
  const panel = createVideoPanel({ videoElement, viewerState });
  panel.setTimeline([0, 1, 2], [0, 0.1, 0.2]);

  // Setting currentTime from 0 to 0 does not reliably emit `seeked` in browsers.
  viewerState.publish({ frameIndex: 0, timestamp: 0, source: "episode" });
  videoElement.currentTime = 0.1;
  videoElement.dispatchEvent(new Event("timeupdate"));

  assert.deepEqual(viewerState.published, [{ frameIndex: 1, timestamp: 0.1, source: "video" }]);
});

test("video-originated state changes do not seek the video again", () => {
  const videoElement = new FakeVideoElement();
  const viewerState = makeViewerState();
  createVideoPanel({ videoElement, viewerState });

  videoElement.currentTime = 0.25;
  viewerState.publish({ frameIndex: 3, timestamp: 0.3, source: "video" });

  assert.equal(videoElement.currentTime, 0.25);
});
