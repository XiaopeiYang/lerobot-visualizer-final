import assert from "node:assert/strict";
import test from "node:test";

import { createLatestFrameQueue } from "../../static/js/latestFrameQueue.mjs";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function flush() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("keeps one request in flight and coalesces pending frames", async () => {
  const requests = [];
  const rendered = [];
  const queue = createLatestFrameQueue({
    loadFrame(episodeIndex, frameIndex) {
      const result = deferred();
      requests.push({ episodeIndex, frameIndex, result });
      return result.promise;
    },
    onFrame: (frame) => rendered.push(frame.frameIndex),
  });

  queue.requestFrame(0, 0);
  queue.requestFrame(0, 1);
  queue.requestFrame(0, 2);
  assert.deepEqual(requests.map((request) => request.frameIndex), [0]);

  requests[0].result.resolve({ frameIndex: 0 });
  await flush();
  assert.deepEqual(rendered, [0]);
  assert.deepEqual(requests.map((request) => request.frameIndex), [0, 2]);

  requests[1].result.resolve({ frameIndex: 2 });
  await flush();
  assert.deepEqual(rendered, [0, 2]);
});

test("never renders an old episode response after an episode switch", async () => {
  const requests = [];
  const rendered = [];
  const queue = createLatestFrameQueue({
    loadFrame(episodeIndex, frameIndex) {
      const result = deferred();
      requests.push({ episodeIndex, frameIndex, result });
      return result.promise;
    },
    onFrame: (frame) => rendered.push(frame),
  });

  queue.requestFrame(0, 10);
  queue.requestFrame(1, null);
  queue.requestFrame(1, 20);
  requests[0].result.resolve({ episodeIndex: 0, frameIndex: 10 });
  await flush();

  assert.deepEqual(rendered, []);
  assert.deepEqual(
    requests.map(({ episodeIndex, frameIndex }) => [episodeIndex, frameIndex]),
    [
      [0, 10],
      [1, 20],
    ]
  );

  requests[1].result.resolve({ episodeIndex: 1, frameIndex: 20 });
  await flush();
  assert.deepEqual(rendered, [{ episodeIndex: 1, frameIndex: 20 }]);
});

test("continues with the newest pending frame after a load error", async () => {
  const requests = [];
  const errors = [];
  const queue = createLatestFrameQueue({
    loadFrame(episodeIndex, frameIndex) {
      const result = deferred();
      requests.push({ episodeIndex, frameIndex, result });
      return result.promise;
    },
    onFrame() {},
    onError: (error, request) => errors.push([error.message, request.frameIndex]),
  });

  queue.requestFrame(0, 1);
  queue.requestFrame(0, 3);
  requests[0].result.reject(new Error("network failure"));
  await flush();

  assert.deepEqual(errors, [["network failure", 1]]);
  assert.deepEqual(requests.map((request) => request.frameIndex), [1, 3]);
});
