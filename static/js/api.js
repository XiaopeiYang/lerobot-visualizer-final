// Thin fetch wrapper for the read-only JSON API served by webapp.py.

const BASE = "/api";

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.error?.message ?? response.statusText;
    throw new Error(`${response.status} ${path}: ${message}`);
  }
  return response.json();
}

export function getDatasetMetadata() {
  return getJSON(`${BASE}/dataset`);
}

export function listEpisodes() {
  return getJSON(`${BASE}/episodes`);
}

export function getEpisode(episodeIndex) {
  return getJSON(`${BASE}/episodes/${episodeIndex}`);
}

export function getFrame(episodeIndex, frameIndex) {
  return getJSON(`${BASE}/episodes/${episodeIndex}/frames/${frameIndex}`);
}

export function getNearest(episodeIndex, timestamp) {
  return getJSON(`${BASE}/episodes/${episodeIndex}/nearest?timestamp=${encodeURIComponent(timestamp)}`);
}

export function getMetrics(episodeIndex) {
  return getJSON(`${BASE}/episodes/${episodeIndex}/metrics`);
}

export function videoUrl(episodeIndex) {
  return `${BASE}/episodes/${episodeIndex}/video`;
}

export function getAnalysis() {
  return getJSON(`${BASE}/analysis`);
}
