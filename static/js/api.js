// Thin fetch wrappers for the One-Rec API.

async function parseError(response) {
  const text = await response.text().catch(() => "");
  try {
    const data = JSON.parse(text);
    if (data.detail) return String(data.detail);
  } catch { /* not JSON */ }
  return text || `HTTP ${response.status}`;
}

export async function apiGet(path, { signal } = {}) {
  const res = await fetch(path, { headers: { Accept: "application/json" }, signal });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export const searchTracks = (query, limit, signal) =>
  apiGet(`/api/v1/search?q=${encodeURIComponent(query)}&limit=${limit}`, { signal });

export const importPlaylist = (url) =>
  apiPost("/api/v1/playlist/import", { url });

export const getRecommendations = (tracks, seed, limit = 10) =>
  apiPost("/api/v1/playlist/recommendations", { tracks, seed: seed || undefined, limit });
