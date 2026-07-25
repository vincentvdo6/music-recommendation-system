// Thin fetch wrappers for the One-Rec API.

export function formatApiDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (typeof item === "string") return item;
      const location = Array.isArray(item?.loc)
        ? item.loc.filter((part) => part !== "body").join(".")
        : "request";
      return `${location || "request"}: ${item?.msg || JSON.stringify(item)}`;
    }).join("; ");
  }
  if (detail && typeof detail === "object") {
    return detail.message || detail.error || JSON.stringify(detail);
  }
  return String(detail || "Request failed");
}

async function parseError(response) {
  const text = await response.text().catch(() => "");
  try {
    const data = JSON.parse(text);
    if (data.detail) return formatApiDetail(data.detail);
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

export const getRecommendations = (tracks, seed, limit = 10, sessionId, profile = "familiar") =>
  apiPost("/api/v1/playlist/recommendations", {
    tracks,
    seed: seed || undefined,
    limit,
    session_id: sessionId,
    profile,
  });

export const sendFeedback = (feedback) =>
  apiPost("/api/v1/feedback", feedback);
