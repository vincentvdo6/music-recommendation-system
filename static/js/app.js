// One-Rec app: the cover (seed search + optional playlist) tweens into the
// result sleeve — one recommendation at a time.

import {
  getRecommendations,
  importPlaylist,
  searchTracks,
  sendFeedback,
} from "./api.js";
import { watchArtwork } from "./palette.js";
import { Player } from "./player.js";
import { initScrollScrub } from "./scrub.js";

const $ = (id) => document.getElementById(id);

const els = {
  playlistUrl: $("playlistUrlInput"),
  importBtn: $("playlistImportBtn"),
  playlistStatus: $("playlistStatus"),
  songInput: $("songInput"),
  searchBtn: $("searchBtn"),
  searchForm: $("searchForm"),
  suggestions: $("suggestions"),
  errorBanner: $("errorBanner"),
  importView: $("importView"),
  seedView: $("seedView"),
  resultView: $("resultView"),
  rail: $("deckRail"),
  seedKicker: $("seedKicker"),
  seedBackBtn: $("seedBackBtn"),
  backBtn: $("backBtn"),
  sleeve: $("sleeve"),
  seedSummary: $("seedSummary"),
  artwork: $("recArtwork"),
  recIndex: $("recIndex"),
  recTitle: $("recTitle"),
  recArtist: $("recArtist"),
  metaAlbum: $("metaAlbum"),
  metaYear: $("metaYear"),
  metaPopularity: $("metaPopularity"),
  matchRow: $("matchRow"),
  matchFill: $("matchFill"),
  matchValue: $("matchValue"),
  whyLine: $("whyLine"),
  spotifyLink: $("spotifyLink"),
  appleMusicLink: $("appleMusicLink"),
  prevBtn: $("prevBtn"),
  nextBtn: $("nextBtn"),
  seedCycleBtn: $("seedCycleBtn"),
  likeBtn: $("likeBtn"),
  neutralBtn: $("neutralBtn"),
  dislikeBtn: $("dislikeBtn"),
  feedbackStatus: $("feedbackStatus"),
  moreLikeBtn: $("moreLikeBtn"),
  moreSimilarBtn: $("moreSimilarBtn"),
  preferenceProfile: $("preferenceProfile"),
};

const audioEl = $("audioPlayer");
const player = new Player({
  audioEl,
  playBtn: $("playBtn"),
  metaEl: $("playerMeta"),
  volumeEl: $("volumeSlider"),
  embedWrap: $("embedWrap"),
  onEvent: (event) => recordFeedback(event),
});

watchArtwork(els.artwork);

function browserSessionId() {
  const key = "one-rec-session-id";
  let value = null;
  try {
    value = sessionStorage.getItem(key);
  } catch {
    // Privacy modes can disable storage; the in-memory id still keeps the
    // current page session coherent.
  }
  if (!value) {
    value = globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    try {
      sessionStorage.setItem(key, value);
    } catch {
      // Best-effort persistence only; recommendation serving must still work.
    }
  }
  return value;
}

function storedProfile() {
  try {
    const version = sessionStorage.getItem("one-rec-profile-version");
    if (version !== "similarity-v2") {
      sessionStorage.setItem("one-rec-profile-version", "similarity-v2");
      sessionStorage.setItem("one-rec-profile", "familiar");
      return "familiar";
    }
    const value = sessionStorage.getItem("one-rec-profile");
    if (["familiar", "balanced", "explorer"].includes(value)) return value;
  } catch {
    // Storage is optional; Closest match is the safe default.
  }
  return "familiar";
}

const state = {
  playlist: [],            // entries from /playlist/import
  searchedTrack: null,     // the track the user searched for
  seedIndex: -1,           // -1 = searched track, otherwise index into playlist
  recommendations: [],
  recIndex: 0,
  impressionId: null,
  viewed: new Set(),
  recShownAt: 0,
  sessionId: browserSessionId(),
  profile: storedProfile(),
};
els.preferenceProfile.value = state.profile;

const ART_PLACEHOLDER =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="100%" height="100%" fill="#17150f"/><circle cx="256" cy="256" r="170" fill="none" stroke="#f6f3ec" stroke-width="2"/><circle cx="256" cy="256" r="24" fill="#f6f3ec"/></svg>`
  );

/* ------------------------------------------------------------- messaging */

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.classList.remove("hidden");
}

function clearError() {
  els.errorBanner.classList.add("hidden");
}

function setStatus(message, positive = false) {
  els.playlistStatus.textContent = message;
  els.playlistStatus.classList.toggle("positive", positive);
}

/* ------------------------------------------------------------ view switch */

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const views = [els.importView, els.seedView, els.resultView];
let viewTransitionActive = false;

// Same-document page swap. Elements tagged with view-transition-name in the
// CSS (wordmark, seed, sleeve) morph between their per-view positions;
// browsers without startViewTransition just get an instant swap.
const setDeck = (v) => document.documentElement.style.setProperty("--deck", String(v));

function transitionTo(index, after) {
  const apply = () => {
    views.forEach((view, i) => view.classList.toggle("hidden", i !== index));
    window.scrollTo(0, 0);
    after?.();
  };
  if (!document.startViewTransition || reducedMotion.matches) {
    apply();
    setDeck(index);
    return;
  }
  viewTransitionActive = true;
  const vt = document.startViewTransition(apply);
  // Slide the rail thumb after the morph so it doesn't fight the snapshot.
  vt.finished.finally(() => {
    viewTransitionActive = false;
    setDeck(index);
  });
}

function focusSearch() {
  // Skip on touch devices — popping the keyboard after a swipe is hostile.
  if (!window.matchMedia("(hover: hover)").matches) return;
  els.songInput.focus();
  els.songInput.select();
}

function goToSeed() {
  if (els.resultView.classList.contains("hidden") || scrub.isActive()) return;
  audioEl.pause();
  transitionTo(1, focusSearch);
}

function goToImport() {
  if (els.seedView.classList.contains("hidden") || scrub.isActive()) return;
  transitionTo(0);
}

// Scroll gestures scrub between neighbouring views instead of scrolling
// the document (see scrub.js). Moving past the cover requires an imported
// playlist; moving past the seed page requires recommendations.
const scrub = initScrollScrub({
  page: document.querySelector(".page"),
  views,
  coverWordmark: document.querySelector(".cover-wordmark"),
  seedWordmark: document.querySelector("#seedView .page-head .wordmark"),
  canAdvance: (from) =>
    from === 0 ? state.playlist.length > 0 : state.recommendations.length > 0,
  isBusy: () => viewTransitionActive,
  onSettled: (index) => {
    if (index !== 2) audioEl.pause();
    if (index === 1) focusSearch();
  },
});

/* --------------------------------------------------------------- import */

async function handleImport() {
  const url = els.playlistUrl.value.trim();
  if (!url) {
    els.playlistUrl.focus();
    setStatus("Paste a Spotify playlist link.");
    return;
  }

  els.importBtn.disabled = true;
  els.importBtn.textContent = "Importing…";
  setStatus("Importing playlist…");
  clearError();

  try {
    const data = await importPlaylist(url);
    state.playlist = (data.tracks || []).map((t, i) => ({ ...t, seed: i === 0 }));
    state.seedIndex = -1;
    if (!state.playlist.length) throw new Error("Playlist has no usable tracks");

    const name = data.summary?.name;
    setStatus(`${name ? name + " — " : ""}${state.playlist.length} tracks imported.`, true);
    els.seedKicker.textContent = `${name || "Playlist"} — ${state.playlist.length} tracks`;
    els.rail.classList.remove("hidden");
  } catch (err) {
    state.playlist = [];
    els.rail.classList.add("hidden");
    setStatus(`Import failed — ${err.message}. The playlist must be public.`);
  } finally {
    els.importBtn.disabled = false;
    els.importBtn.textContent = "Import";
  }
}

/* ------------------------------------------------------------ suggestions */

let debounceTimer = null;
let searchController = null;
let activeIndex = -1;

// Session cache: query -> tracks. Backspacing or retyping a prefix renders
// instantly instead of re-hitting the API.
const suggestCache = new Map();

function cacheSuggestions(query, tracks) {
  suggestCache.set(query.toLowerCase(), tracks);
  if (suggestCache.size > 80) suggestCache.delete(suggestCache.keys().next().value);
}

// Stale-while-revalidate: filter the longest cached prefix of the query so
// something relevant renders on the keystroke itself, then the network
// response replaces it. After a few characters the target track is usually
// already in this list, so autocomplete feels instant.
function prefixFallback(query) {
  const q = query.toLowerCase();
  for (let end = q.length - 1; end >= 2; end--) {
    const cached = suggestCache.get(q.slice(0, end));
    if (!cached) continue;
    const words = q.split(/\s+/).filter(Boolean);
    return cached.filter((t) =>
      words.every((w) => `${t.name} ${t.artist}`.toLowerCase().includes(w))
    );
  }
  return null;
}

function cancelSuggestionSearch() {
  clearTimeout(debounceTimer);
  debounceTimer = null;
  searchController?.abort();
  searchController = null;
}

function hideSuggestions() {
  cancelSuggestionSearch();
  els.suggestions.classList.add("hidden");
  activeIndex = -1;
}

function renderSuggestions(tracks) {
  els.suggestions.textContent = "";
  if (!tracks.length) {
    hideSuggestions();
    return;
  }
  tracks.forEach((track) => {
    const row = document.createElement("div");
    row.className = "suggestion";
    row.setAttribute("role", "option");

    const img = document.createElement("img");
    img.src = track.image_url || ART_PLACEHOLDER;
    img.alt = "";

    const info = document.createElement("div");
    info.className = "suggestion-info";
    const title = document.createElement("div");
    title.className = "suggestion-title";
    title.textContent = track.name || "Unknown";
    const artist = document.createElement("div");
    artist.className = "suggestion-artist";
    artist.textContent = track.artist || "Unknown artist";
    info.append(title, artist);

    row.append(img, info);
    row.addEventListener("mousedown", (e) => e.preventDefault()); // keep input focus
    row.addEventListener("click", () => {
      els.songInput.value = `${track.name} — ${track.artist}`;
      hideSuggestions();
      recommendFor(track);
    });
    els.suggestions.appendChild(row);
  });
  els.suggestions.classList.remove("hidden");
}

function onSearchInput() {
  cancelSuggestionSearch();
  const query = els.songInput.value.trim();
  if (query.length < 2) {
    hideSuggestions();
    return;
  }
  const cached = suggestCache.get(query.toLowerCase());
  if (cached) {
    renderSuggestions(cached);
    return;
  }
  const approx = prefixFallback(query);
  if (approx?.length) renderSuggestions(approx);
  debounceTimer = setTimeout(async () => {
    debounceTimer = null;
    const controller = new AbortController();
    searchController = controller;
    try {
      const data = await searchTracks(query, 5, controller.signal);
      if (controller.signal.aborted) return;
      const tracks = (data.results || []).map((hit) => hit.track);
      cacheSuggestions(query, tracks);
      if (els.songInput.value.trim() !== query) return;
      renderSuggestions(tracks);
    } catch (err) {
      if (err.name !== "AbortError") hideSuggestions();
    } finally {
      if (searchController === controller) searchController = null;
    }
  }, 30);
}

function onSearchKeydown(e) {
  const items = els.suggestions.querySelectorAll(".suggestion");
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeIndex = Math.min(activeIndex + 1, items.length - 1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeIndex = Math.max(activeIndex - 1, -1);
  } else if (e.key === "Enter" && activeIndex >= 0) {
    e.preventDefault();
    items[activeIndex].click();
    return;
  } else if (e.key === "Escape") {
    // When the dropdown is open, Escape only closes it — don't let the
    // document handler also navigate back a page.
    if (!els.suggestions.classList.contains("hidden")) e.stopPropagation();
    hideSuggestions();
    return;
  } else {
    return;
  }
  items.forEach((item, i) => item.classList.toggle("active", i === activeIndex));
}

async function handleSearchSubmit(e) {
  e.preventDefault();
  const query = els.songInput.value.trim();
  if (!query) return;

  hideSuggestions();
  els.searchBtn.disabled = true;
  els.searchBtn.textContent = "Finding…";
  clearError();
  try {
    // Reuse the suggestion results (client cache, then the server's search
    // cache — same limit keeps the cache key identical) before a cold call.
    let track = suggestCache.get(query.toLowerCase())?.[0];
    if (!track) {
      const data = await searchTracks(query, 5);
      track = data.results?.[0]?.track;
    }
    if (!track) throw new Error(`No tracks found for “${query}”`);
    await recommendFor(track);
  } catch (err) {
    showError(err.message);
  } finally {
    els.searchBtn.disabled = false;
    els.searchBtn.textContent = "Recommend";
  }
}

/* --------------------------------------------------------- recommendation */

function seedTrackForIndex() {
  if (state.seedIndex < 0) return state.searchedTrack;
  return state.playlist[state.seedIndex];
}

function seedUriOf(track) {
  if (!track) return undefined;
  if (track.uri) return track.uri;
  const id = track.id || track.spotify_id;
  return id ? `spotify:track:${id}` : undefined;
}

async function recommendFor(searchedTrack) {
  state.searchedTrack = searchedTrack;
  state.seedIndex = -1;
  await requestRecommendations();
}

let recommendationRequestId = 0;
let recommendationBusy = false;
let feedbackActionInFlight = false;

function setFeedbackBusy(busy) {
  [
    els.prevBtn,
    els.nextBtn,
    els.likeBtn,
    els.neutralBtn,
    els.dislikeBtn,
    els.moreLikeBtn,
    els.moreSimilarBtn,
    els.preferenceProfile,
  ].forEach((element) => { element.disabled = busy; });
}

// Visible busy state on the seed page — without it, a slow recommendation
// request looks like a dead click and invites double-submits.
function setSearchBusy(busy) {
  els.searchBtn.disabled = busy;
  els.searchBtn.textContent = busy ? "Finding…" : "Recommend";
  els.songInput.disabled = busy;
  els.seedCycleBtn.disabled = busy || !state.playlist.length;
}

async function requestRecommendations() {
  if (recommendationBusy) return;
  const seedTrack = seedTrackForIndex();
  if (!seedTrack) return;
  recommendationBusy = true;
  const requestId = ++recommendationRequestId;

  setSearchBusy(true);
  setFeedbackBusy(true);
  setStatus(
    state.playlist.length
      ? `Ranking against ${state.playlist.length} playlist tracks…`
      : "Ranking by seed similarity…",
    true
  );
  clearError();

  try {
    const data = await getRecommendations(
      state.playlist.map((t) => ({ ...t, seed: !!t.seed })),
      seedUriOf(seedTrack),
      10,
      state.sessionId,
      state.profile
    );
    if (requestId !== recommendationRequestId) return;
    if (!data.recommendations?.length) throw new Error("No recommendations returned");

    state.recommendations = data.recommendations;
    state.recIndex = 0;
    state.impressionId = data.impression_id || null;
    state.viewed = new Set();
    els.seedCycleBtn.disabled = !state.playlist.length;

    const fromPlaylist = state.seedIndex >= 0 ? " (from your playlist)" : "";
    els.seedSummary.textContent =
      `${seedTrack.name || "that track"}${seedTrack.artist ? " by " + seedTrack.artist : ""}${fromPlaylist}`;

    const inModel = data.playlist?.tracks_in_model;
    const size = data.playlist?.playlist_size ?? state.playlist.length;
    setStatus(
      size === 0
        ? "No playlist — ranked from the seed alone."
        : typeof inModel === "number"
          ? `${size} tracks · ${inModel} known to the model.`
          : `${size} tracks analysed.`,
      true
    );
    if (size > 0) {
      els.seedKicker.textContent =
        typeof inModel === "number"
          ? `${size} tracks · ${inModel} known to the model`
          : `${size} tracks`;
    }

    renderRecommendation(true);
    // Give the artwork a beat to decode so the morph reveals real art,
    // but never stall the transition on a slow image.
    await Promise.race([
      els.artwork.decode().catch(() => {}),
      new Promise((resolve) => setTimeout(resolve, 400)),
    ]);
    if (requestId !== recommendationRequestId) return;
    // The third page is now part of the deck: the rail switches to thirds.
    document.documentElement.style.setProperty("--deck-count", "3");
    transitionTo(2);
  } catch (err) {
    if (requestId !== recommendationRequestId) return;
    showError(err.message);
    setStatus("Recommendation failed — adjust the input and retry.");
    // The error banner lives on the seed page; make sure it's visible.
    if (!els.resultView.classList.contains("hidden")) transitionTo(1);
  } finally {
    if (requestId === recommendationRequestId) {
      recommendationBusy = false;
      setSearchBusy(false);
      setFeedbackBusy(false);
    }
  }
}

let renderTimer = null;

function currentRecommendation() {
  return state.recommendations[state.recIndex] || null;
}

function recordFeedback(event, extra = {}) {
  const rec = currentRecommendation();
  if (!rec || !state.impressionId) return Promise.resolve();
  return sendFeedback({
    impression_id: state.impressionId,
    track_id: rec.id,
    event,
    position: state.recIndex,
    ...extra,
  }).catch(() => {}); // feedback must never interrupt playback/navigation
}

function recordDismiss(event = "dismiss") {
  if (!state.recShownAt) return;
  recordFeedback(event, { dwell_ms: Math.max(0, Date.now() - state.recShownAt) });
}

async function chooseFeedback(event) {
  if (feedbackActionInFlight) return;
  feedbackActionInFlight = true;
  setFeedbackBusy(true);
  try {
    await recordFeedback(event);
    player.markHandled();
    const messages = {
      like: "Marked as a fit.",
      neutral: "Kept neutral — no preference shift.",
      dislike: "Marked not for you.",
      more_like_this: "Using this to tune the session.",
      not_similar_enough: "Tightening results around the seed.",
    };
    els.feedbackStatus.textContent = messages[event] || "Feedback saved.";
    els.likeBtn.setAttribute("aria-pressed", String(event === "like"));
    els.neutralBtn.setAttribute("aria-pressed", String(event === "neutral"));
    els.dislikeBtn.setAttribute("aria-pressed", String(event === "dislike"));
    els.moreLikeBtn.setAttribute("aria-pressed", String(event === "more_like_this"));
    els.moreSimilarBtn.setAttribute("aria-pressed", String(event === "not_similar_enough"));
    if (event === "more_like_this") await requestRecommendations();
    if (event === "not_similar_enough") {
      state.profile = "familiar";
      els.preferenceProfile.value = state.profile;
      try {
        sessionStorage.setItem("one-rec-profile", state.profile);
        sessionStorage.setItem("one-rec-profile-version", "similarity-v2");
      } catch {
        // The in-memory selection is still applied.
      }
      await requestRecommendations();
    }
  } finally {
    feedbackActionInFlight = false;
    if (!recommendationBusy) setFeedbackBusy(false);
  }
}

function renderRecommendation(firstRender = false) {
  const rec = state.recommendations[state.recIndex];
  if (!rec) return;
  clearTimeout(renderTimer);
  renderTimer = null;
  els.sleeve.classList.remove("swapping");

  const update = () => {
    els.artwork.src = rec.image_url || ART_PLACEHOLDER;
    els.recTitle.textContent = rec.name;
    els.recArtist.textContent = rec.artist;
    els.recIndex.textContent =
      `${String(state.recIndex + 1).padStart(2, "0")} / ${String(state.recommendations.length).padStart(2, "0")}`;

    els.metaAlbum.textContent = rec.album || "—";
    const year = rec.release_date ? new Date(rec.release_date).getFullYear() : NaN;
    els.metaYear.textContent = Number.isFinite(year) ? year : "—";
    els.metaPopularity.textContent =
      typeof rec.popularity === "number" && rec.popularity > 0 ? `${rec.popularity} / 100` : "—";

    // Honest similarity: seed co-listen cosine from the model, hidden when the
    // seed wasn't in the embedding vocabulary (score ~ 0).
    const sim = rec.similarity_score ?? 0;
    if (sim > 0.05) {
      els.matchRow.classList.remove("hidden");
      els.matchValue.textContent = `${Math.round(sim * 100)}%`;
      els.matchFill.style.width = `${Math.round(Math.min(1, sim) * 100)}%`;
    } else {
      els.matchRow.classList.add("hidden");
    }

    const factors = rec.explanation?.top_factors || [];
    els.whyLine.textContent = factors.length ? `Why: ${factors.join(" · ")}` : "";

    const query = encodeURIComponent(`${rec.name || ""} ${rec.artist || ""}`.trim());
    els.spotifyLink.href = rec.external_urls?.spotify || `https://open.spotify.com/search/${query}`;
    els.appleMusicLink.href = `https://music.apple.com/us/search?term=${query}`;

    player.load(rec);
    state.recShownAt = Date.now();
    els.feedbackStatus.textContent = "";
    els.likeBtn.setAttribute("aria-pressed", "false");
    els.neutralBtn.setAttribute("aria-pressed", "false");
    els.dislikeBtn.setAttribute("aria-pressed", "false");
    els.moreLikeBtn.setAttribute("aria-pressed", "false");
    els.moreSimilarBtn.setAttribute("aria-pressed", "false");
    const viewKey = `${state.impressionId}:${rec.id}`;
    if (!state.viewed.has(viewKey)) {
      state.viewed.add(viewKey);
      recordFeedback("view");
    }
  };

  if (firstRender) {
    update();
    return;
  }
  els.sleeve.classList.add("swapping");
  renderTimer = setTimeout(() => {
    update();
    els.sleeve.classList.remove("swapping");
    renderTimer = null;
  }, 200);
}

function step(delta) {
  if (recommendationBusy || !state.recommendations.length) return;
  if (!player.wasHandled()) {
    recordDismiss(player.wasSkipped() ? "skip" : "dismiss");
  }
  state.recIndex =
    (state.recIndex + delta + state.recommendations.length) % state.recommendations.length;
  renderRecommendation();
}

function cycleSeed() {
  if (!state.searchedTrack || !state.playlist.length) return;
  state.seedIndex += 1;
  if (state.seedIndex >= state.playlist.length) state.seedIndex = -1;
  requestRecommendations();
}

/* ---------------------------------------------------------------- wiring */

els.importBtn.addEventListener("click", handleImport);
els.playlistUrl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    handleImport();
  }
});

els.songInput.addEventListener("input", onSearchInput);
els.songInput.addEventListener("keydown", onSearchKeydown);
els.songInput.addEventListener("blur", () => setTimeout(hideSuggestions, 150));
els.searchForm.addEventListener("submit", handleSearchSubmit);

els.backBtn.addEventListener("click", goToSeed);
els.seedBackBtn.addEventListener("click", goToImport);

els.prevBtn.addEventListener("click", () => step(-1));
els.nextBtn.addEventListener("click", () => step(1));
els.seedCycleBtn.addEventListener("click", cycleSeed);
els.likeBtn.addEventListener("click", () => chooseFeedback("like"));
els.neutralBtn.addEventListener("click", () => chooseFeedback("neutral"));
els.dislikeBtn.addEventListener("click", () => chooseFeedback("dislike"));
els.moreLikeBtn.addEventListener("click", () => chooseFeedback("more_like_this"));
els.moreSimilarBtn.addEventListener("click", () => chooseFeedback("not_similar_enough"));
els.preferenceProfile.addEventListener("change", () => {
  state.profile = els.preferenceProfile.value;
  try {
    sessionStorage.setItem("one-rec-profile", state.profile);
    sessionStorage.setItem("one-rec-profile-version", "similarity-v2");
  } catch {
    // Keep the selection in memory when storage is unavailable.
  }
  if (state.recommendations.length) requestRecommendations();
});
els.spotifyLink.addEventListener("click", () => recordFeedback("open_spotify"));
els.appleMusicLink.addEventListener("click", () => recordFeedback("open_apple"));

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!els.resultView.classList.contains("hidden")) goToSeed();
    else if (!els.seedView.classList.contains("hidden")) goToImport();
    return;
  }
  if (els.resultView.classList.contains("hidden") || e.target instanceof HTMLInputElement) return;
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});
