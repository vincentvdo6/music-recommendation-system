// One-Rec app: playlist import -> seed search -> one recommendation at a time.

import { getRecommendations, importPlaylist, searchTracks } from "./api.js";
import { watchArtwork } from "./palette.js";
import { Player } from "./player.js";

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
  resultSection: $("resultSection"),
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
};

const player = new Player({
  audioEl: $("audioPlayer"),
  playBtn: $("playBtn"),
  metaEl: $("playerMeta"),
  volumeEl: $("volumeSlider"),
  embedWrap: $("embedWrap"),
});

watchArtwork(els.artwork);

const state = {
  playlist: [],            // entries from /playlist/import
  searchedTrack: null,     // the track the user searched for
  seedIndex: -1,           // -1 = searched track, otherwise index into playlist
  recommendations: [],
  recIndex: 0,
};

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

/* --------------------------------------------------------------- import */

async function handleImport() {
  const url = els.playlistUrl.value.trim();
  if (!url) {
    els.playlistUrl.focus();
    setStatus("Paste a Spotify playlist link to begin.");
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
    setStatus(
      `${name ? name + " — " : ""}${state.playlist.length} tracks ready. Now search a seed song.`,
      true
    );
    els.songInput.disabled = false;
    els.searchBtn.disabled = false;
    els.songInput.focus();
  } catch (err) {
    state.playlist = [];
    els.songInput.disabled = true;
    els.searchBtn.disabled = true;
    setStatus("Import failed — make sure the playlist is public.");
    showError(err.message);
  } finally {
    els.importBtn.disabled = false;
    els.importBtn.textContent = "Import";
  }
}

/* ------------------------------------------------------------ suggestions */

let debounceTimer = null;
let searchController = null;
let activeIndex = -1;

function hideSuggestions() {
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
  clearTimeout(debounceTimer);
  const query = els.songInput.value.trim();
  if (query.length < 2) {
    hideSuggestions();
    return;
  }
  debounceTimer = setTimeout(async () => {
    searchController?.abort();
    searchController = new AbortController();
    try {
      const data = await searchTracks(query, 5, searchController.signal);
      renderSuggestions((data.results || []).map((hit) => hit.track));
    } catch (err) {
      if (err.name !== "AbortError") hideSuggestions();
    }
  }, 150);
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
  if (!query || !state.playlist.length) return;

  els.searchBtn.disabled = true;
  els.searchBtn.textContent = "Finding…";
  clearError();
  try {
    const data = await searchTracks(query, 1);
    const track = data.results?.[0]?.track;
    if (!track) throw new Error(`No tracks found for “${query}”`);
    hideSuggestions();
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

async function requestRecommendations() {
  const seedTrack = seedTrackForIndex();
  if (!seedTrack || !state.playlist.length) return;

  setStatus(`Ranking against ${state.playlist.length} playlist tracks…`, true);
  clearError();

  try {
    const data = await getRecommendations(
      state.playlist.map((t) => ({ ...t, seed: !!t.seed })),
      seedUriOf(seedTrack),
      10
    );
    if (!data.recommendations?.length) throw new Error("No recommendations returned");

    state.recommendations = data.recommendations;
    state.recIndex = 0;

    const fromPlaylist = state.seedIndex >= 0 ? " (from your playlist)" : "";
    els.seedSummary.textContent =
      `${seedTrack.name || "that track"}${seedTrack.artist ? " by " + seedTrack.artist : ""}${fromPlaylist}`;

    const inModel = data.playlist?.tracks_in_model;
    const size = data.playlist?.playlist_size ?? state.playlist.length;
    setStatus(
      typeof inModel === "number"
        ? `${size} tracks · ${inModel} known to the model.`
        : `${size} tracks analysed.`,
      true
    );

    renderRecommendation(true);
    els.resultSection.classList.remove("hidden");
    els.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message);
    setStatus("Recommendation failed — adjust the input and retry.");
  }
}

function renderRecommendation(firstRender = false) {
  const rec = state.recommendations[state.recIndex];
  if (!rec) return;

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
  };

  if (firstRender) {
    update();
    return;
  }
  els.sleeve.classList.add("swapping");
  setTimeout(() => {
    update();
    els.sleeve.classList.remove("swapping");
  }, 200);
}

function step(delta) {
  if (!state.recommendations.length) return;
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

els.prevBtn.addEventListener("click", () => step(-1));
els.nextBtn.addEventListener("click", () => step(1));
els.seedCycleBtn.addEventListener("click", cycleSeed);

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || els.resultSection.classList.contains("hidden")) return;
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});
