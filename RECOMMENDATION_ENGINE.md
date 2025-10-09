# Contextual Recommendation Engine

## Overview

The music recommendation system now requires a user-supplied playlist to
personalise results. Instead of collaborative filtering, it combines a curated
knowledge base of expertly tagged tracks with a profile that is inferred from a
playlist the listener copies and pastes into the product. The playlist is
never served back verbatim; its sole purpose is to teach the engine about the
listener's preferred moods, activities, genres, and audio characteristics before
the final recommendations are generated from the curated catalogue.

The engine remains accurate even when there is no prior history on the platform,
making it ideal for fresh users, episodic sessions, or embedded experiences
where personalised storage is not available.

Key pillars:

- **Playlist personalisation** that derives an averaged audio profile and tag
  preferences from any playlist the listener provides.
- **Curated catalogue** of tracks that include audio features, mood/activity
  tags, temporal preferences, and geographic relevance.
- **Context interpreter** that normalises optional overrides (`mood`, `activity`,
  `time_of_day`, `energy`, `tempo`, `era`, `region`) into canonical targets.
- **Audio similarity** via Spotify-style feature vectors when a seed track or
  features are available.
- **Transparent scoring** with per-track explanations so results can be audited
  and tuned without guesswork.

## Architecture

### 0. Playlist Personalisation

File: `services/recommendation/contextual_engine.py` (`build_user_profile`)

- Accepts any playlist the user pastes into the UI (Spotify URLs, IDs, or raw
  "Song - Artist" strings).
- Resolves the tracks via Spotify, normalises their audio features, and derives
  dominant moods, activities, genres, regions, energy band, tempo bucket, and
  preferred era.
- Produces a `user_profile` object that is required for every recommendation
  request.

### 1. Curated Track Catalogue

File: `data/catalogue/tracks.json`

Each entry contains:

- Core metadata (`id`, `name`, `artist`, `album`, `uri`)
- Audio feature vector (valence, energy, danceability, acousticness, tempo,
  loudness, instrumentalness, speechiness, liveness)
- Popularity and release year for freshness weighting
- Semantic tags grouped into `genres`, `moods`, `activities`, `time_of_day`,
  and `regions`

The loader (`TrackCatalogue`) provides fast lookups by ID or by fuzzy matching
on name/artist and exposes an iterable collection for scoring.

### 2. Context Interpreter

Module: `services/recommendation/contextual_engine.py`

- Normalises loose user input (e.g. "late night", "gym", "rnb") into canonical
  categories.
- Maps context to target audio profiles using handcrafted heuristics (e.g.
  `workout` → high energy, `focus` → lower energy + higher instrumentalness).
- Supports tempo buckets, era presets (`current`, `2010s`, `classic`, etc.), and
  regional hints.

### 3. Audio Similarity Layer

The existing `AudioSimilarityEngine` is reused to compute weighted similarity
between the target profile and catalogue entries. If no seed audio data is
available the engine falls back to context-only profiles and uses a neutral
baseline to keep scores meaningful.

### 4. Scoring & Diversity

For each candidate track the engine computes the following components:

| Component   | Description                                                     | Default Weight |
|-------------|-----------------------------------------------------------------|----------------|
| similarity  | Audio feature distance to the inferred target profile           | 0.45           |
| context     | Alignment between catalogue tags and contextual hints           | 0.35           |
| popularity  | Normalised popularity (guard rails against overly obscure picks)| 0.15           |
| freshness   | Release year alignment with requested era / recency preference | 0.05           |
| penalty     | Diversity nudges (e.g. small penalty for repeating the seed artist) | applied directly |

After scoring the engine enforces diversity by limiting each artist to a maximum
of two appearances in the final list while preserving order. If all candidates
are filtered out (e.g. empty catalogue) the engine returns trending tracks
sorted by popularity.

### 5. Explanations

Every recommended track carries an embedded explanation:

```json
{
  "id": "catalog:track:midnight_drive",
  "name": "Midnight Drive",
  "artist": "Neon Skyline",
  "recommendation": {
    "score": 0.8125,
    "components": {
      "similarity": 0.86,
      "context": 0.79,
      "popularity": 0.82,
      "freshness": 0.58
    }
  }
}
```

This makes it straightforward to tune the catalogue, adjust weights, or explain
results to end-users.

## API Usage

```bash
POST /api/v1/playlist/recommendations
```

```json
{
  "tracks": [
    {"raw": "Song One - Artist A"},
    {"spotify_id": "1AbCxyz123"},
    {"name": "Song Two", "artist": "Artist B", "seed": true}
  ],
  "limit": 6,
  "min_popularity": 30,
  "context": {
    "energy": "high",
    "regions": ["uk"]
  }
}
```

- `tracks` *(required)*: raw playlist entries pasted by the user. Each entry may
  be a Spotify URL/URI/ID, `"Song - Artist"` string, or structured metadata.
- `seed` *(optional)*: Spotify URI to bias towards (defaults to the first track
  marked with `"seed": true`).
- `context` *(optional)*: additional overrides layered on top of the playlist
  profile (same keys as before: `moods`, `activities`, `time_of_day`, `energy`,
  `tempo`, `era`, `regions`, etc.).

The legacy `GET /api/v1/recommendations` endpoint now returns `400` with a
message directing callers to the playlist workflow.

## Extending the Catalogue

1. Add or edit entries in `data/catalogue/tracks.json`.
2. Maintain consistent audio feature scaling (`tempo` in BPM, `loudness` in dB).
3. Add descriptive tags to improve context matching; the engine performs
   case-insensitive matching and supports light fuzzy comparisons.
4. Reload the service — no additional preprocessing is required.

### Generating a real-song catalogue automatically

The repo now includes `scripts/build_catalogue.py`, which pulls live Spotify
metadata (tracks, audio features, artist genres) and writes a ready-to-use
catalogue file for the contextual engine.

Steps:

1. Ensure `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` are present in `.env`.
2. Run the script with playlists, queries, or explicit track IDs you want to
   seed:

   ```bash
   python scripts/build_catalogue.py \
     --playlist 37i9dQZF1DXcBWIGoYBM5M \
     --playlist 37i9dQZF1DWXRqgorJj26U \
     --query "focus lo-fi" --search-limit 40 \
     --min-popularity 45 \
     --overwrite
   ```

   - `--playlist` may be repeated to aggregate editorial or personal playlists.
   - `--query` uses Spotify search to grab top tracks for the provided phrases.
   - `--track` (optional) adds one-off track IDs.
   - Use `--output` to direct the generated JSON elsewhere instead of overwriting
     the default catalogue.

3. Restart the API (or reload Uvicorn) so `TrackCatalogue` reads the refreshed
   data on startup.

## Cold Start & Offline Behaviour

- No user profile storage is required; the engine works immediately after start-up
  as soon as a playlist is provided.
- If external APIs are unavailable the system still serves recommendations from
  the local catalogue after the playlist profile has been derived (cached audio
  features are required for the playlist tracks).
- When the personalised scoring cannot find suitable candidates the engine falls
  back to curated trending tracks, still excluding the playlist songs.

## Future Tweaks

- Integrate live charts or editorial playlists to auto-refresh catalogue data.
- Add optional machine-learned models to refine context → feature translation.
- Capture lightweight session feedback (without persistent profiles) to adapt
  weighting during a single listening session.
