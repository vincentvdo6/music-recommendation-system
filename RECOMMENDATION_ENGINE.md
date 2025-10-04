# Contextual Recommendation Engine

## Overview

The music recommendation system now prioritises contextual awareness over user
history. Instead of collaborative filtering, it combines a curated knowledge base
of expertly tagged tracks with on-the-fly interpretation of the listener's
intent. The engine remains accurate even when there is no listening history,
making it ideal for fresh users, episodic sessions, or embedded experiences
where personalised storage is not available.

Key pillars:

- **Curated catalogue** of tracks that include audio features, mood/activity
  tags, temporal preferences, and geographic relevance.
- **Context interpreter** that normalises free-form hints (`mood`, `activity`,
  `time_of_day`, `energy`, `tempo`, `era`, `region`) into canonical targets.
- **Audio similarity** via Spotify-style feature vectors when a seed track or
  features are available.
- **Transparent scoring** with per-track explanations so results can be audited
  and tuned without guesswork.

## Architecture

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
GET /api/v1/recommendations?seed=spotify:track:1AbCxyz&limit=8&mood=upbeat&activity=workout&time_of_day=evening
```

- `seed` *(optional)*: Spotify/Apple/catalog identifier. Used for audio
  similarity if features are available. Leave empty for pure context-based
  picks.
- `mood`, `activity`, `time_of_day`, `energy`, `tempo`, `era`, `genres`,
  `regions`: contextual hints. Singular or plural forms are supported.
- `user_id`: retained for compatibility but no longer affects scoring.

Response shape mirrors previous versions but `source` will now read
`"contextual"` when the curated engine is used.

## Extending the Catalogue

1. Add or edit entries in `data/catalogue/tracks.json`.
2. Maintain consistent audio feature scaling (`tempo` in BPM, `loudness` in dB).
3. Add descriptive tags to improve context matching; the engine performs
   case-insensitive matching and supports light fuzzy comparisons.
4. Reload the service — no additional preprocessing is required.

## Cold Start & Offline Behaviour

- No user profile storage is required; the engine works immediately after start-up.
- If external APIs are unavailable the system still serves recommendations from
  the local catalogue.
- When both seed and context are missing the engine returns the most popular
  curated tracks, ensuring deterministic behaviour for smoke tests and demos.

## Future Tweaks

- Integrate live charts or editorial playlists to auto-refresh catalogue data.
- Add optional machine-learned models to refine context → feature translation.
- Capture lightweight session feedback (without persistent profiles) to adapt
  weighting during a single listening session.

