"""Utilities for building catalogue entries from Spotify track metadata."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def slugify(tag: str) -> str:
    """Convert free-form tags into the snake-style tokens used in the catalogue."""
    return "".join(ch if ch.isalnum() else "_" for ch in (tag or "").lower()).strip("_")


def unique(values: Iterable[str]) -> List[str]:
    """Return values preserving order while removing duplicates/empties."""
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_release_year(release_date: Optional[str]) -> Optional[int]:
    """Extract the release year from a Spotify release_date string."""
    if not release_date:
        return None
    year = release_date.split("-")[0]
    if len(year) == 4 and year.isdigit():
        return int(year)
    return None


def mood_tags(features: Dict[str, Any]) -> List[str]:
    energy = float(features.get("energy", 0.0) or 0.0)
    valence = float(features.get("valence", 0.0) or 0.0)
    acousticness = float(features.get("acousticness", 0.0) or 0.0)
    danceability = float(features.get("danceability", 0.0) or 0.0)

    moods: List[str] = []
    if energy >= 0.82 and valence >= 0.55:
        moods.append("energetic")
    if valence >= 0.68:
        moods.append("upbeat")
    if 0.45 <= valence <= 0.65 and acousticness >= 0.5:
        moods.append("dreamy")
    if valence <= 0.32:
        moods.append("melancholic")
    if energy <= 0.38:
        moods.append("calm")
    if 0.4 <= energy <= 0.65 and danceability >= 0.55:
        moods.append("chill")
    if 0.55 <= valence <= 0.75 and 0.4 <= energy <= 0.6:
        moods.append("hopeful")

    return unique(moods)


def activity_tags(features: Dict[str, Any]) -> List[str]:
    energy = float(features.get("energy", 0.0) or 0.0)
    danceability = float(features.get("danceability", 0.0) or 0.0)
    instrumentalness = float(features.get("instrumentalness", 0.0) or 0.0)
    acousticness = float(features.get("acousticness", 0.0) or 0.0)
    tempo = float(features.get("tempo", 0.0) or 0.0)

    activities: List[str] = []
    if energy >= 0.8 and danceability >= 0.65:
        activities.extend(["workout", "party"])
    if energy >= 0.7 and danceability >= 0.6 and tempo >= 120:
        activities.append("running")
    if instrumentalness >= 0.35 and energy <= 0.55:
        activities.extend(["focus", "study"])
    if acousticness >= 0.6 and energy <= 0.45:
        activities.extend(["relax", "sleep"])
    if 0.45 <= energy <= 0.65 and danceability >= 0.5:
        activities.append("coding")
    if 0.5 <= energy <= 0.7:
        activities.append("commute")

    return unique(activities)


def time_of_day_tags(features: Dict[str, Any]) -> List[str]:
    energy = float(features.get("energy", 0.0) or 0.0)
    acousticness = float(features.get("acousticness", 0.0) or 0.0)

    tags: List[str] = []
    if energy >= 0.75:
        tags.append("afternoon")
    if 0.5 <= energy < 0.75:
        tags.append("evening")
    if energy < 0.5:
        tags.append("night")
    if acousticness >= 0.55:
        tags.append("morning")

    return unique(tags)


def genres_for_track(track: Dict[str, Any], artist_map: Dict[str, Dict[str, Any]]) -> List[str]:
    artist_ids = track.get("metadata", {}).get("spotify_artist_ids") or []
    genres: List[str] = []
    for artist_id in artist_ids:
        artist = artist_map.get(artist_id)
        if not artist:
            continue
        genres.extend(artist.get("genres", []) or [])
    return genres


def normalise_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Restrict features to the numeric subset used by the contextual engine."""
    keys = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ]
    cleaned = {key: features.get(key) for key in keys if key in features}
    for extra in ("key", "mode", "time_signature"):
        if extra in features:
            cleaned[extra] = features[extra]
    return cleaned


def build_entry(
    track: Dict[str, Any],
    features: Dict[str, Any],
    genres: Sequence[str],
) -> Dict[str, Any]:
    """Convert Spotify track metadata into a catalogue entry."""

    tags = {
        "genres": unique(slugify(tag) for tag in genres)[:5],
        "moods": mood_tags(features),
        "activities": activity_tags(features),
        "time_of_day": time_of_day_tags(features),
        "regions": ["global"],
    }

    entry = {
        "id": track["id"],
        "uri": track.get("uri") or f"spotify:track:{track['id']}",
        "provider": "catalog",
        "name": track.get("name", ""),
        "artist": track.get("artist", ""),
        "album": track.get("album", ""),
        "duration_ms": track.get("duration_ms", 0) or 0,
        "popularity": track.get("popularity", 0) or 0,
        "release_year": extract_release_year(track.get("release_date")),
        "preview_url": track.get("preview_url") or "",
        "image_url": track.get("image_url") or "",
        "external_urls": track.get("external_urls", {}),
        "audio_features": features,
        "tags": tags,
    }

    metadata = track.get("metadata") or {}
    if metadata:
        entry["metadata"] = metadata

    return entry


def build_entries(
    tracks_map: Dict[str, Dict[str, Any]],
    features_map: Dict[str, Dict[str, Any]],
    artist_map: Dict[str, Dict[str, Any]],
    *,
    min_popularity: int = 0,
) -> List[Dict[str, Any]]:
    """Build catalogue-ready entries for the supplied tracks."""

    entries: List[Dict[str, Any]] = []
    for track_id, track in tracks_map.items():
        features_raw = features_map.get(track_id)
        if not features_raw:
            continue
        if track.get("popularity", 0) < min_popularity:
            continue

        features = normalise_features(features_raw)
        entry = build_entry(track, features, genres_for_track(track, artist_map))
        entries.append(entry)

    return entries
