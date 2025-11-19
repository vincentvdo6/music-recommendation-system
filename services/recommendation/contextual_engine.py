"""Context-aware recommendation engine that avoids user-history dependency."""

from __future__ import annotations

import logging
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from services.recommendation.audio_similarity import AudioSimilarityEngine
from services.recommendation.catalogue import TrackCatalogue

logger = logging.getLogger(__name__)


def _canonical(value: str) -> str:
    """Normalise text to a canonical, comparison-friendly form."""
    return value.replace("-", " ").replace("_", " ").casefold().strip() if value else ""


class ContextualRecommendationEngine:
    """Blend curated catalogue knowledge with audio similarity and situational context."""

    DEFAULT_WEIGHTS = {
        "audio": 0.45,
        "context": 0.35,
        "popularity": 0.15,
        "freshness": 0.05,
    }

    MOOD_FEATURE_TARGETS = {
        "upbeat": {"valence": 0.85, "energy": 0.72, "danceability": 0.78},
        "energetic": {"valence": 0.7, "energy": 0.9, "danceability": 0.82},
        "calm": {"valence": 0.45, "energy": 0.3, "acousticness": 0.6},
        "melancholic": {"valence": 0.2, "energy": 0.35, "acousticness": 0.55},
        "dreamy": {"valence": 0.55, "energy": 0.5, "danceability": 0.58},
        "romantic": {"valence": 0.7, "energy": 0.45, "danceability": 0.62},
        "intense": {"valence": 0.45, "energy": 0.95},
        "nostalgic": {"valence": 0.52, "energy": 0.55},
        "chill": {"valence": 0.5, "energy": 0.35, "danceability": 0.5},
        "hopeful": {"valence": 0.74, "energy": 0.55},
        "moody": {"valence": 0.32, "energy": 0.52},
        "sensual": {"valence": 0.58, "energy": 0.48},
    }

    ACTIVITY_FEATURE_TARGETS = {
        "focus": {"energy": 0.45, "danceability": 0.4, "instrumentalness": 0.25},
        "study": {"energy": 0.35, "danceability": 0.42, "instrumentalness": 0.3},
        "coding": {"energy": 0.55, "danceability": 0.52, "instrumentalness": 0.18},
        "workout": {"energy": 0.92, "danceability": 0.82},
        "running": {"energy": 0.9, "danceability": 0.74},
        "yoga": {"energy": 0.32, "acousticness": 0.62, "instrumentalness": 0.35},
        "sleep": {"energy": 0.2, "acousticness": 0.7, "instrumentalness": 0.4},
        "party": {"energy": 0.88, "danceability": 0.86, "speechiness": 0.12},
        "commute": {"energy": 0.6, "danceability": 0.6},
        "gaming": {"energy": 0.8, "danceability": 0.58},
        "relax": {"energy": 0.4, "acousticness": 0.5},
    }

    TIME_OF_DAY_FEATURE_TARGETS = {
        "morning": {"energy": 0.55, "valence": 0.68},
        "afternoon": {"energy": 0.65, "valence": 0.62},
        "evening": {"energy": 0.58, "valence": 0.6},
        "night": {"energy": 0.4, "valence": 0.45, "acousticness": 0.5},
        "late": {"energy": 0.35, "valence": 0.4, "acousticness": 0.55},
    }

    ENERGY_LEVEL_FEATURES = {
        "high": {"energy": 0.88},
        "medium": {"energy": 0.6},
        "low": {"energy": 0.35},
    }

    TEMPO_BPM_TARGETS = {
        "slow": 72.0,
        "medium": 104.0,
        "fast": 128.0,
        "very fast": 150.0,
    }

    MOOD_ALIASES = {
        "happy": "upbeat",
        "joyful": "upbeat",
        "sad": "melancholic",
        "relaxed": "calm",
        "chilled": "chill",
        "energetic": "energetic",
        "focused": "focus",
        "romance": "romantic",
        "sexy": "sensual",
        "dark": "moody",
        "motivated": "energetic",
    }

    MOOD_GENRE_ADJACENCIES = {
        "upbeat": ["pop", "dance", "synthpop"],
        "energetic": ["rock", "hip-hop", "electronic"],
        "calm": ["ambient", "acoustic", "modern classical"],
        "melancholic": ["indie", "folk", "acoustic", "lofi", "r&b", "neo-soul"],
        "dreamy": ["dream pop", "indietronica", "synthwave"],
        "romantic": ["r&b", "neo-soul", "pop"],
        "intense": ["rock", "industrial", "drum_and_bass"],
        "nostalgic": ["synthwave", "surf", "folk"],
        "chill": ["lofi", "chillhop", "ambient"],
        "hopeful": ["indie", "pop", "folk"],
        "moody": ["dark pop", "electronic", "industrial"],
        "sensual": ["r&b", "neo-soul", "electropop"],
    }

    ACTIVITY_ALIASES = {
        "studying": "study",
        "study": "study",
        "work": "focus",
        "working": "focus",
        "gym": "workout",
        "exercise": "workout",
        "run": "running",
        "jogging": "running",
        "sleeping": "sleep",
        "night drive": "night_drive",
        "night-drive": "night_drive",
        "nightdrive": "night_drive",
        "drive": "driving",
        "driving": "driving",
        "commuting": "commute",
        "chilling": "relax",
        "relaxing": "relax",
        "coding": "coding",
        "gaming": "gaming",
    }

    GENRE_ALIASES = {
        "hip hop": "hip-hop",
        "hiphop": "hip-hop",
        "rnb": "r&b",
        "randb": "r&b",
        "edm": "electronic",
        "dnb": "drum_and_bass",
    }

    TIME_OF_DAY_ALIASES = {
        "late night": "night",
        "midnight": "night",
        "midday": "afternoon",
        "dawn": "morning",
        "sunrise": "morning",
        "sunset": "evening",
    }

    ERA_PRESETS = {
        "current": (2020, 2024),
        "recent": (2016, 2024),
        "2010s": (2010, 2019),
        "2000s": (2000, 2009),
        "classic": (1970, 1999),
        "2015+": (2015, 2024),
    }

    def __init__(
        self,
        catalogue: TrackCatalogue,
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.catalogue = catalogue
        self.audio_similarity = AudioSimilarityEngine()
        self.weights = weights or self.DEFAULT_WEIGHTS

    # ------------------------------------------------------------------
    def build_user_profile(self, playlist_entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Derive a preference profile from a user-supplied playlist."""

        entries = [entry for entry in playlist_entries if entry]
        if not entries:
            raise ValueError("Cannot build user profile from an empty playlist")

        feature_totals: Dict[str, float] = {}
        feature_counts: Dict[str, int] = {}

        tag_counters = {
            "moods": Counter(),
            "activities": Counter(),
            "genres": Counter(),
            "time_of_day": Counter(),
            "regions": Counter(),
        }

        release_years: List[int] = []
        canonical_tracks: Set[Tuple[str, str]] = set()
        playlist_track_ids: Set[str] = set()

        for entry in entries:
            track_id = entry.get("id")
            if track_id:
                playlist_track_ids.add(str(track_id))

            name_key = (
                _canonical(entry.get("name", "")),
                _canonical(entry.get("artist", "")),
            )
            if any(name_key):
                canonical_tracks.add(name_key)

            features = entry.get("audio_features") or {}
            for key, value in features.items():
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                feature_totals[key] = feature_totals.get(key, 0.0) + numeric
                feature_counts[key] = feature_counts.get(key, 0) + 1

            tags = entry.get("tags") or {}
            for group, counter in tag_counters.items():
                values = tags.get(group)
                if not values:
                    continue
                if isinstance(values, str):
                    values = [values]
                for tag in values:
                    if not tag:
                        continue
                    counter[_canonical(tag)] += 1

            release_year = entry.get("release_year")
            if isinstance(release_year, int):
                release_years.append(release_year)

        audio_features = {
            key: feature_totals[key] / feature_counts[key]
            for key in feature_totals
            if feature_counts.get(key)
        }

        if not audio_features:
            raise ValueError("Playlist tracks are missing audio features; unable to build profile")

        def top_list(counter: Counter, limit: int) -> List[str]:
            return [item for item, _ in counter.most_common(limit)]

        average_year = sum(release_years) / len(release_years) if release_years else None
        inferred_era = self._infer_era_from_year(average_year) if average_year else None

        average_tempo = audio_features.get("tempo")
        tempo_bucket = self._tempo_bucket(average_tempo) if average_tempo else None

        average_energy = audio_features.get("energy")
        energy_level = self._energy_level_from_feature(average_energy) if average_energy is not None else None

        context = {
            "moods": top_list(tag_counters["moods"], 3),
            "activities": top_list(tag_counters["activities"], 3),
            "genres": top_list(tag_counters["genres"], 5),
            "regions": top_list(tag_counters["regions"], 3),
        }

        time_of_day = top_list(tag_counters["time_of_day"], 1)
        if time_of_day:
            context["time_of_day"] = time_of_day[0]
        if tempo_bucket:
            context["tempo"] = tempo_bucket
        if energy_level:
            context["energy_level"] = energy_level
        if inferred_era:
            context["era"] = inferred_era

        return {
            "audio_features": audio_features,
            "context": context,
            "canonical_tracks": canonical_tracks,
            "track_ids": playlist_track_ids,
        }

    # ------------------------------------------------------------------
    async def get_recommendations(
        self,
        *,
        seed_track_id: Optional[str] = None,
        seed_metadata: Optional[Dict[str, Any]] = None,
        seed_features: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return context-aware recommendations using a playlist-personalised profile."""

        if not user_profile:
            raise ValueError(
                "user_profile is required. Build it from a playlist before requesting recommendations."
            )

        limit = max(1, min(limit, 50))

        merged_context = self._merge_context(user_profile.get("context") or {}, context or {})
        normalised_context = self._normalise_context(merged_context)

        seed_entry = self._resolve_seed_entry(seed_track_id, seed_metadata)
        if not seed_features and seed_metadata and seed_metadata.get("audio_features"):
            seed_features = seed_metadata["audio_features"]

        if not seed_features and seed_entry:
            seed_features = seed_entry.get("audio_features")

        target_profile = self._build_target_profile(
            seed_features,
            merged_context,
            seed_entry,
            user_profile=user_profile,
        )

        candidates = self.catalogue.all_tracks()
        scored: List[Dict[str, Any]] = []

        playlist_track_ids: Set[str] = set(user_profile.get("track_ids", set()))
        canonical_exclusions: Set[Tuple[str, str]] = set(user_profile.get("canonical_tracks", set()))

        for candidate in candidates:
            candidate_id = candidate.get("id")
            if candidate_id and candidate_id in playlist_track_ids:
                continue

            if canonical_exclusions:
                name_key = (
                    _canonical(candidate.get("name", "")),
                    _canonical(candidate.get("artist", "")),
                )
                if name_key[0] and name_key in canonical_exclusions:
                    continue

            if seed_entry and candidate.get("id") == seed_entry.get("id"):
                continue

            components = self._score_candidate(
                candidate,
                target_profile=target_profile,
                context=normalised_context,
                seed_entry=seed_entry,
            )

            final_score = (
                self.weights["audio"] * components["similarity"]
                + self.weights["context"] * components["context"]
                + self.weights["popularity"] * components["popularity"]
                + self.weights["freshness"] * components["freshness"]
                + components["penalty"]
            )

            if final_score <= 0:
                continue

            scored.append(
                {
                    "track": candidate,
                    "score": final_score,
                    "components": components,
                }
            )

        if not scored:
            logger.warning("Contextual engine could not score any candidates; returning trending picks")
            return self._fallback_trending(limit)

        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = self._apply_diversity(scored, limit)
        return [self._augment_track(item) for item in selected]

    # ------------------------------------------------------------------
    def _resolve_seed_entry(
        self,
        seed_track_id: Optional[str],
        seed_metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Locate a catalogue entry that best represents the provided seed."""

        entry: Optional[Dict[str, Any]] = None

        if seed_track_id:
            entry = self.catalogue.get_by_id(seed_track_id)

        if not entry and seed_metadata:
            name = seed_metadata.get("name")
            artist = seed_metadata.get("artist")
            entry = self.catalogue.find_best_match(name, artist)

        return entry

    def _build_target_profile(
        self,
        seed_features: Optional[Dict[str, Any]],
        context: Dict[str, Any],
        seed_entry: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """Blend seed features and context-derived hints into a target profile."""

        components: List[Tuple[Dict[str, Any], float]] = []

        if user_profile and user_profile.get("audio_features"):
            components.append((user_profile["audio_features"], 1.2))

        if seed_features:
            components.append((seed_features, 1.0))

        context_features = self._features_from_context(context)
        if context_features:
            components.append((context_features, 0.8))

        if seed_entry and seed_entry.get("audio_features") and not seed_features:
            components.append((seed_entry["audio_features"], 0.6))

        if not components:
            return None

        totals: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        for feature_dict, weight in components:
            for key, value in feature_dict.items():
                if value is None:
                    continue
                totals[key] = totals.get(key, 0.0) + float(value) * weight
                weights[key] = weights.get(key, 0.0) + weight

        profile = {key: totals[key] / weights[key] for key in totals if weights[key] > 0}
        return profile or None

    def _features_from_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Translate context intent into approximate audio feature targets."""

        features: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        def merge(feature_map: Dict[str, float], weight: float) -> None:
            for key, value in feature_map.items():
                features[key] = features.get(key, 0.0) + value * weight
                weights[key] = weights.get(key, 0.0) + weight

        moods: Iterable[str] = context.get("moods", [])
        for mood in moods:
            target = self.MOOD_FEATURE_TARGETS.get(mood)
            if target:
                merge(target, 1.0)

        activities: Iterable[str] = context.get("activities", [])
        for activity in activities:
            target = self.ACTIVITY_FEATURE_TARGETS.get(activity)
            if target:
                merge(target, 0.9)

        time_of_day = context.get("time_of_day")
        if time_of_day:
            target = self.TIME_OF_DAY_FEATURE_TARGETS.get(time_of_day)
            if target:
                merge(target, 0.7)

        energy_level = context.get("energy_level")
        if energy_level:
            target = self.ENERGY_LEVEL_FEATURES.get(energy_level)
            if target:
                merge(target, 0.8)

        tempo_pref = context.get("tempo")
        if tempo_pref:
            bpm = self.TEMPO_BPM_TARGETS.get(tempo_pref)
            if bpm:
                merge({"tempo": bpm}, 0.6)

        result = {key: features[key] / weights[key] for key in features if weights[key] > 0}
        return result

    def _expand_adjacent_genres(self, moods: Iterable[str], existing_genres: Iterable[str]) -> List[str]:
        """Derive genre hints from the requested moods without overriding explicit genre picks."""

        expanded: List[str] = []
        seen: Set[str] = set()
        base = {_canonical(genre) for genre in existing_genres if genre}

        for mood in moods or []:
            for genre in self.MOOD_GENRE_ADJACENCIES.get(mood, []):
                canonical = self.GENRE_ALIASES.get(_canonical(genre), _canonical(genre))
                if not canonical:
                    continue
                canonical_key = _canonical(canonical)
                if canonical_key in base or canonical in seen:
                    continue
                expanded.append(canonical)
                seen.add(canonical)

        return expanded

    def _normalise_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise raw context dictionary into canonical categories."""

        normalised: Dict[str, Any] = {
            "moods": [],
            "activities": [],
            "genres": [],
            "time_of_day": None,
            "energy_level": None,
            "tempo": None,
            "era": None,
            "regions": [],
            "adjacent_genres": [],
        }

        moods = context.get("moods") or context.get("mood")
        if isinstance(moods, str):
            moods = [moods]
        if moods:
            for mood in moods:
                canonical = self.MOOD_ALIASES.get(_canonical(mood), _canonical(mood))
                if canonical in self.MOOD_FEATURE_TARGETS:
                    normalised["moods"].append(canonical)

        activities = context.get("activities") or context.get("activity")
        if isinstance(activities, str):
            activities = [activities]
        if activities:
            for activity in activities:
                canonical = self.ACTIVITY_ALIASES.get(_canonical(activity), _canonical(activity))
                normalised["activities"].append(canonical)

        genres = context.get("genres") or context.get("genre")
        if isinstance(genres, str):
            genres = [genres]
        if genres:
            for genre in genres:
                canonical = self.GENRE_ALIASES.get(_canonical(genre), _canonical(genre))
                normalised["genres"].append(canonical)

        adjacent_genres = self._expand_adjacent_genres(normalised["moods"], normalised["genres"])
        if adjacent_genres:
            normalised["adjacent_genres"] = adjacent_genres

        tod = context.get("time_of_day") or context.get("time")
        if isinstance(tod, str):
            canonical = self.TIME_OF_DAY_ALIASES.get(_canonical(tod), _canonical(tod))
            if canonical in self.TIME_OF_DAY_FEATURE_TARGETS:
                normalised["time_of_day"] = canonical

        energy = context.get("energy_level") or context.get("energy")
        if isinstance(energy, str):
            canonical = _canonical(energy)
            if canonical in self.ENERGY_LEVEL_FEATURES:
                normalised["energy_level"] = canonical

        tempo = context.get("tempo")
        if isinstance(tempo, (int, float)):
            # Map explicit BPM to bucket
            bpm = float(tempo)
            if bpm <= 90:
                normalised["tempo"] = "slow"
            elif bpm <= 115:
                normalised["tempo"] = "medium"
            elif bpm <= 135:
                normalised["tempo"] = "fast"
            else:
                normalised["tempo"] = "very fast"
        elif isinstance(tempo, str):
            canonical = _canonical(tempo)
            if canonical in self.TEMPO_BPM_TARGETS:
                normalised["tempo"] = canonical

        era = context.get("era") or context.get("decade")
        if isinstance(era, str):
            canonical = era.strip().lower()
            if canonical in self.ERA_PRESETS:
                normalised["era"] = canonical

        regions = context.get("regions") or context.get("region")
        if isinstance(regions, str):
            regions = [regions]
        if regions:
            normalised["regions"] = [_canonical(region) for region in regions]

        return normalised

    def _merge_context(
        self,
        user_context: Dict[str, Any],
        request_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge context inferred from the playlist with explicit request overrides."""

        merged: Dict[str, Any] = {}

        def extend_list(key: str, value: Any) -> None:
            existing = merged.get(key) or []
            if isinstance(existing, str):
                existing = [existing]
            values: List[str]
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, Iterable):
                values = list(value)
            else:
                return

            combined = existing[:]
            for item in values:
                if item is None:
                    continue
                if item not in combined:
                    combined.append(item)
            merged[key] = combined

        def set_if_empty(key: str, value: Any) -> None:
            if value is None:
                return
            if key not in merged or merged[key] in (None, "", [], ()):
                merged[key] = value

        for context_source in (user_context or {}, request_context or {}):
            for key, value in (context_source or {}).items():
                if key in {"moods", "mood", "activities", "activity", "genres", "genre", "regions", "region"}:
                    canonical_key = key
                    if key == "mood":
                        canonical_key = "moods"
                    elif key == "activity":
                        canonical_key = "activities"
                    elif key == "genre":
                        canonical_key = "genres"
                    elif key == "region":
                        canonical_key = "regions"
                    extend_list(canonical_key, value)
                elif key in {"time_of_day", "time"}:
                    set_if_empty("time_of_day", value)
                elif key in {"energy_level", "energy"}:
                    set_if_empty("energy_level", value)
                elif key == "tempo":
                    set_if_empty("tempo", value)
                elif key in {"era", "decade"}:
                    set_if_empty("era", value)
                else:
                    merged[key] = value

        return merged

    def _infer_era_from_year(self, year: float) -> Optional[str]:
        """Map an average release year to one of the configured era buckets."""
        if year is None:
            return None

        year_int = int(round(year))
        ranking = [
            ("current", self.ERA_PRESETS["current"]),
            ("recent", self.ERA_PRESETS["recent"]),
            ("2015+", self.ERA_PRESETS["2015+"]),
            ("2010s", self.ERA_PRESETS["2010s"]),
            ("2000s", self.ERA_PRESETS["2000s"]),
            ("classic", self.ERA_PRESETS["classic"]),
        ]
        for label, (start, end) in ranking:
            if start <= year_int <= end:
                return label

        if year_int >= 2020:
            return "current"
        if year_int >= 2016:
            return "recent"
        if year_int >= 2010:
            return "2010s"
        if year_int >= 2000:
            return "2000s"
        return "classic"

    @staticmethod
    def _tempo_bucket(tempo: float) -> str:
        """Derive the closest tempo bucket for a numeric tempo value."""
        if tempo <= 90:
            return "slow"
        if tempo <= 115:
            return "medium"
        if tempo <= 135:
            return "fast"
        return "very fast"

    @staticmethod
    def _energy_level_from_feature(energy: float) -> str:
        """Convert raw energy into the discrete buckets used by the engine."""
        if energy >= 0.7:
            return "high"
        if energy >= 0.45:
            return "medium"
        return "low"

    def _score_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        target_profile: Optional[Dict[str, float]],
        context: Dict[str, Any],
        seed_entry: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate the score components for a candidate track."""

        similarity = self._similarity_score(candidate, target_profile)
        context_score = self._context_alignment(candidate, context, seed_entry)
        popularity = self._popularity_score(candidate)
        freshness = self._freshness_score(candidate, context)
        penalty = self._artist_penalty(candidate, seed_entry)

        return {
            "similarity": similarity,
            "context": context_score,
            "popularity": popularity,
            "freshness": freshness,
            "penalty": penalty,
        }

    def _similarity_score(
        self,
        candidate: Dict[str, Any],
        target_profile: Optional[Dict[str, float]],
    ) -> float:
        if not target_profile:
            return 0.55  # neutral baseline when we can't compare features

        candidate_features = candidate.get("audio_features") or {}
        if not candidate_features:
            return 0.5

        return float(self.audio_similarity.calculate_similarity(target_profile, candidate_features))

    def _context_alignment(
        self,
        candidate: Dict[str, Any],
        context: Dict[str, Any],
        seed_entry: Optional[Dict[str, Any]],
    ) -> float:
        tags = candidate.get("tags", {})
        score = 0.0
        weight = 0.0

        moods = context.get("moods")
        if moods:
            weight += 1.0
            score += self._tag_score(tags.get("moods", []), moods, fuzz=True)

        activities = context.get("activities")
        if activities:
            weight += 0.9
            score += self._tag_score(tags.get("activities", []), activities, fuzz=True)

        genres = context.get("genres")
        if genres:
            weight += 0.7
            score += self._tag_score(tags.get("genres", []), genres)

        adjacent_genres = context.get("adjacent_genres")
        if adjacent_genres:
            weight += 0.4
            score += self._tag_score(tags.get("genres", []), adjacent_genres, fuzz=True)

        regions = context.get("regions")
        if regions:
            weight += 0.3
            score += self._tag_score(tags.get("regions", []), regions)

        time_of_day = context.get("time_of_day")
        if time_of_day:
            weight += 0.6
            score += self._tag_score(tags.get("time_of_day", []), [time_of_day])

        if seed_entry and not weight:
            # If no explicit context, softly encourage stylistic continuity
            seed_genres = seed_entry.get("tags", {}).get("genres", [])
            if seed_genres:
                weight += 0.5
                score += self._tag_score(tags.get("genres", []), seed_genres)

        if weight == 0:
            return 0.5

        return min(1.0, max(0.0, score / weight))

    def _tag_score(self, candidate_tags: Iterable[str], desired: Iterable[str], *, fuzz: bool = False) -> float:
        candidate_norm = {_canonical(tag) for tag in candidate_tags if tag}
        if not desired:
            return 0.5

        hits = 0.0
        for tag in desired:
            canon = _canonical(tag)
            if canon in candidate_norm:
                hits += 1.0
                continue
            if fuzz:
                for candidate_tag in candidate_norm:
                    if canon in candidate_tag or candidate_tag in canon:
                        hits += 0.6
                        break
        return hits / max(len(list(desired)), 1)

    def _popularity_score(self, candidate: Dict[str, Any]) -> float:
        popularity = candidate.get("popularity", 50)
        try:
            numeric = float(popularity)
        except (TypeError, ValueError):
            numeric = 50.0
        return max(0.0, min(1.0, numeric / 100.0))

    def _freshness_score(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
        release_year = candidate.get("release_year")
        if not release_year:
            return 0.5

        era = context.get("era")
        if not era:
            # Light preference for more recent titles by default
            return max(0.3, min(1.0, (release_year - 2000) / 24.0))

        target_range = self.ERA_PRESETS.get(era)
        if not target_range:
            return 0.5

        start, end = target_range
        if start <= release_year <= end:
            return 0.9

        # Penalise based on distance from target era
        if release_year < start:
            gap = start - release_year
        else:
            gap = release_year - end

        return max(0.2, 0.9 - (gap / 15.0))

    def _artist_penalty(self, candidate: Dict[str, Any], seed_entry: Optional[Dict[str, Any]]) -> float:
        if not seed_entry:
            return 0.0

        seed_artist = _canonical(seed_entry.get("artist", ""))
        cand_artist = _canonical(candidate.get("artist", ""))

        if seed_artist and cand_artist and seed_artist == cand_artist:
            return -0.15  # Encourage variety beyond the seed artist

        return 0.0

    def _apply_diversity(self, scored: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        artist_counts: Dict[str, int] = {}

        for item in scored:
            track = item["track"]
            artist = _canonical(track.get("artist", ""))

            if artist_counts.get(artist, 0) >= 2:
                continue

            selected.append(item)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

            if len(selected) >= limit:
                break

        return selected

    def _augment_track(self, item: Dict[str, Any]) -> Dict[str, Any]:
        track = deepcopy(item["track"])
        track.setdefault("album", "Unknown")
        track.setdefault("duration_ms", 0)
        track.setdefault("popularity", 0)
        track.setdefault("preview_url", "")
        track.setdefault("image_url", "")
        track.setdefault("external_urls", {})

        track_id = track.get("id")
        if track_id and not track.get("uri"):
            track["uri"] = track_id

        # Derive release_date from release_year where possible
        release_year = track.get("release_year")
        if not track.get("release_date"):
            if isinstance(release_year, int):
                track["release_date"] = f"{release_year}-01-01"
            else:
                track["release_date"] = ""

        track["recommendation"] = {
            "score": round(item["score"], 4),
            "components": {
                key: round(value, 4)
                for key, value in item["components"].items()
                if key != "penalty"
            },
        }
        return track

    def _fallback_trending(self, limit: int) -> List[Dict[str, Any]]:
        trending = sorted(
            (track for track in self.catalogue.iterate()),
            key=lambda entry: entry.get("popularity", 0),
            reverse=True,
        )
        return [self._augment_track({"track": track, "score": 0.6, "components": {"similarity": 0.5, "context": 0.5, "popularity": self._popularity_score(track), "freshness": self._freshness_score(track, {}), "penalty": 0.0}}) for track in trending[:limit]]
