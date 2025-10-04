"""Context-aware recommendation engine that avoids user-history dependency."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    async def get_recommendations(
        self,
        *,
        seed_track_id: Optional[str] = None,
        seed_metadata: Optional[Dict[str, Any]] = None,
        seed_features: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return context-aware recommendations without relying on user history."""

        context = context or {}
        limit = max(1, min(limit, 50))

        seed_entry = self._resolve_seed_entry(seed_track_id, seed_metadata)
        if not seed_features and seed_entry:
            seed_features = seed_entry.get("audio_features")

        target_profile = self._build_target_profile(seed_features, context, seed_entry)
        normalised_context = self._normalise_context(context)

        candidates = self.catalogue.all_tracks()
        scored: List[Dict[str, Any]] = []

        for candidate in candidates:
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
    ) -> Optional[Dict[str, float]]:
        """Blend seed features and context-derived hints into a target profile."""

        components: List[Tuple[Dict[str, Any], float]] = []

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
