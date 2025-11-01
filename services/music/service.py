"""High level orchestration for music data providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.apple.client import AppleMusicClient
from services.spotify.client import SpotifyClient
from services.recommendation.catalogue import TrackCatalogue
from services.recommendation.playlist_builder import build_entries
from api.engine_factory import get_engine

logger = logging.getLogger(__name__)


class MusicService:
    """Coordinates Spotify and Apple providers to keep media complete."""

    def __init__(
        self,
        *,
        spotify: Optional[SpotifyClient] = None,
        apple: Optional[AppleMusicClient] = None,
    ) -> None:
        self.spotify = spotify
        self.apple = apple or AppleMusicClient()
        self.catalogue: Optional[TrackCatalogue] = None
        self.context_engine = None  # Will be HybridRecommendationEngine or ContextualRecommendationEngine

        try:
            # Use factory to load hybrid engine with ML models (if available)
            self.context_engine = get_engine()
            self.catalogue = self.context_engine.catalogue

            # Log which engine was loaded
            engine_type = type(self.context_engine).__name__
            logger.info(
                "✓ %s initialized with %d curated tracks",
                engine_type,
                self.catalogue.size() if self.catalogue else 0,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error(
                "✗ Failed to initialise recommendation engine (%s) — falling back to provider heuristics",
                exc,
            )

    async def start(self) -> None:
        tasks = []
        if self.spotify:
            tasks.append(self.spotify.start())
        if self.apple:
            tasks.append(self.apple.start())
        if tasks:
            await asyncio.gather(*tasks)

    async def close(self) -> None:
        tasks = []
        if self.spotify:
            tasks.append(self.spotify.close())
        if self.apple:
            tasks.append(self.apple.close())
        if tasks:
            await asyncio.gather(*tasks)

    async def search_tracks(
        self,
        query: str,
        *,
        limit: int = 10,
        include_features: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """Return tracks, optional feature map, and source identifier."""
        tracks: List[Dict[str, Any]] = []
        features: Dict[str, Any] = {}
        source = "apple"

        if self.spotify and not self.spotify.demo_mode:
            try:
                tracks = await self.spotify.search_tracks(query, limit=limit)
                source = "spotify"
                if include_features and tracks:
                    ids = [track["id"] for track in tracks]
                    features = await self.spotify.get_tracks_features_bulk(ids)
                tracks = await self._ensure_media_assets(tracks)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Spotify search failed (%s), falling back to Apple", exc)
                tracks = []

        if not tracks:
            tracks = await self.apple.search_tracks(query, limit=limit)
            source = "apple"

        return tracks, features, source

    async def get_recommendations(
        self,
        seed: str,
        *,
        limit: int = 5,
        track_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Get recommendation list plus the provider source."""
        raise RuntimeError(
            "Playlist personalization required. Submit a playlist via the playlist recommendations endpoint to tailor results."
        )

    async def recommend_from_playlist(
        self,
        tracks_payload: List[Dict[str, Any]],
        *,
        seed: Optional[str] = None,
        limit: int = 5,
        context: Optional[Dict[str, Any]] = None,
        min_popularity: int = 0,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Generate recommendations from a user-supplied playlist."""
        import sys
        print("=" * 80, flush=True)
        print(f"[DEBUG] recommend_from_playlist CALLED with {len(tracks_payload)} tracks, limit={limit}", flush=True)
        print("=" * 80, flush=True)
        sys.stdout.flush()

        if not tracks_payload:
            return [], "playlist", {"playlist_size": 0}

        if not self.spotify or self.spotify.demo_mode:
            raise RuntimeError("Spotify credentials required for playlist-based recommendations")

        normalised_inputs = self._normalise_playlist_inputs(tracks_payload)

        spotify_ids: List[str] = []
        search_descriptors: List[Dict[str, Any]] = []

        for item in normalised_inputs:
            spotify_id = item.get("spotify_id")
            if spotify_id:
                if spotify_id not in spotify_ids:
                    spotify_ids.append(spotify_id)
                continue

            if item.get("name"):
                search_descriptors.append(item)

        tracks_map: Dict[str, Dict[str, Any]] = {}
        if spotify_ids:
            fetched = await self.spotify.get_tracks_bulk(spotify_ids)
            tracks_map.update(fetched)

        search_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        for descriptor in search_descriptors:
            name = descriptor.get("name", "").strip()
            artist = descriptor.get("artist")
            cache_key = (name.casefold(), (artist or "").casefold())

            track: Optional[Dict[str, Any]] = search_cache.get(cache_key)
            if track is None:
                query = f"{name} {artist}".strip()
                results = await self.spotify.search_tracks(query, limit=5)
                track = None
                if artist:
                    artist_norm = artist.casefold()
                    for candidate in results:
                        if candidate.get("artist", "").casefold() == artist_norm:
                            track = candidate
                            break
                if not track and results:
                    track = results[0]
                search_cache[cache_key] = track

            if track:
                descriptor["spotify_id"] = track["id"]
                if track["id"] not in tracks_map:
                    tracks_map[track["id"]] = track

        if not tracks_map:
            raise RuntimeError("Could not resolve any tracks from supplied playlist data")

        track_ids = list(tracks_map.keys())
        print(f"[DEBUG] Fetching features for {len(track_ids)} tracks: {track_ids[:3]}...", flush=True)
        features_map = await self.spotify.get_tracks_features_bulk(track_ids)
        print(f"[DEBUG] Received features for {len(features_map)} tracks", flush=True)

        artist_ids: List[str] = []
        for track in tracks_map.values():
            artist_ids.extend(track.get("metadata", {}).get("spotify_artist_ids", []))

        if not tracks_map:
            # Use normalised inputs as a fallback source of metadata
            for item in normalised_inputs:
                spotify_id = item.get("spotify_id")
                if not spotify_id:
                    continue
                tracks_map.setdefault(
                    spotify_id,
                    {
                        "id": spotify_id,
                        "name": item.get("name", ""),
                        "artist": item.get("artist", ""),
                        "album": item.get("album", ""),
                        "duration_ms": item.get("duration_ms") or 0,
                        "popularity": item.get("popularity") or 50,
                        "preview_url": item.get("preview_url"),
                        "image_url": item.get("image_url"),
                        "external_urls": item.get("external_urls", {}),
                        "metadata": item.get("metadata", {}),
                    },
                )

        artist_map = await self.spotify.get_artists(artist_ids)

        print(f"[DEBUG] tracks_map has {len(tracks_map)} tracks", flush=True)
        print(f"[DEBUG] features_map has {len(features_map)} features", flush=True)
        print(f"[DEBUG] artist_map has {len(artist_map)} artists", flush=True)

        entries = build_entries(
            tracks_map,
            features_map,
            artist_map,
            min_popularity=min_popularity,
        )

        print(f"[DEBUG] Built {len(entries)} entries from tracks", flush=True)

        if not entries:
            # Audio features failed (likely Spotify 403). Use item2vec-only path (no audio features needed)
            print(f"[DEBUG] No entries built (audio features failed). Using item2vec-only recommendation path...", flush=True)

            if hasattr(self.context_engine, 'recommend') and callable(getattr(self.context_engine, 'recommend')):
                # Use HybridRecommendationEngine.recommend() which only needs track IDs
                playlist_track_ids = list(tracks_map.keys())
                print(f"[DEBUG] Calling engine.recommend() with {len(playlist_track_ids)} track IDs", flush=True)

                ml_recommendations = self.context_engine.recommend(
                    playlist_tracks=playlist_track_ids,
                    limit=limit * 2  # Get 2x for diversity deduplication
                )

                print(f"[DEBUG] Got {len(ml_recommendations)} ML recommendations", flush=True)

                # If no ML recommendations (cold-start), use popularity-based fallback
                if not ml_recommendations:
                    print(f"[DEBUG] No ML recommendations (cold-start). Using catalogue popularity fallback...", flush=True)

                    # Get all tracks from the item2vec model
                    if hasattr(self.context_engine, 'embedding_service') and self.context_engine.embedding_service:
                        all_model_track_ids = list(self.context_engine.embedding_service.item2vec.wv.index_to_key)
                        print(f"[DEBUG] Model has {len(all_model_track_ids)} tracks. Fetching top popular tracks...", flush=True)

                        # Fetch metadata for top N popular tracks from the model
                        sample_size = min(200, len(all_model_track_ids))
                        import random
                        sample_ids = random.sample(all_model_track_ids, sample_size)

                        model_tracks = await self.spotify.get_tracks_bulk(sample_ids)

                        # Sort by popularity and take top
                        popular_tracks = sorted(
                            model_tracks.values(),
                            key=lambda t: t.get('popularity', 0),
                            reverse=True
                        )[:limit * 2]

                        ml_recommendations = [
                            {
                                **track,
                                "rank_score": track.get('popularity', 0) / 100.0  # Normalize to 0-1
                            }
                            for track in popular_tracks
                        ]

                        print(f"[DEBUG] Using {len(ml_recommendations)} popular tracks as fallback", flush=True)

                if ml_recommendations:
                    # Enrich with Spotify metadata
                    enriched = await self._enrich_with_spotify_metadata(ml_recommendations)
                    enriched = await self._ensure_media_assets(enriched)
                    diverse = self._deduplicate_by_artist(enriched, max_per_artist=2, limit=limit)

                    playlist_info = self._build_playlist_info(
                        total_tracks=len(tracks_map),
                        resolved_tracks=len(tracks_map),
                        profile_context=None,
                    )

                    print(f"[DEBUG] Returning {len(diverse)} diverse ML recommendations", flush=True)
                    return diverse, "ml-playlist", playlist_info

            # Final fallback to Apple
            fallback_recommendations = await self._apple_playlist_fallback(
                normalised_inputs,
                limit=limit,
            )

            if fallback_recommendations:
                fallback_recommendations = await self._ensure_media_assets(fallback_recommendations)
                playlist_info = self._build_playlist_info(
                    total_tracks=len(normalised_inputs),
                    resolved_tracks=len(tracks_map) or len(normalised_inputs),
                    profile_context=None,
                )
                return fallback_recommendations, "apple-playlist", playlist_info

            raise RuntimeError(
                "Playlist tracks were filtered out (missing features or below popularity threshold)"
            )

        resolved_count = len(tracks_map) or len(entries)
        user_profile = self.context_engine.build_user_profile(entries) if self.context_engine else None
        profile_context = user_profile.get("context", {}) if user_profile else None
        playlist_info = self._build_playlist_info(
            total_tracks=len(entries),
            resolved_tracks=resolved_count,
            profile_context=profile_context,
        )

        prefer_apple = (self.spotify and self.spotify.demo_mode) or not self.context_engine
        print(f"[DEBUG] prefer_apple={prefer_apple}, demo_mode={self.spotify.demo_mode if self.spotify else None}, has_engine={self.context_engine is not None}", flush=True)
        if prefer_apple:
            fallback_recommendations = await self._apple_playlist_fallback(normalised_inputs, limit=limit)
            if fallback_recommendations:
                fallback_recommendations = await self._ensure_media_assets(fallback_recommendations)
                return fallback_recommendations, "apple-playlist", playlist_info
            if not self.context_engine:
                raise RuntimeError("Curated engine unavailable; cannot personalise recommendations")

        if not user_profile:
            user_profile = self.context_engine.build_user_profile(entries)
            profile_context = user_profile.get("context", {})
            playlist_info = self._build_playlist_info(
                total_tracks=len(entries),
                resolved_tracks=resolved_count,
                profile_context=profile_context,
            )

        seed_id = self._extract_spotify_id(seed) if seed else None
        if not seed_id:
            for item in normalised_inputs:
                if item.get("seed") and item.get("spotify_id"):
                    seed_id = item["spotify_id"]
                    break

        seed_metadata = tracks_map.get(seed_id) if seed_id else None
        if not seed_metadata and tracks_map:
            seed_metadata = next(iter(tracks_map.values()))
            seed_id = seed_metadata.get("id")

        seed_features = features_map.get(seed_id) if seed_id else None

        print(f"[DEBUG] Calling context_engine.get_recommendations with seed_id={seed_id}, limit={limit * 2}", flush=True)

        # Get 2x recommendations to account for artist deduplication after enrichment
        recommendations = await self.context_engine.get_recommendations(
            seed_track_id=seed_id,
            seed_metadata=seed_metadata,
            seed_features=seed_features,
            context=context,
            user_profile=user_profile,
            limit=limit * 2,  # Request extra
        )

        # Fetch real Spotify metadata for ML-recommended tracks
        print(f"[DEBUG] Before enrichment: {len(recommendations)} recommendations", flush=True)
        recommendations = await self._enrich_with_spotify_metadata(recommendations)
        print(f"[DEBUG] After enrichment: {len(recommendations)} recommendations", flush=True)

        # Re-enrich recommendations with Spotify/Apple assets if available
        recommendations = await self._ensure_media_assets(recommendations)

        # Apply artist diversity AFTER enrichment (when we have real artist names)
        print(f"[DEBUG] Before deduplication: {len(recommendations)} recommendations", flush=True)
        print(f"[DEBUG] Artists before dedup: {[r.get('artist', 'Unknown')[:30] for r in recommendations[:10]]}", flush=True)
        recommendations = self._deduplicate_by_artist(recommendations, max_per_artist=2, limit=limit)
        print(f"[DEBUG] After deduplication: {len(recommendations)} recommendations", flush=True)
        print(f"[DEBUG] Artists after dedup: {[r.get('artist', 'Unknown')[:30] for r in recommendations[:10]]}", flush=True)

        if recommendations:
            return recommendations, "playlist", playlist_info

        fallback_recommendations = await self._apple_playlist_fallback(normalised_inputs, limit=limit)
        if fallback_recommendations:
            fallback_recommendations = await self._ensure_media_assets(fallback_recommendations)
            return fallback_recommendations, "apple-playlist", playlist_info

        return recommendations, "playlist", playlist_info

    async def import_playlist_from_url(
        self,
        playlist_ref: str,
        *,
        limit: int = 300,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Resolve a Spotify playlist URL/ID into playlist entries for the UI."""

        if not self.spotify:
            raise RuntimeError("Spotify client not configured for playlist import")

        try:
            result = await self.spotify.fetch_playlist_tracks_open(playlist_ref, limit=limit)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(result.get("tracks", [])):
            spotify_id = track.get("id")
            if not spotify_id:
                continue

            raw = track.get("name") or ""
            artist = track.get("artist") or ""
            if raw and artist:
                raw = f"{raw} - {artist}"

            entry = {
                "raw": raw or spotify_id,
                "spotify_id": spotify_id,
                "name": track.get("name"),
                "artist": artist or None,
                "album": track.get("album"),
                "image_url": track.get("image_url"),
                "uri": f"spotify:track:{spotify_id}",
                "url": (track.get("external_urls") or {}).get("spotify"),
                "seed": idx == 0,
            }
            tracks.append(entry)

        if not tracks:
            raise RuntimeError("Playlist does not contain any playable tracks")

        summary = result.get("playlist") or {}
        summary.setdefault("total_tracks", len(tracks))
        summary["loaded_tracks"] = len(tracks)

        return tracks, summary

    def track_user_interaction(
        self,
        user_id: str,
        track_id: str,
        interaction_type: str,
        track_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Interactions are logged for analytics; recommendations are context-driven only."""
        logger.debug(
            "User %s performed %s on %s", user_id, interaction_type, track_id
        )

    def save_recommendation_state(self) -> None:
        """Save recommendation engine state."""
        logger.debug("Contextual engine is stateless; nothing to persist")

    def _deduplicate_by_artist(
        self,
        tracks: List[Dict[str, Any]],
        max_per_artist: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Apply artist diversity after enrichment with real metadata."""
        logger.info(f"🔍 _deduplicate_by_artist called with {len(tracks)} tracks, limit={limit}")

        artist_counts = {}
        diverse_tracks = []

        # Count all artists first for debugging
        all_artists = {}
        for track in tracks:
            artist = track.get("artist", "Unknown")
            all_artists[artist] = all_artists.get(artist, 0) + 1

        logger.info(f"🎵 Artist distribution in candidates: {all_artists}")

        for track in tracks:
            artist = track.get("artist", "Unknown")
            count = artist_counts.get(artist, 0)

            if count < max_per_artist:
                diverse_tracks.append(track)
                artist_counts[artist] = count + 1

                if len(diverse_tracks) >= limit:
                    break

        logger.info(f"✅ After deduplication: {len(diverse_tracks)} tracks with artists: {artist_counts}")
        return diverse_tracks

    async def _enrich_with_spotify_metadata(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch real Spotify metadata for ML-recommended tracks (replaces stub data)."""
        if not tracks or not self.spotify:
            return tracks

        logger.info(f"🔧 _enrich_with_spotify_metadata called with {len(tracks)} tracks")

        # Identify tracks that need Spotify data (stub tracks with fake artists)
        track_ids_to_fetch = []
        for track in tracks:
            artist = track.get("artist", "")
            logger.debug(f"Track {track.get('id')[:20]}... has artist: {artist}")
            # Check if it's a stub track (fake artist name)
            if artist.startswith("Artist_") or artist == track.get("id"):
                track_ids_to_fetch.append(track["id"])

        logger.info(f"🎯 Found {len(track_ids_to_fetch)} stub tracks to enrich")

        if not track_ids_to_fetch:
            return tracks

        # Fetch real Spotify data in bulk
        logger.info(f"🌐 Fetching Spotify metadata for {len(track_ids_to_fetch)} ML-recommended tracks")
        spotify_tracks = await self.spotify.get_tracks_bulk(track_ids_to_fetch)

        # Replace stub data with real Spotify data
        enriched_tracks = []
        for track in tracks:
            track_id = track.get("id")
            if track_id in spotify_tracks:
                # Replace with real Spotify track data
                real_track = spotify_tracks[track_id]
                # Preserve the recommendation metadata
                real_track["recommendation"] = track.get("recommendation")
                real_track["i2v_embedding"] = track.get("i2v_embedding")
                enriched_tracks.append(real_track)
            else:
                enriched_tracks.append(track)

        return enriched_tracks

    async def _ensure_media_assets(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not tracks or not self.apple:
            return tracks

        tasks = []
        for track in tracks:
            if track.get("provider") == "apple":
                continue
            if track.get("preview_url") and track.get("image_url"):
                continue
            tasks.append(self.apple.enrich_track(track))

        if tasks:
            await asyncio.gather(*tasks)
        return tracks

    def _build_playlist_info(
        self,
        *,
        total_tracks: int,
        resolved_tracks: int,
        profile_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        info = {
            "playlist_size": total_tracks,
            "resolved_tracks": resolved_tracks,
            "top_moods": [],
            "top_activities": [],
            "top_genres": [],
            "regions": [],
            "time_of_day": None,
            "energy_level": None,
            "tempo": None,
            "era": None,
        }

        if profile_context:
            info["top_moods"] = (profile_context.get("moods") or [])[:3]
            info["top_activities"] = (profile_context.get("activities") or [])[:3]
            info["top_genres"] = (profile_context.get("genres") or [])[:5]
            info["regions"] = (profile_context.get("regions") or [])[:3]
            info["time_of_day"] = profile_context.get("time_of_day")
            info["energy_level"] = profile_context.get("energy_level")
            info["tempo"] = profile_context.get("tempo")
            info["era"] = profile_context.get("era")

        return info

    async def _apple_playlist_fallback(
        self,
        playlist_inputs: List[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.apple or not playlist_inputs:
            return []

        limit = max(1, limit)
        playlist_keys: set[Tuple[str, str]] = set()
        for item in playlist_inputs:
            name = (item.get("name") or "").casefold()
            artist = (item.get("artist") or "").casefold()
            if name:
                playlist_keys.add((name, artist))

        seen_tracks: set[str] = set()
        seen_pairs: set[Tuple[str, str]] = set()
        recommendations: List[Dict[str, Any]] = []

        unique_artists: List[str] = []
        seen_artists: set[str] = set()
        for item in playlist_inputs:
            artist = item.get("artist")
            if artist:
                key = artist.casefold()
                if key not in seen_artists:
                    seen_artists.add(key)
                    unique_artists.append(artist)

        for item in playlist_inputs:
            track_name = item.get("name")
            artist_name = item.get("artist")
            if not track_name and not artist_name:
                continue

            candidates = await self.apple.get_related_tracks(
                track_name=track_name,
                artist_name=artist_name,
                limit=min(limit * 2, 12),
                exclude_ids=seen_tracks,
            )

            for candidate in candidates:
                candidate_id = str(candidate.get("id"))
                pair = (
                    (candidate.get("name") or "").casefold(),
                    (candidate.get("artist") or "").casefold(),
                )

                if not pair[0]:
                    continue
                if pair in playlist_keys or pair in seen_pairs:
                    continue
                if candidate_id in seen_tracks:
                    continue

                seen_tracks.add(candidate_id)
                seen_pairs.add(pair)
                recommendations.append(candidate)

                if len(recommendations) >= limit:
                    return recommendations[:limit]

        for artist in unique_artists:
            candidates = await self.apple.search_tracks(f"{artist} essentials", limit=min(limit * 2, 20))
            for candidate in candidates:
                candidate_id = str(candidate.get("id"))
                pair = (
                    (candidate.get("name") or "").casefold(),
                    (candidate.get("artist") or "").casefold(),
                )

                if not pair[0]:
                    continue
                if pair in playlist_keys or pair in seen_pairs:
                    continue
                if candidate_id in seen_tracks:
                    continue

                seen_tracks.add(candidate_id)
                seen_pairs.add(pair)
                recommendations.append(candidate)

                if len(recommendations) >= limit:
                    return recommendations[:limit]

        return recommendations[:limit]

    @staticmethod
    def _parse_seed(seed: str) -> Tuple[str, str]:
        if not seed:
            return "", ""

        parts = seed.split(":")
        if len(parts) >= 2:
            scheme = parts[0].lower()
            identifier = parts[-1]
            if scheme == "spotify":
                return "spotify", identifier
            if scheme in {"itunes", "apple"}:
                return "apple", identifier
        return "", seed

    @staticmethod
    def _post_process_recommendations(
        tracks: List[Dict[str, Any]],
        *,
        seed_track_id: Optional[str],
        seed_name: Optional[str],
        seed_artist: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Deduplicate results and remove the seed track itself."""
        if not tracks:
            return []

        normalized_seed_name = seed_name.lower().strip() if seed_name else None
        normalized_seed_artist = seed_artist.lower().strip() if seed_artist else None
        seed_track_id = seed_track_id or ""

        unique: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for track in tracks:
            track_id = str(track.get("id")) if track.get("id") is not None else ""
            if not track_id:
                continue

            if seed_track_id and track_id == seed_track_id:
                continue

            if track_id in seen_ids:
                continue

            name = track.get("name", "").lower().strip()
            artist = track.get("artist", "").lower().strip()

            if normalized_seed_name:
                same_name = name == normalized_seed_name
                same_artist = normalized_seed_artist and artist == normalized_seed_artist
                if same_name and (not normalized_seed_artist or same_artist):
                    # Skip exact match to the seed
                    continue

            unique.append(track)
            seen_ids.add(track_id)

            if len(unique) >= limit:
                break

        return unique

    @staticmethod
    def _extract_spotify_id(value: Optional[str]) -> Optional[str]:
        if not value:
            return None

        candidate = value.strip()
        if not candidate:
            return None

        if candidate.startswith("spotify:track:"):
            return candidate.split(":")[-1]

        if "open.spotify.com/track/" in candidate:
            fragment = candidate.split("/track/")[-1]
            fragment = fragment.split("?")[0].split("#")[0]
            return fragment or None

        candidate = candidate.split("?")[0].split("#")[0]
        if len(candidate) == 22 and candidate.replace("-", "").isalnum():
            return candidate

        return None

    @staticmethod
    def _split_track_line(raw: str) -> Tuple[Optional[str], Optional[str]]:
        text = raw.strip()
        if not text:
            return None, None

        separators = [
            " – ",
            " — ",
            " - ",
            " | ",
            " : ",
            " by ",
        ]

        for sep in separators:
            if sep in text:
                left, right = text.split(sep, 1)
                return left.strip() or None, right.strip() or None

        if "-" in text:
            left, right = text.split("-", 1)
            return left.strip() or None, right.strip() or None

        return text, None

    def _normalise_playlist_inputs(self, tracks_payload: List[Any]) -> List[Dict[str, Any]]:
        normalised: List[Dict[str, Any]] = []

        for item in tracks_payload:
            if isinstance(item, str):
                descriptor: Dict[str, Any] = {"raw": item}
            elif isinstance(item, dict):
                descriptor = {k: v for k, v in item.items()}
            else:
                continue

            raw_value = descriptor.get("raw")
            spotify_id = (
                descriptor.get("spotify_id")
                or descriptor.get("id")
                or self._extract_spotify_id(descriptor.get("uri"))
                or self._extract_spotify_id(descriptor.get("url"))
            )

            if not spotify_id and isinstance(raw_value, str):
                spotify_id = self._extract_spotify_id(raw_value)

            if spotify_id:
                descriptor["spotify_id"] = spotify_id

            name = descriptor.get("name")
            artist = descriptor.get("artist")

            if isinstance(raw_value, str):
                guessed_name, guessed_artist = self._split_track_line(raw_value)
                name = name or guessed_name
                artist = artist or guessed_artist

            descriptor["name"] = name
            descriptor["artist"] = artist
            descriptor["seed"] = bool(descriptor.get("seed"))

            normalised.append(descriptor)

        return normalised
