"""User profile and listening history tracking for personalized recommendations."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class UserProfile:
    """Tracks user listening history and preferences for collaborative filtering."""

    def __init__(self, storage_path: str = "data/user_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory data structures for fast access
        self.user_plays: Dict[str, List[Dict]] = defaultdict(list)
        self.user_likes: Dict[str, Set[str]] = defaultdict(set)
        self.user_skips: Dict[str, Set[str]] = defaultdict(set)
        self.track_plays: Dict[str, int] = defaultdict(int)
        self.track_users: Dict[str, Set[str]] = defaultdict(set)

        # Load existing data
        self._load_profiles()

    def track_play(self, user_id: str, track_id: str, track_data: Dict,
                   duration_played_ms: Optional[int] = None) -> None:
        """Record that a user played a track."""
        interaction = {
            "track_id": track_id,
            "track_name": track_data.get("name"),
            "artist": track_data.get("artist"),
            "timestamp": datetime.utcnow().isoformat(),
            "duration_ms": duration_played_ms,
            "provider": track_data.get("provider"),
        }

        self.user_plays[user_id].append(interaction)
        self.track_plays[track_id] += 1
        self.track_users[track_id].add(user_id)

        # Keep only last 500 plays per user to prevent unbounded growth
        if len(self.user_plays[user_id]) > 500:
            self.user_plays[user_id] = self.user_plays[user_id][-500:]

    def track_like(self, user_id: str, track_id: str) -> None:
        """Record that a user liked a track."""
        self.user_likes[user_id].add(track_id)

        # Remove from skips if previously skipped
        self.user_skips[user_id].discard(track_id)

    def track_skip(self, user_id: str, track_id: str) -> None:
        """Record that a user skipped a track."""
        self.user_skips[user_id].add(track_id)

        # Keep only last 200 skips to prevent unbounded growth
        if len(self.user_skips[user_id]) > 200:
            skip_list = list(self.user_skips[user_id])
            self.user_skips[user_id] = set(skip_list[-200:])

    def get_user_taste_profile(self, user_id: str) -> Dict:
        """Get aggregated taste profile for a user based on listening history."""
        if user_id not in self.user_plays:
            return {}

        plays = self.user_plays[user_id]
        recent_plays = plays[-50:]  # Focus on recent behavior

        # Extract patterns
        top_artists = defaultdict(int)
        play_times = []

        for play in recent_plays:
            if play.get("artist"):
                top_artists[play["artist"]] += 1
            if play.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(play["timestamp"])
                    play_times.append(ts.hour)
                except Exception:
                    pass

        # Determine preferred listening time
        preferred_time = None
        if play_times:
            avg_hour = sum(play_times) / len(play_times)
            if 6 <= avg_hour < 12:
                preferred_time = "morning"
            elif 12 <= avg_hour < 17:
                preferred_time = "afternoon"
            elif 17 <= avg_hour < 22:
                preferred_time = "evening"
            else:
                preferred_time = "night"

        return {
            "total_plays": len(plays),
            "recent_plays": len(recent_plays),
            "top_artists": dict(sorted(top_artists.items(), key=lambda x: x[1], reverse=True)[:10]),
            "liked_tracks": len(self.user_likes[user_id]),
            "preferred_time": preferred_time,
        }

    def get_similar_users(self, user_id: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Find users with similar listening patterns (collaborative filtering)."""
        if user_id not in self.user_plays:
            return []

        user_tracks = {play["track_id"] for play in self.user_plays[user_id][-100:]}
        if not user_tracks:
            return []

        # Calculate Jaccard similarity with other users
        similarities = []
        for other_user_id in self.user_plays:
            if other_user_id == user_id:
                continue

            other_tracks = {play["track_id"] for play in self.user_plays[other_user_id][-100:]}
            if not other_tracks:
                continue

            intersection = len(user_tracks & other_tracks)
            union = len(user_tracks | other_tracks)

            if union > 0 and intersection >= 2:  # At least 2 tracks in common
                similarity = intersection / union
                similarities.append((other_user_id, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]

    def get_collaborative_recommendations(self, user_id: str, limit: int = 50) -> List[str]:
        """Get track recommendations based on similar users (collaborative filtering)."""
        similar_users = self.get_similar_users(user_id, limit=10)
        if not similar_users:
            return []

        user_tracks = {play["track_id"] for play in self.user_plays[user_id]}
        skipped_tracks = self.user_skips[user_id]

        # Collect tracks from similar users, weighted by similarity
        track_scores = defaultdict(float)
        for similar_user_id, similarity in similar_users:
            for play in self.user_plays[similar_user_id][-50:]:  # Recent plays from similar users
                track_id = play["track_id"]

                # Skip tracks user already heard or explicitly skipped
                if track_id in user_tracks or track_id in skipped_tracks:
                    continue

                track_scores[track_id] += similarity

        # Sort by score and return top candidates
        ranked = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        return [track_id for track_id, _ in ranked[:limit]]

    def get_popular_tracks(self, limit: int = 50, min_plays: int = 3) -> List[str]:
        """Get globally popular tracks as fallback."""
        popular = [(tid, count) for tid, count in self.track_plays.items() if count >= min_plays]
        popular.sort(key=lambda x: x[1], reverse=True)
        return [track_id for track_id, _ in popular[:limit]]

    def save_profiles(self) -> None:
        """Persist user profiles to disk."""
        try:
            data = {
                "user_plays": {uid: plays for uid, plays in self.user_plays.items()},
                "user_likes": {uid: list(likes) for uid, likes in self.user_likes.items()},
                "user_skips": {uid: list(skips) for uid, skips in self.user_skips.items()},
                "track_plays": dict(self.track_plays),
                "track_users": {tid: list(users) for tid, users in self.track_users.items()},
            }

            filepath = self.storage_path / "profiles.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("Saved user profiles to %s", filepath)
        except Exception as exc:
            logger.error("Failed to save user profiles: %s", exc)

    def _load_profiles(self) -> None:
        """Load user profiles from disk."""
        try:
            filepath = self.storage_path / "profiles.json"
            if not filepath.exists():
                logger.info("No existing profiles found, starting fresh")
                return

            with open(filepath, "r") as f:
                data = json.load(f)

            self.user_plays = defaultdict(list, data.get("user_plays", {}))
            self.user_likes = defaultdict(set, {
                uid: set(likes) for uid, likes in data.get("user_likes", {}).items()
            })
            self.user_skips = defaultdict(set, {
                uid: set(skips) for uid, skips in data.get("user_skips", {}).items()
            })
            self.track_plays = defaultdict(int, data.get("track_plays", {}))
            self.track_users = defaultdict(set, {
                tid: set(users) for tid, users in data.get("track_users", {}).items()
            })

            logger.info("Loaded user profiles from %s", filepath)
        except Exception as exc:
            logger.error("Failed to load user profiles: %s", exc)
