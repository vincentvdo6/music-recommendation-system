# Hybrid Recommendation Engine Documentation

## Overview

This music recommendation system implements a state-of-the-art hybrid approach combining collaborative filtering, content-based methods, and context-aware personalization. The architecture is based on research from industry leaders like Spotify, Pandora, YouTube Music, and SoundCloud.

## Architecture

### Three-Pillar Hybrid System

1. **Collaborative Filtering**
   - User-user similarity based on listening patterns
   - Jaccard similarity for finding similar users
   - Weighted recommendations from users with similar taste
   - Tracks user interactions: plays, likes, skips

2. **Content-Based Filtering**
   - Audio feature analysis using Spotify's features:
     - Valence (mood positivity) - weight 1.2
     - Energy (intensity) - weight 1.1
     - Danceability - weight 1.0
     - Acousticness - weight 0.9
     - Instrumentalness - weight 0.8
     - Speechiness - weight 0.7
     - Tempo - weight 0.6
     - Loudness - weight 0.4
   - User taste profile aggregation from listening history
   - Exponential decay similarity function for natural distribution

3. **Context-Aware Personalization**
   - Time of day matching (morning/afternoon/evening/night)
   - Mood-based filtering (upbeat/calm/energetic)
   - Session pattern recognition

### Two-Stage Ranking

#### Stage 1: Candidate Generation (Fast, Broad)
- Collaborative filtering recommendations (from similar users)
- Content-based recommendations (from Spotify's algorithm)
- Popular tracks (fallback)
- Targets ~100 candidates for efficiency

#### Stage 2: Hybrid Scoring & Re-ranking (Precise, Detailed)
- Collaborative score (40% weight)
- Content similarity score (40% weight)
- Popularity score (10% weight)
- Context match score (10% weight)
- Diversity injection to prevent filter bubble

## API Usage

### Get Personalized Recommendations

```bash
GET /api/v1/recommendations?seed=spotify:track:abc123&user_id=user_xyz&limit=10&time_of_day=evening&mood=calm
```

**Parameters:**
- `seed` (required): Track URI or ID to base recommendations on
- `user_id` (optional): User ID for personalized recommendations
- `limit` (optional): Number of recommendations (1-20, default 5)
- `track_name` (optional): Seed track name for non-Spotify IDs
- `artist_name` (optional): Seed artist name
- `time_of_day` (optional): Context - morning/afternoon/evening/night
- `mood` (optional): Context - upbeat/calm/energetic

**Response:**
```json
{
  "seed": "spotify:track:abc123",
  "total": 10,
  "algorithm": "Hybrid (Collaborative + Content-Based + Context-Aware)",
  "source": "hybrid",
  "recommendations": [
    {
      "id": "track_id",
      "name": "Song Name",
      "artist": "Artist Name",
      "similarity_score": 0.92,
      "audio_features": {...},
      "explanation": {
        "top_factors": ["High energy", "Upbeat mood"],
        "similarity_reason": "Based on audio features and listening patterns"
      }
    }
  ]
}
```

### Track User Interactions

To improve personalization, track user interactions:

```bash
POST /api/v1/track-interaction?user_id=user_xyz&track_id=abc123&interaction=play&track_name=Song&artist_name=Artist
```

**Interaction Types:**
- `play`: User listened to the track
- `like`: User explicitly liked the track
- `skip`: User skipped the track

## How It Works

### User Profile Building

1. **Listening History**: Stores last 500 plays per user
2. **Taste Profile**: Aggregates audio features from recent plays (last 50 tracks)
3. **Preference Learning**: Learns patterns like:
   - Preferred artists
   - Typical listening times
   - Energy/mood preferences
   - Skip patterns

### Similarity Calculation

**Audio Feature Similarity:**
```python
# Weighted Euclidean distance with exponential decay
distance = sqrt(sum((feature1 - feature2)^2 * weight) / total_weight)
similarity = exp(-2.0 * distance)
```

**User Similarity (Collaborative Filtering):**
```python
# Jaccard similarity between user listening sets
intersection = len(user1_tracks & user2_tracks)
union = len(user1_tracks | user2_tracks)
similarity = intersection / union
```

### Hybrid Score Formula

```python
final_score = (
    collaborative_score * 0.4 +
    content_score * 0.4 +
    popularity_score * 0.1 +
    context_score * 0.1
)
```

### Diversity Mechanism

To prevent filter bubbles, the system:
- Takes top-scoring candidate
- Alternates between:
  - 70% probability: next highest-scoring track
  - 30% probability: diverse pick from middle range
- Ensures variety while maintaining relevance

## Cold Start Handling

### New Users
- Default to popular tracks
- Use audio features once first track is played
- Build collaborative profile after 2+ tracks in common with other users

### New Tracks
- Use audio features for content-based matching
- Leverage Spotify's recommendation API
- Track initial plays to build collaborative signal

## Performance Optimizations

1. **Caching**
   - Spotify search cache: 2000 entries, 5min TTL
   - Audio features cache: 10000 entries, 2hrs TTL
   - Batch audio feature fetching (up to 100 tracks at once)

2. **Persistence**
   - User profiles saved to disk on shutdown
   - Automatic loading on startup
   - JSON format for easy inspection

3. **Two-Stage Architecture**
   - Stage 1 generates ~100 candidates quickly
   - Stage 2 only scores/ranks those 100
   - Prevents expensive computation on full catalog

## Data Storage

User data is stored in `data/user_profiles/profiles.json`:
```json
{
  "user_plays": {
    "user_id": [
      {
        "track_id": "abc123",
        "track_name": "Song",
        "artist": "Artist",
        "timestamp": "2025-01-15T10:30:00",
        "provider": "spotify"
      }
    ]
  },
  "user_likes": {"user_id": ["track_id1", "track_id2"]},
  "user_skips": {"user_id": ["track_id3"]},
  "track_plays": {"track_id": 42},
  "track_users": {"track_id": ["user1", "user2"]}
}
```

## Research Foundation

This implementation is based on techniques from:

- **Spotify**: Audio feature analysis, hybrid collaborative + content filtering, ANN search
- **Pandora**: Music Genome Project approach, detailed audio attributes
- **YouTube Music**: Transformer-based sequential modeling, context awareness
- **SoundCloud**: Cold start solutions using audio AI analysis

Key papers and resources:
- Collaborative filtering with matrix factorization
- CNN-based audio feature extraction
- RNN/Transformer sequential recommendation
- Hybrid recommender systems
- ANN search for real-time retrieval (Annoy, HNSW)

## Future Enhancements

Potential improvements:
1. **ANN Search**: Implement HNSW for faster similarity lookups at scale
2. **Deep Learning**: Train custom audio embedding models
3. **Graph-Based CF**: Use graph neural networks for user-item interactions
4. **Bandit Algorithms**: Balance exploration vs exploitation
5. **Session-Based**: RNN/Transformer for sequence modeling
6. **Explicit Feedback**: Incorporate user ratings
7. **Multi-Armed Bandits**: A/B testing for algorithm improvements
