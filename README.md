# 🎵 Music Recommendation System

A contextual music discovery web app that blends curated knowledge with optional
Spotify data. The recommendation engine no longer depends on user history—it
works instantly using mood, activity, and other situational hints.

## ✨ Features

- 📋 **Playlist-First Personalisation**: Paste any playlist to build a one-time profile
- 🧭 **Context-Aware Engine**: Blends playlist-derived tastes with curated catalogue tags
- 🧠 **Transparent Scoring**: Every recommendation exposes its score breakdown
- 📊 **Insightful Summaries**: See the moods, activities, tempo, and era detected from your playlist
- 📴 **Offline Ready**: Works with the built-in catalogue even without Spotify access

## 🚀 Quick Start

### Option 1: Batch File (Windows)
```bash
start.bat
```

### Option 2: Python Script
```bash
python run_local.py
```

The server will start automatically and open your browser to http://localhost:8000

## 🎶 How to Use

1. **Paste your playlist**: Drop Spotify links or `Song – Artist` lines into the text box.
2. **Submit once**: The backend confirms the playlist was resolved and builds a profile.
3. **Review the summary**: See the detected moods, activities, energy, tempo, and era.
4. **Play recommendations**: Explore the returned tracks—each comes from the curated catalogue and excludes your pasted songs.

## 📁 Project Structure

```
Music_Recommendation/
├── api/
│   ├── main.py                  # FastAPI application entrypoint
│   └── routers/
│       └── search.py            # Search and recommendation endpoints
├── services/
│   ├── music/
│   │   └── service.py           # Orchestrates providers + contextual engine
│   ├── recommendation/
│   │   ├── audio_similarity.py  # Feature-based similarity utilities
│   │   ├── catalogue.py         # Curated track catalogue loader
│   │   └── contextual_engine.py # Context-first recommendation engine
│   └── spotify/
│       └── client.py            # Spotify Web API client (optional)
├── data/
│   └── catalogue/tracks.json    # Built-in recommendation knowledge base
├── static/
│   └── index.html               # Web interface
├── requirements.txt             # Python dependencies
├── start.bat                    # Windows startup script
└── run_local.py                 # Python startup script
```

## 🛠️ Requirements

- **Python 3.11+** 
- **Dependencies**: FastAPI, Uvicorn, HTTPX (auto-installed)

## 🎯 What Makes It Special

- **Context-first**: Accurate recommendations from the first request—no history required
- **Curated Knowledge**: Tunable dataset with rich tags for explainable results
- **Audio Features**: Uses Spotify-style feature vectors for musical similarity
- **Transparent**: Recommendation payloads include score components for debugging
- **Flexible Setup**: Runs fully offline using the catalogue and upgrades seamlessly with Spotify

## 🔧 Configuration (secure)

1. Create a `.env` file based on `.env.example`:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
ALLOWED_ORIGIN=http://localhost:8000
```

2. Never hardcode secrets in code. The app reads them from env via `python-dotenv`.

## 📝 License

MIT License - feel free to use and modify!
