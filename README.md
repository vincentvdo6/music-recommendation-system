# 🎵 Music Recommendation System

A simple music recommendation web application powered by Spotify's database and audio features.

## ✨ Features

- 🔍 **Song Search**: Search for any song or artist from Spotify's catalog
- 🎯 **Smart Recommendations**: Get personalized recommendations based on audio features
- 🎨 **Beautiful Interface**: Modern, responsive web design
- 🎧 **Real Data**: Uses actual Spotify tracks, not mock data
- 📊 **Audio Analysis**: Shows tempo, energy, danceability, and mood
- 🎯 **Smart Analysis**: Understand why each song was recommended

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

1. **Search for a song**: Type any song name or artist (e.g., "Bohemian Rhapsody", "Ed Sheeran")
2. **Select a track**: Click on any song from the search results  
3. **Get recommendations**: Click "Get Recommendations" to find similar songs
4. **Explore**: See why each song was recommended and click Spotify links to listen

## 📁 Project Structure

```
Music_Recommendation/
├── api/
│   ├── main.py              # FastAPI application
│   └── routers/
│       └── search.py        # Search and recommendations API
├── services/
│   └── spotify/
│       └── client.py        # Spotify Web API client
├── static/
│   └── index.html          # Web interface
├── requirements.txt        # Python dependencies
├── start.bat              # Windows startup script
└── run_local.py           # Python startup script
```

## 🛠️ Requirements

- **Python 3.11+** 
- **Dependencies**: FastAPI, Uvicorn, HTTPX (auto-installed)

## 🎯 What Makes It Special

- **Real Music Data**: Uses Spotify's actual database of songs
- **Audio Features**: Analyzes tempo, energy, key, mood, and danceability  
- **Smart Matching**: Finds similar songs based on musical characteristics
- **Clean & Simple**: Minimal codebase, easy to understand and modify
- **No API Keys Needed**: Works in demo mode with mock data when no Spotify credentials

## 🔧 Customization

To use real Spotify API (optional):
1. Get Spotify API credentials from https://developer.spotify.com/
2. Add them to `services/spotify/client.py`
3. Replace `client_id = "demo"` with your actual credentials

## 📝 License

MIT License - feel free to use and modify!